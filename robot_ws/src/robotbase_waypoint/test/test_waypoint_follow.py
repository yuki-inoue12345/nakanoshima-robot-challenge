"""Focused tests for waypoint follower protocol helpers."""

from array import array
import ast
from pathlib import Path
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
import numpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from robotbase_waypoint.waypoint_follow import (
    goal_uuids_equal,
    GoalRecord,
    person_area_arrival_qos,
    person_search_response_qos,
    waypoint_arrival_threshold,
    WaypointFollower,
)
from robotbase_waypoint.waypoint_io import Waypoint


def test_goal_uuids_equal_accepts_numpy_arrays():
    """Verify NumPy equality does not leak an ambiguous truth-valued array."""
    left = numpy.arange(16, dtype=numpy.uint8)
    right = numpy.arange(16, dtype=numpy.uint8)

    assert goal_uuids_equal(left, right)


def test_goal_uuids_equal_handles_ros_sequence_variants():
    """List and array representations of the same ROS UUID compare equal."""
    values = list(range(16))

    assert goal_uuids_equal(values, array('B', values))
    assert not goal_uuids_equal(values, [*range(15), 99])


def make_waypoint(**changes):
    """Create a waypoint with concise test defaults."""
    values = {
        'number': 1,
        'x': 1.0,
        'y': 0.0,
        'angle_radians': 0.0,
    }
    values.update(changes)
    return Waypoint(**values)


class FakeTime:
    """Minimal ROS-time stand-in for deterministic timeout tests."""

    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


class FakeClock:
    def __init__(self, seconds=0.0):
        self.set_time(seconds)

    def set_time(self, seconds):
        self.current = FakeTime(seconds)

    def now(self):
        return self.current


def make_person_search_follower(*, now=0.0, waiting=True):
    """Create a callback-only follower without ROS publishers or navigation."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.clock = FakeClock(now)
    follower.get_clock = lambda: follower.clock
    follower.index = 0
    follower.waypoints = [
        make_waypoint(person_area=True),
        make_waypoint(number=2, x=2.0),
    ]
    follower.person_search_timeout_sec = 60.0
    follower.person_search_ack_timeout_sec = 5.0
    follower.person_search_waiting = waiting
    follower.person_search_faulted = False
    follower.person_search_fault_reason = ''
    follower.person_search_goal_index = 0 if waiting else None
    follower.person_search_started_at = FakeTime(now) if waiting else None
    follower.person_search_session_id = 1 if waiting else 0
    follower.person_search_arrival_acknowledged = False
    follower.person_search_ack_received_at = None
    follower.person_search_last_detail = ''
    follower.holding_stop = True
    follower.finished = False
    follower.events = []
    follower.logs = []
    follower._is_shutting_down = lambda: False

    def commit_state(**changes):
        follower.__dict__.update(changes)
        return True

    follower._commit_state = commit_state
    follower._publish_person_area_arrival = (
        lambda value: follower.events.append(('arrival', value)))
    follower._publish_safety_zero = lambda: follower.events.append(('zero', None))
    follower._publish_stop = lambda value: (
        follower.events.append(('stop', value)) or True)
    follower._advance = lambda: follower.events.append(('advance', None))
    follower._complete_or_loop = lambda: follower.events.append(('complete', None))
    follower.get_logger = lambda: SimpleNamespace(
        info=lambda message, **_kwargs: follower.logs.append(('info', message)),
        warning=lambda message, **_kwargs: follower.logs.append(('warning', message)),
        error=lambda message, **_kwargs: follower.logs.append(('error', message)),
    )
    return follower


def test_arrival_threshold_prefers_waypoint_value_then_default():
    """Use a waypoint threshold when present and the route default otherwise."""
    assert waypoint_arrival_threshold(
        make_waypoint(threshold=0.7), 1.0, 0.35) == 0.7
    assert waypoint_arrival_threshold(make_waypoint(), 1.0, 0.35) == 1.0


def make_timer_follower(waypoints):
    """Create the state needed to exercise one distance-check timer tick."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.index = 0
    follower.waypoints = waypoints
    follower.default_threshold = 1.0
    follower.precise_threshold = 0.35
    follower.navigation_only = False
    follower.map_frame = 'map'
    follower.base_frame = 'base_footprint'
    follower.goal_handle = object()
    follower.waiting_until = None
    follower.person_area_stop_pending = False
    follower.holding_stop = False
    follower.finished = False
    follower.person_search_waiting = False
    follower.person_search_faulted = False
    follower.transition_in_progress = False
    follower._is_shutting_down = lambda: False
    follower.get_logger = lambda: SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    translation = SimpleNamespace(x=0.0, y=0.0)
    transform = SimpleNamespace(
        transform=SimpleNamespace(translation=translation))
    follower.tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args, **_kwargs: transform)
    return follower


def test_timer_switches_normal_intermediate_waypoint_continuously():
    """Entering a normal intermediate threshold dispatches an in-motion advance."""
    follower = make_timer_follower([
        make_waypoint(x=0.5, threshold=0.6),
        make_waypoint(number=2, x=2.0),
    ])
    calls = []
    follower._advance = lambda: calls.append('advance')
    follower._arrive_at_waypoint = lambda **_kwargs: calls.append('arrive')

    follower._timer_callback()

    assert calls == ['advance']


def test_timer_routes_stop_and_wait_waypoints_to_stopping_arrival():
    """Stop and wait markers retain the cancel-and-hold arrival path."""
    for marker in (
            make_waypoint(x=0.2, stop=True),
            make_waypoint(x=0.2, wait_time=2.0),
    ):
        follower = make_timer_follower([
            marker,
            make_waypoint(number=2, x=2.0),
        ])
        calls = []
        follower._advance = lambda: calls.append('advance')
        follower._arrive_at_waypoint = (
            lambda **kwargs: calls.append(('arrive', kwargs['exact_arrival'])))

        follower._timer_callback()

        assert calls == [('arrive', False)]


def test_advance_sends_next_goal_without_cancel_or_zero():
    """A continuous transition only advances state and dispatches the next goal."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.index = 0
    follower.goal_handle = object()
    follower.goal_result_future = object()
    follower._is_shutting_down = lambda: False
    calls = []

    def commit_state(**changes):
        follower.__dict__.update(changes)
        calls.append(('commit', changes))
        return True

    follower._commit_state = commit_state
    follower._send_current_goal = lambda: calls.append(('send', follower.index))
    follower._ensure_cancel_requested = lambda _record: calls.append(('cancel', None))
    follower._publish_safety_zero = lambda: calls.append(('zero', None))

    follower._advance()

    assert follower.index == 1
    assert calls[-1] == ('send', 1)
    assert not any(name in ('cancel', 'zero') for name, _value in calls)


def test_stop_and_wait_arrivals_keep_cancel_transition():
    """Stopping waypoint arrivals still cancel before entering their wait."""
    for waypoint in (
            make_waypoint(stop=True),
            make_waypoint(wait_time=2.0),
    ):
        follower = WaypointFollower.__new__(WaypointFollower)
        follower.navigation_mode = 'to_pose'
        follower.manual_paused = False
        follower.index = 0
        follower.waypoints = [waypoint, make_waypoint(number=2)]
        follower.goal_handle = object()
        follower.goal_serial = 4
        follower.cancel_transition_generation = 1
        follower.navigation_only = False
        follower.stop_duration = 5.0
        follower._is_shutting_down = lambda: False
        follower.get_clock = lambda: SimpleNamespace(now=lambda: object())
        follower._commit_state = lambda **changes: (
            follower.__dict__.update(changes) is None)
        record = GoalRecord(handle=follower.goal_handle)
        follower._record_for_handle = lambda _handle: record
        cancel_future = SimpleNamespace(add_done_callback=lambda callback: None)
        follower._ensure_cancel_requested = lambda _record: cancel_future

        follower._arrive_at_waypoint(exact_arrival=False)

        assert follower.transition_in_progress
        assert follower.goal_serial == 5
        assert record.handle is follower.goal_handle


def test_final_arrival_and_loop_restart_are_preserved():
    """Completing a loop resets to waypoint one and dispatches it again."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.loop = True
    follower.index = 2
    follower.goal_handle = object()
    follower._is_shutting_down = lambda: False
    follower.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    sent = []

    def commit_state(**changes):
        follower.__dict__.update(changes)
        return True

    follower._commit_state = commit_state
    follower._send_current_goal = lambda: sent.append(follower.index)

    follower._complete_or_loop()

    assert follower.index == 0
    assert follower.goal_handle is None
    assert sent == [0]


def test_final_waypoint_waits_for_current_nav2_success():
    """A normal final waypoint advances only from its current Nav2 result."""
    waypoint = make_waypoint(x=0.2)
    follower = make_timer_follower([waypoint])
    arrivals = []
    follower._advance = lambda: arrivals.append(('advance', None))
    follower._arrive_at_waypoint = (
        lambda **kwargs: arrivals.append(('arrive', kwargs['exact_arrival'])))

    follower._timer_callback()
    assert arrivals == []

    follower.goal_serial = 3
    follower.goal_records = {3: GoalRecord()}
    follower._goal_result(
        SimpleNamespace(
            result=lambda: SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED)),
        serial=3,
    )

    assert arrivals == [('arrive', True)]


def test_delayed_old_goal_response_cannot_replace_current_goal():
    """A late response records ownership but cannot overwrite current state."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    current_handle = object()
    current_result = object()
    follower.goal_serial = 2
    follower.goal_handle = current_handle
    follower.goal_result_future = current_result
    follower.goal_records = {1: GoalRecord()}
    follower._is_shutting_down = lambda: False
    cancel_calls = []
    follower._ensure_cancel_requested = lambda record: cancel_calls.append(record)

    old_result = SimpleNamespace(add_done_callback=lambda callback: None)
    old_handle = SimpleNamespace(
        accepted=True,
        goal_id=SimpleNamespace(uuid=list(range(16))),
        get_result_async=lambda: old_result,
    )
    response = SimpleNamespace(result=lambda: old_handle)

    follower._goal_response(response, serial=1)

    assert follower.goal_handle is current_handle
    assert follower.goal_result_future is current_result
    assert follower.goal_records[1].handle is old_handle
    assert cancel_calls == []


def test_delayed_old_goal_result_cannot_advance_waypoint():
    """A stale successful result is observed without triggering arrival."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.goal_serial = 2
    follower.goal_records = {1: GoalRecord()}
    follower.finished = False
    follower.waiting_until = None
    follower._is_shutting_down = lambda: False
    arrivals = []
    follower._arrive_at_waypoint = lambda **kwargs: arrivals.append(kwargs)
    result = SimpleNamespace(
        result=lambda: SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED))

    follower._goal_result(result, serial=1)

    assert follower.goal_records[1].terminal_status == GoalStatus.STATUS_SUCCEEDED
    assert arrivals == []


def test_person_search_does_not_advance_when_stop_release_fails():
    """Keep navigation stopped if the person-search stop lock cannot be released."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.navigation_mode = 'to_pose'
    follower.manual_paused = False
    follower.index = 0
    follower.waypoints = [
        make_waypoint(person_area=True),
        make_waypoint(number=2, x=2.0),
    ]
    follower.person_search_waiting = True
    follower.person_search_goal_index = 0
    follower.person_search_started_at = object()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_faulted = False
    follower.person_search_fault_reason = ''
    follower.holding_stop = True
    follower.finished = False
    follower._is_shutting_down = lambda: False

    events = []

    def commit_state(**changes):
        follower.__dict__.update(changes)
        return True

    def publish_stop(value):
        events.append(('stop', value))
        return bool(value)

    follower._commit_state = commit_state
    follower._publish_person_area_arrival = (
        lambda value: events.append(('arrival', value)))
    follower._publish_stop = publish_stop
    follower._publish_safety_zero = lambda: events.append(('zero', None))
    follower._advance = lambda: events.append(('advance', None))
    follower.get_logger = lambda: SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )

    follower._finish_person_search(True)

    assert ('stop', False) in events
    assert ('advance', None) not in events
    assert ('zero', None) in events
    assert ('stop', True) in events
    assert follower.holding_stop
    assert follower.person_search_faulted
    assert follower.person_search_fault_reason == 'navigation_release_failed'


def test_person_search_start_begins_unacknowledged_ack_wait():
    follower = make_person_search_follower(waiting=False)

    follower._activate_person_search()

    assert follower.person_search_waiting
    assert not follower.person_search_arrival_acknowledged
    assert follower.person_search_ack_received_at is None
    assert follower.person_search_started_at is follower.clock.current
    assert follower.events == [('arrival', True)]
    assert any('waiting for Camera arrival ACK' in text for _level, text in follower.logs)


def test_person_search_before_ack_timeout_does_not_fault():
    follower = make_person_search_follower()
    follower.clock.set_time(5.0)

    assert not follower._check_person_search_timeout()
    assert follower.person_search_waiting
    assert not follower.person_search_faulted


def test_person_search_without_ack_times_out_fail_closed():
    follower = make_person_search_follower()
    follower.clock.set_time(5.000001)

    assert follower._check_person_search_timeout()
    assert follower.person_search_fault_reason == 'person_search_ack_timeout'
    assert follower.holding_stop
    assert ('arrival', False) in follower.events
    assert ('zero', None) in follower.events
    assert ('stop', True) in follower.events
    assert ('stop', False) not in follower.events
    assert ('advance', None) not in follower.events


def test_acknowledged_search_does_not_use_ack_timeout():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(4.0)
    follower.clock.set_time(6.0)

    assert not follower._check_person_search_timeout()
    assert follower.person_search_waiting
    assert not follower.person_search_faulted


def test_first_true_ack_records_search_timeout_origin():
    follower = make_person_search_follower(now=3.0)

    follower._person_search_ack_callback(SimpleNamespace(data=True))

    assert follower.person_search_arrival_acknowledged
    assert follower.person_search_ack_received_at is follower.clock.current
    assert any('Camera arrival ACK received' in text for _level, text in follower.logs)
    assert any(
        'waiting for Camera safe-to-advance' in text
        for _level, text in follower.logs)


def test_search_timeout_is_not_raised_before_sixty_seconds_after_ack():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(4.0)
    follower.clock.set_time(64.0)

    assert not follower._check_person_search_timeout()
    assert follower.person_search_waiting


def test_search_timeout_is_raised_after_sixty_seconds_after_ack():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(4.0)
    follower.clock.set_time(64.000001)

    assert follower._check_person_search_timeout()
    assert follower.person_search_fault_reason == 'person_search_timeout'
    assert follower.holding_stop
    assert ('advance', None) not in follower.events


def test_duplicate_ack_does_not_reset_search_timeout_origin():
    follower = make_person_search_follower(now=2.0)
    follower._person_search_ack_callback(SimpleNamespace(data=True))
    first_ack_at = follower.person_search_ack_received_at
    follower.clock.set_time(10.0)

    follower._person_search_ack_callback(SimpleNamespace(data=True))

    assert follower.person_search_ack_received_at is first_ack_at
    assert any('duplicate Camera arrival ACK ignored' in text
               for _level, text in follower.logs)


def test_done_true_after_ack_completes_and_advances():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(1.0)

    follower._person_search_done_callback(SimpleNamespace(data=True))

    assert not follower.person_search_waiting
    assert not follower.person_search_faulted
    assert ('arrival', False) in follower.events
    assert ('stop', False) in follower.events
    assert ('advance', None) in follower.events


def test_done_false_after_ack_fails_closed():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(1.0)

    follower._person_search_done_callback(SimpleNamespace(data=False))

    assert follower.person_search_faulted
    assert follower.person_search_fault_reason == 'camera_reported_unsafe_completion'
    assert follower.holding_stop
    assert ('zero', None) in follower.events
    assert ('stop', True) in follower.events
    assert ('advance', None) not in follower.events


def test_person_search_completion_resets_ack_state_and_timestamp():
    for safe_to_advance in (True, False):
        follower = make_person_search_follower()
        follower.person_search_arrival_acknowledged = True
        follower.person_search_ack_received_at = FakeTime(1.0)

        follower._finish_person_search(safe_to_advance)

        assert not follower.person_search_arrival_acknowledged
        assert follower.person_search_ack_received_at is None
        assert follower.person_search_started_at is None


def test_next_session_does_not_inherit_previous_ack():
    follower = make_person_search_follower(waiting=False)
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(1.0)

    follower._activate_person_search()

    assert not follower.person_search_arrival_acknowledged
    assert follower.person_search_ack_received_at is None


def test_shutdown_callbacks_do_not_change_person_search_state():
    follower = make_person_search_follower()
    follower._is_shutting_down = lambda: True
    original = (
        follower.person_search_waiting,
        follower.person_search_arrival_acknowledged,
        follower.person_search_ack_received_at,
    )

    follower._person_search_ack_callback(SimpleNamespace(data=True))
    follower._person_search_done_callback(SimpleNamespace(data=True))

    assert original == (
        follower.person_search_waiting,
        follower.person_search_arrival_acknowledged,
        follower.person_search_ack_received_at,
    )
    assert follower.events == []


def test_done_before_ack_is_ignored_without_navigation_advance():
    for result in (True, False):
        follower = make_person_search_follower()

        follower._person_search_done_callback(SimpleNamespace(data=result))

        assert follower.person_search_waiting
        assert not follower.person_search_faulted
        assert ('advance', None) not in follower.events
        assert ('stop', False) not in follower.events
        assert any('ignored before Camera arrival ACK' in text
                   for _level, text in follower.logs)


def test_person_search_session_invalidation_resets_ack_state():
    follower = make_person_search_follower()
    follower.person_search_arrival_acknowledged = True
    follower.person_search_ack_received_at = FakeTime(1.0)
    follower._invalidate_person_area_stop = lambda: None

    assert follower._invalidate_person_search_session()
    assert not follower.person_search_arrival_acknowledged
    assert follower.person_search_ack_received_at is None
    assert follower.person_search_started_at is None


def test_person_search_timeout_parameter_defaults_are_preserved_and_added():
    source_path = (
        Path(__file__).parents[1]
        / 'robotbase_waypoint'
        / 'waypoint_follow.py'
    )
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    declarations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != 'declare_parameter':
            continue
        if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
            name = ast.literal_eval(node.args[0])
            if name in {'person_search_timeout_sec', 'person_search_ack_timeout_sec'}:
                declarations[name] = ast.literal_eval(node.args[1])

    assert declarations['person_search_timeout_sec'] == 60.0
    assert declarations['person_search_ack_timeout_sec'] == 5.0


def test_person_search_timeout_parameters_require_positive_values():
    source_path = (
        Path(__file__).parents[1]
        / 'robotbase_waypoint'
        / 'waypoint_follow.py'
    )
    source = source_path.read_text(encoding='utf-8')
    assert 'person_search_timeout_sec must be positive' in source
    assert 'person_search_ack_timeout_sec must be positive' in source


def test_person_search_ros_topic_names_are_unchanged():
    source_path = (
        Path(__file__).parents[1]
        / 'robotbase_waypoint'
        / 'waypoint_follow.py'
    )
    source = source_path.read_text(encoding='utf-8')
    for topic in (
            '/person_area_arrived',
            '/person_search/arrival_received',
            '/person_search_done',
            '/person_search_done_detail'):
        assert topic in source


def test_person_search_ros_qos_contract_is_unchanged():
    arrival_qos = person_area_arrival_qos()
    response_qos = person_search_response_qos()

    assert arrival_qos.depth == 1
    assert arrival_qos.reliability == ReliabilityPolicy.RELIABLE
    assert arrival_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert response_qos.depth == 10
    assert response_qos.reliability == ReliabilityPolicy.RELIABLE
    assert response_qos.durability == DurabilityPolicy.VOLATILE
