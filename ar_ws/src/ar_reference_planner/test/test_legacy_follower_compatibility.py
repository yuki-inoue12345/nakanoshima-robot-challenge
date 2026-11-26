"""Verify fail-closed interoperability with the Launcher2 legacy follower."""

from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from action_msgs.msg import GoalStatus, GoalStatusArray
from ar_reference_planner.follower_state import (
    FollowerSnapshot,
    validate_snapshot,
)
from ar_reference_planner.legacy_compatibility import (
    conservative_safe_end_index,
    resolve_legacy_progress,
    unique_waypoint_index,
)
from ar_reference_planner.legacy_completion import (
    LegacyCompletion,
    LegacyCompletionGate,
)
from ar_reference_planner.legacy_follower_bridge import LegacyFollowerBridge
from ar_reference_planner.route import load_route
from builtin_interfaces.msg import Time
import pytest
from rcl_interfaces.msg import Log, ParameterType, ParameterValue
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robotbase_waypoint_interfaces.msg import WaypointFollowerState
import yaml


def _route(tmp_path: Path, points=None):
    if points is None:
        points = [
            (1, 1.0, 0.0, {}),
            (2, 3.0, 0.0, {}),
            (3, 6.0, 1.0, {'person_area': True}),
            (4, 9.0, 1.0, {}),
        ]
    path = tmp_path / 'route.yaml'
    path.write_text(yaml.safe_dump({'waypoints': [
        {
            'number': number,
            'x': x,
            'y': y,
            'angle_radians': 0.0,
            **flags,
        }
        for number, x, y, flags in points
    ]}), encoding='utf-8')
    return load_route(str(path))


def test_to_pose_plan_endpoint_resolves_only_one_exact_route_goal(tmp_path):
    route = _route(tmp_path)

    progress = resolve_legacy_progress(
        route, 'to_pose', 3.04, -0.02, tolerance_m=0.40,
        separation_m=0.20,
    )

    assert progress.next_index == 1
    assert progress.endpoint_index == 1
    # The display-only reference may look ahead, but never beyond the first
    # explicit wait/person/stop boundary in the verified route.
    assert progress.safe_end_index == 2


def test_through_poses_requires_bound_remaining_count(tmp_path):
    route = _route(tmp_path)

    with pytest.raises(ValueError, match='feedback_required'):
        resolve_legacy_progress(route, 'through_poses', 6.0, 1.0)

    progress = resolve_legacy_progress(
        route, 'through_poses', 6.0, 1.0, poses_remaining=2
    )
    assert progress.next_index == 1
    assert progress.endpoint_index == 2
    assert progress.safe_end_index == 2


def test_ambiguous_or_off_route_plan_endpoint_is_rejected(tmp_path):
    route = _route(tmp_path, [
        (1, 1.0, 0.0, {}),
        (2, 1.1, 0.0, {}),
        (3, 4.0, 0.0, {}),
    ])

    with pytest.raises(ValueError, match='ambiguous'):
        unique_waypoint_index(
            route, 1.05, 0.0, tolerance_m=0.40, separation_m=0.20
        )
    with pytest.raises(ValueError, match='not_on_route'):
        unique_waypoint_index(
            route, 20.0, 20.0, tolerance_m=0.40, separation_m=0.20
        )


def test_safe_end_never_crosses_stop_wait_or_person_boundary(tmp_path):
    route = _route(tmp_path)

    assert conservative_safe_end_index(route, 0) == 2
    assert conservative_safe_end_index(route, 2) == 2
    assert conservative_safe_end_index(route, 3) == 3


def test_action_status_must_be_fresh_unique_and_single_mode():
    values = {
        'action_status_timeout_sec': 2.5,
    }

    class Fake:
        _navigation_mode = 'to_pose'
        _active_goals = {
            'to_pose': ('01' * 16,),
            'through_poses': (),
        }
        _status_received_at = {'to_pose': 10.0, 'through_poses': None}
        _to_pose_feedback = None
        _through_feedback = None

        @staticmethod
        def _feedback_liveness(mode, goal_uuid):
            return LegacyFollowerBridge._feedback_liveness(
                fake, mode, goal_uuid
            )

        @staticmethod
        def get_parameter(name):
            return type('Parameter', (), {'value': values[name]})()

    fake = Fake()
    goal, reason = LegacyFollowerBridge._active_goal(fake, 11.0)
    assert goal == '01' * 16
    assert reason == 'legacy_nav2_goal_active'

    fake._active_goals['through_poses'] = ('02' * 16,)
    assert LegacyFollowerBridge._active_goal(fake, 11.0) == (
        None, 'legacy_nav2_action_mode_ambiguous'
    )
    fake._active_goals['through_poses'] = ()
    assert LegacyFollowerBridge._active_goal(fake, 13.0) == (
        None, 'legacy_action_status_stale'
    )


def test_matching_to_pose_feedback_bridges_status_heartbeat_gap():
    """Fresh feedback for the same known UUID prevents a false HOLD."""
    goal_uuid = '01' * 16
    values = {'action_status_timeout_sec': 2.5}
    fake = SimpleNamespace(
        _navigation_mode='to_pose',
        _active_goals={'to_pose': (goal_uuid,), 'through_poses': ()},
        _status_received_at={'to_pose': 10.0, 'through_poses': None},
        _to_pose_feedback=(goal_uuid, 12.8),
        _through_feedback=None,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )
    fake._feedback_liveness = lambda mode, goal: (
        LegacyFollowerBridge._feedback_liveness(fake, mode, goal)
    )

    assert LegacyFollowerBridge._active_goal(fake, 13.0) == (
        goal_uuid, 'legacy_nav2_goal_active'
    )

    fake._to_pose_feedback = ('02' * 16, 12.9)
    assert LegacyFollowerBridge._active_goal(fake, 13.0) == (
        None, 'legacy_action_status_stale'
    )


def test_feedback_never_creates_goal_without_status_identity():
    """Feedback is liveness only; it cannot authorize an unknown goal."""
    goal_uuid = '01' * 16
    fake = SimpleNamespace(
        _navigation_mode='to_pose',
        _active_goals={'to_pose': (), 'through_poses': ()},
        _status_received_at={'to_pose': None, 'through_poses': None},
        _to_pose_feedback=(goal_uuid, 12.9),
        _through_feedback=None,
        get_parameter=lambda _name: SimpleNamespace(value=2.5),
    )
    fake._feedback_liveness = lambda mode, goal: (
        LegacyFollowerBridge._feedback_liveness(fake, mode, goal)
    )

    assert LegacyFollowerBridge._active_goal(fake, 13.0) == (
        None, 'legacy_nav2_goal_inactive'
    )


def test_old_plan_is_rejected_after_goal_generation_change(tmp_path):
    route = _route(tmp_path)
    values = {
        'plan_timeout_sec': 6.5,
        'action_status_timeout_sec': 2.5,
        'endpoint_tolerance_m': 0.40,
        'endpoint_separation_m': 0.20,
    }

    class Fake:
        _goal_uuid = '01' * 16
        _plan_endpoint = (3.0, 0.0, 7)
        _minimum_plan_sequence = 8
        _plan_received_at = 10.0
        _navigation_mode = 'to_pose'
        _through_feedback = None
        _route = route

        @staticmethod
        def _active_goal(_now):
            return '01' * 16, 'legacy_nav2_goal_active'

        @staticmethod
        def get_parameter(name):
            return type('Parameter', (), {'value': values[name]})()

    fake = Fake()
    progress, reason = LegacyFollowerBridge._resolve_progress(fake, 10.5)
    assert progress is None
    assert reason == 'waiting_for_plan_after_nav2_goal_change'

    fake._plan_endpoint = (3.0, 0.0, 8)
    progress, reason = LegacyFollowerBridge._resolve_progress(fake, 10.5)
    assert progress.next_index == 1
    assert reason == 'legacy_progress_verified'


def test_goal_change_immediately_invalidates_previous_verified_progress():
    status = GoalStatus()
    status.status = GoalStatus.STATUS_EXECUTING
    status.goal_info.goal_id.uuid = [2] * 16
    message = GoalStatusArray(status_list=[status])
    invalidations = []
    reasons = []
    fake = SimpleNamespace(
        _navigation_mode='to_pose',
        _active_goals={'to_pose': (), 'through_poses': ()},
        _status_received_at={'to_pose': None, 'through_poses': None},
        _goal_uuid='01' * 16,
        _goal_generation=4,
        _plan_endpoint=(3.0, 0.0, 7),
        _plan_received_at=10.0,
        _to_pose_feedback=None,
        _through_feedback=None,
        _last_progress='verified-old-progress',
        _minimum_plan_sequence=7,
        _plan_sequence=7,
        _publish_state=lambda progress, activity, allowed: invalidations.append(
            (progress, activity, allowed)
        ),
        _log_reason=reasons.append,
    )

    LegacyFollowerBridge._on_status(fake, 'to_pose', message)

    assert fake._goal_generation == 5
    assert fake._last_progress is None
    assert invalidations == [(
        'verified-old-progress',
        WaypointFollowerState.ACTIVITY_TRANSITIONING,
        False,
    )]
    assert reasons == ['waiting_for_plan_after_nav2_goal_change']


def test_action_status_subscriptions_use_ros_action_qos():
    """Late bridge startup must receive Nav2's transient-local status."""
    package_source = (
        Path(__file__).parents[1]
        / 'ar_reference_planner'
        / 'legacy_follower_bridge.py'
    )
    text = package_source.read_text(encoding='utf-8')

    assert text.count('qos_profile_action_status_default') >= 3


def test_verified_legacy_state_is_accepted_by_strict_planner_contract(tmp_path):
    route = _route(tmp_path)
    progress = resolve_legacy_progress(route, 'to_pose', 3.0, 0.0)
    published = []

    class ClockValue:
        @staticmethod
        def to_msg():
            return Time(sec=20, nanosec=0)

    fake = SimpleNamespace(
        _publisher=SimpleNamespace(publish=published.append),
        _route=route,
        _map_frame='map',
        _navigation_mode='to_pose',
        _goal_generation=7,
        get_clock=lambda: SimpleNamespace(now=lambda: ClockValue()),
    )
    LegacyFollowerBridge._publish_state(
        fake,
        progress,
        WaypointFollowerState.ACTIVITY_NAVIGATING,
        True,
    )

    snapshot = FollowerSnapshot.from_message(published[-1])
    assert validate_snapshot(
        snapshot, route, 'map', 20_500_000_000, 2.5
    ) == ''
    assert snapshot.next_index == 1
    assert snapshot.goal_generation == 7
    assert snapshot.safe_end_index == 2


def test_finished_publisher_uses_terminal_index_and_disallows_reference(tmp_path):
    route = _route(tmp_path)
    published = []

    class ClockValue:
        @staticmethod
        def to_msg():
            return Time(sec=30, nanosec=0)

    fake = SimpleNamespace(
        _publisher=SimpleNamespace(publish=published.append),
        _route=route, _map_frame='map', _navigation_mode='to_pose',
        get_clock=lambda: SimpleNamespace(now=lambda: ClockValue()),
    )
    completion = LegacyCompletion(
        route.sha256, 8, '01' * 16, len(route.waypoints) - 1,
        route.waypoints[-1].number, len(route.waypoints),
    )
    LegacyFollowerBridge._publish_finished_state(fake, completion)

    message = published[-1]
    assert message.activity == WaypointFollowerState.ACTIVITY_FINISHED
    assert message.next_waypoint_index == len(route.waypoints)
    assert message.next_waypoint_number == route.waypoints[-1].number
    assert message.reference_allowed is False


def test_terminal_follower_log_publishes_completion_without_timer_delay(tmp_path):
    route = _route(tmp_path)
    published = []
    reasons = []
    gate = LegacyCompletionGate('koko_waypoint_follower', 2.0)
    goal_uuid = '01' * 16
    gate.bind_final_goal(
        route_sha256=route.sha256,
        goal_generation=8,
        goal_uuid=goal_uuid,
        final_index=len(route.waypoints) - 1,
        final_number=route.waypoints[-1].number,
        waypoint_count=len(route.waypoints),
    )
    gate.observe_status(goal_uuid, GoalStatus.STATUS_SUCCEEDED, monotonic())

    class ClockValue:
        @staticmethod
        def to_msg():
            return Time(sec=30, nanosec=0)

    fake = SimpleNamespace(
        _completion_gate=gate,
        _publisher=SimpleNamespace(publish=published.append),
        _route=route,
        _map_frame='map',
        _navigation_mode='to_pose',
        get_clock=lambda: SimpleNamespace(now=lambda: ClockValue()),
        _log_reason=reasons.append,
    )
    first = Log(name='koko_waypoint_follower')
    first.msg = f'[WP:{route.waypoints[-1].number}] reached by Nav2'
    second = Log(name='koko_waypoint_follower')
    second.msg = 'All waypoints completed'

    LegacyFollowerBridge._on_follower_log(fake, first)
    assert published == []
    LegacyFollowerBridge._on_follower_log(fake, second)
    assert published[-1].activity == WaypointFollowerState.ACTIVITY_FINISHED
    assert reasons[-1] == 'legacy_route_completion_verified'


def test_modern_follower_parameter_disables_compatibility_publisher():
    class Future:
        @staticmethod
        def result():
            return SimpleNamespace(result=SimpleNamespace(
                names=['reference_state_topic'],
            ))

    future = Future()
    removed = []
    reasons = []
    fake = SimpleNamespace(
        _parameter_future=future,
        _follower_parameters_available=False,
        _legacy_verified=True,
        _modern_follower=False,
        _remove_publisher=lambda: removed.append(True),
        _log_reason=reasons.append,
    )

    LegacyFollowerBridge._on_parameter_list(fake, future)

    assert fake._modern_follower is True
    assert fake._legacy_verified is False
    assert fake._follower_parameters_available is True
    assert removed == [True]
    assert reasons == ['modern_follower_native_state']


def test_legacy_parameter_list_requests_only_declared_required_names():
    callbacks = []

    class ParameterFuture:
        @staticmethod
        def add_done_callback(callback):
            callbacks.append(callback)

    parameter_future = ParameterFuture()
    requested = []
    list_future = SimpleNamespace(
        result=lambda: SimpleNamespace(
            result=SimpleNamespace(names=['waypoint_file', 'map_frame']),
        ),
    )
    fake = SimpleNamespace(
        _parameter_future=list_future,
        _on_parameters=lambda future: None,
        _parameter_client=SimpleNamespace(
            get_parameters=lambda names: (
                requested.append(tuple(names)) or parameter_future
            ),
        ),
        _modern_follower=True,
        _parameter_transport_failed=lambda reason: pytest.fail(reason),
    )

    LegacyFollowerBridge._on_parameter_list(fake, list_future)

    assert fake._modern_follower is False
    assert requested == [(
        'waypoint_file', 'navigation_mode', 'map_frame',
    )]
    assert fake._parameter_future is parameter_future
    assert callbacks == [fake._on_parameters]


def test_legacy_follower_parameters_load_exact_route_without_manual_selection(
    tmp_path,
):
    route = _route(tmp_path)
    values = [
        ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value=str(route.path),
        ),
        ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value='to_pose',
        ),
        ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value='map',
        ),
    ]

    class Future:
        @staticmethod
        def result():
            return SimpleNamespace(values=values)

    future = Future()
    events = []
    fake = SimpleNamespace(
        _parameter_future=future,
        _follower_parameters_available=False,
        _legacy_verified=False,
        _modern_follower=False,
        _route=None,
        _navigation_mode=None,
        _map_frame='map',
        _ensure_publisher=lambda: events.append('publisher'),
        _reset_goal_binding=lambda reason: events.append(('reset', reason)),
        _log_reason=lambda reason: events.append(('reason', reason)),
    )

    LegacyFollowerBridge._on_parameters(fake, future)

    assert fake._legacy_verified is True
    assert fake._modern_follower is False
    assert fake._follower_parameters_available is True
    assert fake._route.sha256 == route.sha256
    assert fake._navigation_mode == 'to_pose'
    assert events[0] == 'publisher'
    assert events[1] == ('reset', 'legacy_follower_configuration_changed')
    assert events[2][0] == 'reason'
    assert events[2][1].startswith('legacy_follower_verified route=')


def test_parameter_service_loss_republishes_safe_hold():
    progress = SimpleNamespace(next_index=1, safe_end_index=2)
    published = []
    fake = SimpleNamespace(
        _legacy_verified=True,
        _modern_follower=False,
        _follower_parameters_available=False,
        _last_progress=progress,
        _query_parameters=lambda now: None,
        _publish_state=lambda item, activity, allowed: published.append(
            (item, activity, allowed)
        ),
    )

    LegacyFollowerBridge._tick(fake)

    assert published == [(
        progress,
        WaypointFollowerState.ACTIVITY_HOLDING_STOP,
        False,
    )]


def test_disappearing_parameter_service_marks_transport_unavailable():
    reasons = []
    fake = SimpleNamespace(
        _parameter_future=None,
        _next_parameter_query=0.0,
        _parameter_client=SimpleNamespace(
            services_are_ready=lambda: False,
        ),
        _follower_parameters_available=True,
        get_parameter=lambda name: SimpleNamespace(value=2.0),
        _log_reason=reasons.append,
    )

    LegacyFollowerBridge._query_parameters(fake, 10.0)

    assert fake._next_parameter_query == 12.0
    assert fake._follower_parameters_available is False
    assert reasons == ['follower_parameter_service_unavailable']


def test_parameter_service_recovery_restores_verified_navigation():
    progress = SimpleNamespace(next_index=1, safe_end_index=2)
    published = []
    reasons = []
    fake = SimpleNamespace(
        _legacy_verified=True,
        _modern_follower=False,
        _follower_parameters_available=True,
        _last_progress=progress,
        _query_parameters=lambda now: None,
        _resolve_progress=lambda now: (progress, 'legacy_progress_verified'),
        _publish_state=lambda item, activity, allowed: published.append(
            (item, activity, allowed)
        ),
        _log_reason=reasons.append,
    )

    LegacyFollowerBridge._tick(fake)

    assert published == [(
        progress,
        WaypointFollowerState.ACTIVITY_NAVIGATING,
        True,
    )]
    assert reasons == ['legacy_progress_verified']


def test_parameter_query_failure_preserves_route_for_safe_hold():
    class Future:
        @staticmethod
        def result():
            raise RuntimeError('service disconnected')

    future = Future()
    reasons = []
    fake = SimpleNamespace(
        _parameter_future=future,
        _follower_parameters_available=True,
        _legacy_verified=True,
        _parameter_transport_failed=lambda reason: (
            setattr(fake, '_follower_parameters_available', False),
            reasons.append(reason),
        ),
        _log_reason=reasons.append,
    )

    LegacyFollowerBridge._on_parameters(fake, future)

    assert fake._follower_parameters_available is False
    assert fake._legacy_verified is True
    assert reasons == [
        'follower_parameter_query_failed service disconnected'
    ]


def test_empty_parameter_response_is_diagnostic_and_fail_closed():
    class Future:
        @staticmethod
        def result():
            return SimpleNamespace(values=[])

    future = Future()
    reasons = []
    fake = SimpleNamespace(
        _parameter_future=future,
        _follower_parameters_available=True,
        _legacy_verified=True,
        _parameter_transport_failed=lambda reason: (
            setattr(fake, '_follower_parameters_available', False),
            reasons.append(reason),
        ),
    )

    LegacyFollowerBridge._on_parameters(fake, future)

    assert fake._legacy_verified is True
    assert fake._follower_parameters_available is False
    assert reasons == [
        'follower_parameter_response_incomplete expected=3 actual=0'
    ]


def test_real_jazzy_parameter_service_recovers_after_legacy_follower_start(
    tmp_path, monkeypatch,
):
    route = _route(tmp_path)
    log_dir = tmp_path / 'ros-log'
    log_dir.mkdir()
    monkeypatch.setenv('ROS_LOG_DIR', str(log_dir))
    rclpy.init(args=[
        '--ros-args',
        '-p', 'follower_node_name:=/legacy_parameter_test_follower',
        '-p', 'parameter_poll_sec:=0.05',
        '-p', 'publish_period_sec:=0.05',
    ])
    bridge = LegacyFollowerBridge()
    follower = None
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)
    try:
        unavailable_deadline = monotonic() + 0.25
        while monotonic() < unavailable_deadline:
            executor.spin_once(timeout_sec=0.05)
        assert bridge._follower_parameters_available is False
        assert bridge._publisher is None

        follower = Node('legacy_parameter_test_follower')
        follower.declare_parameter('waypoint_file', str(route.path))
        follower.declare_parameter('navigation_mode', 'to_pose')
        follower.declare_parameter('map_frame', 'map')
        executor.add_node(follower)

        ready_deadline = monotonic() + 3.0
        while (
            (not bridge._legacy_verified or bridge._publisher is None)
            and monotonic() < ready_deadline
        ):
            executor.spin_once(timeout_sec=0.05)

        assert bridge._legacy_verified is True
        assert bridge._follower_parameters_available is True
        assert bridge._modern_follower is False
        assert bridge._route.sha256 == route.sha256
        assert bridge._publisher is not None
        assert bridge.count_publishers(
            '/waypoint_follower/reference_state'
        ) == 1
    finally:
        for timer in bridge.timers:
            timer.cancel()
        drain_deadline = monotonic() + 1.0
        while (
            bridge._parameter_future is not None
            and monotonic() < drain_deadline
        ):
            executor.spin_once(timeout_sec=0.05)
        for _ in range(4):
            executor.spin_once(timeout_sec=0.05)
        executor.remove_node(bridge)
        if follower is not None:
            executor.remove_node(follower)
        executor.shutdown()
        bridge.destroy_node()
        if follower is not None:
            follower.destroy_node()
        rclpy.try_shutdown()


def test_dedicated_launch_starts_bridge_without_control_interfaces():
    package = Path(__file__).parents[1]
    launch = (
        package / 'launch' / 'launcher2_reference_ar.launch.py'
    ).read_text(encoding='utf-8')
    bridge = (
        package / 'ar_reference_planner' / 'legacy_follower_bridge.py'
    ).read_text(encoding='utf-8')

    assert "executable='legacy_follower_reference_bridge'" in launch
    assert "'plan_topic': '/plan'" in launch
    assert 'ActionClient' not in bridge
    assert 'cmd_vel' not in bridge
    assert 'send_goal' not in bridge
    assert "'reference_state_topic'" in bridge
    assert 'modern_follower_native_state' in bridge


def test_scoped_follower_list_service_skips_unrelated_parameter_services():
    # The bridge needs only read-only ListParameters, not set services.
    from types import SimpleNamespace
    requests = []

    class Future:
        def add_done_callback(self, callback):
            self.callback = callback

    future = Future()
    scoped = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda req: requests.append(req) or future,
    )
    fake = SimpleNamespace(
        _parameter_future=None, _next_parameter_query=0.0,
        _parameter_list_service=scoped,
        _parameter_client=SimpleNamespace(
            services_are_ready=lambda: False,
        ),
        _follower_parameters_available=False,
        get_parameter=lambda name: SimpleNamespace(value=1.0),
        _log_reason=lambda reason: None,
        _on_parameter_list=lambda result: None,
    )
    LegacyFollowerBridge._query_parameters(fake, 10.0)
    assert len(requests) == 1
    assert requests[0].prefixes == ['reference_state_topic']
    assert requests[0].depth == 1
    assert fake._parameter_future is future


def test_scoped_follower_get_service_preserves_legacy_parameter_names():
    from types import SimpleNamespace
    requested = []

    class Future:
        def add_done_callback(self, callback):
            self.callback = callback

    list_future = SimpleNamespace(
        result=lambda: SimpleNamespace(result=SimpleNamespace(names=[]))
    )
    get_future = Future()
    fake = SimpleNamespace(
        _parameter_future=list_future,
        _parameter_get_service=SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=lambda req: requested.append(list(req.names)) or get_future,
        ),
        _parameter_client=SimpleNamespace(
            get_parameters=lambda names: (_ for _ in ()).throw(
                AssertionError('legacy generic client was incorrectly used')
            ),
        ),
        _modern_follower=True,
        _on_parameters=lambda result: None,
        _parameter_transport_failed=lambda reason: (_ for _ in ()).throw(
            AssertionError(reason)
        ),
    )
    LegacyFollowerBridge._on_parameter_list(fake, list_future)
    assert requested == [['waypoint_file', 'navigation_mode', 'map_frame']]
    assert fake._parameter_future is get_future
