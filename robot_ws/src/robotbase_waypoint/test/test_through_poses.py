"""Continuous sections must preserve progress, stop tasks and goal ownership."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateThroughPoses
import pytest
from rclpy.time import Time as RosTime

import robotbase_waypoint.waypoint_follow as module
from robotbase_waypoint.waypoint_follow import limit_section_length
from robotbase_waypoint.waypoint_follow import navigation_segment_end, WaypointFollower
from robotbase_waypoint.waypoint_follow import split_ambiguous_section
from robotbase_waypoint.waypoint_io import Waypoint


def wp(number, *, stop=False, wait_time=0.0, person_area=False, threshold=-1.0):
    """Place ordered points two metres apart on a straight route."""
    return Waypoint(number, float(number * 2), 0.0, 0.0, threshold,
                    stop, wait_time, person_area)


class Handle:
    """Action handle whose acceptance/result/cancellation can arrive separately."""

    accepted = True

    def __init__(self, goal_id):
        self.goal_id = goal_id
        self.result = Future()
        self.cancel = Future()
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel


@pytest.fixture
def follower():
    """Exercise the real state machine with deterministic action transports."""
    node = WaypointFollower.__new__(WaypointFollower)
    node._init_shutdown_state()
    node.navigation_mode = 'through_poses'
    node.navigation_only = False
    node.max_section_length = 40.0
    node.section_switch_distance = 4.0
    node.single_goal_recovery = False
    node._bt_directory = None
    node.manual_paused = False
    node.manual_pause_started_at = None
    node.waypoints = [wp(1), wp(2), wp(3)]
    node.index = 0
    node.loop = False
    node.default_threshold = 1.0
    node.precise_threshold = .35
    node.stop_duration = 5.0
    node.stop_on_failure = False
    node.person_area_cancel_timeout_sec = 3.0
    node.person_area_zero_settle_sec = .5
    node.map_frame = 'map'
    node.base_frame = 'robot/base_footprint'
    node.position = (0.0, 0.0)
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
        transform=SimpleNamespace(translation=SimpleNamespace(
            x=node.position[0], y=node.position[1]))))
    node.goal_handle = node.goal_result_future = None
    node.goal_serial = 0
    node.transition_in_progress = False
    node.cancel_transition_generation = 0
    node.active_cancel_transition_generation = None
    node.person_area_cancel_started_at = None
    node.person_area_stop_pending = False
    node.person_area_stop_started_at = None
    node.person_area_stop_starts_search = False
    node.holding_stop = False
    node.waiting_until = None
    node.waiting_finishes_route = False
    node.person_search_waiting = False
    node.person_search_faulted = False
    node.finished = False
    node.now = 0.0
    node.get_clock = lambda: SimpleNamespace(now=lambda: RosTime(seconds=node.now))
    node.events = []
    node.get_logger = lambda: SimpleNamespace(
        info=lambda text, **kw: node.events.append(('info', text)),
        warning=lambda text, **kw: node.events.append(('warning', text)),
        error=lambda text, **kw: node.events.append(('error', text)))
    node.stop_publisher = SimpleNamespace(
        publish=lambda msg: node.events.append(('stop', msg.data)))
    node.safety_cmd_vel_publisher = SimpleNamespace(
        publish=lambda msg: node.events.append(('zero', msg.linear.x)))
    node.sent = []

    def send(goal, *, goal_uuid, feedback_callback):
        future = Future()
        node.sent.append(SimpleNamespace(goal=goal, future=future, goal_id=goal_uuid,
                                         feedback=feedback_callback))
        return future

    node.action_client = SimpleNamespace(send_goal_async=send)
    node._through_poses_tree = lambda: '/tmp/test_through.xml'
    yield node
    if node._bt_directory is not None:
        node._bt_directory.cleanup()


def accept(sent):
    """Resolve one request into a live Nav2 goal."""
    handle = Handle(sent.goal_id)
    sent.future.set_result(handle)
    return handle


def feedback(sent, remaining):
    sent.feedback(SimpleNamespace(feedback=SimpleNamespace(
        number_of_poses_remaining=remaining)))


def finish(handle, status=GoalStatus.STATUS_SUCCEEDED):
    handle.result.set_result(SimpleNamespace(status=status))


def test_sections_include_stop_and_person_endpoints_and_respect_navigation_only():
    points = [wp(1), wp(2, stop=True), wp(3), wp(4, person_area=True), wp(5)]
    assert navigation_segment_end(points, 0) == 1
    assert navigation_segment_end(points, 1) == 1
    assert navigation_segment_end(points, 2) == 3
    assert navigation_segment_end(points, 4) == 4
    assert navigation_segment_end(points, 2, navigation_only=True) == 4


def test_straight_section_stays_continuous_and_hairpin_is_split():
    points = [wp(i) for i in range(1, 7)]
    assert split_ambiguous_section(points, 0, 5) == 5
    points = [Waypoint(i + 1, x, 0., 0.) for i, x in enumerate((0., 2., 4., 6., 4., 2.))]
    assert split_ambiguous_section(points, 0, 5) == 3
    assert split_ambiguous_section(points, 4, 5) == 5


def test_length_limit_counts_polyline_and_keeps_exact_boundary():
    points = [Waypoint(i + 1, x, y, 0.) for i, (x, y) in enumerate(
        ((10., 0.), (30., 0.), (30., 20.), (30., 30.)))]
    assert limit_section_length(points, 0, 3, (0., 0.), 50.) == 2
    assert limit_section_length(points, 0, 3, (0., 0.), 49.) == 1
    assert limit_section_length(points, 1, 3, (20., 0.), 30.) == 2
    assert limit_section_length(points, 0, 3, (-100., 0.), 50.) == 0


def test_fifty_metre_sections_preserve_remaining_points_and_wait(follower):
    follower.max_section_length = 50.0
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.,
                                   wait_time=2.0 if i == 8 else 0.0)
                          for i in range(9)]
    follower._send_current_goal()
    assert [p.pose.position.x for p in follower.sent[0].goal.poses] == [
        0., 10., 20., 30., 40., 50.]
    follower.position = (50., 0.)
    finish(accept(follower.sent[0]))
    assert follower.index == 6
    assert [p.pose.position.x for p in follower.sent[1].goal.poses] == [60., 70., 80.]
    follower.position = (80., 0.)
    finish(accept(follower.sent[1]))
    assert follower.waiting_until is not None


def test_large_smooth_loop_does_not_end_near_early_part_of_path():
    import math
    points = [Waypoint(i + 1, 5 * math.cos(i * math.pi / 12),
                       5 * math.sin(i * math.pi / 12), 0.) for i in range(25)]
    assert split_ambiguous_section(points, 0, 24) == 12


def test_hairpin_sections_preserve_order_and_wait_marker(follower):
    follower.waypoints = [Waypoint(i + 1, x, 0., 0., wait_time=2.0 if i == 5 else 0.)
                          for i, x in enumerate((0., 2., 4., 6., 4., 2.))]
    follower._send_current_goal()
    assert [p.pose.position.x for p in follower.sent[0].goal.poses] == [0., 2., 4., 6.]
    follower.position = (6., 0.)
    finish(accept(follower.sent[0]))
    assert follower.index == 4
    assert [p.pose.position.x for p in follower.sent[1].goal.poses] == [4., 2.]
    follower.position = (2., 0.)
    finish(accept(follower.sent[1]))
    assert follower.index == 5
    assert follower.waiting_until is not None


def test_midpoint_feedback_does_not_resend_or_cancel_or_command_zero(follower):
    follower.waypoints = [wp(1), wp(2), wp(3, wait_time=2.0), wp(4)]
    follower._send_current_goal()
    sent = follower.sent[0]
    handle = accept(sent)
    assert isinstance(sent.goal, NavigateThroughPoses.Goal)
    assert [pose.pose.position.x for pose in sent.goal.poses] == [2., 4., 6.]
    feedback(sent, 2)
    assert follower.index == 1
    feedback(sent, 3)  # Old/reordered feedback cannot move backwards.
    assert follower.index == 1
    follower._timer_callback()  # No TF lookup is needed for intermediate points.
    assert len(follower.sent) == 1
    assert handle.cancel_calls == 0
    assert not any(name in ('stop', 'zero') for name, _ in follower.events)


@pytest.mark.parametrize('remaining', [1, 2])
@pytest.mark.parametrize('status', [GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED])
def test_early_extension_retains_old_endpoint_and_ignores_stale_result(
        follower, remaining, status):
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    old = follower.sent[0]
    handle = accept(old)
    feedback(old, remaining)
    follower.position = (35.9, 0.)
    follower._timer_callback()
    assert len(follower.sent) == 1
    follower.position = (36., 0.)
    follower._timer_callback()
    assert follower.index == 5 - remaining
    expected = [40., 50., 60., 70.] if remaining == 1 else [30., 40., 50., 60.]
    assert [p.pose.position.x for p in follower.sent[1].goal.poses] == expected
    follower._timer_callback()  # Do not re-send while the new request is pending.
    assert len(follower.sent) == 2
    finish(handle, status)
    assert follower.index == 5 - remaining
    assert not follower.finished
    assert handle.cancel_calls == 0
    assert not any(name in ('stop', 'zero') for name, _ in follower.events)


def test_pause_during_extension_cancels_both_owned_goals(follower, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, 'monotonic', lambda: now[0])
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    old = accept(follower.sent[0])
    feedback(follower.sent[0], 1)
    follower.position = (36., 0.)
    follower._timer_callback()
    response = follower._set_paused(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success
    new = accept(follower.sent[1])
    follower._timer_callback()
    assert old.cancel_calls == new.cancel_calls == 1
    assert follower.index == 4
    response = follower._set_paused(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    finish(old, GoalStatus.STATUS_CANCELED)
    finish(new, GoalStatus.STATUS_CANCELED)
    now[0] = 1.0
    response = follower._set_paused(SimpleNamespace(data=False), SimpleNamespace())
    assert response.success
    assert [p.pose.position.x for p in follower.sent[-1].goal.poses] == [40., 50., 60., 70.]
    feedback(follower.sent[1], 1)
    assert follower.index == 4


@pytest.mark.parametrize('pose_available', [True, False])
def test_replacement_success_at_old_endpoint_never_skips_remaining_points(
        follower, pose_available):
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    accept(follower.sent[0])
    feedback(follower.sent[0], 1)
    follower.position = (36., 0.)
    follower._timer_callback()
    updated = accept(follower.sent[1])
    follower.position = (40., 0.)
    if not pose_available:
        def missing_pose(*args):
            raise module.TransformException('No current pose')
        follower.tf_buffer.lookup_transform = missing_pose
    finish(updated)  # Nav2 reports success for 70 m while still at 40 m.
    assert follower.index == 4
    assert len(follower.sent) == 3
    assert [p.pose.position.x for p in follower.sent[2].goal.poses] == [
        40., 50., 60., 70., 80.]
    assert not follower.finished


def test_replacement_success_at_new_endpoint_advances_normally(follower):
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    accept(follower.sent[0])
    feedback(follower.sent[0], 1)
    follower.position = (36., 0.)
    follower._timer_callback()
    updated = accept(follower.sent[1])
    follower.position = (70., 0.)
    finish(updated)
    assert follower.index == 8
    assert [p.pose.position.x for p in follower.sent[2].goal.poses] == [80.]


def test_extension_only_resends_when_endpoint_can_advance(follower):
    """Proximity alone cannot repeatedly send the same horizon."""
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    accept(follower.sent[0])
    follower.position = (36., 0.)
    # No passage feedback: retaining all old points leaves the same endpoint.
    for _ in range(3):
        follower._timer_callback()
    assert len(follower.sent) == 1 and follower.index == 0
    feedback(follower.sent[0], 1)
    follower._timer_callback()
    accept(follower.sent[1])
    for _ in range(3):
        follower._timer_callback()
    assert len(follower.sent) == 2
    feedback(follower.sent[0], 1)
    assert follower.index == 4


def test_shutdown_during_extension_retains_and_cancels_late_goal(follower):
    """Shutdown owns both actions, even if replacement acceptance is delayed."""
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    follower._send_current_goal()
    old = accept(follower.sent[0])
    feedback(follower.sent[0], 1)
    follower.position = (36., 0.)
    follower._timer_callback()
    follower.request_shutdown('test cancellation during update')
    for record in follower.goal_records.values():
        follower._ensure_cancel_requested(record)
    new = accept(follower.sent[1])
    assert old.cancel_calls == new.cancel_calls == 1
    feedback(follower.sent[1], 1)
    finish(old)
    finish(new)
    assert follower.index == 4 and len(follower.sent) == 2
    assert all(record.resolved for record in follower.goal_records.values())


def test_nearby_endpoint_split_never_extends_early(follower):
    """A smooth closed section retains its explicit arrival boundary."""
    import math

    follower.max_section_length = 100.
    follower.position = (5., 0.)
    follower.waypoints = [Waypoint(i + 1, 5 * math.cos(i * math.pi / 12),
                                   5 * math.sin(i * math.pi / 12), 0.)
                          for i in range(25)]
    follower._send_current_goal()
    accept(follower.sent[0])
    record = follower.goal_records[follower.goal_serial]
    assert record.end_index < 24 and not record.allow_early_extension
    follower.position = (follower.waypoints[record.end_index].x,
                         follower.waypoints[record.end_index].y)
    follower._timer_callback()
    assert len(follower.sent) == 1


@pytest.mark.parametrize('task', ['stop', 'wait', 'person', 'final', 'hairpin'])
def test_early_extension_never_bypasses_protected_endpoint(follower, task):
    follower.waypoints = [Waypoint(i + 1, float(i * 10), 0., 0.) for i in range(9)]
    endpoint = follower.waypoints[4]
    if task == 'stop':
        follower.waypoints[4] = replace(endpoint, stop=True)
    elif task == 'wait':
        follower.waypoints[4] = replace(endpoint, wait_time=2.0)
    elif task == 'person':
        follower.waypoints[4] = replace(endpoint, person_area=True)
    elif task == 'final':
        follower.waypoints = follower.waypoints[:5]
    else:
        follower.waypoints[5] = replace(follower.waypoints[5], x=30.0)
    follower._send_current_goal()
    accept(follower.sent[0])
    feedback(follower.sent[0], 1)
    follower.position = (36., 0.)
    follower._timer_callback()
    assert len(follower.sent) == 1
    assert not follower.goal_records[follower.goal_serial].allow_early_extension


def test_section_success_waits_then_starts_next_section(follower):
    follower.waypoints = [wp(1), wp(2, wait_time=3.0), wp(3), wp(4)]
    follower._send_current_goal()
    handle = accept(follower.sent[0])
    finish(handle)
    assert follower.index == 1
    assert follower.holding_stop
    assert follower.waiting_until is not None
    assert len(follower.sent) == 1
    follower.now = 3.1
    follower._timer_callback()
    assert follower.index == 2
    assert not follower.holding_stop
    assert [pose.pose.position.x for pose in follower.sent[-1].goal.poses] == [6., 8.]


def test_starting_in_middle_keeps_only_remaining_points(follower):
    follower.waypoints = [wp(1), wp(2), wp(3), wp(4)]
    follower.index = 2
    follower._send_current_goal()
    assert [pose.pose.position.x for pose in follower.sent[0].goal.poses] == [6., 8.]
    feedback(follower.sent[0], 1)
    assert follower.index == 3


def test_loop_restarts_without_immediate_success_at_current_endpoint(follower):
    follower.loop = True
    follower._send_current_goal()
    follower.position = (6., 0.)
    finish(accept(follower.sent[0]))
    # Return to the start first; re-sending all poses would end at the
    # robot's current position and could succeed without traversing the lap.
    assert [pose.pose.position.x for pose in follower.sent[1].goal.poses] == [2.]
    follower.position = (2., 0.)
    finish(accept(follower.sent[1]))
    assert [pose.pose.position.x for pose in follower.sent[2].goal.poses] == [4., 6.]


def test_closed_section_keeps_all_points_in_order(follower):
    follower.waypoints = [wp(1), wp(2), Waypoint(3, 0., 0., 0.)]
    follower._send_current_goal()
    assert [pose.pose.position.x for pose in follower.sent[0].goal.poses] == [2., 4.]
    follower.position = (4., 0.)
    finish(accept(follower.sent[0]))
    assert [pose.pose.position.x for pose in follower.sent[1].goal.poses] == [0.]


def test_person_search_is_started_at_section_end_and_blocks_next_section(follower):
    follower.waypoints = [wp(1), wp(2, person_area=True), wp(3)]
    follower._start_person_search = lambda: follower.events.append(('search', follower.index))
    follower._send_current_goal()
    finish(accept(follower.sent[0]))
    assert follower.index == 1
    assert ('search', 1) in follower.events
    assert len(follower.sent) == 1


def test_batch_failure_retries_next_point_without_skipping_task(follower):
    follower.waypoints = [wp(1), wp(2, person_area=True), wp(3)]
    follower._send_current_goal()
    finish(accept(follower.sent[0]), GoalStatus.STATUS_ABORTED)
    assert follower.index == 0
    assert len(follower.sent[1].goal.poses) == 1
    finish(accept(follower.sent[1]), GoalStatus.STATUS_ABORTED)
    assert follower.index == 1
    assert follower.sent[-1].goal.poses[0].pose.position.x == 4.0
    assert len(follower.sent[-1].goal.poses) == 1


def test_manual_pause_owns_late_acceptance_and_resumes_unpassed_points(follower, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, 'monotonic', lambda: now[0])
    follower._send_current_goal()
    sent = follower.sent[0]
    feedback(sent, 2)
    response = follower._set_paused(SimpleNamespace(data=True), SimpleNamespace())
    assert response.success and follower.manual_paused
    handle = accept(sent)  # Pause was requested before the server accepted.
    follower._timer_callback()
    assert handle.cancel_calls == 1
    response = follower._set_paused(SimpleNamespace(data=False), SimpleNamespace())
    assert not response.success
    finish(handle, GoalStatus.STATUS_CANCELED)
    now[0] = 1.0
    follower._timer_callback()
    response = follower._set_paused(SimpleNamespace(data=False), SimpleNamespace())
    assert response.success and not follower.manual_paused
    assert [pose.pose.position.x for pose in follower.sent[-1].goal.poses] == [4., 6.]
    feedback(sent, 1)  # A late message from the cancelled goal is harmless.
    assert follower.index == 1


def test_current_tree_checks_passage_outside_rate_controller(follower):
    del follower._through_poses_tree
    path = follower._through_poses_tree()
    tree = ET.parse(path)
    passage = tree.find('.//PipelineSequence/RemoveTraversedGoals')
    assert passage is not None and 'radius' not in passage.attrib
    assert tree.find('.//RemovePassedGoals') is None
    assert tree.find('.//PathLongerOnApproach').get('length_factor') == '3.0'


def test_to_pose_dispatch_never_uses_section_splitting(follower, monkeypatch):
    """The original mode sends exactly one pose even across a task boundary."""
    from nav2_msgs.action import NavigateToPose

    follower.navigation_mode = 'to_pose'
    follower.waypoints = [wp(1), wp(2, stop=True), wp(3)]

    def forbidden(*args, **kwargs):
        raise AssertionError('to_pose must not use through_poses section management')

    monkeypatch.setattr(module, 'navigation_segment_end', forbidden)
    monkeypatch.setattr(module, 'split_ambiguous_section', forbidden)
    monkeypatch.setattr(module, 'split_nearby_endpoint', forbidden)
    follower._through_poses_tree = forbidden
    sent = []

    def send(goal, **kwargs):
        sent.append(goal)
        assert 'feedback_callback' not in kwargs
        return Future()

    follower.action_client.send_goal_async = send
    follower._send_current_goal()
    assert len(sent) == 1
    assert isinstance(sent[0], NavigateToPose.Goal)
    assert sent[0].pose.pose.position.x == 2.0
    assert sent[0].behavior_tree == ''
    assert follower.goal_records[1].end_index is None


def test_through_intermediate_thresholds_cannot_trigger_arrival(follower):
    """Huge route/YAML thresholds cannot discard intermediate goals locally."""
    follower.waypoints = [wp(1, threshold=100.), wp(2, threshold=100.), wp(3)]
    follower.default_threshold = 100.
    follower._send_current_goal()
    handle = accept(follower.sent[0])
    follower.position = (1.99, 0.)
    follower._timer_callback()
    assert follower.index == 0
    assert len(follower.sent) == 1
    assert handle.cancel_calls == 0
    assert not any(name in ('stop', 'zero') for name, _ in follower.events)


def test_unspecified_navigation_mode_defaults_to_original_mode():
    """Keep direct invocations without navigation_mode backward compatible."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text())
    defaults = [ast.literal_eval(node.args[1]) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'declare_parameter' and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 'navigation_mode']
    assert defaults == ['to_pose']
