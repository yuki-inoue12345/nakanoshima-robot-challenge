"""Validate the typed follower-to-reference-planner safety contract."""

from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from ar_reference_planner.follower_state import FollowerSnapshot, validate_snapshot
from ar_reference_planner.request_node import ReferencePathRequestNode
from ar_reference_planner.route import load_route, select_horizon_by_index
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from robotbase_waypoint_interfaces.msg import WaypointFollowerState
import yaml


NOW_NS = 20_000_000_000


def route_file(tmp_path: Path) -> Path:
    path = tmp_path / 'route.yaml'
    points = [
        {'number': index + 10, 'x': float(index * 2), 'y': 0.0,
         'angle_radians': 0.0}
        for index in range(8)
    ]
    path.write_text(yaml.safe_dump({'waypoints': points}), encoding='utf-8')
    return path


def state_for(route, *, mode='to_pose', index=1, generation=3):
    message = WaypointFollowerState()
    message.header.stamp.sec = NOW_NS // 1_000_000_000
    message.header.frame_id = 'map'
    message.schema_version = WaypointFollowerState.SCHEMA_VERSION
    message.route_sha256 = route.sha256
    message.next_waypoint_index = index
    message.next_waypoint_number = route.waypoints[index].number
    message.navigation_mode = mode
    message.goal_generation = generation
    message.activity = WaypointFollowerState.ACTIVITY_NAVIGATING
    message.reference_allowed = True
    message.safe_section_end_index = 7
    message.waypoint_count = len(route.waypoints)
    return message


def test_to_pose_and_through_poses_states_are_accepted(tmp_path):
    route = load_route(str(route_file(tmp_path)))
    for mode in ('to_pose', 'through_poses'):
        snapshot = FollowerSnapshot.from_message(state_for(route, mode=mode))

        assert validate_snapshot(snapshot, route, 'map', NOW_NS, 2.5) == ''


def test_index_and_generation_change_request_identity(tmp_path):
    route = load_route(str(route_file(tmp_path)))
    first = FollowerSnapshot.from_message(state_for(route, index=1, generation=3))
    advanced = FollowerSnapshot.from_message(state_for(route, index=2, generation=3))
    regenerated = FollowerSnapshot.from_message(state_for(route, index=1, generation=4))

    assert first.identity != advanced.identity
    assert first.identity != regenerated.identity
    assert select_horizon_by_index(route, 1, 7, 11.0).waypoint_numbers == (
        11, 12, 13, 14, 15, 16)
    assert select_horizon_by_index(route, 2, 7, 11.0).waypoint_numbers == (
        12, 13, 14, 15, 16, 17)


def test_route_change_stale_state_and_bad_boundary_are_rejected(tmp_path):
    route = load_route(str(route_file(tmp_path)))

    wrong_hash = state_for(route)
    wrong_hash.route_sha256 = 'f' * 64
    assert validate_snapshot(
        FollowerSnapshot.from_message(wrong_hash), route, 'map', NOW_NS, 2.5
    ) == 'route_hash_mismatch'

    stale = FollowerSnapshot.from_message(state_for(route))
    assert validate_snapshot(
        stale, route, 'map', NOW_NS + 3_000_000_000, 2.5
    ).startswith('stale_state')

    bad_end = state_for(route, index=2)
    bad_end.safe_section_end_index = 1
    assert validate_snapshot(
        FollowerSnapshot.from_message(bad_end), route, 'map', NOW_NS, 2.5
    ) == 'safe_section_end_index_out_of_range'


def test_pause_and_person_search_cannot_be_marked_reference_allowed(tmp_path):
    route = load_route(str(route_file(tmp_path)))
    for activity in (
            WaypointFollowerState.ACTIVITY_MANUAL_PAUSED,
            WaypointFollowerState.ACTIVITY_PERSON_SEARCH_WAITING,
            WaypointFollowerState.ACTIVITY_TASK_WAITING):
        message = state_for(route)
        message.activity = activity
        message.reference_allowed = False
        snapshot = FollowerSnapshot.from_message(message)

        assert validate_snapshot(
            snapshot, route, 'map', NOW_NS, 2.5
        ).startswith('reference_disallowed')


def test_late_result_from_old_goal_generation_is_never_adopted():
    node = ReferencePathRequestNode.__new__(ReferencePathRequestNode)
    node._active_generation = 8
    node._active_state_identity = ('route', 1, 2)
    node._state_identity = ('route', 1, 3)
    node._follower_state = None
    node._request_started = monotonic()
    events = []
    node._invalidate = lambda reason: events.append(('invalidate', reason))
    node._finish_cycle = lambda _now, immediate=False: events.append(
        ('finish', immediate))
    future = SimpleNamespace(result=lambda: SimpleNamespace(result=object()))

    node._result(future, 8)

    assert events[0][0] == 'invalidate'
    assert 'outdated_result_rejected' in events[0][1]
    assert events[1] == ('finish', True)


def test_outdated_request_is_canceled_once_while_ownership_is_retained():
    node = ReferencePathRequestNode.__new__(ReferencePathRequestNode)
    node._active_generation = 4
    node._active_state_identity = ('old',)
    node._state_identity = ('new',)
    node._cancel_requested_generation = None
    calls = []
    node._goal_handle = SimpleNamespace(
        cancel_goal_async=lambda: calls.append('cancel'))

    node._cancel_outdated_request()
    node._cancel_outdated_request()

    assert calls == ['cancel']
    assert node._active_generation == 4


def test_atomic_reference_message_binds_path_to_follower_identity(tmp_path):
    route = load_route(str(route_file(tmp_path)))
    snapshot = FollowerSnapshot.from_message(state_for(route, index=2, generation=9))
    path = PathMessage()
    path.header.frame_id = 'map'
    path.header.stamp.sec = 20
    for x in (1.0, 4.0, 7.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        path.poses.append(pose)

    published = []
    node = ReferencePathRequestNode.__new__(ReferencePathRequestNode)
    node._map_frame = 'map'
    node._state_publisher = SimpleNamespace(
        publish=lambda message: published.append(message))
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(
            to_msg=lambda: Time(sec=21, nanosec=0)))

    node._publish_state(
        valid=True,
        reason='',
        path=path,
        state=snapshot,
        planner_generation=12,
        path_length_m=6.0,
    )

    assert len(published) == 1
    message = published[0]
    assert message.valid
    assert message.path == path
    assert message.route_sha256 == route.sha256
    assert message.next_waypoint_index == 2
    assert message.goal_generation == 9
    assert message.planner_generation == 12
    assert message.path_length_m == 6.0
    assert message.follower_state_stamp.sec == 20
