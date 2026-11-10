"""Read-only follower state must mirror, never drive, mission state."""

from types import SimpleNamespace

from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time

from robotbase_waypoint.waypoint_follow import reference_state_qos, WaypointFollower
from robotbase_waypoint.waypoint_io import Waypoint
from robotbase_waypoint_interfaces.msg import WaypointFollowerState


def follower_for_state(mode='to_pose'):
    """Create only the committed fields consumed by the state snapshot."""
    follower = WaypointFollower.__new__(WaypointFollower)
    follower.route_sha256 = 'a' * 64
    follower.map_frame = 'map'
    follower.waypoints = [
        Waypoint(10, 1.0, 0.0, 0.0),
        Waypoint(20, 2.0, 0.0, 0.0, person_area=True),
        Waypoint(30, 3.0, 0.0, 0.0),
    ]
    follower.index = 0
    follower.navigation_mode = mode
    follower.navigation_only = False
    follower.goal_serial = 7
    follower.goal_records = {7: object()}
    follower.finished = False
    follower.manual_paused = False
    follower.person_search_faulted = False
    follower.person_search_waiting = False
    follower.person_area_stop_pending = False
    follower.waiting_until = None
    follower.transition_in_progress = False
    follower.holding_stop = False
    follower._is_shutting_down = lambda: False
    follower.get_clock = lambda: SimpleNamespace(
        now=lambda: Time(seconds=12.5))
    return follower


def test_reference_state_qos_is_reliable_and_transient_local():
    qos = reference_state_qos()

    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_to_pose_state_uses_existing_next_index_and_task_boundary():
    follower = follower_for_state('to_pose')

    message = follower._reference_state_message()

    assert message.schema_version == WaypointFollowerState.SCHEMA_VERSION
    assert message.next_waypoint_index == 0
    assert message.next_waypoint_number == 10
    assert message.safe_section_end_index == 1
    assert message.goal_generation == 7
    assert message.navigation_mode == 'to_pose'
    assert message.reference_allowed


def test_through_poses_state_uses_feedback_updated_index_without_rederiving_it():
    follower = follower_for_state('through_poses')
    follower.index = 1

    message = follower._reference_state_message()

    assert message.next_waypoint_index == 1
    assert message.next_waypoint_number == 20
    assert message.safe_section_end_index == 1
    assert message.navigation_mode == 'through_poses'


def test_pause_person_search_and_wait_states_disallow_reference_paths():
    cases = (
        ('manual_paused', WaypointFollowerState.ACTIVITY_MANUAL_PAUSED),
        ('person_search_waiting', WaypointFollowerState.ACTIVITY_PERSON_SEARCH_WAITING),
        ('waiting_until', WaypointFollowerState.ACTIVITY_TASK_WAITING),
    )
    for field, expected in cases:
        follower = follower_for_state()
        setattr(follower, field, object() if field == 'waiting_until' else True)

        message = follower._reference_state_message()

        assert message.activity == expected
        assert not message.reference_allowed
