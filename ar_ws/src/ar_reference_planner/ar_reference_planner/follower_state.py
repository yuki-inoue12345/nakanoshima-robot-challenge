"""Validate the typed read-only waypoint follower state contract."""

from dataclasses import dataclass
import re

from robotbase_waypoint_interfaces.msg import WaypointFollowerState

from .route import Route


HASH_PATTERN = re.compile(r'^[0-9a-f]{64}$')
VALID_MODES = {'to_pose', 'through_poses'}
VALID_ACTIVITIES = {
    WaypointFollowerState.ACTIVITY_STARTING,
    WaypointFollowerState.ACTIVITY_NAVIGATING,
    WaypointFollowerState.ACTIVITY_TRANSITIONING,
    WaypointFollowerState.ACTIVITY_TASK_WAITING,
    WaypointFollowerState.ACTIVITY_MANUAL_PAUSED,
    WaypointFollowerState.ACTIVITY_PERSON_SEARCH_STOPPING,
    WaypointFollowerState.ACTIVITY_PERSON_SEARCH_WAITING,
    WaypointFollowerState.ACTIVITY_PERSON_SEARCH_FAULT,
    WaypointFollowerState.ACTIVITY_HOLDING_STOP,
    WaypointFollowerState.ACTIVITY_FINISHED,
    WaypointFollowerState.ACTIVITY_SHUTTING_DOWN,
}


@dataclass(frozen=True)
class FollowerSnapshot:
    """Immutable follower state used to bind one planner request."""

    stamp_ns: int
    frame_id: str
    route_sha256: str
    next_index: int
    next_number: int
    navigation_mode: str
    goal_generation: int
    activity: int
    reference_allowed: bool
    safe_end_index: int
    waypoint_count: int

    @classmethod
    def from_message(cls, message: WaypointFollowerState):
        """Copy ROS sequences/scalars into a stable validation object."""
        if message.schema_version != WaypointFollowerState.SCHEMA_VERSION:
            raise ValueError(f'unsupported_schema={message.schema_version}')
        if not HASH_PATTERN.fullmatch(message.route_sha256):
            raise ValueError('route_sha256 must be 64 lowercase hex characters')
        if message.navigation_mode not in VALID_MODES:
            raise ValueError(f'invalid_navigation_mode={message.navigation_mode!r}')
        if message.activity not in VALID_ACTIVITIES:
            raise ValueError(f'invalid_activity={message.activity}')
        if message.reference_allowed != (
                message.activity == WaypointFollowerState.ACTIVITY_NAVIGATING):
            raise ValueError('reference_allowed disagrees with activity')
        stamp_ns = message.header.stamp.sec * 1_000_000_000
        stamp_ns += message.header.stamp.nanosec
        return cls(
            stamp_ns=stamp_ns,
            frame_id=str(message.header.frame_id),
            route_sha256=str(message.route_sha256),
            next_index=int(message.next_waypoint_index),
            next_number=int(message.next_waypoint_number),
            navigation_mode=str(message.navigation_mode),
            goal_generation=int(message.goal_generation),
            activity=int(message.activity),
            reference_allowed=bool(message.reference_allowed),
            safe_end_index=int(message.safe_section_end_index),
            waypoint_count=int(message.waypoint_count),
        )

    @property
    def identity(self):
        """Identify all follower fields that bind a planner result."""
        return (
            self.route_sha256,
            self.next_index,
            self.next_number,
            self.navigation_mode,
            self.goal_generation,
            self.safe_end_index,
        )


def validate_snapshot(
        snapshot: FollowerSnapshot, route: Route, map_frame: str,
        now_ns: int, maximum_age_sec: float) -> str:
    """Return an empty string only when a snapshot is safe for planning."""
    if snapshot.frame_id != map_frame:
        return f'frame_id={snapshot.frame_id!r} expected={map_frame!r}'
    if snapshot.route_sha256 != route.sha256:
        return 'route_hash_mismatch'
    if snapshot.waypoint_count != len(route.waypoints):
        return 'waypoint_count_mismatch'
    if not 0 <= snapshot.next_index < len(route.waypoints):
        return 'next_waypoint_index_out_of_range'
    if route.waypoints[snapshot.next_index].number != snapshot.next_number:
        return 'next_waypoint_number_mismatch'
    if not snapshot.next_index <= snapshot.safe_end_index < len(route.waypoints):
        return 'safe_section_end_index_out_of_range'
    if snapshot.stamp_ns <= 0:
        return 'zero_timestamp'
    age_ns = now_ns - snapshot.stamp_ns
    if age_ns < -500_000_000:
        return 'future_timestamp'
    if age_ns > int(maximum_age_sec * 1_000_000_000):
        return f'stale_state age_sec={age_ns / 1e9:.3f}'
    if not snapshot.reference_allowed:
        return f'reference_disallowed activity={snapshot.activity}'
    return ''
