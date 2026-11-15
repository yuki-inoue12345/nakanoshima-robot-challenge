"""Fail-closed display gate for remaining real Nav2 Path length."""

from dataclasses import dataclass
import math
import re


_ROUTE_HASH = re.compile(r'^[0-9a-f]{64}$')
_SCHEMA_VERSION = 1
_NAVIGATING_ACTIVITY = 1
_ACTIVITY_STATUS = {
    0: 'WAITING FOR NAV',
    2: 'UPDATING ROUTE',
    3: 'TASK WAIT',
    4: 'PAUSED',
    5: 'PERSON SEARCH',
    6: 'PERSON SEARCH',
    7: 'PERSON SEARCH FAULT',
    8: 'STOPPED',
    9: 'ROUTE COMPLETE',
    10: 'STOPPING',
}


def _stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass(frozen=True)
class WaypointGoalView:
    route_sha256: str
    next_index: int
    next_number: int
    navigation_mode: str
    goal_generation: int
    stamp_sec: float
    minimum_path_generation: int


@dataclass(frozen=True)
class FollowerActivityView:
    activity: int
    stamp_sec: float


class WaypointProgressGate:
    """Authorize a plan-remaining label without changing Follower control.

    The distance comes only from the current real ``/plan`` polyline.  The
    Follower state is used as a safety/activity gate, but no claim is made that
    the Path endpoint is the next Waypoint.
    """

    def __init__(self, timeout_sec: float = 2.5, expected_frame: str = 'map'):
        if timeout_sec <= 0.0 or not math.isfinite(timeout_sec):
            raise ValueError('waypoint follower timeout must be finite and positive')
        self.timeout_sec = float(timeout_sec)
        self.expected_frame = str(expected_frame)
        self._view: WaypointGoalView | None = None
        self._activity: FollowerActivityView | None = None
        self.reason = 'waiting_for_follower_state'

    @staticmethod
    def _identity(message) -> tuple:
        return (
            str(message.route_sha256),
            int(message.next_waypoint_index),
            int(message.next_waypoint_number),
            str(message.navigation_mode),
            int(message.goal_generation),
        )

    def clear(self, reason: str, *, clear_activity: bool = True) -> None:
        self._view = None
        if clear_activity:
            self._activity = None
        self.reason = str(reason)

    def update(self, message, now_ros_sec: float, path_generation: int) -> bool:
        try:
            if int(message.schema_version) != _SCHEMA_VERSION:
                raise ValueError('unsupported_follower_schema')
            if str(message.header.frame_id) != self.expected_frame:
                raise ValueError('follower_frame_mismatch')
            route_hash, index, number, mode, goal_generation = self._identity(message)
            if not _ROUTE_HASH.fullmatch(route_hash):
                raise ValueError('invalid_route_hash')
            waypoint_count = int(message.waypoint_count)
            if waypoint_count <= 0 or not 0 <= index < waypoint_count:
                raise ValueError('next_waypoint_index_out_of_range')
            if mode not in ('to_pose', 'through_poses'):
                raise ValueError('invalid_navigation_mode')
            stamp = _stamp_sec(message.header.stamp)
            age = float(now_ros_sec) - stamp
            if stamp <= 0.0:
                raise ValueError('follower_zero_timestamp')
            if age < -0.5:
                raise ValueError('follower_future_timestamp')
            if age > self.timeout_sec:
                raise ValueError(f'follower_state_stale age={age:.3f}s')
            activity = int(message.activity)
            if activity not in range(0, 11):
                raise ValueError(f'invalid_follower_activity activity={activity}')
        except (AttributeError, TypeError, ValueError) as exc:
            self.clear(str(exc))
            return False

        if activity != _NAVIGATING_ACTIVITY or not bool(message.reference_allowed):
            self._view = None
            self._activity = FollowerActivityView(activity, stamp)
            self.reason = f'follower_not_navigating activity={activity}'
            return False

        identity = (route_hash, index, number, mode, goal_generation)
        previous_identity = None
        minimum_generation = int(path_generation) + 1
        if self._view is not None:
            previous_identity = (
                self._view.route_sha256,
                self._view.next_index,
                self._view.next_number,
                self._view.navigation_mode,
                self._view.goal_generation,
            )
            if identity == previous_identity:
                minimum_generation = self._view.minimum_path_generation
        self._view = WaypointGoalView(
            route_hash,
            index,
            number,
            mode,
            goal_generation,
            stamp,
            minimum_generation,
        )
        self._activity = None
        self.reason = (
            'waiting_for_plan_after_goal_change'
            if identity != previous_identity
            else 'valid'
        )
        return True

    def navigation_hold_status(self, now_ros_sec: float) -> str | None:
        """Return a truthful UI label for a fresh explicit Follower hold."""
        activity = self._activity
        if activity is None:
            return None
        age = float(now_ros_sec) - activity.stamp_sec
        if age < -0.5 or age > self.timeout_sec:
            self.clear(f'follower_state_stale age={age:.3f}s')
            return None
        return _ACTIVITY_STATUS.get(activity.activity, 'STOPPED')

    def label(
        self,
        now_ros_sec: float,
        path_generation: int,
        remaining_path_m: float | None,
        anchor_distance_m: float | None,
        maximum_anchor_distance_m: float,
    ) -> str | None:
        view = self._view
        if view is None:
            return None
        age = float(now_ros_sec) - view.stamp_sec
        if age < -0.5 or age > self.timeout_sec:
            self.clear(f'follower_state_stale age={age:.3f}s')
            return None
        if int(path_generation) < view.minimum_path_generation:
            self.reason = 'waiting_for_plan_after_goal_change'
            return None
        if (
            remaining_path_m is None
            or anchor_distance_m is None
            or not math.isfinite(remaining_path_m)
            or not math.isfinite(anchor_distance_m)
            or remaining_path_m < 0.0
            or anchor_distance_m > maximum_anchor_distance_m
        ):
            self.reason = 'path_progress_not_reliable'
            return None
        self.reason = 'valid'
        if remaining_path_m < 1.0:
            distance = f'{int(round(remaining_path_m * 100.0))} cm'
        else:
            distance = f'{remaining_path_m:.1f} m'
        return f'PLAN LEFT: {distance}'

    def authorizes_plan(self, now_ros_sec: float, path_generation: int) -> bool:
        """Confirm that guidance belongs to a fresh, navigating Follower goal."""
        view = self._view
        if view is None:
            return False
        age = float(now_ros_sec) - view.stamp_sec
        if age < -0.5 or age > self.timeout_sec:
            self.clear(f'follower_state_stale age={age:.3f}s')
            return False
        if int(path_generation) < view.minimum_path_generation:
            self.reason = 'waiting_for_plan_after_goal_change'
            return False
        self.reason = 'valid'
        return True
