"""Fail-closed display gate for remaining real Nav2 Path length."""

from dataclasses import dataclass
import math
import re


_ROUTE_HASH = re.compile(r'^[0-9a-f]{64}$')
_SCHEMA_VERSION = 1
_NAVIGATING_ACTIVITY = 1


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

    def clear(self, reason: str) -> None:
        self._view = None
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
            if (
                int(message.activity) != _NAVIGATING_ACTIVITY
                or not bool(message.reference_allowed)
            ):
                raise ValueError(f'follower_not_navigating activity={message.activity}')
        except (AttributeError, TypeError, ValueError) as exc:
            self.clear(str(exc))
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
        self.reason = (
            'waiting_for_plan_after_goal_change'
            if identity != previous_identity
            else 'valid'
        )
        return True

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
