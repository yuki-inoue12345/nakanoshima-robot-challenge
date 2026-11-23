"""Fail-closed binding between a reference Path and follower progress state."""

from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Callable

import numpy as np

from .path_adapter import PathSnapshot, PathStore, path_message_to_array


_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VALID_MODES = {"to_pose", "through_poses"}
_FOLLOWER_SCHEMA_VERSION = 1
_REFERENCE_SCHEMA_VERSION = 2
_NAVIGATING_ACTIVITY = 1


def _stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass(frozen=True)
class FollowerIdentity:
    """Fields that must exactly match the state used for reference planning."""

    route_sha256: str
    next_index: int
    next_number: int
    navigation_mode: str
    goal_generation: int
    safe_end_index: int
    waypoint_count: int


@dataclass(frozen=True)
class FollowerView:
    identity: FollowerIdentity
    stamp_sec: float


@dataclass(frozen=True)
class ReferenceMetadata:
    identity: FollowerIdentity
    planner_generation: int
    source_state_stamp_sec: float
    declared_length_m: float


class ReferencePathGate:
    """Accept a reference path only while its exact follower identity is current."""

    def __init__(
        self,
        timeout_sec: float = 6.5,
        follower_timeout_sec: float = 2.5,
        expected_frame: str = "map",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if follower_timeout_sec <= 0.0 or not math.isfinite(follower_timeout_sec):
            raise ValueError("reference_follower_timeout_sec must be finite and positive")
        self.path_store = PathStore(timeout_sec, clock)
        self.follower_timeout_sec = float(follower_timeout_sec)
        self.expected_frame = PathStore.validate_frame_id(expected_frame)
        self._follower: FollowerView | None = None
        self._metadata: ReferenceMetadata | None = None
        self._pending_reference: tuple[object, float] | None = None
        self._lock = threading.RLock()
        self.reason = "waiting_for_follower_state"

    @staticmethod
    def _identity(message) -> FollowerIdentity:
        route_hash = str(message.route_sha256)
        mode = str(message.navigation_mode)
        if not _HASH_PATTERN.fullmatch(route_hash):
            raise ValueError("invalid_route_hash")
        if mode not in _VALID_MODES:
            raise ValueError(f"invalid_navigation_mode={mode!r}")
        waypoint_count = int(message.waypoint_count)
        next_index = int(message.next_waypoint_index)
        safe_end = int(message.safe_section_end_index)
        if waypoint_count <= 0:
            raise ValueError("invalid_waypoint_count")
        if not 0 <= next_index < waypoint_count:
            raise ValueError("next_waypoint_index_out_of_range")
        if not next_index <= safe_end < waypoint_count:
            raise ValueError("safe_section_end_index_out_of_range")
        return FollowerIdentity(
            route_hash,
            next_index,
            int(message.next_waypoint_number),
            mode,
            int(message.goal_generation),
            safe_end,
            waypoint_count,
        )

    def clear(self, reason: str) -> None:
        with self._lock:
            self.path_store.clear()
            self._metadata = None
            self._pending_reference = None
            self.reason = str(reason)

    def update_follower(self, message, now_ros_sec: float) -> bool:
        """Update the live authorization and immediately clear on pause/change."""
        with self._lock:
            return self._update_follower_locked(message, now_ros_sec)

    def _update_follower_locked(self, message, now_ros_sec: float) -> bool:
        try:
            if int(message.schema_version) != _FOLLOWER_SCHEMA_VERSION:
                raise ValueError(f"unsupported_follower_schema={message.schema_version}")
            if str(message.header.frame_id) != self.expected_frame:
                raise ValueError(
                    f"follower_frame={message.header.frame_id!r} "
                    f"expected={self.expected_frame!r}"
                )
            identity = self._identity(message)
            stamp = _stamp_sec(message.header.stamp)
            age = float(now_ros_sec) - stamp
            if stamp <= 0.0:
                raise ValueError("follower_zero_timestamp")
            if age < -0.5:
                raise ValueError("follower_future_timestamp")
            if age > self.follower_timeout_sec:
                raise ValueError(f"follower_state_stale age={age:.3f}s")
            if (
                int(message.activity) != _NAVIGATING_ACTIVITY
                or not bool(message.reference_allowed)
            ):
                raise ValueError(f"reference_disallowed activity={message.activity}")
        except (AttributeError, TypeError, ValueError) as exc:
            self._follower = None
            self.clear(str(exc))
            return False

        self._follower = FollowerView(identity, stamp)
        if self._metadata is not None and self._metadata.identity != identity:
            self.clear("reference_identity_changed")
        elif self._metadata is None:
            self.reason = "waiting_for_reference_path"
        pending = self._pending_reference
        if pending is not None:
            pending_message, pending_received_sec = pending
            self._pending_reference = None
            # A reference result can arrive just before the corresponding
            # periodic Follower state.  Revalidate the complete message now;
            # no route is made visible while the identity/stamp is unproven.
            self.update_reference(pending_message, pending_received_sec)
        return True

    def update_reference(self, message, now_ros_sec: float) -> bool:
        """Atomically validate metadata and Path before replacing visible state."""
        with self._lock:
            return self._update_reference_locked(message, now_ros_sec)

    def _update_reference_locked(self, message, now_ros_sec: float) -> bool:
        try:
            if int(message.schema_version) != _REFERENCE_SCHEMA_VERSION:
                raise ValueError(f"unsupported_reference_schema={message.schema_version}")
            if not bool(message.valid):
                reason = str(message.invalid_reason) or "planner_marked_invalid"
                self.clear(reason)
                return False
            if self._follower is None:
                raise ValueError("no_current_follower_state")
            if str(message.header.frame_id) != self.expected_frame:
                raise ValueError(
                    f"reference_frame={message.header.frame_id!r} "
                    f"expected={self.expected_frame!r}"
                )
            envelope_stamp = _stamp_sec(message.header.stamp)
            envelope_age = float(now_ros_sec) - envelope_stamp
            if envelope_stamp <= 0.0:
                raise ValueError("reference_zero_timestamp")
            if envelope_age < -0.5:
                raise ValueError("reference_future_timestamp")
            if envelope_age > self.path_store.timeout_sec:
                raise ValueError(f"reference_message_stale age={envelope_age:.3f}s")
            identity = self._identity(message)
            if identity != self._follower.identity:
                raise ValueError("reference_identity_mismatch")
            source_stamp = _stamp_sec(message.follower_state_stamp)
            if source_stamp <= 0.0:
                raise ValueError("reference_source_state_zero_timestamp")
            if source_stamp > self._follower.stamp_sec + 1e-6:
                # DDS delivery order between the two topics is not atomic.
                # Keep one candidate only when its exact route/goal identity
                # already matches.  It remains invisible until a fresh
                # Follower message reaches source_stamp and all checks below
                # are rerun.
                self._pending_reference = (message, float(now_ros_sec))
                self.reason = "waiting_for_matching_follower_state"
                return False
            points, frame = path_message_to_array(message.path)
            if frame != self.expected_frame:
                raise ValueError(
                    f"path_frame={frame!r} expected={self.expected_frame!r}"
                )
            path_stamp = _stamp_sec(message.path.header.stamp)
            path_age = float(now_ros_sec) - path_stamp
            if path_stamp <= 0.0:
                raise ValueError("reference_path_zero_timestamp")
            if path_age < -0.5:
                raise ValueError("reference_path_future_timestamp")
            if path_age > self.path_store.timeout_sec:
                raise ValueError(f"reference_path_stamp_stale age={path_age:.3f}s")
            for index, pose in enumerate(message.path.poses):
                pose_frame = str(pose.header.frame_id)
                if pose_frame not in ("", self.expected_frame):
                    raise ValueError(
                        f"reference_pose_{index}_frame={pose_frame!r}"
                    )
            if len(points) < 2:
                raise ValueError("reference_path_has_fewer_than_two_points")
            declared_length = float(message.path_length_m)
            measured_length = float(
                np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum()
            )
            if not math.isfinite(declared_length) or declared_length <= 0.0:
                raise ValueError("invalid_declared_path_length")
            if abs(declared_length - measured_length) > max(0.05, measured_length * 0.01):
                raise ValueError(
                    "reference_path_length_mismatch "
                    f"declared={declared_length:.3f} measured={measured_length:.3f}"
                )
            planner_generation = int(message.planner_generation)
            if planner_generation <= 0:
                raise ValueError("invalid_planner_generation")
        except (AttributeError, TypeError, ValueError) as exc:
            self.clear(str(exc))
            return False

        self.path_store.replace(
            points,
            frame,
            header_stamp_sec=path_stamp,
            received_ros_sec=float(now_ros_sec),
        )
        self._metadata = ReferenceMetadata(
            identity,
            planner_generation,
            source_stamp,
            declared_length,
        )
        self._pending_reference = None
        self.reason = "valid"
        return True

    def state(
        self, now_ros_sec: float, now_monotonic: float | None = None
    ) -> tuple[str, PathSnapshot | None, str]:
        with self._lock:
            return self._state_locked(now_ros_sec, now_monotonic)

    def _state_locked(
        self, now_ros_sec: float, now_monotonic: float | None = None
    ) -> tuple[str, PathSnapshot | None, str]:
        if self._follower is None:
            return "INVALID", None, self.reason
        follower_age = float(now_ros_sec) - self._follower.stamp_sec
        if follower_age < -0.5:
            self.clear("follower_future_timestamp")
            return "INVALID", None, self.reason
        if follower_age > self.follower_timeout_sec:
            self.clear(f"follower_state_stale age={follower_age:.3f}s")
            return "INVALID", None, self.reason
        if self._metadata is None or self._metadata.identity != self._follower.identity:
            return "INVALID", None, self.reason
        path_state, snapshot = self.path_store.state(now_monotonic)
        if path_state == "PATH STALE":
            self.clear("reference_path_expired")
            return "INVALID", None, self.reason
        if path_state != "ACTIVE":
            return "INVALID", None, self.reason
        return "ACTIVE", snapshot, "valid"

    def age(self, now_monotonic: float | None = None) -> float:
        with self._lock:
            return self.path_store.age(now_monotonic)

    def state_for_ros_time(
        self, ros_time_sec: float
    ) -> tuple[str, PathSnapshot | None]:
        with self._lock:
            return self.path_store.state_for_ros_time(ros_time_sec)

    @property
    def metadata(self) -> ReferenceMetadata | None:
        with self._lock:
            return self._metadata
