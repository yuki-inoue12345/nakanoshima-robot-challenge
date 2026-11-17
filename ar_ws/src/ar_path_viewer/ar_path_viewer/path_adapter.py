"""Validation and atomically replaced state for incoming nav_msgs/Path data."""

from collections import deque
from dataclasses import dataclass
import re
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np


_FRAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_/]*$")


@dataclass(frozen=True)
class PathSnapshot:
    points: np.ndarray
    frame_id: str
    received_monotonic: float
    generation: int
    header_stamp_sec: Optional[float] = None
    received_ros_sec: Optional[float] = None


class PathStore:
    """Thread-safe whole-Path exchange with monotonic staleness tracking."""

    def __init__(
        self,
        timeout_sec: float = 6.5,
        clock: Callable[[], float] = time.monotonic,
        history_size: int = 8,
    ):
        if timeout_sec <= 0.0 or not np.isfinite(timeout_sec):
            raise ValueError("timeout_sec must be finite and positive")
        self.timeout_sec = float(timeout_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: Optional[PathSnapshot] = None
        if int(history_size) < 1:
            raise ValueError("history_size must be positive")
        self._history = deque(maxlen=int(history_size))
        self._generation = 0

    @staticmethod
    def validate_frame_id(frame_id: str) -> str:
        clean = str(frame_id).strip()
        if not clean or clean.startswith("/") or not _FRAME_PATTERN.fullmatch(clean):
            raise ValueError(f"invalid frame_id: {frame_id!r}")
        return clean

    def replace(
        self, points: np.ndarray, frame_id: str,
        received_at: float | None = None,
        *, header_stamp_sec: float | None = None,
        received_ros_sec: float | None = None,
    ) -> PathSnapshot:
        frame = self.validate_frame_id(frame_id)
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            array = np.empty((0, 3), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("path points must have shape (N, 3)")
        if not np.isfinite(array).all():
            raise ValueError("path points contain NaN or Inf")
        timestamp = self._clock() if received_at is None else float(received_at)
        if not np.isfinite(timestamp):
            raise ValueError("received_at must be finite")
        if header_stamp_sec is not None and (
            not np.isfinite(header_stamp_sec) or header_stamp_sec <= 0.0
        ):
            raise ValueError("Path header stamp must be finite and positive")
        if received_ros_sec is not None and not np.isfinite(received_ros_sec):
            raise ValueError("Path ROS receipt time must be finite")
        immutable = np.array(array, copy=True)
        immutable.setflags(write=False)
        with self._lock:
            self._generation += 1
            if len(immutable) == 0:
                self._history.clear()
            self._snapshot = PathSnapshot(
                immutable, frame, timestamp, self._generation,
                header_stamp_sec, received_ros_sec,
            )
            self._history.append(self._snapshot)
            return self._snapshot

    def snapshot(self) -> Optional[PathSnapshot]:
        with self._lock:
            return self._snapshot

    def clear(self) -> None:
        """Atomically remove the current path after an invalid/empty external state."""
        with self._lock:
            self._generation += 1
            self._snapshot = None
            self._history.clear()

    def state_for_ros_time(
        self, image_stamp_sec: float, now: float | None = None
    ) -> Tuple[str, Optional[PathSnapshot]]:
        """Return the newest Path that had arrived by an image exposure time.

        This is deliberately based on the immutable ROS receipt timestamp, not
        the Path header (which describes plan generation).  It prevents a new
        plan from being drawn onto an older delayed camera frame.  Invalid or
        empty updates clear the complete history, so this method cannot revive
        a route across a fail-closed boundary.
        """
        stamp = float(image_stamp_sec)
        if not np.isfinite(stamp) or stamp <= 0.0:
            return "NO PATH", None
        with self._lock:
            candidates = tuple(self._history)
        snapshot = next(
            (
                item for item in reversed(candidates)
                if item.received_ros_sec is not None
                and item.received_ros_sec <= stamp + 1e-9
            ),
            None,
        )
        if snapshot is None or len(snapshot.points) == 0:
            return "NO PATH", snapshot
        timestamp = self._clock() if now is None else float(now)
        if timestamp - snapshot.received_monotonic > self.timeout_sec:
            return "PATH STALE", snapshot
        return "ACTIVE", snapshot

    def state(self, now: float | None = None) -> Tuple[str, Optional[PathSnapshot]]:
        snapshot = self.snapshot()
        if snapshot is None or len(snapshot.points) == 0:
            return "NO PATH", snapshot
        timestamp = self._clock() if now is None else float(now)
        if timestamp - snapshot.received_monotonic > self.timeout_sec:
            return "PATH STALE", snapshot
        return "ACTIVE", snapshot

    def age(self, now: float | None = None) -> float:
        snapshot = self.snapshot()
        if snapshot is None:
            return 0.0
        timestamp = self._clock() if now is None else float(now)
        return timestamp - snapshot.received_monotonic


def path_message_to_array(message) -> Tuple[np.ndarray, str]:
    """Convert a nav_msgs/Path-like object without depending on ROS in tests."""
    frame = PathStore.validate_frame_id(message.header.frame_id)
    points = np.array(
        [
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            for pose in message.poses
        ],
        dtype=np.float64,
    )
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("path points contain NaN or Inf")
    return points, frame
