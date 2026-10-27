"""Validation and atomically replaced state for incoming nav_msgs/Path data."""

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


class PathStore:
    """Thread-safe whole-Path exchange with monotonic staleness tracking."""

    def __init__(self, timeout_sec: float = 6.5, clock: Callable[[], float] = time.monotonic):
        if timeout_sec <= 0.0 or not np.isfinite(timeout_sec):
            raise ValueError("timeout_sec must be finite and positive")
        self.timeout_sec = float(timeout_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: Optional[PathSnapshot] = None
        self._generation = 0

    @staticmethod
    def validate_frame_id(frame_id: str) -> str:
        clean = str(frame_id).strip()
        if not clean or clean.startswith("/") or not _FRAME_PATTERN.fullmatch(clean):
            raise ValueError(f"invalid frame_id: {frame_id!r}")
        return clean

    def replace(self, points: np.ndarray, frame_id: str, received_at: float | None = None) -> PathSnapshot:
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
        immutable = np.array(array, copy=True)
        immutable.setflags(write=False)
        with self._lock:
            self._generation += 1
            self._snapshot = PathSnapshot(immutable, frame, timestamp, self._generation)
            return self._snapshot

    def snapshot(self) -> Optional[PathSnapshot]:
        with self._lock:
            return self._snapshot

    def clear(self) -> None:
        """Atomically remove the current path after an invalid/empty external state."""
        with self._lock:
            self._generation += 1
            self._snapshot = None

    def state(self, now: float | None = None) -> Tuple[str, Optional[PathSnapshot]]:
        snapshot = self.snapshot()
        if snapshot is None or len(snapshot.points) == 0:
            return "NO PATH", snapshot
        timestamp = self._clock() if now is None else float(now)
        if timestamp - snapshot.received_monotonic > self.timeout_sec:
            return "PATH STALE", snapshot
        return "ACTIVE", snapshot


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
