"""Bounded, immutable, exact-TF-ready Costmap frames for direct C920 video.

This is deliberately display/semantic selection only: it never generates a
Costmap, estimates TF, extrapolates a future grid, or modifies Nav2 control.
A selected grid must still pass the Viewer's image-time TF and motion checks.
"""

from collections import deque
import math


class CostmapReadyHistory:
    """Select the last exact-TF-ready grid that predates one camera exposure.

    Each item comes from ``CostmapTransformQueue.resolve``.  A grid that was
    merely received, or was awaiting TF, must never enter this history.
    A frame-id epoch change immediately removes all prior snapshots.
    """

    def __init__(self, capacity: int = 4):
        if isinstance(capacity, bool) or not 2 <= capacity <= 16:
            raise ValueError("Costmap history capacity must be in 2..16")
        self.capacity = int(capacity)
        self._items = deque(maxlen=capacity)
        self._frame_id = None

    def clear(self):
        self._items.clear()
        self._frame_id = None

    def invalidate_if_frame_changed(self, frame_id: str):
        """An incoming map-frame epoch change invalidates old ready grids."""
        if self._frame_id is not None and str(frame_id) != self._frame_id:
            self.clear()

    @property
    def count(self):
        return len(self._items)

    def add(self, synchronized):
        """Accept only newer same-frame ready snapshots, never raw messages."""
        snapshot = getattr(synchronized, "snapshot", None)
        if snapshot is None:
            return False
        stamp = float(getattr(snapshot, "stamp_sec", float("nan")))
        frame_id = str(getattr(snapshot, "frame_id", ""))
        if not frame_id or not math.isfinite(stamp) or stamp <= 0.0:
            return False
        if self._frame_id is not None and frame_id != self._frame_id:
            self.clear()
        if self._items and stamp <= self._items[-1].snapshot.stamp_sec:
            return False
        self._frame_id = frame_id
        self._items.append(synchronized)
        return True

    def for_image(
        self, image_stamp_sec: float, ros_now_sec: float,
        *, maximum_image_age_sec: float, maximum_ros_age_sec: float,
    ):
        """Return a real older grid, or None. No future/expired admission."""
        values = (
            image_stamp_sec, ros_now_sec,
            maximum_image_age_sec, maximum_ros_age_sec,
        )
        if (
            not all(math.isfinite(float(v)) for v in values)
            or image_stamp_sec <= 0.0
            or maximum_image_age_sec <= 0.0
            or maximum_ros_age_sec <= 0.0
        ):
            return None
        # A reset/rewind in ROS time invalidates the history epoch.
        if self._items and (
            ros_now_sec < self._items[-1].snapshot.stamp_sec - 0.5
        ):
            self.clear()
            return None
        for item in reversed(self._items):
            snap = item.snapshot
            if snap.frame_id != self._frame_id:
                continue
            image_age = image_stamp_sec - snap.stamp_sec
            ros_age = ros_now_sec - snap.stamp_sec
            if (
                0.0 <= image_age <= maximum_image_age_sec
                and -0.02 <= ros_age <= maximum_ros_age_sec
            ):
                return item
        return None
