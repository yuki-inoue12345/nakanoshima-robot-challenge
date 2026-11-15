"""Fail-closed Local Costmap sampling and camera projection.

The input is Nav2's ``nav_msgs/OccupancyGrid`` view of the local costmap.
Values are kept semantic instead of being described as generic danger:

* ``-1``: unknown
* ``high_cost_min .. lethal_cost_min - 1``: high navigation cost
* ``lethal_cost_min .. 100``: lethal-cost cells in the Nav2 grid

The module has no ROS node dependency, which keeps geometry and freshness
tests reproducible without connecting to a robot.
"""

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from .projection import CameraIntrinsics, CameraMount, world_to_optical
from .tf_adapter import quaternion_matrix_xyzw


UNKNOWN_CLASS = 0
HIGH_COST_CLASS = 1
LETHAL_COST_CLASS = 2


@dataclass(frozen=True)
class CostmapOverlaySettings:
    """Validation, classification, and rendering workload limits."""

    timeout_sec: float = 1.5
    stamp_max_age_sec: float = 1.5
    future_tolerance_sec: float = 0.5
    high_cost_min: int = 60
    lethal_cost_min: int = 100
    max_points_per_class: int = 450
    plane_height_m: float = 0.02

    def validate(self) -> None:
        values = (
            self.timeout_sec,
            self.stamp_max_age_sec,
            self.future_tolerance_sec,
            self.plane_height_m,
        )
        if not np.isfinite(values).all():
            raise ValueError("costmap overlay settings must be finite")
        if self.timeout_sec <= 0.0 or self.stamp_max_age_sec <= 0.0:
            raise ValueError("costmap overlay timeouts must be positive")
        if self.future_tolerance_sec < 0.0:
            raise ValueError("costmap future tolerance must be non-negative")
        if not 1 <= self.high_cost_min < self.lethal_cost_min <= 100:
            raise ValueError(
                "costmap thresholds must satisfy 1 <= high < lethal <= 100"
            )
        if self.max_points_per_class < 1:
            raise ValueError("costmap max_points_per_class must be positive")


@dataclass(frozen=True)
class CostmapSnapshot:
    """Validated and bounded costmap points in the message frame."""

    frame_id: str
    stamp_sec: float
    received_at: float
    points: np.ndarray
    classes: np.ndarray
    analysis_points: np.ndarray
    analysis_costs: np.ndarray
    unknown_cells: int
    high_cost_cells: int
    lethal_cost_cells: int

    @property
    def sampled_cells(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class ProjectedCostmap:
    """Image pixels grouped by the exact cost class used for rendering."""

    unknown_pixels: np.ndarray
    high_cost_pixels: np.ndarray
    lethal_cost_pixels: np.ndarray
    input_points: int
    visible_points: int


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _bounded_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministically retain spatially distributed row-major cells."""
    if indices.size <= maximum:
        return indices
    selected = np.linspace(0, indices.size - 1, maximum, dtype=np.int64)
    return indices[selected]


def occupancy_grid_to_snapshot(
    message,
    settings: CostmapOverlaySettings,
    *,
    ros_now_sec: float,
    received_at: float,
) -> CostmapSnapshot:
    """Validate and sample one ``nav_msgs/OccupancyGrid`` message."""
    settings.validate()
    frame_id = str(message.header.frame_id).strip()
    if not frame_id:
        raise ValueError("costmap frame_id is empty")
    stamp_sec = _stamp_to_sec(message.header.stamp)
    if not math.isfinite(stamp_sec) or stamp_sec <= 0.0:
        raise ValueError("costmap timestamp is zero or invalid")
    ros_now = float(ros_now_sec)
    if not math.isfinite(ros_now):
        raise ValueError("ROS time must be finite")
    stamp_age = ros_now - stamp_sec
    if stamp_age > settings.stamp_max_age_sec:
        raise ValueError(
            f"costmap timestamp stale age={stamp_age:.3f}s "
            f"limit={settings.stamp_max_age_sec:.3f}s"
        )
    if stamp_age < -settings.future_tolerance_sec:
        raise ValueError(
            f"costmap timestamp future offset={-stamp_age:.3f}s"
        )

    width = int(message.info.width)
    height = int(message.info.height)
    resolution = float(message.info.resolution)
    if width < 1 or height < 1:
        raise ValueError("costmap dimensions must be positive")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("costmap resolution must be finite and positive")
    data = np.asarray(message.data, dtype=np.int16)
    if data.size != width * height:
        raise ValueError(
            f"costmap data length mismatch expected={width * height} "
            f"actual={data.size}"
        )
    if np.any(data < -1) or np.any(data > 100):
        raise ValueError("OccupancyGrid costs must be -1 or within [0, 100]")

    origin = message.info.origin
    translation = np.asarray(
        (origin.position.x, origin.position.y, origin.position.z),
        dtype=np.float64,
    )
    quaternion = (
        origin.orientation.x,
        origin.orientation.y,
        origin.orientation.z,
        origin.orientation.w,
    )
    rotation = quaternion_matrix_xyzw(quaternion)
    # OccupancyGrid is a horizontal 2-D plane.  Reject tilted origins instead
    # of silently projecting cells at invented heights.
    if not np.allclose(rotation[:, 2], (0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("costmap origin is not a horizontal plane")
    if not np.isfinite(translation).all():
        raise ValueError("costmap origin contains NaN or Inf")

    flat_all = np.arange(data.size, dtype=np.int64)
    rows_all, columns_all = np.divmod(flat_all, width)
    local_all = np.column_stack((
        (columns_all.astype(np.float64) + 0.5) * resolution,
        (rows_all.astype(np.float64) + 0.5) * resolution,
        np.full(data.size, settings.plane_height_m, dtype=np.float64),
    ))
    analysis_points = (rotation @ local_all.T).T + translation

    masks = (
        data == -1,
        (data >= settings.high_cost_min) & (data < settings.lethal_cost_min),
        data >= settings.lethal_cost_min,
    )
    class_counts = tuple(int(np.count_nonzero(mask)) for mask in masks)
    points = []
    classes = []
    for class_id, mask in enumerate(masks):
        flat = _bounded_indices(np.flatnonzero(mask), settings.max_points_per_class)
        if flat.size == 0:
            continue
        points.append(analysis_points[flat])
        classes.append(np.full(flat.size, class_id, dtype=np.uint8))

    point_array = (
        np.concatenate(points, axis=0)
        if points else np.empty((0, 3), dtype=np.float64)
    )
    class_array = (
        np.concatenate(classes, axis=0)
        if classes else np.empty((0,), dtype=np.uint8)
    )
    return CostmapSnapshot(
        frame_id=frame_id,
        stamp_sec=stamp_sec,
        received_at=float(received_at),
        points=point_array,
        classes=class_array,
        analysis_points=analysis_points,
        analysis_costs=data.copy(),
        unknown_cells=class_counts[UNKNOWN_CLASS],
        high_cost_cells=class_counts[HIGH_COST_CLASS],
        lethal_cost_cells=class_counts[LETHAL_COST_CLASS],
    )


class CostmapStore:
    """Keep only a fresh, valid costmap snapshot; invalid input clears it."""

    def __init__(
        self,
        settings: CostmapOverlaySettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.clock = clock
        self.snapshot = None
        self.reason = "waiting_for_local_costmap"
        self.receive_count = 0

    def update(self, message, ros_now_sec: float) -> bool:
        self.receive_count += 1
        try:
            snapshot = occupancy_grid_to_snapshot(
                message,
                self.settings,
                ros_now_sec=ros_now_sec,
                received_at=self.clock(),
            )
        except (AttributeError, TypeError, ValueError) as error:
            self.snapshot = None
            self.reason = f"invalid_local_costmap {error}"
            return False
        self.snapshot = snapshot
        self.reason = "local_costmap_ready"
        return True

    def state(self, ros_now_sec: float):
        snapshot = self.snapshot
        if snapshot is None:
            return "UNAVAILABLE", None, self.reason
        receipt_age = max(0.0, self.clock() - snapshot.received_at)
        if receipt_age > self.settings.timeout_sec:
            self.snapshot = None
            self.reason = (
                f"local_costmap_receive_stale age={receipt_age:.3f}s "
                f"limit={self.settings.timeout_sec:.3f}s"
            )
            return "STALE", None, self.reason
        stamp_age = float(ros_now_sec) - snapshot.stamp_sec
        if stamp_age > self.settings.stamp_max_age_sec:
            self.snapshot = None
            self.reason = (
                f"local_costmap_stamp_stale age={stamp_age:.3f}s "
                f"limit={self.settings.stamp_max_age_sec:.3f}s"
            )
            return "STALE", None, self.reason
        if stamp_age < -self.settings.future_tolerance_sec:
            self.snapshot = None
            self.reason = (
                f"local_costmap_stamp_future offset={-stamp_age:.3f}s"
            )
            return "UNAVAILABLE", None, self.reason
        return "ACTIVE", snapshot, "local_costmap_ready"

    def clear(self, reason: str) -> None:
        self.snapshot = None
        self.reason = str(reason)


def project_costmap_points(
    points_in_base: np.ndarray,
    classes: np.ndarray,
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
) -> ProjectedCostmap:
    """Project classified base-frame points into the calibrated image."""
    points = np.asarray(points_in_base, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("costmap points must have shape (N, 3)")
    if class_values.shape != (points.shape[0],):
        raise ValueError("costmap classes must match point count")
    if not math.isfinite(float(near)) or near <= 0.0:
        raise ValueError("near clip must be finite and positive")
    if points.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int32)
        return ProjectedCostmap(empty, empty, empty, 0, 0)
    optical = world_to_optical(points, (0.0, 0.0, 0.0), mount)
    depth = optical[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * optical[:, 0] / depth + intrinsics.cx
        v = intrinsics.fy * optical[:, 1] / depth + intrinsics.cy
    visible = (
        np.isfinite(u)
        & np.isfinite(v)
        & (depth >= near)
        & (u >= 0.0)
        & (u <= intrinsics.width - 1.0)
        & (v >= 0.0)
        & (v <= intrinsics.height - 1.0)
    )
    pixels = np.column_stack((u, v))

    def group(class_id: int) -> np.ndarray:
        selected = visible & (class_values == class_id)
        return np.rint(pixels[selected]).astype(np.int32)

    return ProjectedCostmap(
        unknown_pixels=group(UNKNOWN_CLASS),
        high_cost_pixels=group(HIGH_COST_CLASS),
        lethal_cost_pixels=group(LETHAL_COST_CLASS),
        input_points=int(points.shape[0]),
        visible_points=int(np.count_nonzero(visible)),
    )
