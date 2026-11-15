"""Neutral, evidence-bound alerts from an exact-time Costmap and real Path."""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class CostmapAlertSettings:
    path_corridor_width_m: float = 0.35
    nearby_lethal_distance_m: float = 0.60
    robot_half_length_m: float = 0.30
    robot_half_width_m: float = 0.29
    confirm_sec: float = 0.25
    release_sec: float = 0.40

    def validate(self) -> None:
        values = (
            self.path_corridor_width_m,
            self.nearby_lethal_distance_m,
            self.robot_half_length_m,
            self.robot_half_width_m,
            self.confirm_sec,
            self.release_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("costmap alert settings must be finite")
        if self.path_corridor_width_m <= 0.0:
            raise ValueError("costmap alert corridor must be positive")
        if self.nearby_lethal_distance_m <= 0.0:
            raise ValueError("costmap nearby distance must be positive")
        if self.robot_half_length_m <= 0.0 or self.robot_half_width_m <= 0.0:
            raise ValueError("costmap alert footprint must be positive")
        if self.confirm_sec < 0.0 or self.release_sec < 0.0:
            raise ValueError("costmap alert hysteresis must be non-negative")


@dataclass(frozen=True)
class CostmapAlert:
    label: str
    kind: str
    detail: str


def _minimum_polyline_distance(points_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) < 2:
        return np.full(len(points_xy), np.inf, dtype=np.float64)
    result = np.full(len(points_xy), np.inf, dtype=np.float64)
    # Costmaps contain at most a few thousand cells; chunking avoids a large
    # cells x path-segments allocation on the 20 fps Viewer thread.
    for start in range(0, len(points_xy), 1024):
        chunk = points_xy[start:start + 1024]
        best = np.full(len(chunk), np.inf, dtype=np.float64)
        for first, second in zip(path_xy[:-1], path_xy[1:]):
            vector = second - first
            denominator = float(vector @ vector)
            if denominator <= 1e-12:
                distance = np.linalg.norm(chunk - first, axis=1)
            else:
                fraction = np.clip(((chunk - first) @ vector) / denominator, 0.0, 1.0)
                projected = first + fraction[:, None] * vector
                distance = np.linalg.norm(chunk - projected, axis=1)
            best = np.minimum(best, distance)
        result[start:start + len(chunk)] = best
    return result


def evaluate_costmap_alert(
    points_in_base: np.ndarray,
    costs: np.ndarray,
    path_points_base: np.ndarray,
    *,
    high_cost_min: int,
    lethal_cost_min: int,
    settings: CostmapAlertSettings = CostmapAlertSettings(),
) -> tuple[CostmapAlert | None, str]:
    """Return only observations supported by synchronized numeric inputs."""
    settings.validate()
    points = np.asarray(points_in_base, dtype=np.float64)
    values = np.asarray(costs, dtype=np.int16)
    path = np.asarray(path_points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or values.shape != (len(points),):
        return None, "invalid_costmap_geometry"
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        return None, "real_path_unavailable"
    if not np.isfinite(points).all() or not np.isfinite(path).all():
        return None, "nonfinite_geometry"
    relevant = points[:, 0] >= 0.0
    lethal = relevant & (values >= lethal_cost_min)
    high = relevant & (values >= high_cost_min) & (values < lethal_cost_min)
    if np.any(lethal):
        lethal_xy = points[lethal, :2]
        # Distance to the configured rectangular footprint, not merely to the
        # robot origin.  Zero means the lethal cell intersects the footprint.
        delta_x = np.maximum(
            np.abs(lethal_xy[:, 0]) - settings.robot_half_length_m, 0.0
        )
        delta_y = np.maximum(
            np.abs(lethal_xy[:, 1]) - settings.robot_half_width_m, 0.0
        )
        robot_distance = np.hypot(delta_x, delta_y)
        minimum = float(np.min(robot_distance))
        if minimum <= settings.nearby_lethal_distance_m:
            return CostmapAlert(
                "COSTMAP ALERT: NEARBY", "nearby",
                f"nearest_lethal_m={minimum:.3f}",
            ), "lethal_cost_near_robot"
    if np.any(lethal):
        minimum = float(np.min(_minimum_polyline_distance(
            points[lethal, :2], path[:, :2]
        )))
        if minimum <= settings.path_corridor_width_m:
            return CostmapAlert(
                "LETHAL COST NEAR PATH", "lethal_path",
                f"path_distance_m={minimum:.3f}",
            ), "lethal_cost_near_path"
    if np.any(high):
        minimum = float(np.min(_minimum_polyline_distance(
            points[high, :2], path[:, :2]
        )))
        if minimum <= settings.path_corridor_width_m:
            return CostmapAlert(
                "HIGH COST NEAR PATH", "high_path",
                f"path_distance_m={minimum:.3f}",
            ), "high_cost_near_path"
    return None, "no_cost_near_path"


class CostmapAlertGate:
    """Debounce changing alerts; invalid input clears immediately."""

    def __init__(
        self,
        confirm_sec: float,
        release_sec: float = 0.0,
        clock=time.monotonic,
    ) -> None:
        self.confirm_sec = float(confirm_sec)
        self.release_sec = float(release_sec)
        if (
            not math.isfinite(self.confirm_sec)
            or not math.isfinite(self.release_sec)
            or self.confirm_sec < 0.0
            or self.release_sec < 0.0
        ):
            raise ValueError("alert hysteresis must be finite and non-negative")
        self.clock = clock
        self._candidate: CostmapAlert | None = None
        self._since = 0.0
        self._last_seen = 0.0
        self._visible = False

    def clear(self) -> None:
        self._candidate = None
        self._since = 0.0
        self._last_seen = 0.0
        self._visible = False

    def update(self, alert: CostmapAlert | None) -> CostmapAlert | None:
        if alert is None:
            if (
                self._candidate is not None
                and self._visible
                and self.release_sec > 0.0
                and self.clock() - self._last_seen <= self.release_sec
            ):
                return self._candidate
            self.clear()
            return None
        now = self.clock()
        if self._candidate is None or self._candidate.label != alert.label:
            self._candidate = alert
            self._since = now
            self._visible = False
        else:
            self._candidate = alert
        self._last_seen = now
        if now - self._since + 1e-9 < self.confirm_sec:
            return None
        self._visible = True
        return self._candidate
