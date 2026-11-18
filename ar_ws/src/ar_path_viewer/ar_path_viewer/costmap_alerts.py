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
    first = np.asarray(path_xy[:-1], dtype=np.float64)
    vectors = np.asarray(path_xy[1:] - path_xy[:-1], dtype=np.float64)
    denominator = np.einsum("ij,ij->i", vectors, vectors)
    valid_segment = denominator > 1e-12
    safe_denominator = np.where(valid_segment, denominator, 1.0)
    # Vectorize over Path segments.  The previous Python inner loop made one
    # complete Costmap alert evaluation proportional to hundreds of Python
    # calls per frame.  Bounded chunks preserve exact point-to-segment
    # distances while keeping temporary arrays below a few MiB.
    for start in range(0, len(points_xy), 512):
        chunk = np.asarray(points_xy[start:start + 512], dtype=np.float64)
        delta = chunk[:, None, :] - first[None, :, :]
        fraction = np.einsum("csi,si->cs", delta, vectors)
        fraction = np.clip(fraction / safe_denominator[None, :], 0.0, 1.0)
        projected_delta = delta - fraction[:, :, None] * vectors[None, :, :]
        distance_sq = np.einsum(
            "csi,csi->cs", projected_delta, projected_delta
        )
        if not np.all(valid_segment):
            point_distance_sq = np.einsum("csi,csi->cs", delta, delta)
            distance_sq[:, ~valid_segment] = point_distance_sq[:, ~valid_segment]
        result[start:start + len(chunk)] = np.sqrt(np.min(distance_sq, axis=1))
    return result


def _corridor_candidates(
    points_xy: np.ndarray, path_xy: np.ndarray, corridor_m: float
) -> np.ndarray:
    """Discard only points mathematically unable to reach the corridor."""
    lower = np.min(path_xy, axis=0) - float(corridor_m)
    upper = np.max(path_xy, axis=0) + float(corridor_m)
    selected = np.all((points_xy >= lower) & (points_xy <= upper), axis=1)
    return points_xy[selected]


def evaluate_costmap_alert(
    points_in_base: np.ndarray,
    costs: np.ndarray,
    path_points_base: np.ndarray,
    *,
    high_cost_min: int,
    lethal_cost_min: int,
    settings: CostmapAlertSettings = CostmapAlertSettings(),
) -> tuple[CostmapAlert | None, str]:
    """Return the strongest observation supported by synchronized inputs."""
    nearby, nearby_reason = evaluate_robot_proximity_alert(
        points_in_base,
        costs,
        lethal_cost_min=lethal_cost_min,
        settings=settings,
    )
    if nearby is not None:
        return nearby, nearby_reason
    path_alert, path_reason = evaluate_path_cost_alert(
        points_in_base,
        costs,
        path_points_base,
        high_cost_min=high_cost_min,
        lethal_cost_min=lethal_cost_min,
        settings=settings,
    )
    return path_alert, path_reason


def _validated_costmap_arrays(
    points_in_base: np.ndarray, costs: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    points = np.asarray(points_in_base, dtype=np.float64)
    values = np.asarray(costs, dtype=np.int16)
    if points.ndim != 2 or points.shape[1] != 3 or values.shape != (len(points),):
        return None, None, "invalid_costmap_geometry"
    if not np.isfinite(points).all():
        return None, None, "nonfinite_geometry"
    return points, values, "valid"


def evaluate_robot_proximity_alert(
    points_in_base: np.ndarray,
    costs: np.ndarray,
    *,
    lethal_cost_min: int,
    settings: CostmapAlertSettings = CostmapAlertSettings(),
) -> tuple[CostmapAlert | None, str]:
    """Evaluate exact-time lethal cells near the footprint without a Path.

    This answers only whether the synchronized Local Costmap contains a
    lethal-cost cell close to the robot.  It does not claim that the cell is a
    person, that a collision will occur, or why Nav2 chose a route.
    """
    settings.validate()
    points, values, reason = _validated_costmap_arrays(points_in_base, costs)
    if points is None or values is None:
        return None, reason
    relevant = points[:, 0] >= 0.0
    lethal = relevant & (values >= lethal_cost_min)
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
    return None, "no_lethal_cost_near_robot"


def evaluate_path_cost_alert(
    points_in_base: np.ndarray,
    costs: np.ndarray,
    path_points_base: np.ndarray,
    *,
    high_cost_min: int,
    lethal_cost_min: int,
    settings: CostmapAlertSettings = CostmapAlertSettings(),
) -> tuple[CostmapAlert | None, str]:
    """Evaluate cost near one strictly synchronized real Path."""
    settings.validate()
    points, values, reason = _validated_costmap_arrays(points_in_base, costs)
    if points is None or values is None:
        return None, reason
    path = np.asarray(path_points_base, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        return None, "real_path_unavailable"
    if not np.isfinite(path).all():
        return None, "nonfinite_geometry"
    relevant = points[:, 0] >= 0.0
    lethal = relevant & (values >= lethal_cost_min)
    high = relevant & (values >= high_cost_min) & (values < lethal_cost_min)
    if np.any(lethal):
        candidates = _corridor_candidates(
            points[lethal, :2], path[:, :2], settings.path_corridor_width_m
        )
        if len(candidates):
            minimum = float(np.min(_minimum_polyline_distance(
                candidates, path[:, :2]
            )))
            if minimum <= settings.path_corridor_width_m:
                return CostmapAlert(
                    "LETHAL COST NEAR PATH", "lethal_path",
                    f"path_distance_m={minimum:.3f}",
                ), "lethal_cost_near_path"
    if np.any(high):
        candidates = _corridor_candidates(
            points[high, :2], path[:, :2], settings.path_corridor_width_m
        )
        if len(candidates):
            minimum = float(np.min(_minimum_polyline_distance(
                candidates, path[:, :2]
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


@dataclass(frozen=True)
class CostmapAlertHistory:
    """Display-only record of a recently ended evidence-bound alert."""

    alert: CostmapAlert
    age_sec: float
    current_status: str


class CostmapAlertHistoryGate:
    """Keep a short, explicitly historical notice without extending danger.

    This state is never returned as a current ``CostmapAlert`` and must not be
    used by semantic evaluation or robot control.  Goal identity changes and
    invalid identities clear it immediately.
    """

    _HAZARD_KINDS = {"nearby", "lethal_path", "high_path"}

    def __init__(self, duration_sec: float = 2.5, clock=time.monotonic) -> None:
        self.duration_sec = float(duration_sec)
        if not math.isfinite(self.duration_sec) or self.duration_sec <= 0.0:
            raise ValueError("alert history duration must be finite and positive")
        self.clock = clock
        self._alert: CostmapAlert | None = None
        self._last_active = 0.0
        self._identity: tuple | None = None

    def clear(self) -> None:
        self._alert = None
        self._last_active = 0.0
        self._identity = None

    def update(
        self,
        active_alert: CostmapAlert | None,
        *,
        evaluation_status: str,
        goal_identity: tuple | None,
    ) -> CostmapAlertHistory | None:
        now = float(self.clock())
        if goal_identity is None:
            self.clear()
            return None
        identity = tuple(goal_identity)
        if self._identity is not None and self._identity != identity:
            self.clear()
        self._identity = identity
        if (
            active_alert is not None
            and active_alert.kind in self._HAZARD_KINDS
        ):
            self._alert = active_alert
            self._last_active = now
            return None
        if self._alert is None:
            return None
        age = now - self._last_active
        if not math.isfinite(age) or age < 0.0 or age > self.duration_sec:
            self.clear()
            return None
        status = (
            "CLEARED"
            if evaluation_status == "cleared"
            else "UNCONFIRMED"
        )
        return CostmapAlertHistory(self._alert, age, status)
