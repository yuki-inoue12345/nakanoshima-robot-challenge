"""Conservative, non-geometric guidance for a real Path outside the camera view."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GuidanceSettings:
    enabled: bool = True
    lookahead_m: float = 1.5
    min_remaining_m: float = 0.6
    near_goal_m: float = 0.6
    max_path_start_distance_m: float = 1.0
    straight_angle_deg: float = 15.0
    max_bearing_deg: float = 80.0
    max_bearing_change_deg: float = 65.0

    def validate(self) -> None:
        positive = (
            self.lookahead_m,
            self.min_remaining_m,
            self.near_goal_m,
            self.max_path_start_distance_m,
            self.straight_angle_deg,
            self.max_bearing_deg,
            self.max_bearing_change_deg,
        )
        if not np.isfinite(positive).all() or any(value <= 0.0 for value in positive):
            raise ValueError("off-screen guidance thresholds must be finite and positive")
        if self.straight_angle_deg >= self.max_bearing_deg:
            raise ValueError("guidance straight angle must be below max bearing")
        if self.max_bearing_deg >= 90.0:
            raise ValueError("guidance max bearing must be below 90 degrees")


@dataclass(frozen=True)
class OffscreenGuidance:
    direction: str
    bearing_deg: float
    remaining_path_m: float
    endpoint_distance_m: float

    def detail(self) -> str:
        return (
            f"direction={self.direction} bearing_deg={self.bearing_deg:.1f} "
            f"remaining_path_m={self.remaining_path_m:.2f} "
            f"endpoint_distance_m={self.endpoint_distance_m:.2f}"
        )


@dataclass(frozen=True)
class PathProgress:
    """Remaining distance measured on the received Nav2 Path polyline."""

    remaining_path_m: float
    endpoint_distance_m: float
    anchor_distance_m: float


@dataclass(frozen=True)
class GuidanceDecision:
    guidance: OffscreenGuidance | None
    reason: str

    def detail(self) -> str:
        if self.guidance is None:
            return f"suppressed reason={self.reason}"
        return self.guidance.detail()


def _closest_path_anchor(points_xy: np.ndarray) -> tuple[int, float, np.ndarray, float]:
    """Same first-minimum closest segment as the original; batched in NumPy.

    The received Path is not simplified, reordered, or altered.  Exact tie
    resolution remains first-segment-wins (np.argmin).
    """
    if len(points_xy) < 2:
        raise ValueError(
            "guidance requires at least one non-empty Path segment"
        )
    starts = points_xy[:-1]
    deltas = np.diff(points_xy, axis=0)
    squared_lengths = np.einsum('ij,ij->i', deltas, deltas)
    numerators = -np.einsum('ij,ij->i', starts, deltas)
    fractions = np.divide(
        numerators, squared_lengths,
        out=np.zeros_like(numerators), where=squared_lengths > 1e-12,
    )
    np.clip(fractions, 0.0, 1.0, out=fractions)
    closest = starts + fractions[:, None] * deltas
    distances = np.linalg.norm(closest, axis=1)
    index = int(np.argmin(distances))
    return (
        index, float(fractions[index]),
        closest[index].copy(), float(distances[index]),
    )


def _remaining_polyline(
    points_xy: np.ndarray, index: int, anchor: np.ndarray
) -> np.ndarray:
    """Zero-copy-friendly fast path, with legacy near-duplicate semantics.

    The original compares each point against the *last accepted* point, not
    merely the preceding point.  Keep that exact behavior for sub-nanometer
    repeated/near-repeated points; ordinary ROS Path points take the fast path.
    """
    candidates = np.concatenate((
        np.asarray(anchor, dtype=float).reshape(1, 2),
        np.asarray(points_xy[index + 1:], dtype=float),
    ))
    if len(candidates) <= 1:
        return candidates
    consecutive_distances = np.linalg.norm(np.diff(candidates, axis=0), axis=1)
    if bool(np.all(consecutive_distances > 1e-9)):
        return candidates
    filtered = [candidates[0]]
    for point in candidates[1:]:
        if np.linalg.norm(point - filtered[-1]) > 1e-9:
            filtered.append(point)
    return np.asarray(filtered)


def measure_path_progress(points_base: np.ndarray) -> PathProgress | None:
    """Measure real Path distance from its closest point to its endpoint.

    No waypoint coordinates or distance are inferred here.  This is only the
    remaining length of the received Path polyline.
    """
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return None
    if not np.isfinite(points).all():
        return None
    points_xy = points[:, :2]
    try:
        index, _, anchor, anchor_distance = _closest_path_anchor(points_xy)
    except ValueError:
        return None
    remaining = _remaining_polyline(points_xy, index, anchor)
    if len(remaining) < 2:
        return None
    remaining_length = float(
        np.linalg.norm(np.diff(remaining, axis=0), axis=1).sum()
    )
    return PathProgress(
        remaining_path_m=remaining_length,
        endpoint_distance_m=float(np.linalg.norm(remaining[-1])),
        anchor_distance_m=anchor_distance,
    )


def _point_at_distance(polyline: np.ndarray, distance_m: float) -> np.ndarray:
    walked = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        length = float(np.linalg.norm(end - start))
        if length <= 1e-12:
            continue
        if walked + length >= distance_m:
            fraction = (distance_m - walked) / length
            return start + fraction * (end - start)
        walked += length
    return polyline[-1]


def _bearing(point: np.ndarray) -> float:
    return math.degrees(math.atan2(float(point[1]), float(point[0])))


def _angle_difference(first: float, second: float) -> float:
    return abs(math.degrees(math.atan2(
        math.sin(math.radians(second - first)),
        math.cos(math.radians(second - first)),
    )))


def decide_offscreen_guidance(
    points_base: np.ndarray,
    settings: GuidanceSettings = GuidanceSettings(),
) -> GuidanceDecision:
    """Use only the real remaining Path to choose straight/left/right guidance.

    This does not project, extend, or reshape a Path. The caller must additionally
    require a valid, fresh Path and localization. It may render the result as a
    compact badge beside a visible Path or a larger guide when no segment is visible.
    """
    settings.validate()
    if not settings.enabled:
        return GuidanceDecision(None, "disabled")
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return GuidanceDecision(None, "fewer than two Path points")
    if not np.isfinite(points).all():
        return GuidanceDecision(None, "Path contains NaN or Inf")
    points_xy = points[:, :2]
    try:
        index, _, anchor, anchor_distance = _closest_path_anchor(points_xy)
    except ValueError as exc:
        return GuidanceDecision(None, str(exc))
    if anchor_distance > settings.max_path_start_distance_m:
        return GuidanceDecision(None, "Path start is too far from the robot")

    remaining = _remaining_polyline(points_xy, index, anchor)
    if len(remaining) < 2:
        return GuidanceDecision(None, "no remaining Path after closest point")
    progress = measure_path_progress(points)
    if progress is None:
        return GuidanceDecision(None, "no measurable remaining Path")
    remaining_length = progress.remaining_path_m
    endpoint_distance = progress.endpoint_distance_m
    if endpoint_distance <= settings.near_goal_m:
        return GuidanceDecision(None, "endpoint is near the robot")
    if remaining_length < settings.min_remaining_m:
        return GuidanceDecision(None, "remaining Path is too short")

    lookahead = min(settings.lookahead_m, remaining_length)
    early_distance = min(lookahead, max(settings.min_remaining_m, lookahead * 0.5))
    early_point = _point_at_distance(remaining, early_distance)
    target_point = _point_at_distance(remaining, lookahead)
    if np.linalg.norm(target_point) < settings.min_remaining_m * 0.5:
        return GuidanceDecision(None, "lookahead direction is too small")
    early_bearing = _bearing(early_point)
    target_bearing = _bearing(target_point)
    if abs(target_bearing) > settings.max_bearing_deg:
        return GuidanceDecision(None, "Path points behind or nearly behind the camera")
    if _angle_difference(early_bearing, target_bearing) > settings.max_bearing_change_deg:
        return GuidanceDecision(None, "Path direction changes too sharply")
    if (
        early_bearing * target_bearing < 0.0
        and abs(early_bearing) > settings.straight_angle_deg
        and abs(target_bearing) > settings.straight_angle_deg
    ):
        return GuidanceDecision(None, "Path direction changes sides")

    if target_bearing > settings.straight_angle_deg:
        direction = "LEFT"
    elif target_bearing < -settings.straight_angle_deg:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"
    return GuidanceDecision(
        OffscreenGuidance(
            direction,
            target_bearing,
            remaining_length,
            endpoint_distance,
        ),
        "",
    )
