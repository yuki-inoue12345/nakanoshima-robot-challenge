"""Fail-closed stability gate for visibly disruptive Nav2 Path updates."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PathTransitionDecision:
    visible: bool
    status: str
    detail: str


@dataclass(frozen=True)
class _PathShape:
    points_xy: np.ndarray
    arc_m: np.ndarray


def _shape(points_base: np.ndarray) -> _PathShape | None:
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return None
    xy = points[:, :2]
    if not np.all(np.isfinite(xy)):
        return None
    arc = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    ))
    keep = np.concatenate(([True], np.diff(arc) > 1e-6))
    xy = xy[keep]
    arc = arc[keep]
    if len(xy) < 2 or arc[-1] < 1e-3:
        return None
    return _PathShape(xy, arc)


def _nearest_arc(shape: _PathShape, point: np.ndarray) -> tuple[float, float]:
    best_distance = math.inf
    best_arc = 0.0
    for index, (start, end) in enumerate(
        zip(shape.points_xy[:-1], shape.points_xy[1:])
    ):
        vector = end - start
        length_squared = float(vector @ vector)
        fraction = 0.0 if length_squared <= 1e-12 else float(
            np.clip(((point - start) @ vector) / length_squared, 0.0, 1.0)
        )
        projected = start + fraction * vector
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_arc = float(
                shape.arc_m[index]
                + fraction * (shape.arc_m[index + 1] - shape.arc_m[index])
            )
    return best_arc, best_distance


def _sample(shape: _PathShape, distances: np.ndarray) -> np.ndarray:
    return np.column_stack((
        np.interp(distances, shape.arc_m, shape.points_xy[:, 0]),
        np.interp(distances, shape.arc_m, shape.points_xy[:, 1]),
    ))


def _angle_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_angle = math.atan2(float(first[1]), float(first[0]))
    second_angle = math.atan2(float(second[1]), float(second[0]))
    delta = math.atan2(
        math.sin(second_angle - first_angle),
        math.cos(second_angle - first_angle),
    )
    return abs(math.degrees(delta))


class PathTransitionGate:
    """Suppress only large shape changes until the latest Path is stable.

    The old path is never held on screen. Compatible updates (including normal
    progress along a route) are accepted immediately. A visibly different
    route is hidden briefly and labelled REPLANNING; this avoids presenting a
    transient plan as a settled driving intention without changing Nav2 data.
    """

    def __init__(
        self,
        hold_sec: float = 0.35,
        max_displacement_m: float = 0.60,
        max_heading_change_deg: float = 35.0,
        compare_horizon_m: float = 2.0,
        sample_step_m: float = 0.20,
    ) -> None:
        values = (
            hold_sec,
            max_displacement_m,
            max_heading_change_deg,
            compare_horizon_m,
            sample_step_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("path transition thresholds must be finite")
        if hold_sec < 0.0:
            raise ValueError("path transition hold_sec must be non-negative")
        if max_displacement_m <= 0.0 or max_heading_change_deg <= 0.0:
            raise ValueError("path transition thresholds must be positive")
        if compare_horizon_m <= 0.0 or sample_step_m <= 0.0:
            raise ValueError("path transition sampling values must be positive")
        self.hold_sec = float(hold_sec)
        self.max_displacement_m = float(max_displacement_m)
        self.max_heading_change_deg = float(max_heading_change_deg)
        self.compare_horizon_m = float(compare_horizon_m)
        self.sample_step_m = float(sample_step_m)
        self._confirmed_generation: int | None = None
        self._confirmed: _PathShape | None = None
        self._pending_generation: int | None = None
        self._pending: _PathShape | None = None
        self._pending_since = 0.0

    def clear(self) -> None:
        self._confirmed_generation = None
        self._confirmed = None
        self._pending_generation = None
        self._pending = None
        self._pending_since = 0.0

    def _difference(self, first: _PathShape, second: _PathShape) -> tuple[float, float]:
        first_offset, start_distance = _nearest_arc(first, second.points_xy[0])
        # A normal Nav2 update starts farther along the previous polyline.
        # Comparing from that matched point prevents robot progress and turns
        # already present in the route from looking like a replan. If the new
        # start cannot be matched, compare local shapes as a conservative
        # fallback because map-frame translations alone are not route changes.
        aligned = start_distance <= self.max_displacement_m
        first_remaining = float(first.arc_m[-1]) - first_offset
        horizon = min(
            self.compare_horizon_m,
            first_remaining if aligned else float(first.arc_m[-1]),
            float(second.arc_m[-1]),
        )
        if horizon < 0.10:
            return 0.0, 0.0
        count = max(2, int(math.ceil(horizon / self.sample_step_m)) + 1)
        distances = np.linspace(0.0, horizon, count)
        first_samples = _sample(
            first, distances + (first_offset if aligned else 0.0)
        )
        second_samples = _sample(second, distances)
        if not aligned:
            first_samples = first_samples - first_samples[0]
            second_samples = second_samples - second_samples[0]
        displacement = float(
            np.max(np.linalg.norm(first_samples - second_samples, axis=1))
        )
        heading_delta = _angle_delta_deg(
            first_samples[-1] - first_samples[0],
            second_samples[-1] - second_samples[0],
        )
        return displacement, heading_delta

    def update(
        self, generation: int, points_base: np.ndarray, now_sec: float
    ) -> PathTransitionDecision:
        now = float(now_sec)
        if not math.isfinite(now):
            raise ValueError("path transition time must be finite")
        candidate = _shape(points_base)
        if candidate is None:
            self.clear()
            return PathTransitionDecision(True, "", "path_shape_not_comparable")
        generation = int(generation)
        if self._confirmed is None:
            self._confirmed = candidate
            self._confirmed_generation = generation
            return PathTransitionDecision(True, "", "first_path")
        if generation == self._confirmed_generation and self._pending is None:
            return PathTransitionDecision(True, "", "confirmed_generation")

        if self._pending is not None and generation == self._pending_generation:
            if now - self._pending_since >= self.hold_sec:
                self._confirmed = candidate
                self._confirmed_generation = generation
                self._pending = None
                self._pending_generation = None
                return PathTransitionDecision(True, "", "replan_confirmed")
            return PathTransitionDecision(
                False,
                "PATH CHANGE CHECK",
                f"generation={generation} age={max(0.0, now - self._pending_since):.3f}s",
            )

        had_pending = self._pending is not None
        baseline = self._pending if had_pending else self._confirmed
        displacement, heading_delta = self._difference(baseline, candidate)
        disruptive = (
            displacement > self.max_displacement_m
            or heading_delta > self.max_heading_change_deg
        )
        if not disruptive:
            if had_pending:
                self._pending = candidate
                self._pending_generation = generation
                if now - self._pending_since < self.hold_sec:
                    return PathTransitionDecision(
                        False,
                        "PATH CHANGE CHECK",
                        f"generation={generation} pending_compatible=true "
                        f"age={max(0.0, now - self._pending_since):.3f}s",
                    )
            self._confirmed = candidate
            self._confirmed_generation = generation
            self._pending = None
            self._pending_generation = None
            return PathTransitionDecision(
                True,
                "",
                f"compatible displacement_m={displacement:.3f} "
                f"heading_deg={heading_delta:.1f}",
            )

        self._pending = candidate
        self._pending_generation = generation
        self._pending_since = now
        if self.hold_sec == 0.0:
            self._confirmed = candidate
            self._confirmed_generation = generation
            self._pending = None
            self._pending_generation = None
            return PathTransitionDecision(
                True,
                "",
                f"replan_immediate displacement_m={displacement:.3f} "
                f"heading_deg={heading_delta:.1f}",
            )
        return PathTransitionDecision(
            False,
            "PATH CHANGE CHECK",
            f"generation={generation} displacement_m={displacement:.3f} "
            f"heading_deg={heading_delta:.1f}",
        )
