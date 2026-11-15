"""Conservative explanation of observed Costmap and real-Path geometry.

The labels describe a spatial relationship visible in the same timestamped
inputs.  They do not claim access to the planner's internal causal reasoning,
object identity, image occlusion, or general-purpose danger.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostmapExplanationSettings:
    """Robot-frame regions and evidence thresholds for explanation labels."""

    forward_min_m: float = 0.40
    forward_max_m: float = 2.50
    center_half_width_m: float = 0.45
    side_outer_width_m: float = 1.20
    maximum_unknown_fraction: float = 0.25
    low_cost_max_fraction: float = 0.08
    blocked_lethal_fraction: float = 0.03
    side_cost_min_fraction: float = 0.08
    side_cost_difference: float = 0.04
    minimum_region_cells: int = 80

    def validate(self) -> None:
        values = (
            self.forward_min_m,
            self.forward_max_m,
            self.center_half_width_m,
            self.side_outer_width_m,
            self.maximum_unknown_fraction,
            self.low_cost_max_fraction,
            self.blocked_lethal_fraction,
            self.side_cost_min_fraction,
            self.side_cost_difference,
        )
        if not np.isfinite(values).all():
            raise ValueError("costmap explanation settings must be finite")
        if not 0.0 <= self.forward_min_m < self.forward_max_m:
            raise ValueError("invalid costmap explanation forward range")
        if not 0.0 < self.center_half_width_m < self.side_outer_width_m:
            raise ValueError("invalid costmap explanation lateral range")
        fractions = values[4:]
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("costmap explanation fractions must be in [0, 1]")
        if self.minimum_region_cells < 1:
            raise ValueError("minimum_region_cells must be positive")


@dataclass(frozen=True)
class RegionEvidence:
    cells: int
    unknown_fraction: float
    high_cost_fraction: float
    lethal_fraction: float
    mean_known_cost: float | None


@dataclass(frozen=True)
class CostmapExplanation:
    label: str
    kind: str
    detail: str


@dataclass(frozen=True)
class CostmapExplanationDecision:
    explanation: CostmapExplanation | None
    reason: str
    front: RegionEvidence | None = None
    left: RegionEvidence | None = None
    right: RegionEvidence | None = None


def _region(
    points: np.ndarray,
    costs: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    high_cost_min: int,
    lethal_cost_min: int,
) -> RegionEvidence:
    selected = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    values = costs[selected]
    if values.size == 0:
        return RegionEvidence(0, 1.0, 1.0, 1.0, None)
    known = values >= 0
    known_values = values[known]
    if known_values.size == 0:
        return RegionEvidence(int(values.size), 1.0, 1.0, 1.0, None)
    return RegionEvidence(
        cells=int(values.size),
        unknown_fraction=float(np.mean(values < 0)),
        high_cost_fraction=float(np.mean(known_values >= high_cost_min)),
        lethal_fraction=float(np.mean(known_values >= lethal_cost_min)),
        mean_known_cost=float(np.mean(known_values)),
    )


def explain_costmap_and_path(
    points_in_base: np.ndarray,
    costs: np.ndarray,
    *,
    path_direction: str | None,
    viewer_state: str,
    high_cost_min: int = 60,
    lethal_cost_min: int = 100,
    settings: CostmapExplanationSettings = CostmapExplanationSettings(),
) -> CostmapExplanationDecision:
    """Describe only well-supported Costmap/Path spatial relationships."""
    settings.validate()
    points = np.asarray(points_in_base, dtype=np.float64)
    values = np.asarray(costs, dtype=np.int16)
    if points.ndim != 2 or points.shape[1] != 3:
        return CostmapExplanationDecision(None, "invalid_costmap_points")
    if values.shape != (points.shape[0],):
        return CostmapExplanationDecision(None, "cost_count_mismatch")
    if not np.isfinite(points).all():
        return CostmapExplanationDecision(None, "nonfinite_costmap_points")
    if np.any(values < -1) or np.any(values > 100):
        return CostmapExplanationDecision(None, "invalid_cost_values")
    if viewer_state not in (
        "ACTIVE", "PATH OUT OF VIEW", "REPLANNING", "NAVIGATION HOLD"
    ):
        return CostmapExplanationDecision(None, "viewer_state_not_explainable")

    front = _region(
        points, values,
        settings.forward_min_m, settings.forward_max_m,
        -settings.center_half_width_m, settings.center_half_width_m,
        high_cost_min, lethal_cost_min,
    )
    left = _region(
        points, values,
        settings.forward_min_m, settings.forward_max_m,
        settings.center_half_width_m, settings.side_outer_width_m,
        high_cost_min, lethal_cost_min,
    )
    right = _region(
        points, values,
        settings.forward_min_m, settings.forward_max_m,
        -settings.side_outer_width_m, -settings.center_half_width_m,
        high_cost_min, lethal_cost_min,
    )
    evidence = (front, left, right)
    if any(item.cells < settings.minimum_region_cells for item in evidence):
        return CostmapExplanationDecision(
            None, "insufficient_region_cells", front, left, right
        )
    if any(
        item.unknown_fraction > settings.maximum_unknown_fraction
        for item in evidence
    ):
        return CostmapExplanationDecision(
            None, "unknown_region_fraction_too_high", front, left, right
        )

    front_blocked = (
        front.lethal_fraction >= settings.blocked_lethal_fraction
    )
    detail = (
        f"front_high={front.high_cost_fraction:.3f} "
        f"front_lethal={front.lethal_fraction:.3f} "
        f"left_high={left.high_cost_fraction:.3f} "
        f"right_high={right.high_cost_fraction:.3f} "
        f"path_direction={path_direction or 'none'} state={viewer_state}"
    )
    if viewer_state == "REPLANNING":
        if front_blocked:
            return CostmapExplanationDecision(
                CostmapExplanation(
                    "PATH CHANGING | COST AHEAD", "replanning", detail
                ),
                "replanning_with_front_cost",
                front, left, right,
            )
        return CostmapExplanationDecision(
            None, "replanning_without_cost_evidence", front, left, right
        )
    if viewer_state == "NAVIGATION HOLD":
        if front_blocked:
            return CostmapExplanationDecision(
                CostmapExplanation(
                    "COSTMAP BLOCKED AHEAD", "blocked", detail
                ),
                "hold_with_front_lethal_cost",
                front, left, right,
            )
        return CostmapExplanationDecision(
            None, "hold_without_cost_evidence", front, left, right
        )
    if front_blocked:
        return CostmapExplanationDecision(
            CostmapExplanation("COSTMAP BLOCKED AHEAD", "blocked", detail),
            "front_lethal_cost",
            front, left, right,
        )

    direction = "" if path_direction is None else path_direction.upper()
    if direction == "LEFT":
        right_difference = (
            right.high_cost_fraction - left.high_cost_fraction
        )
        if (
            right.high_cost_fraction >= settings.side_cost_min_fraction
            and right_difference >= settings.side_cost_difference
        ):
            return CostmapExplanationDecision(
                CostmapExplanation("RIGHT COST | PATH LEFT", "side_cost", detail),
                "right_cost_with_left_path",
                front, left, right,
            )
    elif direction == "RIGHT":
        left_difference = left.high_cost_fraction - right.high_cost_fraction
        if (
            left.high_cost_fraction >= settings.side_cost_min_fraction
            and left_difference >= settings.side_cost_difference
        ):
            return CostmapExplanationDecision(
                CostmapExplanation("LEFT COST | PATH RIGHT", "side_cost", detail),
                "left_cost_with_right_path",
                front, left, right,
            )
    elif direction == "STRAIGHT":
        if (
            front.high_cost_fraction <= settings.low_cost_max_fraction
            and front.lethal_fraction < settings.blocked_lethal_fraction
        ):
            return CostmapExplanationDecision(
                CostmapExplanation(
                    "FRONT LOW COST | PATH STRAIGHT", "low_cost", detail
                ),
                "low_cost_front_with_straight_path",
                front, left, right,
            )
    return CostmapExplanationDecision(
        None, "cost_path_relationship_ambiguous", front, left, right
    )
