"""Fail-closed tests for factual Costmap/real-Path explanation labels."""

import cv2
import numpy as np

from ar_path_viewer.costmap_explanation import explain_costmap_and_path
from ar_path_viewer.rendering import render_frame


def _scene():
    xs = np.arange(0.0, 3.05, 0.05)
    ys = np.arange(-1.5, 1.55, 0.05)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.column_stack((
        x_grid.ravel(), y_grid.ravel(), np.zeros(x_grid.size)
    ))
    return points, np.zeros(points.shape[0], dtype=np.int16)


def _set_cost(points, costs, *, y_min, y_max, value, x_min=0.4, x_max=2.5):
    selected = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    costs[selected] = value


def test_straight_path_on_low_cost_front_is_explained_without_causal_claim():
    points, costs = _scene()

    decision = explain_costmap_and_path(
        points, costs, path_direction="STRAIGHT", viewer_state="ACTIVE"
    )

    assert decision.reason == "low_cost_front_with_straight_path"
    assert decision.explanation.label == "FRONT LOW COST | PATH STRAIGHT"
    assert "danger" not in decision.explanation.label.lower()


def test_asymmetric_side_cost_must_agree_with_real_path_direction():
    points, costs = _scene()
    _set_cost(points, costs, y_min=-1.2, y_max=-0.45, value=70)

    left = explain_costmap_and_path(
        points, costs, path_direction="LEFT", viewer_state="ACTIVE"
    )
    wrong_direction = explain_costmap_and_path(
        points, costs, path_direction="RIGHT", viewer_state="ACTIVE"
    )

    assert left.explanation.label == "RIGHT COST | PATH LEFT"
    assert wrong_direction.explanation is None
    assert wrong_direction.reason == "cost_path_relationship_ambiguous"


def test_front_lethal_cost_is_distinguished_for_active_replanning_and_hold():
    points, costs = _scene()
    _set_cost(points, costs, y_min=-0.45, y_max=0.45, value=100)

    active = explain_costmap_and_path(
        points, costs, path_direction="STRAIGHT", viewer_state="ACTIVE"
    )
    replanning = explain_costmap_and_path(
        points, costs, path_direction=None, viewer_state="REPLANNING"
    )
    hold = explain_costmap_and_path(
        points, costs, path_direction=None, viewer_state="NAVIGATION HOLD"
    )

    assert active.explanation.label == "COSTMAP BLOCKED AHEAD"
    assert replanning.explanation.label == "PATH CHANGING | COST AHEAD"
    assert hold.explanation.label == "COSTMAP BLOCKED AHEAD"


def test_unknown_or_stale_evidence_is_suppressed_fail_closed():
    points, costs = _scene()
    _set_cost(points, costs, y_min=-1.2, y_max=1.2, value=-1)

    unknown = explain_costmap_and_path(
        points, costs, path_direction="STRAIGHT", viewer_state="ACTIVE"
    )
    stale = explain_costmap_and_path(
        points, np.zeros_like(costs),
        path_direction="STRAIGHT", viewer_state="PATH STALE",
    )

    assert unknown.explanation is None
    assert unknown.reason == "unknown_region_fraction_too_high"
    assert stale.explanation is None
    assert stale.reason == "viewer_state_not_explainable"


def test_explanation_card_is_compact_and_does_not_cover_lower_paths():
    points, costs = _scene()
    decision = explain_costmap_and_path(
        points, costs, path_direction="STRAIGHT", viewer_state="ACTIVE"
    )
    image = np.zeros((480, 640, 3), np.uint8)
    frame = render_frame(
        image,
        [((100, 400), (540, 300))],
        "ACTIVE",
        costmap_explanation=decision.explanation,
        debug_overlay=False,
    )

    assert np.count_nonzero(frame[50:104, 10:380]) > 500
    assert np.array_equal(frame[150:280], image[150:280])
    assert cv2.countNonZero(frame[290:, :, 1]) > 0
