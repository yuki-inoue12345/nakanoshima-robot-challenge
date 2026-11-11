import numpy as np
import pytest

from ar_path_viewer.offscreen_guidance import (
    GuidanceSettings,
    decide_offscreen_guidance,
    measure_path_progress,
)
from ar_path_viewer.projection import CameraIntrinsics, CameraMount, project_path


@pytest.mark.parametrize(
    ("points", "direction", "bearing_sign"),
    (
        (((0.1, 0.0, 0.0), (2.0, 0.0, 0.0)), "STRAIGHT", 0),
        (
            ((0.1, 0.0, 0.0), (0.7, 0.0, 0.0), (1.2, 1.0, 0.0)),
            "LEFT",
            1,
        ),
        (
            ((0.1, 0.0, 0.0), (0.7, 0.0, 0.0), (1.2, -1.0, 0.0)),
            "RIGHT",
            -1,
        ),
    ),
)
def test_real_path_selects_straight_left_and_right(points, direction, bearing_sign):
    decision = decide_offscreen_guidance(np.asarray(points, dtype=float))
    assert decision.guidance is not None
    assert decision.guidance.direction == direction
    if bearing_sign:
        assert np.sign(decision.guidance.bearing_deg) == bearing_sign
    else:
        assert abs(decision.guidance.bearing_deg) < 1e-9


def test_goal_near_robot_suppresses_potentially_misleading_arrow():
    decision = decide_offscreen_guidance(
        np.array(((0.05, 0.0, 0.0), (0.45, 0.05, 0.0)))
    )
    assert decision.guidance is None
    assert "endpoint is near" in decision.reason


def test_behind_and_ambiguous_paths_do_not_produce_guidance():
    behind = decide_offscreen_guidance(
        np.array(((-0.1, 0.0, 0.0), (-2.0, 0.0, 0.0)))
    )
    assert behind.guidance is None
    assert "behind" in behind.reason

    side_change = decide_offscreen_guidance(
        np.array(((0.1, 0.0, 0.0), (0.7, 0.8, 0.0), (1.1, -1.0, 0.0))),
        GuidanceSettings(lookahead_m=3.0),
    )
    assert side_change.guidance is None
    assert "changes" in side_change.reason


def test_distant_or_disabled_path_is_suppressed():
    points = np.array(((2.0, 2.0, 0.0), (4.0, 2.0, 0.0)))
    distant = decide_offscreen_guidance(points)
    disabled = decide_offscreen_guidance(
        points, GuidanceSettings(enabled=False)
    )
    assert distant.guidance is None
    assert "too far" in distant.reason
    assert disabled.guidance is None
    assert disabled.reason == "disabled"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        GuidanceSettings(max_bearing_deg=90.0).validate()
    with pytest.raises(ValueError):
        GuidanceSettings(straight_angle_deg=30.0, max_bearing_deg=20.0).validate()


def test_november_9_bag_short_path_gets_truthful_straight_guidance():
    # Downsampled from ar_1109_02 at 2026-11-09 13:36:20.635 JST using the
    # recorded map -> robot/base_footprint TF. The original has 100 points,
    # len=2.53m, view=0.00m and is below the image.
    points_base = np.array((
        (-0.0445, -0.0025, 0.0),
        (0.1900, 0.0124, 0.0),
        (0.4148, 0.0233, 0.0),
        (0.6397, 0.0289, 0.0),
        (0.8647, 0.0284, 0.0),
        (1.0895, 0.0213, 0.0),
        (1.3143, 0.0118, 0.0),
        (1.5391, 0.0024, 0.0),
        (1.7640, -0.0069, 0.0),
        (1.9888, -0.0160, 0.0),
        (2.2136, -0.0247, 0.0),
        (2.4623, -0.0756, 0.0),
    ))
    projection = project_path(
        points_base, (0.0, 0.0, 0.0), CameraMount(), CameraIntrinsics()
    )
    decision = decide_offscreen_guidance(points_base)
    assert projection.stats.drawable_segments == 0
    assert projection.stats.out_of_view_reason == (
        "all forward Path points are below the image"
    )
    assert decision.guidance is not None
    assert decision.guidance.direction == "STRAIGHT"
    assert abs(decision.guidance.bearing_deg) < 2.0


def test_path_progress_measures_from_closest_anchor_not_first_pose():
    points = np.array((
        (-1.0, 0.2, 0.0),
        (0.0, 0.2, 0.0),
        (2.0, 0.2, 0.0),
        (2.0, 2.2, 0.0),
    ))
    progress = measure_path_progress(points)
    assert progress is not None
    assert progress.anchor_distance_m == pytest.approx(0.2)
    assert progress.remaining_path_m == pytest.approx(4.0)
    assert progress.endpoint_distance_m == pytest.approx(np.hypot(2.0, 2.2))


@pytest.mark.parametrize(
    "points",
    (
        np.empty((0, 3)),
        np.array(((1.0, 0.0, 0.0),)),
        np.array(((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0))),
    ),
)
def test_path_progress_rejects_unmeasurable_paths(points):
    assert measure_path_progress(points) is None
