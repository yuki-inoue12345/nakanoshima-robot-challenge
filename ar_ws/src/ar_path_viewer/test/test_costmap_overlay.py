"""ROS-independent Local Costmap perception overlay tests."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ar_path_viewer.costmap_overlay import (
    CostmapOverlaySettings,
    CostmapStore,
    HIGH_COST_CLASS,
    LETHAL_COST_CLASS,
    UNKNOWN_CLASS,
    ProjectedCostmap,
    occupancy_grid_to_snapshot,
    project_costmap_points,
)
from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.rendering import draw_costmap_overlay


def _grid(
    data,
    *,
    width=2,
    height=2,
    resolution=1.0,
    stamp=10.0,
    frame="odom",
    origin=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    seconds = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame,
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=int(round((stamp - seconds) * 1e9)),
            ),
        ),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(
                    x=origin[0], y=origin[1], z=origin[2]
                ),
                orientation=SimpleNamespace(
                    x=quaternion[0],
                    y=quaternion[1],
                    z=quaternion[2],
                    w=quaternion[3],
                ),
            ),
        ),
        data=list(data),
    )


def test_grid_cells_keep_unknown_high_and_lethal_semantics():
    settings = CostmapOverlaySettings(
        high_cost_min=60, lethal_cost_min=100, plane_height_m=0.02
    )

    snapshot = occupancy_grid_to_snapshot(
        _grid((-1, 0, 60, 100)),
        settings,
        ros_now_sec=10.1,
        received_at=20.0,
    )

    assert snapshot.unknown_cells == 1
    assert snapshot.high_cost_cells == 1
    assert snapshot.lethal_cost_cells == 1
    assert set(snapshot.classes) == {
        UNKNOWN_CLASS,
        HIGH_COST_CLASS,
        LETHAL_COST_CLASS,
    }
    np.testing.assert_allclose(
        snapshot.points,
        ((0.5, 0.5, 0.02), (0.5, 1.5, 0.02), (1.5, 1.5, 0.02)),
    )
    assert snapshot.analysis_points.shape == (4, 3)
    np.testing.assert_array_equal(snapshot.analysis_costs, (-1, 0, 60, 100))


def test_grid_origin_yaw_is_applied_to_cell_centers():
    half = np.sqrt(0.5)
    snapshot = occupancy_grid_to_snapshot(
        _grid(
            (100, 0),
            width=2,
            height=1,
            origin=(2.0, 3.0, 0.0),
            quaternion=(0.0, 0.0, half, half),
        ),
        CostmapOverlaySettings(),
        ros_now_sec=10.0,
        received_at=10.0,
    )

    np.testing.assert_allclose(snapshot.points[0, :2], (1.5, 3.5), atol=1e-9)


def test_sampling_is_bounded_per_class_without_changing_counts():
    values = [-1] * 100 + [70] * 100 + [100] * 100
    snapshot = occupancy_grid_to_snapshot(
        _grid(values, width=30, height=10),
        CostmapOverlaySettings(max_points_per_class=12),
        ros_now_sec=10.0,
        received_at=10.0,
    )

    assert snapshot.sampled_cells == 36
    assert snapshot.analysis_points.shape == (300, 3)
    assert snapshot.analysis_costs.shape == (300,)
    assert snapshot.unknown_cells == 100
    assert snapshot.high_cost_cells == 100
    assert snapshot.lethal_cost_cells == 100


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (_grid((0, 101, 0, 0)), "within"),
        (_grid((0, 0, 0)), "length mismatch"),
        (_grid((0, 0, 0, 0), frame=""), "frame_id"),
        (_grid((0, 0, 0, 0), stamp=0.0), "timestamp"),
    ],
)
def test_invalid_grid_is_rejected(message, match):
    with pytest.raises(ValueError, match=match):
        occupancy_grid_to_snapshot(
            message,
            CostmapOverlaySettings(),
            ros_now_sec=10.0,
            received_at=10.0,
        )


def test_store_clears_invalid_and_stale_data_then_recovers():
    now = [10.0]
    store = CostmapStore(
        CostmapOverlaySettings(timeout_sec=1.0, stamp_max_age_sec=1.0),
        clock=lambda: now[0],
    )
    assert store.update(_grid((0, 100, 0, 0)), 10.0)
    assert store.state(10.5)[0] == "ACTIVE"

    now[0] = 11.1
    state, snapshot, reason = store.state(11.1)
    assert state == "STALE"
    assert snapshot is None
    assert "receive_stale" in reason

    now[0] = 12.0
    assert store.update(_grid((0, 100, 0, 0), stamp=12.0), 12.0)
    assert not store.update(_grid((0, 101, 0, 0), stamp=12.0), 12.0)
    assert store.snapshot is None
    assert "invalid_local_costmap" in store.reason


def test_projection_preserves_ros_left_right_sign():
    points = np.asarray(((4.0, 1.0, 0.0), (4.0, -1.0, 0.0)))
    projection = project_costmap_points(
        points,
        np.asarray((LETHAL_COST_CLASS, HIGH_COST_CLASS), dtype=np.uint8),
        CameraMount(x=0.0, y=0.0, z=0.0),
        CameraIntrinsics(),
        0.1,
    )

    # Robot-left (+y) is optical-left (smaller u).
    assert projection.lethal_cost_pixels[0, 0] < 319
    assert projection.high_cost_pixels[0, 0] > 319
    assert projection.visible_points == 2


def test_sparse_rendering_keeps_cost_classes_visually_distinct():
    projection = ProjectedCostmap(
        unknown_pixels=np.asarray(((40, 40),), dtype=np.int32),
        high_cost_pixels=np.asarray(((80, 40),), dtype=np.int32),
        lethal_cost_pixels=np.asarray(((120, 40),), dtype=np.int32),
        input_points=3,
        visible_points=3,
    )
    image = np.zeros((100, 160, 3), dtype=np.uint8)

    frame = draw_costmap_overlay(image, projection, alpha=1.0, radius_px=2)

    assert np.any(frame[35:46, 35:46] != 0)
    assert np.any(frame[35:46, 75:86] != 0)
    assert np.any(frame[35:46, 115:126] != 0)
    assert not np.array_equal(frame[40, 40], frame[40, 80])
    assert cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 10
