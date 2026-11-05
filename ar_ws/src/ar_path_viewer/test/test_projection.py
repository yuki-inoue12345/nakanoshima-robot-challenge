import math
from pathlib import Path

import numpy as np
import yaml

from ar_path_viewer.projection import (
    DEFAULT_CAMERA_Y_M,
    CameraIntrinsics,
    CameraMount,
    project_clipped_segments,
    world_to_optical,
)


INTRINSICS = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
LEVEL_MOUNT = CameraMount(x=0.0, y=0.0, z=0.7, pitch_up_deg=0.0)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def pixel_for_point(point, mount=LEVEL_MOUNT):
    optical = world_to_optical(np.asarray([point], dtype=float), (0.0, 0.0, 0.0), mount)[0]
    return np.array((
        INTRINSICS.fx * optical[0] / optical[2] + INTRINSICS.cx,
        INTRINSICS.fy * optical[1] / optical[2] + INTRINSICS.cy,
    ))


def test_forward_point_at_camera_height_projects_to_center():
    assert np.allclose(pixel_for_point((5.0, 0.0, 0.7)), (320.0, 240.0))


def test_ros_left_appears_on_image_left_and_right_on_right():
    left = pixel_for_point((5.0, 1.0, 0.7))
    right = pixel_for_point((5.0, -1.0, 0.7))
    assert left[0] < INTRINSICS.cx < right[0]


def test_robot_left_camera_offset_moves_centerline_to_image_right():
    centered = pixel_for_point((5.0, 0.0, 0.7))
    offset = pixel_for_point(
        (5.0, 0.0, 0.7),
        CameraMount(x=0.0, y=DEFAULT_CAMERA_Y_M, z=0.7, pitch_up_deg=0.0),
    )
    assert offset[0] > centered[0]


def test_runtime_configs_use_measured_camera_y():
    for filename in ("ar_path_viewer.yaml", "ar_path_viewer_c920.yaml"):
        config = yaml.safe_load((PACKAGE_ROOT / "config" / filename).read_text())
        params = config["ar_path_viewer"]["ros__parameters"]
        assert params["camera_y"] == DEFAULT_CAMERA_Y_M
    assert CameraMount().y == DEFAULT_CAMERA_Y_M


def test_positive_pitch_up_moves_level_scene_down_in_image():
    level_v = pixel_for_point((5.0, 0.0, 0.7))[1]
    upward_v = pixel_for_point(
        (5.0, 0.0, 0.7), CameraMount(x=0.0, y=0.0, z=0.7, pitch_up_deg=7.5)
    )[1]
    assert upward_v > level_v


def test_robot_yaw_changes_path_image_side():
    path = np.array(((2.0, 1.0, 0.7), (5.0, 1.0, 0.7)))
    optical_zero = world_to_optical(path, (0.0, 0.0, 0.0), LEVEL_MOUNT)
    optical_left = world_to_optical(path, (0.0, 0.0, math.radians(10.0)), LEVEL_MOUNT)
    u_zero = INTRINSICS.fx * optical_zero[-1, 0] / optical_zero[-1, 2] + INTRINSICS.cx
    u_left = INTRINSICS.fx * optical_left[-1, 0] / optical_left[-1, 2] + INTRINSICS.cx
    assert u_left > u_zero


def test_fully_behind_segment_is_not_projected():
    points = np.array(((-2.0, 0.0, 0.7), (-1.0, 0.0, 0.7)))
    assert project_clipped_segments(points, (0, 0, 0), LEVEL_MOUNT, INTRINSICS) == []


def test_near_plane_crossing_is_clipped_not_dropped():
    points = np.array(((0.01, 0.0, 0.7), (2.0, 0.0, 0.7)))
    segments = project_clipped_segments(points, (0, 0, 0), LEVEL_MOUNT, INTRINSICS, near=0.1)
    assert len(segments) == 1
    assert segments[0][0] == (320, 240)


def test_segment_crossing_image_boundary_is_clipped_to_edge():
    points = np.array(((5.0, 10.0, 0.7), (5.0, 0.0, 0.7)))
    segments = project_clipped_segments(points, (0, 0, 0), LEVEL_MOUNT, INTRINSICS)
    assert len(segments) == 1
    assert segments[0][0][0] == 0
    assert segments[0][1] == (320, 240)


def test_straight_curve_and_corner_have_projected_segments():
    paths = (
        np.array(((1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0))),
        np.array(((1.0, 0.0, 0.0), (3.0, 0.2, 0.0), (6.0, 1.0, 0.0))),
        np.array(((1.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0))),
    )
    for path in paths:
        assert project_clipped_segments(path, (0, 0, 0), CameraMount(), INTRINSICS)


def test_invalid_geometry_is_rejected():
    with np.testing.assert_raises(ValueError):
        world_to_optical(np.array(((math.nan, 0.0, 0.0),)), (0, 0, 0), LEVEL_MOUNT)
    with np.testing.assert_raises(ValueError):
        CameraIntrinsics(fx=0.0).validate()
