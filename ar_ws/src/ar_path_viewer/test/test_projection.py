import math
from pathlib import Path

import numpy as np
import yaml

from ar_path_viewer.projection import (
    DEFAULT_CAMERA_Y_M,
    CameraIntrinsics,
    CameraMount,
    project_path,
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
        assert (params["display_width"], params["display_height"]) == (960, 720)
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


def test_segment_with_both_endpoints_outside_is_preserved_when_it_crosses_view():
    points = np.array(((5.0, 10.0, 0.7), (5.0, -10.0, 0.7)))
    result = project_path(points, (0, 0, 0), LEVEL_MOUNT, INTRINSICS)
    assert result.stats.image_points == 0
    assert result.stats.drawable_segments == 1
    assert result.segments[0][0][0] == 0
    assert result.segments[0][1][0] == INTRINSICS.width - 1
    assert 0.0 < result.stats.visible_path_length_m < result.stats.path_length_m
    assert math.isclose(result.stats.visible_depth_min_m, 5.0, abs_tol=1e-9)
    assert math.isclose(result.stats.visible_depth_max_m, 5.0, abs_tol=1e-9)


def test_default_upward_pitch_ground_visibility_matches_independent_geometry():
    mount = CameraMount()
    intrinsics = CameraIntrinsics()
    theta = math.radians(mount.pitch_up_deg)
    bottom_ratio = (intrinsics.height - 1 - intrinsics.cy) / intrinsics.fy
    camera_distance = mount.z * (
        math.cos(theta) + bottom_ratio * math.sin(theta)
    ) / (bottom_ratio * math.cos(theta) - math.sin(theta))
    base_x = mount.x + camera_distance
    optical = world_to_optical(
        np.array(((base_x, mount.y, 0.0),)), (0.0, 0.0, 0.0), mount
    )[0]
    pixel = np.array((
        intrinsics.fx * optical[0] / optical[2] + intrinsics.cx,
        intrinsics.fy * optical[1] / optical[2] + intrinsics.cy,
    ))
    assert math.isclose(camera_distance, 2.275675, abs_tol=1e-6)
    assert np.allclose(pixel, (intrinsics.cx, intrinsics.height - 1), atol=1e-6)


def test_projection_stats_classify_behind_below_and_visible_paths():
    behind = project_path(
        np.array(((-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        (0, 0, 0), CameraMount(), CameraIntrinsics(),
    )
    assert behind.stats.drawable_segments == 0
    assert behind.stats.out_of_view_reason == "all Path points are behind the camera"

    below = project_path(
        np.array(((0.5, 0.0, 0.0), (1.5, 0.0, 0.0))),
        (0, 0, 0), CameraMount(), CameraIntrinsics(),
    )
    assert below.stats.drawable_segments == 0
    assert below.stats.below_points == below.stats.front_points
    assert below.stats.out_of_view_reason == "all forward Path points are below the image"

    visible = project_path(
        np.array(((2.5, 0.0, 0.0), (4.0, 0.0, 0.0), (6.0, 1.0, 0.0))),
        (0, 0, 0), CameraMount(), CameraIntrinsics(),
    )
    assert visible.stats.front_points == 3
    assert visible.stats.drawable_segments == 2
    assert visible.stats.out_of_view_reason == ""


def test_height_and_pitch_change_ground_entry_without_shifting_pixels_artificially():
    path = np.array(tuple((x, 0.0, 0.0) for x in np.linspace(0.5, 6.0, 40)))
    level_low = project_path(
        path, (0, 0, 0), CameraMount(z=0.4, pitch_up_deg=0.0), INTRINSICS
    )
    upward_high = project_path(
        path, (0, 0, 0), CameraMount(z=0.9, pitch_up_deg=10.0), INTRINSICS
    )
    assert level_low.stats.drawable_segments > upward_high.stats.drawable_segments


def test_diagnostics_distinguish_dense_short_path_from_long_visible_path():
    short = np.column_stack((
        np.linspace(2.5, 2.8, 119), np.zeros(119), np.zeros(119)
    ))
    long = np.column_stack((
        np.linspace(2.5, 20.0, 119), np.zeros(119), np.zeros(119)
    ))
    short_result = project_path(short, (0, 0, 0), CameraMount(), CameraIntrinsics())
    long_result = project_path(long, (0, 0, 0), CameraMount(), CameraIntrinsics())
    assert short_result.stats.transformed_points == long_result.stats.transformed_points == 119
    assert math.isclose(short_result.stats.path_length_m, 0.3, abs_tol=1e-9)
    assert math.isclose(long_result.stats.path_length_m, 17.5, abs_tol=1e-9)
    assert math.isclose(short_result.stats.endpoint_distance_m, 2.8, abs_tol=1e-9)
    assert math.isclose(long_result.stats.endpoint_distance_m, 20.0, abs_tol=1e-9)
    assert math.isclose(
        short_result.stats.camera_forward_path_length_m, 0.3, abs_tol=1e-9
    )
    assert math.isclose(
        long_result.stats.camera_forward_path_length_m, 17.5, abs_tol=1e-9
    )
    assert math.isclose(
        short_result.stats.visible_path_length_m, 0.3, abs_tol=1e-9
    )
    assert math.isclose(
        long_result.stats.visible_path_length_m, 17.5, abs_tol=1e-9
    )
    assert long_result.stats.front_depth_max_m > 19.0
    assert long_result.stats.visible_depth_max_m > 19.0
    assert math.isclose(
        long_result.stats.nonvisible_forward_path_length_m, 0.0, abs_tol=1e-9
    )
    assert long_result.stats.visible_height_px > 140
    assert long_result.stats.visible_pixel_length > 150.0
    assert short_result.stats.visible_height_px < long_result.stats.visible_height_px
    assert short_result.stats.visible_pixel_length < long_result.stats.visible_pixel_length


def test_metric_lengths_split_total_forward_and_camera_visible_portions():
    # Level camera: half the path is behind the camera, while near ground is below view.
    path = np.array(((-2.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    result = project_path(path, (0, 0, 0), LEVEL_MOUNT, INTRINSICS)
    stats = result.stats
    assert math.isclose(stats.path_length_m, 12.0, abs_tol=1e-9)
    assert math.isclose(stats.endpoint_distance_m, 10.0, abs_tol=1e-9)
    assert math.isclose(stats.camera_forward_path_length_m, 10.0, abs_tol=1e-9)
    assert 8.0 < stats.visible_path_length_m < 9.0
    assert 1.0 < stats.nonvisible_forward_path_length_m < 2.0
    assert stats.visible_depth_min_m >= 0.1
    assert math.isclose(stats.visible_depth_max_m, 10.0, abs_tol=1e-9)
    assert stats.drawable_segments == 1
    assert stats.visible_pixel_length > 200.0


def test_empty_path_has_no_endpoint_and_zero_metric_lengths():
    result = project_path(
        np.empty((0, 3)), (0, 0, 0), CameraMount(), CameraIntrinsics()
    )
    assert result.stats.path_length_m == 0.0
    assert result.stats.endpoint_distance_m is None
    assert result.stats.camera_forward_path_length_m == 0.0
    assert result.stats.visible_path_length_m == 0.0
    assert result.stats.nonvisible_forward_path_length_m == 0.0
    assert result.stats.visible_depth_min_m is None
    assert result.stats.visible_depth_max_m is None


def test_straight_curve_and_corner_have_projected_segments():
    paths = (
        np.array(((1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0))),
        np.array(((1.0, 0.0, 0.0), (3.0, 0.2, 0.0), (6.0, 1.0, 0.0))),
        np.array(((1.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0))),
    )
    for path in paths:
        assert project_clipped_segments(path, (0, 0, 0), CameraMount(), INTRINSICS)


def test_mirrored_long_turns_keep_equal_visible_metric_lengths():
    x = np.linspace(2.5, 12.0, 119)
    bend = np.square(np.clip((x - 6.0) / 6.0, 0.0, 1.0)) * 2.0
    left = np.column_stack((x, bend, np.zeros_like(x)))
    right = np.column_stack((x, -bend, np.zeros_like(x)))
    left_result = project_path(left, (0, 0, 0), CameraMount(y=0.0), CameraIntrinsics())
    right_result = project_path(right, (0, 0, 0), CameraMount(y=0.0), CameraIntrinsics())
    assert left_result.stats.drawable_segments > 100
    assert right_result.stats.drawable_segments > 100
    assert math.isclose(
        left_result.stats.path_length_m,
        right_result.stats.path_length_m,
        abs_tol=1e-9,
    )
    assert math.isclose(
        left_result.stats.visible_path_length_m,
        right_result.stats.visible_path_length_m,
        abs_tol=1e-9,
    )


def test_invalid_geometry_is_rejected():
    with np.testing.assert_raises(ValueError):
        world_to_optical(np.array(((math.nan, 0.0, 0.0),)), (0, 0, 0), LEVEL_MOUNT)
    with np.testing.assert_raises(ValueError):
        CameraIntrinsics(fx=0.0).validate()
