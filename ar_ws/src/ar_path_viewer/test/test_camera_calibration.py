from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ar_path_viewer.calibration_tool import (
    CalibrationFrameCandidate,
    DEFAULT_CHECKERBOARD_INNER_COLUMNS,
    DEFAULT_CHECKERBOARD_INNER_ROWS,
    DEFAULT_CHECKERBOARD_SQUARE_SIZE_M,
    DEFAULT_MINIMUM_BOARD_AREA_RATIO,
    OPTICAL_FROM_CAMERA_BODY,
    _rotation_from_mount,
    build_argument_parser,
    characterize_calibration_frame,
    compare_projection,
    detect_checkerboard,
    estimate_extrinsics,
    extract_calibration_video_frames,
    export_runtime_overlay,
    record_calibration_video,
    select_diverse_calibration_frames,
)
from ar_path_viewer.camera_model import CameraUndistorter
from ar_path_viewer.projection import CameraIntrinsics


def intrinsics_result(distortion=None):
    return {
        "image_size": [640, 480],
        "camera_matrix": [
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": distortion or [0.0] * 5,
    }


def synthetic_checkerboard_frame(quad=None, blur=0):
    square = 36
    board = np.zeros((7 * square, 10 * square), np.uint8)
    for row in range(7):
        for column in range(10):
            if (row + column) % 2 == 0:
                board[
                    row * square:(row + 1) * square,
                    column * square:(column + 1) * square,
                ] = 255
    if quad is None:
        quad = ((140, 100), (500, 110), (490, 360), (130, 350))
    source = np.float32([
        (0, 0), (board.shape[1] - 1, 0),
        (board.shape[1] - 1, board.shape[0] - 1),
        (0, board.shape[0] - 1),
    ])
    transform = cv2.getPerspectiveTransform(source, np.float32(quad))
    canvas = np.full((480, 640), 127, np.uint8)
    warped = cv2.warpPerspective(board, transform, (640, 480), borderValue=127)
    mask = cv2.warpPerspective(
        np.full_like(board, 255), transform, (640, 480), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]
    if blur:
        canvas = cv2.GaussianBlur(canvas, (blur, blur), 0)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_checkerboard_detector_finds_9_by_6_inner_corners():
    square = 40
    board = np.zeros((7 * square, 10 * square), np.uint8)
    for row in range(7):
        for column in range(10):
            if (row + column) % 2:
                board[
                    row * square:(row + 1) * square,
                    column * square:(column + 1) * square,
                ] = 255
    canvas = np.full((480, 640), 127, np.uint8)
    canvas[100:380, 120:520] = board
    corners = detect_checkerboard(
        cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 9, 6
    )
    assert corners is not None
    assert corners.shape == (54, 1, 2)


def test_video_frame_characterization_measures_blur_pose_and_board_size():
    sharp = synthetic_checkerboard_frame()
    blurred = synthetic_checkerboard_frame(blur=21)
    sharp_corners = detect_checkerboard(sharp, 9, 6)
    blurred_corners = detect_checkerboard(blurred, 9, 6)
    sharp_result = characterize_calibration_frame(
        sharp, sharp_corners, 9, 6, 10, 0.5
    )
    blurred_result = characterize_calibration_frame(
        blurred, blurred_corners, 9, 6, 11, 0.55
    )
    assert sharp_result.area_ratio > 0.10
    assert 0.3 < sharp_result.center_x < 0.7
    assert sharp_result.sharpness > blurred_result.sharpness * 10.0


def test_diversity_selection_rejects_nearly_identical_compositions():
    frames = [
        synthetic_checkerboard_frame(),
        synthetic_checkerboard_frame(),
        synthetic_checkerboard_frame(
            ((30, 30), (330, 60), (300, 280), (50, 260))
        ),
        synthetic_checkerboard_frame(
            ((300, 160), (610, 100), (590, 390), (280, 350))
        ),
    ]
    candidates = []
    for index, image in enumerate(frames):
        corners = detect_checkerboard(image, 9, 6)
        candidates.append(characterize_calibration_frame(
            image, corners, 9, 6, index, index / 20.0
        ))
    selected = select_diverse_calibration_frames(
        candidates, target_images=4, min_diversity_distance=0.05
    )
    assert len(selected) == 3
    assert {item.frame_index for item in selected} != {0, 1}


def test_saved_video_is_detected_deduplicated_and_exported_as_png(tmp_path):
    video = tmp_path / "board.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (640, 480)
    )
    assert writer.isOpened()
    quads = (
        ((140, 100), (500, 110), (490, 360), (130, 350)),
        ((30, 30), (330, 60), (300, 280), (50, 260)),
        ((300, 160), (610, 100), (590, 390), (280, 350)),
        ((130, 180), (480, 110), (520, 350), (160, 390)),
    )
    for quad in quads:
        frame = synthetic_checkerboard_frame(quad)
        for _ in range(3):
            writer.write(frame)
    blurred = synthetic_checkerboard_frame(blur=21)
    writer.write(blurred)
    writer.write(blurred)
    writer.release()

    recorded = tmp_path / "recorded.mp4"
    recording = record_calibration_video(
        str(video), recorded, fps=20.0, duration_sec=0.001, display=False
    )
    assert recording["width"] == 640
    assert recording["height"] == 480
    assert recording["fps"] == 20.0
    assert recording["frames_written"] >= 1
    recorded_capture = cv2.VideoCapture(str(recorded))
    ok, recorded_frame = recorded_capture.read()
    recorded_capture.release()
    assert ok
    assert recorded_frame.shape[:2] == (480, 640)

    output_dir = tmp_path / "selected"
    report_path = tmp_path / "selection.yaml"
    report = extract_calibration_video_frames(
        video,
        output_dir,
        report_path,
        target_images=8,
        minimum_images=3,
        analysis_fps=20.0,
        min_diversity_distance=0.03,
    )
    assert 3 <= report["selected_count"] <= len(quads)
    assert report["analysis"]["accepted_candidates"] == 12
    assert report["analysis"]["blurred"] == 2
    assert report["checkerboard"]["inner_columns"] == 9
    assert report["checkerboard"]["inner_rows"] == 6
    assert report["checkerboard"]["square_size_m"] == 0.020
    assert report["missing_compositions"]
    assert any(
        "board position missing" in item
        for item in report["missing_compositions"]
    )
    assert len(list(output_dir.glob("*.png"))) == report["selected_count"]
    assert yaml.safe_load(report_path.read_text(encoding="utf-8"))[
        "selected_count"
    ] == report["selected_count"]

    blocked_report = tmp_path / "blocked_selection.yaml"
    blocked_intrinsics = tmp_path / "must_not_exist.yaml"
    with pytest.raises(ValueError, match="coverage is insufficient"):
        extract_calibration_video_frames(
            video,
            tmp_path / "blocked_selected",
            blocked_report,
            target_images=8,
            minimum_images=8,
            analysis_fps=20.0,
            min_diversity_distance=0.03,
            intrinsics_output=blocked_intrinsics,
        )
    assert blocked_report.exists()
    assert not blocked_intrinsics.exists()


def test_intrinsics_cli_defaults_match_actual_10_by_7_board_with_20mm_squares():
    args = build_argument_parser().parse_args([
        "intrinsics",
        "--images", "board_*.png",
        "--output", "intrinsics.yaml",
    ])
    assert args.inner_columns == DEFAULT_CHECKERBOARD_INNER_COLUMNS == 9
    assert args.inner_rows == DEFAULT_CHECKERBOARD_INNER_ROWS == 6
    assert args.square_size_m == DEFAULT_CHECKERBOARD_SQUARE_SIZE_M == 0.020


def test_video_cli_defaults_record_20fps_and_select_30_to_40_images():
    parser = build_argument_parser()
    record = parser.parse_args([
        "record-video", "--device", "0", "--output", "board.mp4",
    ])
    extract = parser.parse_args([
        "extract-video",
        "--video", "board.mp4",
        "--output-dir", "selected",
        "--report", "selection.yaml",
    ])
    assert record.fps == 20.0
    assert record.duration_sec == 90.0
    assert extract.target_images == 36
    assert extract.min_images == 30
    assert extract.analysis_fps == 4.0
    assert extract.minimum_area_ratio == DEFAULT_MINIMUM_BOARD_AREA_RATIO == 0.015


def test_diversity_selection_penalizes_tiny_board_when_pose_is_equivalent():
    descriptor = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    large = CalibrationFrameCandidate(
        0, 0.0, 500.0, 0.080, 0.5, 0.5, 0.0, 0.0, 0.0, descriptor
    )
    anchor = CalibrationFrameCandidate(
        1, 1.0, 500.0, 0.050, 0.2, 0.2, 0.0, 0.0, 0.0,
        (0.2, 0.2, 0.0, 0.0, 0.0, 0.0),
    )
    tiny = CalibrationFrameCandidate(
        2, 2.0, 2000.0, 0.015, 0.8, 0.8, 0.0, 0.0, 0.0,
        (0.8, 0.8, -0.5, 0.0, 0.0, 0.0),
    )
    useful = CalibrationFrameCandidate(
        3, 3.0, 500.0, 0.045, 0.8, 0.8, 0.0, 0.0, 0.0,
        (0.8, 0.8, -0.45, 0.0, 0.0, 0.0),
    )
    selected = select_diverse_calibration_frames(
        (large, anchor, tiny, useful), target_images=3, min_diversity_distance=0.01
    )
    assert useful in selected


def test_undistorter_preserves_640_by_480_and_provides_effective_intrinsics():
    intrinsics = CameraIntrinsics()
    undistorter = CameraUndistorter(
        intrinsics, (-0.20, 0.04, 0.001, -0.001, 0.0), alpha=0.0
    )
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.line(image, (0, 0), (639, 479), (255, 255, 255), 3)
    corrected = undistorter.apply(image)
    assert corrected.shape == image.shape
    assert undistorter.description.output_intrinsics.width == 640
    assert undistorter.description.output_intrinsics.height == 480
    assert not np.array_equal(corrected, image)


def test_synthetic_surveyed_points_recover_mount_convention():
    mount = {
        "camera_x": 0.20,
        "camera_y": 0.094,
        "camera_z": 0.6967,
        "camera_yaw_deg": 2.0,
        "camera_pitch_up_deg": 14.0,
        "camera_roll_deg": -1.0,
    }
    rotation_base_from_camera = _rotation_from_mount(mount)
    rotation_optical_from_base = (
        OPTICAL_FROM_CAMERA_BODY @ rotation_base_from_camera.T
    )
    origin = np.array((mount["camera_x"], mount["camera_y"], mount["camera_z"]))
    translation = -rotation_optical_from_base @ origin
    rotation_vector, _ = cv2.Rodrigues(rotation_optical_from_base)
    object_points = np.array([
        (4.0, -1.0, 0.0), (4.0, 1.0, 0.0),
        (6.0, -1.0, 0.0), (6.0, 1.0, 0.0),
        (8.0, -1.0, 0.0), (8.0, 1.0, 0.0),
        (4.0, -1.0, 1.2), (4.0, 1.0, 1.2),
        (6.0, -1.0, 1.2), (6.0, 1.0, 1.2),
        (8.0, -1.0, 1.2), (8.0, 1.0, 1.2),
    ], dtype=float)
    image_points, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        np.array(intrinsics_result()["camera_matrix"]),
        np.zeros(5),
    )
    data = {"samples": [{
        "image": "synthetic.png",
        "points": [
            {
                "name": f"p{index}",
                "base_xyz_m": point.tolist(),
                "pixel_uv": pixel.tolist(),
            }
            for index, (point, pixel) in enumerate(zip(
                object_points, image_points.reshape(-1, 2)
            ))
        ],
    }]}
    result = estimate_extrinsics(data, intrinsics_result())
    recovered = result["aggregate"]
    for name, expected in mount.items():
        assert abs(recovered[name] - expected) < 1e-4
    assert result["sample_results"][0]["rms_reprojection_error_px"] < 1e-5


def test_export_is_review_only_and_does_not_replace_existing_configuration():
    extrinsics = {
        "aggregate": {
            "camera_x": 0.21,
            "camera_y": 0.09,
            "camera_z": 0.70,
            "camera_yaw_deg": 1.0,
            "camera_pitch_up_deg": 12.0,
            "camera_roll_deg": -0.5,
        }
    }
    overlay = export_runtime_overlay(
        intrinsics_result([-0.2, 0.04, 0.0, 0.0, 0.0]), extrinsics, 0.0
    )
    assert overlay["calibration_overlay_only"]
    assert overlay["review_required_before_merge"]
    params = overlay["ar_path_viewer"]["ros__parameters"]
    assert params["camera_undistort"] is True
    assert params["camera_pitch_up_deg"] == 12.0
    assert "image_input_uri" not in params
    assert "output_video" not in params


def test_post_remount_review_overlay_is_scoped_and_not_production_enabling():
    overlay_path = (
        Path(__file__).parents[1]
        / "config"
        / "c920_fixed_20261111_review_overlay.yaml"
    )
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    params = overlay["ar_path_viewer"]["ros__parameters"]

    assert params["width"] == 640
    assert params["height"] == 480
    assert params["fx"] == pytest.approx(658.6575)
    assert params["fy"] == pytest.approx(657.3184)
    assert params["cx"] == pytest.approx(322.9545)
    assert params["cy"] == pytest.approx(239.1993)
    assert params["camera_undistort"] is True
    assert params["camera_distortion_coefficients"] == pytest.approx(
        [0.0554541, -0.0907945, 0.0, 0.0, 0.0]
    )
    assert params["camera_pitch_up_deg"] == pytest.approx(11.70)
    assert params["calibration_profile_label"] == "C920 REVIEW"
    assert params["path_display_mode"] == "elevated"
    assert params["path_display_height_m"] == pytest.approx(0.50)

    # The overlay cannot silently select devices, destinations, or enable the
    # display-only reference path. Those remain owned by the base/test launch.
    for forbidden in (
        "image_input_uri",
        "output_video",
        "log_output",
        "reference_path_enabled",
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_yaw_deg",
        "camera_roll_deg",
    ):
        assert forbidden not in params


def test_perception_costmap_is_opt_in_and_review_overlay_is_scoped():
    config_directory = Path(__file__).parents[1] / "config"
    for filename in ("ar_path_viewer.yaml", "ar_path_viewer_c920.yaml"):
        document = yaml.safe_load(
            (config_directory / filename).read_text(encoding="utf-8")
        )
        params = document["ar_path_viewer"]["ros__parameters"]
        assert params["perception_costmap_enabled"] is False
        assert params["perception_explanation_enabled"] is False
        assert params["perception_minimap_enabled"] is False
        assert params["perception_alert_enabled"] is False
        assert params["perception_costmap_topic"] == "/local_costmap/costmap"
        assert params["perception_costmap_max_grid_width_cells"] == 512
        assert params["perception_costmap_max_grid_height_cells"] == 512
        assert params["perception_costmap_max_grid_cells"] == 65536
        assert params["perception_camera_sync_max_age_sec"] == 0.40
        assert params["perception_camera_sync_max_capture_duration_sec"] == 0.12
        assert params["perception_camera_sync_max_frame_transport_age_sec"] == 0.12
        assert params["perception_camera_sync_clock_tolerance_sec"] == 0.02
        assert params["perception_camera_sync_max_translation_m"] == 0.20
        assert params["perception_camera_sync_max_rotation_deg"] == 15.0
        assert params["perception_camera_sync_exact_tf_tolerance_sec"] == 0.005

    review = yaml.safe_load(
        (config_directory / "perception_costmap_review_overlay.yaml").read_text(
            encoding="utf-8"
        )
    )
    params = review["ar_path_viewer"]["ros__parameters"]
    expected_review = {
        "perception_costmap_enabled": True,
        "perception_camera_frame_delay_enabled": False,
        "perception_costmap_camera_history_enabled": True,
        "perception_minimap_visual_hold_sec": 1.5,
        "rotation_guidance_enabled": True,
        "perception_explanation_enabled": True,
        "perception_minimap_enabled": True,
        "perception_alert_enabled": True,
        "perception_costmap_camera_overlay_enabled": False,
        "perception_alert_nearby_lethal_distance_m": 0.60,
        "perception_alert_confirm_sec": 0.50,
        "perception_alert_forward_confirm_sec": 0.80,
        "perception_alert_forward_max_m": 2.00,
        "perception_alert_forward_half_width_m": 0.40,
        "perception_alert_forward_min_cost": 85,
        "perception_alert_forward_min_cells": 2,
    }
    assert params == expected_review

    v4l2 = yaml.safe_load(
        (config_directory /
         "perception_costmap_v4l2_review_overlay.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert v4l2["ar_path_viewer"]["ros__parameters"] == {
        **expected_review,
        "image_input_type": "camera_timestamped",
        "perception_costmap_visual_hold_sec": 0.85,
    }


def test_coplanar_survey_points_are_rejected_as_ambiguous():
    points = [
        {
            "base_xyz_m": [float(index), float(index % 2), 0.0],
            "pixel_uv": [100.0 + index, 200.0 + index],
        }
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="coplanar"):
        estimate_extrinsics(
            {"samples": [{"image": "plane.png", "points": points}]},
            intrinsics_result(),
        )


def test_mark_points_command_requires_explicit_input_and_output_paths():
    args = build_argument_parser().parse_args([
        "mark-points",
        "--image", "reference.png",
        "--reference-points", "survey.yaml",
        "--output", "correspondences.yaml",
    ])
    assert args.command == "mark-points"
    assert args.image.name == "reference.png"
    assert args.reference_points.name == "survey.yaml"
    assert args.output.name == "correspondences.yaml"


def test_before_after_projection_comparison_writes_side_by_side_image(tmp_path):
    image_path = tmp_path / "reference.png"
    points_path = tmp_path / "path.csv"
    before_path = tmp_path / "before.yaml"
    overlay_path = tmp_path / "overlay.yaml"
    output_path = tmp_path / "comparison.png"
    assert cv2.imwrite(str(image_path), np.full((480, 640, 3), 30, np.uint8))
    np.savetxt(
        points_path,
        np.array(((0.5, 0.0, 0.0), (8.0, 0.0, 0.0))),
        delimiter=",",
    )
    before = {
        "ar_path_viewer": {"ros__parameters": {
            "width": 640, "height": 480,
            "fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5,
            "camera_x": 0.20, "camera_y": 0.094, "camera_z": 0.6967,
            "camera_pitch_up_deg": 7.5,
        }}
    }
    overlay = export_runtime_overlay(intrinsics_result(), None, 0.0)
    before_path.write_text(yaml.safe_dump(before), encoding="utf-8")
    overlay_path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    compare_projection(
        image_path, points_path, before_path, overlay_path, output_path
    )
    comparison = cv2.imread(str(output_path))
    assert comparison.shape == (480, 1280, 3)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compare_projection(
            image_path, points_path, before_path, overlay_path, output_path
        )


def test_v4l2_timestamped_review_profile_is_not_production_default():
    config_dir = Path(__file__).parents[1] / "config"
    opt_in = yaml.safe_load((config_dir / (
        "perception_costmap_v4l2_review_overlay.yaml"
    )).read_text(encoding="utf-8"))["ar_path_viewer"]["ros__parameters"]
    assert opt_in["image_input_type"] == "camera_timestamped"
    assert opt_in["perception_costmap_enabled"] is True
    for profile in ("ar_path_viewer.yaml", "ar_path_viewer_c920.yaml"):
        base = yaml.safe_load((config_dir / profile).read_text(
            encoding="utf-8"
        ))["ar_path_viewer"]["ros__parameters"]
        assert base["perception_costmap_enabled"] is False
        assert base["image_input_type"] != "camera_timestamped"
