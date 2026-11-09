import cv2
import numpy as np
import pytest
import yaml

from ar_path_viewer.calibration_tool import (
    OPTICAL_FROM_CAMERA_BODY,
    _rotation_from_mount,
    build_argument_parser,
    compare_projection,
    detect_checkerboard,
    estimate_extrinsics,
    export_runtime_overlay,
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
