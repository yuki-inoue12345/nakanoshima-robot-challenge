import cv2
import numpy as np
import pytest

from ar_path_viewer.board_extrinsics import (
    _pose_from_mount,
    calibrate_board_extrinsics,
    checkerboard_points_base,
)
from ar_path_viewer.calibration_tool import build_argument_parser


TRUE_MOUNT = {
    "camera_x": 0.20,
    "camera_y": 0.094,
    "camera_z": 0.6967,
    "camera_yaw_deg": 2.0,
    "camera_pitch_up_deg": 12.0,
    "camera_roll_deg": -1.0,
}


def candidate_intrinsics(focal_delta=0.0):
    return {
        "image_size": [640, 480],
        "camera_matrix": [
            [658.6575 + focal_delta, 0.0, 322.9545],
            [0.0, 657.3184 + focal_delta, 239.1993],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [0.0] * 5,
    }


def _checkerboard_texture():
    square = 60
    texture = np.zeros((7 * square, 10 * square), np.uint8)
    for row in range(7):
        for column in range(10):
            if (row + column) % 2 == 0:
                texture[
                    row * square:(row + 1) * square,
                    column * square:(column + 1) * square,
                ] = 255
    return texture


def _render_placement(path, origin, mount=TRUE_MOUNT):
    intrinsics = candidate_intrinsics()
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
    rotation, translation = _pose_from_mount(mount)
    column_axis = np.array((0.0, -1.0, 0.0))
    row_axis = np.array((0.0, 0.0, -1.0))
    outer_top_left = (
        np.asarray(origin, dtype=float) - 0.020 * column_axis - 0.020 * row_axis
    )
    outer = np.asarray((
        outer_top_left,
        outer_top_left + 0.20 * column_axis,
        outer_top_left + 0.20 * column_axis + 0.14 * row_axis,
        outer_top_left + 0.14 * row_axis,
    ))
    target = cv2.projectPoints(
        outer, rotation, translation, matrix, np.zeros(5)
    )[0].reshape(-1, 2).astype(np.float32)
    texture = _checkerboard_texture()
    source = np.float32((
        (0, 0), (texture.shape[1] - 1, 0),
        (texture.shape[1] - 1, texture.shape[0] - 1),
        (0, texture.shape[0] - 1),
    ))
    transform = cv2.getPerspectiveTransform(source, target)
    canvas = np.full((480, 640), 127, np.uint8)
    warped = cv2.warpPerspective(
        texture, transform, (640, 480), borderValue=127
    )
    mask = cv2.warpPerspective(
        np.full_like(texture, 255), transform, (640, 480), borderValue=0
    )
    canvas[mask > 0] = warped[mask > 0]
    assert cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))


def _synthetic_layout(tmp_path, origins=None):
    origins = origins or (
        (1.20, 0.08, 0.78),
        (1.55, 0.43, 0.79),
        (1.70, -0.27, 0.77),
        (2.00, 0.08, 0.82),
    )
    placements = []
    for index, origin in enumerate(origins):
        image = tmp_path / f"placement_{index + 1}.png"
        _render_placement(image, origin)
        placements.append({
            "id": f"placement_{index + 1}",
            "image": str(image),
            "origin_base_m": list(origin),
            "column_axis_base": [0.0, -1.0, 0.0],
            "row_axis_base": [0.0, 0.0, -1.0],
            "corner_order": "auto",
        })
    return {
        "survey_confirmed": True,
        "units": "m",
        "frame_id": "base_footprint",
        "checkerboard": {
            "inner_columns": 9,
            "inner_rows": 6,
            "square_size_m": 0.020,
        },
        "survey_uncertainty": {
            "position_std_m": 0.002,
            "orientation_std_deg": 0.2,
            "random_seed": 9,
        },
        "placements": placements,
    }


def test_joint_checkerboards_recover_mount_signs_and_generate_review(tmp_path):
    layout = _synthetic_layout(tmp_path)
    previews = tmp_path / "previews"
    result = calibrate_board_extrinsics(
        layout,
        candidate_intrinsics(),
        previews,
        monte_carlo_iterations=20,
        sensitivity_intrinsics=(("focal_plus_8", candidate_intrinsics(8.0)),),
    )
    recovered = result["aggregate"]
    assert abs(recovered["camera_x"] - TRUE_MOUNT["camera_x"]) < 0.003
    assert abs(recovered["camera_y"] - TRUE_MOUNT["camera_y"]) < 0.003
    assert abs(recovered["camera_z"] - TRUE_MOUNT["camera_z"]) < 0.003
    assert abs(recovered["camera_yaw_deg"] - TRUE_MOUNT["camera_yaw_deg"]) < 0.15
    assert abs(
        recovered["camera_pitch_up_deg"] - TRUE_MOUNT["camera_pitch_up_deg"]
    ) < 0.15
    assert abs(recovered["camera_roll_deg"] - TRUE_MOUNT["camera_roll_deg"]) < 0.15
    assert result["joint_rms_reprojection_error_px"] < 0.5
    assert result["review_only"] is True
    assert result["approved_for_runtime"] is False
    assert len(result["placement_results"]) == 4
    assert len(list(previews.glob("*_overlay.png"))) == 4
    assert result["survey_monte_carlo"]["iterations_succeeded"] == 20
    assert result["survey_monte_carlo"]["parameter_spread"][
        "camera_pitch_up_deg"
    ]["stddev"] > 0.0
    assert len(result["intrinsics_sensitivity"]) == 1


def test_globally_coplanar_board_placements_are_rejected(tmp_path):
    layout = _synthetic_layout(tmp_path, origins=(
        (1.50, 0.08, 0.78),
        (1.50, 0.43, 0.79),
        (1.50, -0.27, 0.77),
    ))
    with pytest.raises(ValueError, match="globally coplanar"):
        calibrate_board_extrinsics(
            layout, candidate_intrinsics(), monte_carlo_iterations=0
        )


def test_unconfirmed_survey_template_is_rejected(tmp_path):
    layout = _synthetic_layout(tmp_path)
    layout["survey_confirmed"] = False
    with pytest.raises(ValueError, match="survey_confirmed"):
        calibrate_board_extrinsics(
            layout, candidate_intrinsics(), monte_carlo_iterations=0
        )


def test_checkerboard_survey_rejects_unit_and_axis_mistakes():
    with pytest.raises(ValueError, match="centimetres"):
        checkerboard_points_base(
            (150.0, 0.0, 70.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0), 9, 6, 0.020,
        )
    with pytest.raises(ValueError, match="unit vector"):
        checkerboard_points_base(
            (1.5, 0.0, 0.7), (0.0, -100.0, 0.0),
            (0.0, 0.0, -1.0), 9, 6, 0.020,
        )
    with pytest.raises(ValueError, match="perpendicular"):
        checkerboard_points_base(
            (1.5, 0.0, 0.7), (0.0, -1.0, 0.0),
            (0.0, -1.0, 0.0), 9, 6, 0.020,
        )


def test_board_extrinsics_cli_requires_separate_review_outputs():
    args = build_argument_parser().parse_args([
        "extrinsics-board",
        "--intrinsics", "candidate.yaml",
        "--layout", "placements.yaml",
        "--output", "extrinsics-review.yaml",
        "--preview-dir", "previews",
        "--review-overlay", "overlay-review.yaml",
    ])
    assert args.command == "extrinsics-board"
    assert args.monte_carlo_iterations == 200
    assert args.output.name == "extrinsics-review.yaml"
    assert args.review_overlay.name == "overlay-review.yaml"
