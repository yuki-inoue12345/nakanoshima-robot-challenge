"""Joint fixed-camera extrinsic calibration from surveyed checkerboards.

All surveyed coordinates use the ROS body convention: base x forward, y left,
and z up.  Images remain raw/distorted 640x480 camera frames.  One camera pose
is fitted jointly to every detected checkerboard corner in every placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .calibration_tool import (
    OPTICAL_FROM_CAMERA_BODY,
    _camera_from_result,
    _mount_from_optical_pose,
    _rotation_from_mount,
    detect_checkerboard,
)
from .camera_model import camera_matrix


CURRENT_PROVISIONAL_MOUNT = {
    "camera_x": 0.20,
    "camera_y": 0.094,
    "camera_z": 0.6967,
    "camera_yaw_deg": 0.0,
    "camera_pitch_up_deg": 7.5,
    "camera_roll_deg": 0.0,
}


@dataclass
class BoardObservation:
    identifier: str
    image_path: Path
    image: np.ndarray
    object_points: np.ndarray
    detected_points: np.ndarray
    requested_order: str
    origin_base_m: np.ndarray
    column_axis_base: np.ndarray
    row_axis_base: np.ndarray


def checkerboard_points_base(
    origin_base_m: Sequence[float],
    column_axis_base: Sequence[float],
    row_axis_base: Sequence[float],
    columns: int,
    rows: int,
    square_size_m: float,
) -> np.ndarray:
    """Generate row-major inner-corner coordinates in ``base_footprint``."""
    origin = np.asarray(origin_base_m, dtype=float)
    column_axis = np.asarray(column_axis_base, dtype=float)
    row_axis = np.asarray(row_axis_base, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_base_m must contain three finite metres")
    if np.max(np.abs(origin)) > 20.0:
        raise ValueError(
            "origin_base_m is outside +/-20 m; values may be centimetres, not metres"
        )
    for name, axis in (
        ("column_axis_base", column_axis),
        ("row_axis_base", row_axis),
    ):
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError(f"{name} must contain three finite values")
        if abs(float(np.linalg.norm(axis)) - 1.0) > 0.02:
            raise ValueError(f"{name} must be a unit vector")
    if abs(float(np.dot(column_axis, row_axis))) > 0.02:
        raise ValueError("checkerboard column and row axes must be perpendicular")
    if columns < 2 or rows < 2:
        raise ValueError("checkerboard inner-corner counts must be at least 2x2")
    if not math.isfinite(square_size_m) or not 0.005 <= square_size_m <= 0.10:
        raise ValueError(
            "square_size_m must be in metres between 0.005 and 0.10"
        )
    grid = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return np.asarray([
        origin
        + float(column) * square_size_m * column_axis
        + float(row) * square_size_m * row_axis
        for column, row in grid
    ], dtype=np.float64)


def _board_settings(layout: Mapping[str, Any]) -> tuple[int, int, float]:
    if layout.get("survey_confirmed") is not True:
        raise ValueError(
            "layout survey_confirmed must be true after every position and axis "
            "has been physically measured"
        )
    if str(layout.get("units", "")) != "m":
        raise ValueError("layout units must be explicitly set to 'm'")
    frame = str(layout.get("frame_id", ""))
    if frame not in {"base_footprint", "robot/base_footprint"}:
        raise ValueError("layout frame_id must be base_footprint or robot/base_footprint")
    board = layout.get("checkerboard", {})
    columns = int(board.get("inner_columns", 0))
    rows = int(board.get("inner_rows", 0))
    square_size = float(board.get("square_size_m", 0.0))
    if (columns, rows) != (9, 6):
        raise ValueError("this C920 procedure requires a 9x6 inner-corner board")
    if abs(square_size - 0.020) > 1e-6:
        raise ValueError("this C920 procedure requires measured 0.020 m squares")
    return columns, rows, square_size


def load_board_observations(
    layout: Mapping[str, Any],
) -> tuple[list[BoardObservation], dict[str, Any]]:
    """Load raw images, detect corners, and generate surveyed 3D corners."""
    columns, rows, square_size = _board_settings(layout)
    placements = layout.get("placements")
    if not isinstance(placements, list) or len(placements) < 3:
        raise ValueError("at least three surveyed checkerboard placements are required")
    observations = []
    identifiers = set()
    for index, placement in enumerate(placements):
        identifier = str(placement.get("id", f"placement_{index + 1}"))
        if identifier in identifiers:
            raise ValueError(f"duplicate placement id: {identifier}")
        identifiers.add(identifier)
        image_path = Path(str(placement.get("image", ""))).expanduser()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read placement image: {image_path}")
        if image.shape[:2] != (480, 640):
            raise ValueError(
                f"placement {identifier} must be a raw 640x480 frame, got "
                f"{image.shape[1]}x{image.shape[0]}"
            )
        corners = detect_checkerboard(image, columns, rows)
        if corners is None:
            raise ValueError(f"9x6 checkerboard not detected: {image_path}")
        order = str(placement.get("corner_order", "auto"))
        if order not in {"auto", "normal", "reversed"}:
            raise ValueError(
                f"placement {identifier} corner_order must be auto, normal, or reversed"
            )
        origin = np.asarray(placement.get("origin_base_m", ()), dtype=float)
        column_axis = np.asarray(
            placement.get("column_axis_base", ()), dtype=float
        )
        row_axis = np.asarray(placement.get("row_axis_base", ()), dtype=float)
        object_points = checkerboard_points_base(
            origin, column_axis, row_axis, columns, rows, square_size
        )
        observations.append(BoardObservation(
            identifier=identifier,
            image_path=image_path,
            image=image,
            object_points=object_points,
            detected_points=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            requested_order=order,
            origin_base_m=origin,
            column_axis_base=column_axis,
            row_axis_base=row_axis,
        ))
    all_points = np.vstack([item.object_points for item in observations])
    centered = all_points - np.mean(all_points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-7))
    spans = np.ptp(all_points, axis=0)
    if rank < 3:
        raise ValueError(
            "all checkerboard corners are globally coplanar; use placements at "
            "different forward distances as well as left/right positions"
        )
    centers = np.asarray([
        np.mean(item.object_points, axis=0) for item in observations
    ])
    geometry = {
        "placement_count": len(observations),
        "point_count": int(len(all_points)),
        "global_rank": rank,
        "point_span_base_m": {
            "x": float(spans[0]), "y": float(spans[1]), "z": float(spans[2]),
        },
        "placement_center_span_base_m": {
            "x": float(np.ptp(centers[:, 0])),
            "y": float(np.ptp(centers[:, 1])),
            "z": float(np.ptp(centers[:, 2])),
        },
        "centered_singular_values_m": [float(value) for value in singular],
        "condition_ratio": float(singular[0] / max(singular[-1], 1e-12)),
    }
    return observations, geometry


def _ordered_points(observation: BoardObservation, reversed_order: bool) -> np.ndarray:
    points = observation.detected_points
    return points[::-1].copy() if reversed_order else points.copy()


def _solve_pose(
    observations: Sequence[BoardObservation],
    reversed_orders: Sequence[bool],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    object_points = np.vstack([item.object_points for item in observations])
    image_points = np.vstack([
        _ordered_points(item, reverse)
        for item, reverse in zip(observations, reversed_orders)
    ])
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=4.0,
        confidence=0.999,
        iterationsCount=500,
    )
    if not success:
        raise ValueError("joint checkerboard PnP failed")
    if hasattr(cv2, "solvePnPRefineLM"):
        rotation, translation = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            matrix,
            distortion,
            rotation,
            translation,
        )
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    optical = (rotation_matrix @ object_points.T).T + translation.reshape(1, 3)
    if float(np.min(optical[:, 2])) <= 0.0:
        raise ValueError("estimated camera pose places surveyed points behind the camera")
    projected = cv2.projectPoints(
        object_points, rotation, translation, matrix, distortion
    )[0].reshape(-1, 2)
    delta = projected - image_points
    squared = np.sum(delta * delta, axis=1)
    mount = _mount_from_optical_pose(rotation_matrix, translation.reshape(3))
    return {
        "rotation_vector": rotation.reshape(3),
        "translation": translation.reshape(3),
        "rotation_matrix": rotation_matrix,
        "mount": mount,
        "rms": float(np.sqrt(np.mean(squared))),
        "maximum": float(np.sqrt(np.max(squared))),
        "inlier_count": 0 if inliers is None else int(len(inliers)),
        "projected": projected,
        "image_points": image_points,
        "per_point_squared": squared,
    }


def _select_corner_orders(
    observations: Sequence[BoardObservation],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[list[bool], dict[str, Any]]:
    fixed = []
    automatic = []
    for index, item in enumerate(observations):
        if item.requested_order == "auto":
            fixed.append(False)
            automatic.append(index)
        else:
            fixed.append(item.requested_order == "reversed")
    if len(automatic) > 10:
        raise ValueError(
            "at most ten automatic corner orders are supported; mark excess "
            "placements as normal or reversed after reviewing previews"
        )
    best = None
    best_orders = None
    for flags in itertools.product((False, True), repeat=len(automatic)):
        orders = list(fixed)
        for index, reverse in zip(automatic, flags):
            orders[index] = reverse
        try:
            result = _solve_pose(observations, orders, matrix, distortion)
        except (ValueError, cv2.error):
            continue
        if best is None or result["rms"] < best["rms"]:
            best = result
            best_orders = orders
    if best is None or best_orders is None:
        raise ValueError("no physically valid checkerboard corner ordering was found")
    return best_orders, best


def _per_placement_results(
    observations: Sequence[BoardObservation],
    orders: Sequence[bool],
    solved: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    offset = 0
    for item, reverse in zip(observations, orders):
        count = len(item.object_points)
        squared = np.asarray(solved["per_point_squared"])[offset:offset + count]
        results.append({
            "id": item.identifier,
            "image": str(item.image_path),
            "corner_order": "reversed" if reverse else "normal",
            "point_count": count,
            "rms_reprojection_error_px": float(np.sqrt(np.mean(squared))),
            "max_reprojection_error_px": float(np.sqrt(np.max(squared))),
            "origin_base_m": item.origin_base_m.tolist(),
            "column_axis_base": item.column_axis_base.tolist(),
            "row_axis_base": item.row_axis_base.tolist(),
        })
        offset += count
    return results


def _pose_from_mount(mount: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rotation_base_from_camera = _rotation_from_mount(mount)
    rotation_optical_from_base = (
        OPTICAL_FROM_CAMERA_BODY @ rotation_base_from_camera.T
    )
    origin = np.asarray([
        mount["camera_x"], mount["camera_y"], mount["camera_z"]
    ], dtype=float)
    translation = -rotation_optical_from_base @ origin
    rotation_vector = cv2.Rodrigues(rotation_optical_from_base)[0].reshape(3)
    return rotation_vector, translation


def _draw_cross(image: np.ndarray, point: Sequence[float], color) -> None:
    x, y = (int(round(value)) for value in point)
    cv2.line(image, (x - 4, y), (x + 4, y), color, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - 4), (x, y + 4), color, 1, cv2.LINE_AA)


def write_checkerboard_previews(
    observations: Sequence[BoardObservation],
    orders: Sequence[bool],
    solved: Mapping[str, Any],
    matrix: np.ndarray,
    distortion: np.ndarray,
    output_dir: Path,
    comparison_mount: Mapping[str, float],
) -> list[str]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite preview directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    current_rotation, current_translation = _pose_from_mount(comparison_mount)
    paths = []
    offset = 0
    for item, reverse in zip(observations, orders):
        count = len(item.object_points)
        predicted = np.asarray(solved["projected"])[offset:offset + count]
        observed = _ordered_points(item, reverse)
        current = cv2.projectPoints(
            item.object_points,
            current_rotation,
            current_translation,
            matrix,
            distortion,
        )[0].reshape(-1, 2)
        canvas = item.image.copy()
        for point in observed:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2,
                       (0, 255, 0), -1, cv2.LINE_AA)
        for point in current:
            _draw_cross(canvas, point, (0, 165, 255))
        for point in predicted:
            _draw_cross(canvas, point, (255, 0, 255))
        cv2.circle(canvas, tuple(np.rint(observed[0]).astype(int)), 8,
                   (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{item.identifier} order={'reversed' if reverse else 'normal'}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "green=observed magenta=fit orange=current-7.5deg",
            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        destination = output_dir / f"{item.identifier}_overlay.png"
        if not cv2.imwrite(str(destination), canvas):
            raise OSError(f"failed to write preview: {destination}")
        paths.append(str(destination))
        offset += count
    return paths


def _spread(results: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    names = (
        "camera_x", "camera_y", "camera_z", "camera_yaw_deg",
        "camera_pitch_up_deg", "camera_roll_deg",
    )
    return {
        name: {
            "min": float(min(item[name] for item in results)),
            "max": float(max(item[name] for item in results)),
            "range": float(max(item[name] for item in results)
                           - min(item[name] for item in results)),
            "stddev": float(np.std([item[name] for item in results])),
        }
        for name in names
    }


def _leave_one_out(
    observations: Sequence[BoardObservation],
    orders: Sequence[bool],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    estimates = []
    failures = []
    for omitted in range(len(observations)):
        kept_observations = [
            item for index, item in enumerate(observations) if index != omitted
        ]
        kept_orders = [
            item for index, item in enumerate(orders) if index != omitted
        ]
        points = np.vstack([item.object_points for item in kept_observations])
        if np.linalg.matrix_rank(points - points[0], tol=1e-7) < 3:
            failures.append(observations[omitted].identifier)
            continue
        solved = _solve_pose(
            kept_observations, kept_orders, matrix, distortion
        )
        estimates.append({
            "omitted": observations[omitted].identifier,
            **solved["mount"],
            "rms_reprojection_error_px": solved["rms"],
        })
    return {
        "successful_estimates": estimates,
        "failed_or_degenerate_omissions": failures,
        "parameter_spread": _spread(estimates) if len(estimates) >= 2 else None,
    }


def _monte_carlo_survey(
    observations: Sequence[BoardObservation],
    orders: Sequence[bool],
    matrix: np.ndarray,
    distortion: np.ndarray,
    iterations: int,
    position_std_m: float,
    orientation_std_deg: float,
    seed: int,
) -> dict[str, Any] | None:
    if iterations <= 0 or (position_std_m <= 0.0 and orientation_std_deg <= 0.0):
        return None
    if position_std_m < 0.0 or orientation_std_deg < 0.0:
        raise ValueError("survey uncertainty values cannot be negative")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        perturbed = []
        for item in observations:
            rotation_delta = cv2.Rodrigues(
                rng.normal(0.0, math.radians(orientation_std_deg), 3)
            )[0]
            column_axis = rotation_delta @ item.column_axis_base
            row_axis = rotation_delta @ item.row_axis_base
            origin = item.origin_base_m + rng.normal(0.0, position_std_m, 3)
            clone = BoardObservation(
                item.identifier, item.image_path, item.image,
                checkerboard_points_base(
                    origin, column_axis, row_axis, 9, 6, 0.020
                ),
                item.detected_points, item.requested_order, origin,
                column_axis, row_axis,
            )
            perturbed.append(clone)
        try:
            solved = _solve_pose(perturbed, orders, matrix, distortion)
        except (ValueError, cv2.error):
            continue
        estimates.append(solved["mount"])
    return {
        "iterations_requested": iterations,
        "iterations_succeeded": len(estimates),
        "assumed_independent_position_std_m": position_std_m,
        "assumed_independent_orientation_std_deg": orientation_std_deg,
        "parameter_spread": _spread(estimates) if len(estimates) >= 2 else None,
    }


def calibrate_board_extrinsics(
    layout: Mapping[str, Any],
    intrinsics_data: Mapping[str, Any],
    preview_dir: Path | None = None,
    monte_carlo_iterations: int = 200,
    sensitivity_intrinsics: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> dict[str, Any]:
    """Jointly estimate a fixed C920 mount from surveyed board placements."""
    intrinsics, distortion = _camera_from_result(intrinsics_data)
    if (intrinsics.width, intrinsics.height) != (640, 480):
        raise ValueError("board extrinsics require 640x480 intrinsics")
    matrix = camera_matrix(intrinsics)
    observations, geometry = load_board_observations(layout)
    orders, solved = _select_corner_orders(
        observations, matrix, distortion
    )
    placement_results = _per_placement_results(observations, orders, solved)
    warnings = []
    center_span = geometry["placement_center_span_base_m"]
    if center_span["x"] < 0.25:
        warnings.append(
            "forward-distance span is below 0.25 m; camera depth/angle separation is weak"
        )
    if center_span["y"] < 0.30:
        warnings.append(
            "left/right placement span is below 0.30 m; yaw/lateral separation is weak"
        )
    if geometry["condition_ratio"] > 100.0:
        warnings.append("surveyed 3D geometry is poorly conditioned")
    for item in placement_results:
        if item["rms_reprojection_error_px"] > 1.0:
            warnings.append(
                f"{item['id']} residual exceeds 1 px; recheck its measurements/order"
            )
    comparison_mount = dict(CURRENT_PROVISIONAL_MOUNT)
    comparison_mount.update(layout.get("comparison_mount", {}))
    previews = []
    if preview_dir is not None:
        previews = write_checkerboard_previews(
            observations, orders, solved, matrix, distortion,
            preview_dir, comparison_mount,
        )
    uncertainty = layout.get("survey_uncertainty", {})
    monte_carlo = _monte_carlo_survey(
        observations,
        orders,
        matrix,
        distortion,
        monte_carlo_iterations,
        float(uncertainty.get("position_std_m", 0.0)),
        float(uncertainty.get("orientation_std_deg", 0.0)),
        int(uncertainty.get("random_seed", 920)),
    )
    sensitivity = []
    for label, data in sensitivity_intrinsics:
        alternate_intrinsics, alternate_distortion = _camera_from_result(data)
        alternate_matrix = camera_matrix(alternate_intrinsics)
        alternate_orders, alternate = _select_corner_orders(
            observations, alternate_matrix, alternate_distortion
        )
        alternate_mount = alternate["mount"]
        sensitivity.append({
            "label": label,
            "corner_orders": [
                "reversed" if value else "normal" for value in alternate_orders
            ],
            "rms_reprojection_error_px": alternate["rms"],
            "aggregate": alternate_mount,
            "delta_from_primary": {
                name: float(alternate_mount[name] - solved["mount"][name])
                for name in solved["mount"]
            },
        })
    return {
        "format_version": 1,
        "calibration_type": "joint_surveyed_checkerboard_extrinsics",
        "review_only": True,
        "approved_for_runtime": False,
        "coordinate_convention": (
            "base x forward, y left, z up; camera_x/y/z are optical-center "
            "coordinates in base_footprint"
        ),
        "checkerboard_origin": (
            "origin_base_m is detected corner 0; column_axis points from corner "
            "0 to 8 and row_axis from corner 0 to corner 45"
        ),
        "geometry": geometry,
        "aggregate": solved["mount"],
        "joint_rms_reprojection_error_px": solved["rms"],
        "joint_max_reprojection_error_px": solved["maximum"],
        "ransac_initial_inlier_count": solved["inlier_count"],
        "total_point_count": int(sum(
            len(item.object_points) for item in observations
        )),
        "placement_results": placement_results,
        "leave_one_placement_out": _leave_one_out(
            observations, orders, matrix, distortion
        ),
        "survey_monte_carlo": monte_carlo,
        "intrinsics_sensitivity": sensitivity,
        "comparison_mount": comparison_mount,
        "preview_images": previews,
        "warnings": warnings,
        "required_review": [
            "Verify red-circled corner 0 and auto-selected order in every preview.",
            "Investigate any placement residual above 1 px before export.",
            "Do not merge the review overlay into the NUC YAML automatically.",
            "Re-run extrinsics if the intrinsics model or C920 capture controls change.",
        ],
    }
