"""Offline C920 calibration and before/after projection comparison tools."""

import argparse
from datetime import datetime
import glob
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .camera_model import CameraUndistorter, camera_matrix, validate_distortion
from .projection import CameraIntrinsics, CameraMount, project_path
from .rendering import render_frame


OPTICAL_FROM_CAMERA_BODY = np.array(
    ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_new_yaml(path: Path, data: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False, allow_unicode=True)


def _camera_from_result(data: Mapping[str, Any]) -> tuple[CameraIntrinsics, np.ndarray]:
    width, height = (int(value) for value in data["image_size"])
    matrix = np.asarray(data["camera_matrix"], dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )
    intrinsics.validate()
    distortion = validate_distortion(data["distortion_coefficients"])
    return intrinsics, distortion


def detect_checkerboard(
    image: np.ndarray,
    columns: int,
    rows: int,
) -> np.ndarray | None:
    if columns < 2 or rows < 2:
        raise ValueError("checkerboard inner-corner counts must be at least 2x2")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (columns, rows)
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, candidate = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            corners = candidate
    if corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, candidate = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.001,
            )
            corners = cv2.cornerSubPix(
                gray, candidate, (11, 11), (-1, -1), criteria
            )
    return None if corners is None else np.asarray(corners, dtype=np.float32)


def calibrate_intrinsics(
    image_paths: Sequence[Path],
    columns: int,
    rows: int,
    square_size_m: float,
    min_images: int = 15,
    preview_dir: Path | None = None,
) -> dict[str, Any]:
    if square_size_m <= 0.0 or not math.isfinite(square_size_m):
        raise ValueError("checkerboard square size must be finite and positive")
    if min_images < 3:
        raise ValueError("at least three accepted calibration images are required")
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= float(square_size_m)
    object_points = []
    image_points = []
    accepted = []
    rejected = []
    image_size = None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(path), "reason": "unreadable image"})
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected.append({
                "image": str(path),
                "reason": f"size {current_size} differs from {image_size}",
            })
            continue
        corners = detect_checkerboard(image, columns, rows)
        if corners is None:
            rejected.append({"image": str(path), "reason": "corners not found"})
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(str(path))
        if preview_dir is not None:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, (columns, rows), corners, True)
            destination = preview_dir / f"{path.stem}_corners.png"
            if not cv2.imwrite(str(destination), preview):
                raise OSError(f"failed to write preview: {destination}")

    if image_size is None:
        raise ValueError("no readable calibration images")
    if image_size != (640, 480):
        raise ValueError(
            f"C920 calibration must use 640x480 images, got {image_size}"
        )
    if len(accepted) < min_images:
        raise ValueError(
            f"only {len(accepted)} checkerboards accepted; need at least {min_images}"
        )
    rms, matrix, distortion, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    )
    per_image_errors = []
    for object_set, image_set, rotation, translation in zip(
        object_points, image_points, rotation_vectors, translation_vectors
    ):
        projected, _ = cv2.projectPoints(
            object_set, rotation, translation, matrix, distortion
        )
        error = np.linalg.norm(
            image_set.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
        )
        per_image_errors.append(float(np.sqrt(np.mean(error ** 2))))
    return {
        "format_version": 1,
        "calibration_type": "opencv_pinhole",
        "image_size": [int(image_size[0]), int(image_size[1])],
        "checkerboard": {
            "inner_columns": int(columns),
            "inner_rows": int(rows),
            "square_size_m": float(square_size_m),
        },
        "accepted_images": accepted,
        "rejected_images": rejected,
        "rms_reprojection_error_px": float(rms),
        "per_image_rms_error_px": per_image_errors,
        "mean_per_image_rms_error_px": float(np.mean(per_image_errors)),
        "max_per_image_rms_error_px": float(np.max(per_image_errors)),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
    }


def _mount_from_optical_pose(
    rotation_optical_from_base: np.ndarray,
    translation_optical_from_base: np.ndarray,
) -> dict[str, float]:
    camera_origin_base = -rotation_optical_from_base.T @ translation_optical_from_base
    rotation_base_from_camera = (
        rotation_optical_from_base.T @ OPTICAL_FROM_CAMERA_BODY
    )
    pitch_standard = math.asin(float(np.clip(
        -rotation_base_from_camera[2, 0], -1.0, 1.0
    )))
    yaw = math.atan2(
        float(rotation_base_from_camera[1, 0]),
        float(rotation_base_from_camera[0, 0]),
    )
    roll = math.atan2(
        float(rotation_base_from_camera[2, 1]),
        float(rotation_base_from_camera[2, 2]),
    )
    return {
        "camera_x": float(camera_origin_base[0]),
        "camera_y": float(camera_origin_base[1]),
        "camera_z": float(camera_origin_base[2]),
        "camera_yaw_deg": math.degrees(yaw),
        "camera_pitch_up_deg": -math.degrees(pitch_standard),
        "camera_roll_deg": math.degrees(roll),
    }


def _rotation_from_mount(mount: Mapping[str, float]) -> np.ndarray:
    yaw = math.radians(float(mount["camera_yaw_deg"]))
    pitch = math.radians(-float(mount["camera_pitch_up_deg"]))
    roll = math.radians(float(mount["camera_roll_deg"]))
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rz = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    return rz @ ry @ rx


def estimate_extrinsics(
    correspondence_data: Mapping[str, Any],
    intrinsics_data: Mapping[str, Any],
) -> dict[str, Any]:
    intrinsics, distortion = _camera_from_result(intrinsics_data)
    samples = correspondence_data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("correspondence YAML requires a non-empty samples list")
    results = []
    rotations = []
    origins = []
    for sample in samples:
        image_path = str(sample.get("image", ""))
        points = sample.get("points", [])
        object_points = np.asarray(
            [point["base_xyz_m"] for point in points], dtype=np.float64
        )
        image_points = np.asarray(
            [point["pixel_uv"] for point in points], dtype=np.float64
        )
        if len(object_points) < 6 or object_points.shape != (len(points), 3):
            raise ValueError(f"sample {image_path!r} needs at least six 3D points")
        if image_points.shape != (len(points), 2):
            raise ValueError(f"sample {image_path!r} has invalid pixel points")
        if not np.isfinite(object_points).all() or not np.isfinite(image_points).all():
            raise ValueError(f"sample {image_path!r} contains NaN or Inf")
        if np.linalg.matrix_rank(object_points - object_points[0], tol=1e-7) < 3:
            raise ValueError(
                f"sample {image_path!r} points are coplanar; use surveyed "
                "floor and vertical targets at multiple depths/heights"
            )
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix(intrinsics),
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=200,
        )
        if not success or inliers is None or len(inliers) < 6:
            raise ValueError(f"PnP failed or had fewer than six inliers: {image_path}")
        inlier_indices = inliers.reshape(-1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rotation_vector, translation = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix(intrinsics),
                distortion,
                rotation_vector,
                translation,
            )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        projected, _ = cv2.projectPoints(
            object_points, rotation_vector, translation,
            camera_matrix(intrinsics), distortion,
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1
        )
        mount = _mount_from_optical_pose(rotation, translation.reshape(3))
        rotations.append(_rotation_from_mount(mount))
        origins.append([mount["camera_x"], mount["camera_y"], mount["camera_z"]])
        results.append({
            "image": image_path,
            "point_count": int(len(points)),
            "inlier_count": int(len(inliers)),
            "rms_reprojection_error_px": float(np.sqrt(np.mean(errors ** 2))),
            "max_reprojection_error_px": float(np.max(errors)),
            **mount,
        })

    mean_rotation = np.mean(np.asarray(rotations), axis=0)
    u_matrix, _, vh_matrix = np.linalg.svd(mean_rotation)
    aggregate_rotation = u_matrix @ vh_matrix
    if np.linalg.det(aggregate_rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        aggregate_rotation = u_matrix @ vh_matrix
    aggregate_origin = np.median(np.asarray(origins), axis=0)
    rotation_optical_from_base = (
        OPTICAL_FROM_CAMERA_BODY @ aggregate_rotation.T
    )
    translation = -rotation_optical_from_base @ aggregate_origin
    aggregate = _mount_from_optical_pose(
        rotation_optical_from_base, translation
    )
    spread = {
        key + "_range": float(max(item[key] for item in results) - min(
            item[key] for item in results
        ))
        for key in aggregate
    }
    return {
        "format_version": 1,
        "calibration_type": "base_to_camera_extrinsics",
        "sample_results": results,
        "aggregate": aggregate,
        "sample_spread": spread,
        "warning": (
            "Do not apply until surveyed base_xyz_m values and reprojection "
            "errors have been independently reviewed."
        ),
    }


def export_runtime_overlay(
    intrinsics_data: Mapping[str, Any],
    extrinsics_data: Mapping[str, Any] | None,
    alpha: float,
) -> dict[str, Any]:
    intrinsics, distortion = _camera_from_result(intrinsics_data)
    params: dict[str, Any] = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "camera_undistort": True,
        "camera_undistort_alpha": float(alpha),
        "camera_distortion_coefficients": [
            float(value) for value in distortion
        ],
    }
    if extrinsics_data is not None:
        aggregate = extrinsics_data.get("aggregate", {})
        for name in (
            "camera_x",
            "camera_y",
            "camera_z",
            "camera_yaw_deg",
            "camera_pitch_up_deg",
            "camera_roll_deg",
        ):
            params[name] = float(aggregate[name])
    return {
        "calibration_overlay_only": True,
        "review_required_before_merge": True,
        "ar_path_viewer": {"ros__parameters": params},
    }


def _runtime_parameters(data: Mapping[str, Any]) -> dict[str, Any]:
    if "ar_path_viewer" in data:
        return dict(data["ar_path_viewer"]["ros__parameters"])
    return dict(data)


def _load_path_points(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = _read_yaml(path)
        values = data.get("points_base", data) if isinstance(data, dict) else data
        points = np.asarray(values, dtype=float)
    else:
        points = np.loadtxt(path, delimiter=",", dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3} or len(points) < 2:
        raise ValueError("Path points must be an Nx2 or Nx3 array")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    if not np.isfinite(points).all():
        raise ValueError("Path points contain NaN or Inf")
    return points


def _render_configuration(
    source_image: np.ndarray,
    points_base: np.ndarray,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    source_intrinsics = CameraIntrinsics(
        width=int(params["width"]),
        height=int(params["height"]),
        fx=float(params["fx"]),
        fy=float(params["fy"]),
        cx=float(params["cx"]),
        cy=float(params["cy"]),
    )
    image = cv2.resize(
        source_image, (source_intrinsics.width, source_intrinsics.height)
    )
    intrinsics = source_intrinsics
    if bool(params.get("camera_undistort", False)):
        undistorter = CameraUndistorter(
            source_intrinsics,
            params["camera_distortion_coefficients"],
            float(params.get("camera_undistort_alpha", 0.0)),
        )
        image = undistorter.apply(image)
        intrinsics = undistorter.description.output_intrinsics
    mount = CameraMount(
        x=float(params["camera_x"]),
        y=float(params["camera_y"]),
        z=float(params["camera_z"]),
        yaw_deg=float(params.get("camera_yaw_deg", 0.0)),
        pitch_up_deg=float(params.get("camera_pitch_up_deg", 0.0)),
        roll_deg=float(params.get("camera_roll_deg", 0.0)),
    )
    projection = project_path(
        points_base,
        (0.0, 0.0, 0.0),
        mount,
        intrinsics,
        float(params.get("near_clip_m", 0.1)),
    )
    state = "ACTIVE" if projection.segments else "PATH OUT OF VIEW"
    detail = projection.stats.compact_detail()
    return render_frame(image, projection.segments, state, detail=detail), detail


def compare_projection(
    image_path: Path,
    path_points_path: Path,
    before_config_path: Path,
    after_overlay_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing comparison: {output_path}"
        )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read comparison image: {image_path}")
    points = _load_path_points(path_points_path)
    before = _runtime_parameters(_read_yaml(before_config_path))
    after = dict(before)
    after.update(_runtime_parameters(_read_yaml(after_overlay_path)))
    before_frame, _ = _render_configuration(image, points, before)
    after_frame, _ = _render_configuration(image, points, after)
    cv2.putText(
        before_frame, "BEFORE", (12, before_frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        after_frame, "CALIBRATED", (12, after_frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    comparison = np.hstack((before_frame, after_frame))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"failed to write comparison: {output_path}")


def capture_images(device: str, output_dir: Path, fps: float) -> int:
    try:
        source: int | str = int(device)
    except ValueError:
        source = device
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, float(fps))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"cannot open camera: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    window = "C920 calibration capture - SPACE/C save, Q/F10 exit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                raise OSError("camera stopped while capturing calibration images")
            if image.shape[:2] != (480, 640):
                image = cv2.resize(image, (640, 480))
            display = image.copy()
            cv2.putText(
                display, f"saved={saved}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKeyEx(20)
            low = key & 0xFF
            if low in (ord("q"), ord("Q")) or key in (
                0x00790000, 0x0000FFC7, 0x01000039,
            ):
                break
            if low in (ord("c"), ord("C"), ord(" ")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                destination = output_dir / f"c920_calibration_{timestamp}.png"
                if not cv2.imwrite(str(destination), image):
                    raise OSError(f"failed to save image: {destination}")
                saved += 1
    finally:
        capture.release()
        cv2.destroyWindow(window)
    return saved


def mark_reference_points(
    image_path: Path,
    reference_points_path: Path,
    output_path: Path,
) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read reference image: {image_path}")
    references = _read_yaml(reference_points_path)
    points = references.get("points", []) if isinstance(references, dict) else []
    if len(points) < 6:
        raise ValueError("reference YAML needs at least six surveyed points")
    for point in points:
        xyz = np.asarray(point.get("base_xyz_m", ()), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("every reference point needs finite base_xyz_m[3]")
    clicked: list[tuple[int, int]] = []
    window = "Reference points - click in order, Backspace undo, Enter save, Q cancel"

    def mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONUP and len(clicked) < len(points):
            clicked.append((int(x), int(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            display = image.copy()
            for index, pixel in enumerate(clicked):
                cv2.circle(display, pixel, 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    display, str(index + 1), (pixel[0] + 7, pixel[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                )
            if len(clicked) < len(points):
                prompt = f"{len(clicked) + 1}/{len(points)}: {points[len(clicked)]['name']}"
            else:
                prompt = "All points marked - press Enter to save"
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, 38), (20, 20, 20), -1)
            cv2.putText(
                display, prompt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKeyEx(20)
            low = key & 0xFF
            if low in (ord("q"), ord("Q")):
                raise RuntimeError("reference-point marking cancelled")
            if low in (8, 127) and clicked:
                clicked.pop()
            if low in (10, 13) and len(clicked) == len(points):
                break
    finally:
        cv2.destroyWindow(window)
    output = {
        "format_version": 1,
        "coordinate_convention": "base x forward, y left, z up; metres",
        "samples": [{
            "image": str(image_path),
            "points": [
                {
                    "name": str(reference["name"]),
                    "base_xyz_m": [float(value) for value in reference["base_xyz_m"]],
                    "pixel_uv": [int(pixel[0]), int(pixel[1])],
                }
                for reference, pixel in zip(points, clicked)
            ],
        }],
    }
    _write_new_yaml(output_path, output)
    return len(clicked)


def _expand_images(patterns: Sequence[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern)))
    return list(dict.fromkeys(paths))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-camera C920 calibration and AR comparison"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--device", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--fps", type=float, default=20.0)

    intrinsics = commands.add_parser("intrinsics")
    intrinsics.add_argument("--images", nargs="+", required=True)
    intrinsics.add_argument("--inner-columns", type=int, default=9)
    intrinsics.add_argument("--inner-rows", type=int, default=6)
    intrinsics.add_argument("--square-size-m", type=float, default=0.030)
    intrinsics.add_argument("--min-images", type=int, default=20)
    intrinsics.add_argument("--preview-dir", type=Path)
    intrinsics.add_argument("--output", type=Path, required=True)

    extrinsics = commands.add_parser("extrinsics")
    extrinsics.add_argument("--intrinsics", type=Path, required=True)
    extrinsics.add_argument("--correspondences", type=Path, required=True)
    extrinsics.add_argument("--output", type=Path, required=True)

    mark = commands.add_parser("mark-points")
    mark.add_argument("--image", type=Path, required=True)
    mark.add_argument("--reference-points", type=Path, required=True)
    mark.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export")
    export.add_argument("--intrinsics", type=Path, required=True)
    export.add_argument("--extrinsics", type=Path)
    export.add_argument("--undistort-alpha", type=float, default=0.0)
    export.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--image", type=Path, required=True)
    compare.add_argument("--path-points", type=Path, required=True)
    compare.add_argument("--before-config", type=Path, required=True)
    compare.add_argument("--after-overlay", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "capture":
        count = capture_images(args.device, args.output_dir, args.fps)
        print(f"saved {count} calibration images")
    elif args.command == "intrinsics":
        paths = _expand_images(args.images)
        result = calibrate_intrinsics(
            paths,
            args.inner_columns,
            args.inner_rows,
            args.square_size_m,
            args.min_images,
            args.preview_dir,
        )
        _write_new_yaml(args.output, result)
        print(
            f"accepted {len(result['accepted_images'])} images; "
            f"RMS={result['rms_reprojection_error_px']:.4f}px"
        )
    elif args.command == "extrinsics":
        result = estimate_extrinsics(
            _read_yaml(args.correspondences), _read_yaml(args.intrinsics)
        )
        _write_new_yaml(args.output, result)
        print(yaml.safe_dump(result["aggregate"], sort_keys=False))
    elif args.command == "mark-points":
        count = mark_reference_points(
            args.image, args.reference_points, args.output
        )
        print(f"marked {count} surveyed reference points")
    elif args.command == "export":
        extrinsics = _read_yaml(args.extrinsics) if args.extrinsics else None
        result = export_runtime_overlay(
            _read_yaml(args.intrinsics), extrinsics, args.undistort_alpha
        )
        _write_new_yaml(args.output, result)
        print(f"wrote review-only overlay: {args.output}")
    elif args.command == "compare":
        compare_projection(
            args.image,
            args.path_points,
            args.before_config,
            args.after_overlay,
            args.output,
        )
        print(f"wrote comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
