"""Calibrated camera-image correction shared by runtime and offline tools."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .projection import CameraIntrinsics


SUPPORTED_DISTORTION_LENGTHS = {4, 5, 8, 12, 14}


def camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    intrinsics.validate()
    return np.array(
        (
            (intrinsics.fx, 0.0, intrinsics.cx),
            (0.0, intrinsics.fy, intrinsics.cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def validate_distortion(coefficients: Sequence[float]) -> np.ndarray:
    distortion = np.asarray(tuple(coefficients), dtype=np.float64).reshape(-1)
    if len(distortion) not in SUPPORTED_DISTORTION_LENGTHS:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_DISTORTION_LENGTHS))
        raise ValueError(f"distortion coefficient count must be one of: {allowed}")
    if not np.isfinite(distortion).all():
        raise ValueError("distortion coefficients must be finite")
    return distortion


@dataclass(frozen=True)
class UndistortDescription:
    source_intrinsics: CameraIntrinsics
    output_intrinsics: CameraIntrinsics
    distortion: tuple[float, ...]
    alpha: float
    valid_roi: tuple[int, int, int, int]


class CameraUndistorter:
    """Precomputed OpenCV remap that preserves the configured output size."""

    def __init__(
        self,
        source_intrinsics: CameraIntrinsics,
        distortion_coefficients: Sequence[float],
        alpha: float = 0.0,
    ) -> None:
        source_intrinsics.validate()
        distortion = validate_distortion(distortion_coefficients)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("camera undistort alpha must be between 0 and 1")
        size = (source_intrinsics.width, source_intrinsics.height)
        source_matrix = camera_matrix(source_intrinsics)
        output_matrix, roi = cv2.getOptimalNewCameraMatrix(
            source_matrix,
            distortion,
            size,
            float(alpha),
            size,
        )
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            source_matrix,
            distortion,
            None,
            output_matrix,
            size,
            cv2.CV_32FC1,
        )
        output_intrinsics = CameraIntrinsics(
            width=source_intrinsics.width,
            height=source_intrinsics.height,
            fx=float(output_matrix[0, 0]),
            fy=float(output_matrix[1, 1]),
            cx=float(output_matrix[0, 2]),
            cy=float(output_matrix[1, 2]),
        )
        output_intrinsics.validate()
        self.description = UndistortDescription(
            source_intrinsics,
            output_intrinsics,
            tuple(float(value) for value in distortion),
            float(alpha),
            tuple(int(value) for value in roi),
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        expected = (
            self.description.source_intrinsics.height,
            self.description.source_intrinsics.width,
        )
        if image.ndim != 3 or image.shape[:2] != expected:
            raise ValueError(
                f"undistortion requires a {expected[1]}x{expected[0]} color image"
            )
        return cv2.remap(
            image,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
