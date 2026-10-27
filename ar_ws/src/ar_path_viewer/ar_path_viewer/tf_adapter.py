"""Small TF adapters. Pure matrix functions remain usable without ROS."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def quaternion_matrix_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-15 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        (
            (1.0 - yy - zz, xy - wz, xz + wy),
            (xy + wz, 1.0 - xx - zz, yz - wx),
            (xz - wy, yz + wx, 1.0 - xx - yy),
        ),
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray,
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Apply a target-from-source rigid transform to Nx3 source points."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points must be a finite Nx3 array")
    offset = np.asarray(translation, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("translation must be a finite 3-vector")
    return (quaternion_matrix_xyzw(quaternion_xyzw) @ array.T).T + offset


def transform_message_points(points: np.ndarray, transform_message) -> np.ndarray:
    transform = transform_message.transform
    return transform_points(
        points,
        (transform.translation.x, transform.translation.y, transform.translation.z),
        (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w),
    )
