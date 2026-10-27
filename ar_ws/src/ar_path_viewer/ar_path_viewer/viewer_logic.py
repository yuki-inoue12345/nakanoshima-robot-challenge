"""ROS-independent orchestration for path transform and projection states."""

from typing import Callable, List, Tuple

import numpy as np

from .projection import CameraIntrinsics, CameraMount, project_clipped_segments


def project_path_state(
    points: np.ndarray,
    path_frame: str,
    base_frame: str,
    transform_to_base: Callable[[np.ndarray, str], np.ndarray],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
) -> Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    if path_frame == base_frame:
        points_base = points
    else:
        try:
            points_base = transform_to_base(points, path_frame)
        except Exception:
            return "TF UNAVAILABLE", []
    segments = project_clipped_segments(
        points_base, (0.0, 0.0, 0.0), mount, intrinsics, near
    )
    if not segments:
        return "PATH OUT OF VIEW", []
    return "ACTIVE", segments
