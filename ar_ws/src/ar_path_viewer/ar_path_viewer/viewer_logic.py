"""ROS-independent orchestration for path transform and projection states."""

from typing import Callable, List, Tuple

import numpy as np

from .projection import (
    CameraIntrinsics,
    CameraMount,
    ProjectionStats,
    project_clipped_segments,
    project_path,
)


def project_path_state_detailed(
    points: np.ndarray,
    path_frame: str,
    base_frame: str,
    transform_to_base: Callable[[np.ndarray, str], np.ndarray],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
    on_tf_error: Callable[[Exception], None] | None = None,
) -> Tuple[
    str,
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    ProjectionStats | None,
]:
    if path_frame == base_frame:
        points_base = points
    else:
        try:
            points_base = transform_to_base(points, path_frame)
        except Exception as exc:
            if on_tf_error is not None:
                on_tf_error(exc)
            return "TF UNAVAILABLE", [], None
    projection = project_path(
        points_base, (0.0, 0.0, 0.0), mount, intrinsics, near
    )
    segments = list(projection.segments)
    if not segments:
        return "PATH OUT OF VIEW", [], projection.stats
    return "ACTIVE", segments, projection.stats


def project_path_state(
    points: np.ndarray,
    path_frame: str,
    base_frame: str,
    transform_to_base: Callable[[np.ndarray, str], np.ndarray],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
    on_tf_error: Callable[[Exception], None] | None = None,
) -> Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    state, segments, _ = project_path_state_detailed(
        points, path_frame, base_frame, transform_to_base,
        mount, intrinsics, near, on_tf_error,
    )
    return state, segments
