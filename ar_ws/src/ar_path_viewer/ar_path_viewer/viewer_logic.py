"""ROS-independent orchestration for path transform and projection states."""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .offscreen_guidance import (
    GuidanceDecision,
    GuidanceSettings,
    OffscreenGuidance,
    decide_offscreen_guidance,
)
from .projection import (
    CameraIntrinsics,
    CameraMount,
    ProjectionStats,
    project_clipped_segments as project_clipped_segments,
    project_path,
)


@dataclass(frozen=True)
class PathProjectionState:
    state: str
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    stats: ProjectionStats | None
    guidance: OffscreenGuidance | None
    guidance_detail: str = ""


def evaluate_path_state(
    points: np.ndarray,
    path_frame: str,
    base_frame: str,
    transform_to_base: Callable[[np.ndarray, str], np.ndarray],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
    guidance_settings: GuidanceSettings = GuidanceSettings(enabled=False),
    on_tf_error: Callable[[Exception], None] | None = None,
) -> PathProjectionState:
    if path_frame == base_frame:
        points_base = points
    else:
        try:
            points_base = transform_to_base(points, path_frame)
        except Exception as exc:
            if on_tf_error is not None:
                on_tf_error(exc)
            return PathProjectionState("TF UNAVAILABLE", [], None, None)
    projection = project_path(
        points_base, (0.0, 0.0, 0.0), mount, intrinsics, near
    )
    segments = list(projection.segments)
    if segments:
        return PathProjectionState("ACTIVE", segments, projection.stats, None)
    decision: GuidanceDecision = decide_offscreen_guidance(
        points_base, guidance_settings
    )
    return PathProjectionState(
        "PATH OUT OF VIEW",
        [],
        projection.stats,
        decision.guidance,
        decision.detail(),
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
    result = evaluate_path_state(
        points,
        path_frame,
        base_frame,
        transform_to_base,
        mount,
        intrinsics,
        near,
        on_tf_error=on_tf_error,
    )
    return result.state, result.segments, result.stats


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
