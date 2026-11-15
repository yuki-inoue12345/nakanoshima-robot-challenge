"""ROS-independent orchestration for path transform and projection states."""

from dataclasses import dataclass, replace
from typing import Callable, List, Tuple

import numpy as np

from .offscreen_guidance import (
    GuidanceDecision,
    GuidanceSettings,
    OffscreenGuidance,
    PathProgress,
    decide_offscreen_guidance,
    measure_path_progress,
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
    progress: PathProgress | None = None
    points_base: np.ndarray | None = None


@dataclass(frozen=True)
class PathDisplayGeometry:
    """Display-only Z geometry that never changes the Nav2 Path XY shape.

    Nav2 global paths normally lie on the ground plane.  With the fixed,
    upward-facing C920, a short ground path can be physically below the image.
    ``elevated`` preserves every transformed X/Y coordinate and only places the
    rendered polyline on a constant virtual height.  Guidance decisions still
    use the unmodified path.
    """

    mode: str = "ground"
    height_m: float = 0.5

    def validate(self) -> None:
        if self.mode not in ("ground", "elevated"):
            raise ValueError("path_display_mode must be 'ground' or 'elevated'")
        if not np.isfinite(self.height_m) or not 0.0 <= self.height_m <= 2.0:
            raise ValueError("path_display_height_m must be finite and in [0, 2]")

    @property
    def label(self) -> str:
        if self.mode == "elevated":
            return f"AR FLOAT {self.height_m:.2f}m"
        return "AR GROUND"

    def apply(self, points_base: np.ndarray) -> np.ndarray:
        self.validate()
        rendered = np.asarray(points_base, dtype=float).copy()
        if rendered.ndim != 2 or rendered.shape[1] != 3:
            raise ValueError("path points must be an Nx3 array")
        rendered[:, 2] = self.height_m if self.mode == "elevated" else 0.0
        return rendered


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
    display_geometry: PathDisplayGeometry = PathDisplayGeometry(),
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
    rendered_points = display_geometry.apply(points_base)
    projection = project_path(
        rendered_points, (0.0, 0.0, 0.0), mount, intrinsics, near
    )
    # Keep the route metrics truthful: only visibility/depth fields describe
    # the display geometry. Path length and endpoint distance remain those of
    # the transformed Nav2 Path, not the virtual height used for rendering.
    path_length_m = float(
        np.linalg.norm(np.diff(points_base, axis=0), axis=1).sum()
    ) if len(points_base) >= 2 else 0.0
    endpoint_distance_m = (
        float(np.linalg.norm(points_base[-1])) if len(points_base) else None
    )
    stats = replace(
        projection.stats,
        path_length_m=path_length_m,
        endpoint_distance_m=endpoint_distance_m,
    )
    segments = list(projection.segments)
    progress = measure_path_progress(points_base)
    decision: GuidanceDecision = decide_offscreen_guidance(
        points_base, guidance_settings
    )
    if segments:
        return PathProjectionState(
            "ACTIVE", segments, stats, decision.guidance,
            decision.detail(), progress, points_base,
        )
    return PathProjectionState(
        "PATH OUT OF VIEW",
        [],
        stats,
        decision.guidance,
        decision.detail(),
        progress,
        points_base,
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
    display_geometry: PathDisplayGeometry = PathDisplayGeometry(),
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
        mount, intrinsics, near,
        on_tf_error=on_tf_error,
        display_geometry=display_geometry,
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
    display_geometry: PathDisplayGeometry = PathDisplayGeometry(),
) -> Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    state, segments, _ = project_path_state_detailed(
        points, path_frame, base_frame, transform_to_base,
        mount, intrinsics, near, on_tf_error,
        display_geometry=display_geometry,
    )
    return state, segments
