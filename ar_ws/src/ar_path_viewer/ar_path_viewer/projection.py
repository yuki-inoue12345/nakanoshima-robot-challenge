"""ROS-independent camera geometry and clipped path projection.

ROS body axes are x forward, y left, z up.  The intermediate camera body
frame uses the same convention.  OpenCV optical axes are x right, y down,
z forward, therefore ``(x_o, y_o, z_o) = (-y_c, -z_c, x_c)``.

``pitch_up`` is deliberately human-readable: positive values point the lens
upward.  This is the negative of a standard right-handed rotation about the
camera/body +y axis.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_CAMERA_Y_M = 0.094


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width < 2 or self.height < 2:
            raise ValueError("image dimensions must be at least 2x2")
        if self.fx <= 0.0 or self.fy <= 0.0 or not np.isfinite(values).all():
            raise ValueError("camera intrinsics must be finite with positive focal lengths")


@dataclass(frozen=True)
class CameraMount:
    # x remains provisional. y is measured from the upper-frame CAD centerline.
    x: float = 0.20
    y: float = DEFAULT_CAMERA_Y_M
    z: float = 0.6967
    yaw_deg: float = 0.0
    pitch_up_deg: float = 7.5
    roll_deg: float = 0.0

    def validate(self) -> None:
        if not np.isfinite((self.x, self.y, self.z, self.yaw_deg,
                            self.pitch_up_deg, self.roll_deg)).all():
            raise ValueError("camera mount values must be finite")


@dataclass(frozen=True)
class ProjectionStats:
    """Counts that explain how a transformed Path relates to the camera frustum."""

    transformed_points: int
    input_segments: int
    front_points: int
    image_points: int
    behind_points: int
    near_points: int
    left_points: int
    right_points: int
    above_points: int
    below_points: int
    drawable_segments: int
    path_length_m: float
    endpoint_distance_m: float | None
    camera_forward_path_length_m: float
    visible_path_length_m: float
    nonvisible_forward_path_length_m: float
    front_depth_min_m: float | None
    front_depth_max_m: float | None
    visible_depth_min_m: float | None
    visible_depth_max_m: float | None
    visible_pixel_length: float
    visible_width_px: int
    visible_height_px: int
    out_of_view_reason: str = ""

    def compact_detail(self) -> str:
        return (
            f"pts={self.transformed_points} len={self.path_length_m:.2f}m "
            f"end={self._format_distance(self.endpoint_distance_m)} "
            f"front={self.camera_forward_path_length_m:.2f}m "
            f"view={self.visible_path_length_m:.2f}m "
            f"seg={self.drawable_segments} px={self.visible_pixel_length:.0f}"
        )

    @staticmethod
    def _format_distance(value: float | None) -> str:
        return "none" if value is None else f"{value:.2f}m"

    def log_detail(self) -> str:
        depth_min = (
            "none" if self.front_depth_min_m is None else f"{self.front_depth_min_m:.3f}"
        )
        depth_max = (
            "none" if self.front_depth_max_m is None else f"{self.front_depth_max_m:.3f}"
        )
        visible_depth_min = (
            "none"
            if self.visible_depth_min_m is None
            else f"{self.visible_depth_min_m:.3f}"
        )
        visible_depth_max = (
            "none"
            if self.visible_depth_max_m is None
            else f"{self.visible_depth_max_m:.3f}"
        )
        endpoint_distance = (
            "none" if self.endpoint_distance_m is None else f"{self.endpoint_distance_m:.3f}"
        )
        return (
            f"transformed_points={self.transformed_points} "
            f"input_segments={self.input_segments} front_points={self.front_points} "
            f"image_points={self.image_points} behind_points={self.behind_points} "
            f"near_points={self.near_points} left_points={self.left_points} "
            f"right_points={self.right_points} above_points={self.above_points} "
            f"below_points={self.below_points} "
            f"drawable_segments={self.drawable_segments} "
            f"path_length_m={self.path_length_m:.3f} "
            f"endpoint_distance_m={endpoint_distance} "
            f"camera_forward_path_length_m={self.camera_forward_path_length_m:.3f} "
            f"visible_path_length_m={self.visible_path_length_m:.3f} "
            f"nonvisible_forward_path_length_m="
            f"{self.nonvisible_forward_path_length_m:.3f} "
            f"front_depth_min_m={depth_min} "
            f"front_depth_max_m={depth_max} "
            f"visible_depth_min_m={visible_depth_min} "
            f"visible_depth_max_m={visible_depth_max} "
            f"visible_pixel_length={self.visible_pixel_length:.1f} "
            f"visible_width_px={self.visible_width_px} "
            f"visible_height_px={self.visible_height_px}"
        )


@dataclass(frozen=True)
class ProjectionResult:
    segments: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]
    stats: ProjectionStats


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def world_to_optical(
    points_world: np.ndarray,
    robot_xy_yaw: Sequence[float],
    mount: CameraMount,
) -> np.ndarray:
    """Transform Nx3 ground/world points into the OpenCV optical frame."""
    mount.validate()
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points_world must be finite")
    rx, ry, yaw = (float(v) for v in robot_xy_yaw)
    if not np.isfinite((rx, ry, yaw)).all():
        raise ValueError("robot pose must be finite")

    # World -> robot body. Robot is assumed level on the z=0 ground plane.
    p_body = (_rot_z(-yaw) @ (points - np.array((rx, ry, 0.0))).T).T
    camera_origin_body = np.array((mount.x, mount.y, mount.z))

    # Camera body -> robot body. Positive pitch_up requires standard Ry(-pitch).
    yaw_m = np.deg2rad(mount.yaw_deg)
    pitch_m = np.deg2rad(-mount.pitch_up_deg)
    roll_m = np.deg2rad(mount.roll_deg)
    rotation_body_from_camera = _rot_z(yaw_m) @ _rot_y(pitch_m) @ _rot_x(roll_m)
    p_camera_body = (rotation_body_from_camera.T @ (p_body - camera_origin_body).T).T

    # camera-body (forward, left, up) -> optical (right, down, forward)
    return np.column_stack(
        (-p_camera_body[:, 1], -p_camera_body[:, 2], p_camera_body[:, 0])
    )


def _frustum_planes(
    intrinsics: CameraIntrinsics, near: float
) -> Tuple[Tuple[np.ndarray, float], ...]:
    xmin = (0.0 - intrinsics.cx) / intrinsics.fx
    xmax = ((intrinsics.width - 1.0) - intrinsics.cx) / intrinsics.fx
    ymin = (0.0 - intrinsics.cy) / intrinsics.fy
    ymax = ((intrinsics.height - 1.0) - intrinsics.cy) / intrinsics.fy
    return (
        (np.array((0.0, 0.0, 1.0)), -near),
        (np.array((1.0, 0.0, -xmin)), 0.0),
        (np.array((-1.0, 0.0, xmax)), 0.0),
        (np.array((0.0, 1.0, -ymin)), 0.0),
        (np.array((0.0, -1.0, ymax)), 0.0),
    )


def _clip_segment_parameters(
    a: np.ndarray,
    b: np.ndarray,
    planes: Sequence[Tuple[np.ndarray, float]],
) -> Tuple[float, float] | None:
    """Return the original segment fractions inside all supplied half spaces."""
    direction = b - a
    enter, leave = 0.0, 1.0
    # Every plane is represented by dot(normal, p) + offset >= 0.
    for normal, offset in planes:
        f0 = float(np.dot(normal, a) + offset)
        fd = float(np.dot(normal, direction))
        if abs(fd) < 1e-12:
            if f0 < 0.0:
                return None
            continue
        crossing = -f0 / fd
        if fd > 0.0:
            enter = max(enter, crossing)
        else:
            leave = min(leave, crossing)
        if enter > leave:
            return None
    return enter, leave


def project_path(
    points_world: np.ndarray,
    robot_xy_yaw: Sequence[float],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float = 0.10,
) -> ProjectionResult:
    """Project a polyline and report why points or segments are outside the view."""
    intrinsics.validate()
    if near <= 0.0 or not np.isfinite(near):
        raise ValueError("near must be finite and positive")
    optical = world_to_optical(points_world, robot_xy_yaw, mount)
    depth = optical[:, 2]
    behind = depth <= 0.0
    near_only = (depth > 0.0) & (depth < near)
    front = depth >= near

    xmin = (0.0 - intrinsics.cx) / intrinsics.fx
    xmax = ((intrinsics.width - 1.0) - intrinsics.cx) / intrinsics.fx
    ymin = (0.0 - intrinsics.cy) / intrinsics.fy
    ymax = ((intrinsics.height - 1.0) - intrinsics.cy) / intrinsics.fy
    left = front & (optical[:, 0] < xmin * depth)
    right = front & (optical[:, 0] > xmax * depth)
    above = front & (optical[:, 1] < ymin * depth)
    below = front & (optical[:, 1] > ymax * depth)
    inside = front & ~(left | right | above | below)

    result: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    camera_forward_path_length_m = 0.0
    visible_path_length_m = 0.0
    visible_depths: List[float] = []
    forward_plane = ((np.array((0.0, 0.0, 1.0)), 0.0),)
    frustum_planes = _frustum_planes(intrinsics, near)
    for index in range(max(0, len(optical) - 1)):
        segment_length_m = float(np.linalg.norm(optical[index + 1] - optical[index]))
        forward_parameters = _clip_segment_parameters(
            optical[index], optical[index + 1], forward_plane
        )
        if forward_parameters is not None:
            camera_forward_path_length_m += (
                segment_length_m * (forward_parameters[1] - forward_parameters[0])
            )
        visible_parameters = _clip_segment_parameters(
            optical[index], optical[index + 1], frustum_planes
        )
        if visible_parameters is None:
            continue
        enter, leave = visible_parameters
        direction = optical[index + 1] - optical[index]
        clipped = (
            optical[index] + enter * direction,
            optical[index] + leave * direction,
        )
        visible_path_length_m += segment_length_m * (leave - enter)
        visible_depths.extend((float(clipped[0][2]), float(clipped[1][2])))
        pixels = []
        for point in clipped:
            u = intrinsics.fx * point[0] / point[2] + intrinsics.cx
            v = intrinsics.fy * point[1] / point[2] + intrinsics.cy
            pixels.append(
                (
                    int(np.clip(np.rint(u), 0, intrinsics.width - 1)),
                    int(np.clip(np.rint(v), 0, intrinsics.height - 1)),
                )
            )
        result.append((pixels[0], pixels[1]))

    point_count = len(optical)
    front_count = int(np.count_nonzero(front))
    path_length_m = (
        float(
            np.linalg.norm(
                np.diff(np.asarray(points_world, dtype=float), axis=0), axis=1
            ).sum()
        )
        if point_count >= 2
        else 0.0
    )
    if point_count:
        robot_origin = np.array(
            (float(robot_xy_yaw[0]), float(robot_xy_yaw[1]), 0.0), dtype=float
        )
        endpoint_distance_m = float(
            np.linalg.norm(np.asarray(points_world, dtype=float)[-1] - robot_origin)
        )
    else:
        endpoint_distance_m = None
    front_depths = depth[front]
    front_depth_min = float(front_depths.min()) if front_count else None
    front_depth_max = float(front_depths.max()) if front_count else None
    visible_depth_min = min(visible_depths) if visible_depths else None
    visible_depth_max = max(visible_depths) if visible_depths else None
    nonvisible_forward_path_length_m = max(
        0.0, camera_forward_path_length_m - visible_path_length_m
    )
    visible_pixel_length = float(sum(
        np.hypot(end[0] - start[0], end[1] - start[1]) for start, end in result
    ))
    if result:
        visible_pixels = np.array(
            [point for segment in result for point in segment], dtype=int
        )
        visible_width = int(visible_pixels[:, 0].max() - visible_pixels[:, 0].min() + 1)
        visible_height = int(visible_pixels[:, 1].max() - visible_pixels[:, 1].min() + 1)
    else:
        visible_width = 0
        visible_height = 0
    reason = ""
    if not result:
        if point_count < 2:
            reason = "fewer than two transformed Path points"
        elif front_count == 0:
            if int(np.count_nonzero(behind)) == point_count:
                reason = "all Path points are behind the camera"
            else:
                reason = "all Path points are behind or inside the near clip plane"
        elif int(np.count_nonzero(below)) == front_count:
            reason = "all forward Path points are below the image"
        elif int(np.count_nonzero(above)) == front_count:
            reason = "all forward Path points are above the image"
        elif int(np.count_nonzero(left)) == front_count:
            reason = "all forward Path points are left of the image"
        elif int(np.count_nonzero(right)) == front_count:
            reason = "all forward Path points are right of the image"
        else:
            reason = "Path segments do not intersect the image frustum"

    stats = ProjectionStats(
        transformed_points=point_count,
        input_segments=max(0, point_count - 1),
        front_points=front_count,
        image_points=int(np.count_nonzero(inside)),
        behind_points=int(np.count_nonzero(behind)),
        near_points=int(np.count_nonzero(near_only)),
        left_points=int(np.count_nonzero(left)),
        right_points=int(np.count_nonzero(right)),
        above_points=int(np.count_nonzero(above)),
        below_points=int(np.count_nonzero(below)),
        drawable_segments=len(result),
        path_length_m=path_length_m,
        endpoint_distance_m=endpoint_distance_m,
        camera_forward_path_length_m=camera_forward_path_length_m,
        visible_path_length_m=visible_path_length_m,
        nonvisible_forward_path_length_m=nonvisible_forward_path_length_m,
        front_depth_min_m=front_depth_min,
        front_depth_max_m=front_depth_max,
        visible_depth_min_m=visible_depth_min,
        visible_depth_max_m=visible_depth_max,
        visible_pixel_length=visible_pixel_length,
        visible_width_px=visible_width,
        visible_height_px=visible_height,
        out_of_view_reason=reason,
    )
    return ProjectionResult(tuple(result), stats)


def project_clipped_segments(
    points_world: np.ndarray,
    robot_xy_yaw: Sequence[float],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float = 0.10,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Project a polyline, preserving segments that cross view boundaries."""
    return list(project_path(points_world, robot_xy_yaw, mount, intrinsics, near).segments)
