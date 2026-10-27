"""ROS-independent camera geometry and clipped path projection.

ROS body axes are x forward, y left, z up.  The intermediate camera body
frame uses the same convention.  OpenCV optical axes are x right, y down,
z forward, therefore ``(x_o, y_o, z_o) = (-y_c, -z_c, x_c)``.

``pitch_up`` is deliberately human-readable: positive values point the lens
upward.  This is the negative of a standard right-handed rotation about the
camera/body +y axis.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


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
    # x and y are explicit provisional values until measured in base_footprint.
    x: float = 0.20
    y: float = 0.0
    z: float = 0.6967
    yaw_deg: float = 0.0
    pitch_up_deg: float = 7.5
    roll_deg: float = 0.0

    def validate(self) -> None:
        if not np.isfinite((self.x, self.y, self.z, self.yaw_deg,
                            self.pitch_up_deg, self.roll_deg)).all():
            raise ValueError("camera mount values must be finite")


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


def _clip_segment_to_frustum(
    a: np.ndarray,
    b: np.ndarray,
    intrinsics: CameraIntrinsics,
    near: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Clip one optical-frame line segment against near plane and image frustum."""
    xmin = (0.0 - intrinsics.cx) / intrinsics.fx
    xmax = ((intrinsics.width - 1.0) - intrinsics.cx) / intrinsics.fx
    ymin = (0.0 - intrinsics.cy) / intrinsics.fy
    ymax = ((intrinsics.height - 1.0) - intrinsics.cy) / intrinsics.fy

    # Every plane is represented by dot(normal, p) + offset >= 0.
    planes = (
        (np.array((0.0, 0.0, 1.0)), -near),
        (np.array((1.0, 0.0, -xmin)), 0.0),
        (np.array((-1.0, 0.0, xmax)), 0.0),
        (np.array((0.0, 1.0, -ymin)), 0.0),
        (np.array((0.0, -1.0, ymax)), 0.0),
    )
    direction = b - a
    enter, leave = 0.0, 1.0
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
    return a + enter * direction, a + leave * direction


def project_clipped_segments(
    points_world: np.ndarray,
    robot_xy_yaw: Sequence[float],
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float = 0.10,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Project a world polyline, preserving segments that cross view boundaries."""
    intrinsics.validate()
    if near <= 0.0 or not np.isfinite(near):
        raise ValueError("near must be finite and positive")
    optical = world_to_optical(points_world, robot_xy_yaw, mount)
    result: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for index in range(max(0, len(optical) - 1)):
        clipped = _clip_segment_to_frustum(
            optical[index], optical[index + 1], intrinsics, near
        )
        if clipped is None:
            continue
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
    return result
