"""ROS-message-shaped runtime gate for selected map and Keepout readiness."""

from __future__ import annotations

import hashlib
import math

from .keepout_config import MapSignature


def _yaw_from_quaternion(quaternion) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        return float('nan')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def occupancy_data_sha256(data) -> str:
    """Hash signed int8 OccupancyGrid data in a transport-independent form."""
    return hashlib.sha256(bytes(int(value) & 0xff for value in data)).hexdigest()


class KeepoutReadinessGate:
    """Require exact map/mask content and active shared filter servers."""

    def __init__(
        self,
        map_signature: MapSignature,
        *,
        keepout_required: bool,
        mask_signature: MapSignature | None,
        filter_info_topic: str,
        mask_topic: str,
    ) -> None:
        if keepout_required and mask_signature is None:
            raise ValueError('required Keepout needs an expected mask signature')
        self.map_signature = map_signature
        self.mask_signature = mask_signature
        self.keepout_required = bool(keepout_required)
        self.filter_info_topic = str(filter_info_topic)
        self.mask_topic = str(mask_topic)
        self.map_valid = False
        self.mask_valid = not self.keepout_required
        self.info_valid = not self.keepout_required
        self.mask_server_active = not self.keepout_required
        self.info_server_active = not self.keepout_required
        self.map_error = 'map_not_received'
        self.mask_error = 'mask_not_received' if self.keepout_required else ''
        self.info_error = 'filter_info_not_received' if self.keepout_required else ''

    def _geometry_error(self, message, expected: MapSignature) -> str:
        info = message.info
        origin = info.origin
        yaw = _yaw_from_quaternion(origin.orientation)
        actual = (
            int(info.width), int(info.height), float(info.resolution),
            float(origin.position.x), float(origin.position.y), yaw,
        )
        wanted = (
            expected.width, expected.height, expected.resolution,
            expected.origin_x, expected.origin_y, expected.origin_yaw,
        )
        if not all(math.isfinite(value) for value in actual[2:]):
            return 'nonfinite_geometry'
        if actual[:2] != wanted[:2] or any(
            not math.isclose(a, b, abs_tol=1e-6)
            for a, b in zip(actual[2:], wanted[2:])
        ):
            return f'geometry_mismatch actual={actual} expected={wanted}'
        return ''

    def observe_map(self, message) -> None:
        error = self._geometry_error(message, self.map_signature)
        if not error and occupancy_data_sha256(message.data) != self.map_signature.data_sha256:
            error = 'map_content_mismatch'
        self.map_valid = not error
        self.map_error = error

    def observe_mask(self, message) -> None:
        if not self.keepout_required:
            return
        error = self._geometry_error(message, self.mask_signature)
        if not error and occupancy_data_sha256(message.data) != self.mask_signature.data_sha256:
            error = 'mask_content_mismatch'
        self.mask_valid = not error
        self.mask_error = error

    def observe_filter_info(self, message) -> None:
        if not self.keepout_required:
            return
        error = ''
        if int(message.type) != 0:
            error = f'filter_type={message.type} expected=0'
        elif str(message.filter_mask_topic) != self.mask_topic:
            error = (
                f'mask_topic={message.filter_mask_topic!r} '
                f'expected={self.mask_topic!r}'
            )
        elif not math.isclose(float(message.base), 0.0, abs_tol=1e-6):
            error = f'filter_base={message.base} expected=0'
        elif not math.isclose(float(message.multiplier), 1.0, abs_tol=1e-6):
            error = f'filter_multiplier={message.multiplier} expected=1'
        self.info_valid = not error
        self.info_error = error

    def set_lifecycle(self, server: str, active: bool) -> None:
        if server == 'mask':
            self.mask_server_active = bool(active)
        elif server == 'info':
            self.info_server_active = bool(active)
        else:
            raise ValueError(f'unknown Keepout server: {server}')

    @property
    def ready(self) -> bool:
        return (
            self.map_valid
            and self.mask_valid
            and self.info_valid
            and self.mask_server_active
            and self.info_server_active
        )

    @property
    def reason(self) -> str:
        if not self.map_valid:
            return self.map_error or 'map_not_ready'
        if not self.keepout_required:
            return 'keepout_disabled_map_verified'
        if not self.mask_valid:
            return self.mask_error or 'mask_not_ready'
        if not self.info_valid:
            return self.info_error or 'filter_info_not_ready'
        if not self.mask_server_active:
            return 'filter_mask_server_not_active'
        if not self.info_server_active:
            return 'costmap_filter_info_server_not_active'
        return 'ready'
