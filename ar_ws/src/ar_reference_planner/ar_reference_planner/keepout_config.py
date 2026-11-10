"""
Fail-closed selected-map and Keepout configuration validation.

The robot-control launch remains the owner of the root map and Keepout filter
servers.  This module verifies the same on-disk selection before the isolated
reference costmap is allowed to consume those shared, read-only topics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from PIL import Image
import yaml


class KeepoutConfigurationError(ValueError):
    """Selected map or saved Keepout data is unsafe to use."""


@dataclass(frozen=True)
class MapSignature:
    """Map geometry and exact OccupancyGrid data signature."""

    yaml_path: Path
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    data_sha256: str

    def metadata(self) -> dict:
        """Return the format stored by the existing waypoint editor."""
        return {
            'resolution': self.resolution,
            'origin': [self.origin_x, self.origin_y, self.origin_yaw],
            'size': [self.width, self.height],
        }


@dataclass(frozen=True)
class KeepoutSelection:
    """Validated Keepout state belonging to one selected map."""

    map_signature: MapSignature
    enabled: bool
    reason: str
    settings_path: Path | None = None
    mask_signature: MapSignature | None = None


def _finite_float(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise KeepoutConfigurationError(f'{name} must be numeric') from error
    if not math.isfinite(number):
        raise KeepoutConfigurationError(f'{name} must be finite')
    return number


def _map_data_sha256(image: Image.Image, metadata: dict) -> str:
    mode = str(metadata.get('mode', 'trinary')).lower()
    if mode != 'trinary':
        raise KeepoutConfigurationError(
            f'unsupported map mode {mode!r}; only trinary is verified'
        )
    if image.mode in ('RGBA', 'LA'):
        alpha = image.getchannel('A')
        if alpha.getextrema() != (255, 255):
            raise KeepoutConfigurationError(
                'map image contains transparency that cannot be verified safely'
            )
    gray = image.convert('L')
    width, height = gray.size
    pixels = gray.load()
    negate = int(metadata.get('negate', 0))
    if negate not in (0, 1):
        raise KeepoutConfigurationError('map negate must be 0 or 1')
    occupied = _finite_float(metadata.get('occupied_thresh', 0.65), 'occupied_thresh')
    free = _finite_float(metadata.get('free_thresh', 0.196), 'free_thresh')
    if not 0.0 <= free < occupied <= 1.0:
        raise KeepoutConfigurationError(
            'map thresholds must satisfy 0 <= free < occupied <= 1'
        )
    output = bytearray()
    # nav_msgs/OccupancyGrid starts at the map's lower-left while image files
    # start at the upper-left, so rows are reversed like nav2_map_server.
    for row in range(height - 1, -1, -1):
        for column in range(width):
            shade = pixels[column, row] / 255.0
            occupancy = shade if negate else 1.0 - shade
            if occupancy > occupied:
                value = 100
            elif occupancy < free:
                value = 0
            else:
                value = -1
            output.append(value & 0xff)
    return hashlib.sha256(output).hexdigest()


def load_map_signature(yaml_path: str | Path) -> MapSignature:
    """Load and hash one trinary Nav2 map without starting ROS."""
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise KeepoutConfigurationError(f'map YAML is missing: {path}')
    try:
        metadata = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as error:
        raise KeepoutConfigurationError(f'cannot read map YAML {path}: {error}') from error
    if not isinstance(metadata, dict) or not metadata.get('image'):
        raise KeepoutConfigurationError(f'map YAML has no image: {path}')
    image_path = (path.parent / str(metadata['image'])).resolve()
    if not image_path.is_file():
        raise KeepoutConfigurationError(f'map image is missing: {image_path}')
    origin = metadata.get('origin')
    if not isinstance(origin, list) or len(origin) != 3:
        raise KeepoutConfigurationError(f'map origin must contain x, y, yaw: {path}')
    resolution = _finite_float(metadata.get('resolution'), 'resolution')
    if resolution <= 0.0:
        raise KeepoutConfigurationError('map resolution must be positive')
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            data_hash = _map_data_sha256(image, metadata)
    except (OSError, ValueError) as error:
        raise KeepoutConfigurationError(
            f'cannot verify map image {image_path}: {error}'
        ) from error
    return MapSignature(
        yaml_path=path,
        resolution=resolution,
        width=int(width),
        height=int(height),
        origin_x=_finite_float(origin[0], 'origin x'),
        origin_y=_finite_float(origin[1], 'origin y'),
        origin_yaw=_finite_float(origin[2], 'origin yaw'),
        data_sha256=data_hash,
    )


def _metadata_matches(actual: object, expected: dict) -> bool:
    if not isinstance(actual, dict):
        return False
    try:
        resolution = float(actual['resolution'])
        origin = [float(value) for value in actual['origin']]
        size = [int(value) for value in actual['size']]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isclose(resolution, expected['resolution'], abs_tol=1e-12)
        and len(origin) == 3
        and all(math.isclose(a, b, abs_tol=1e-12)
                for a, b in zip(origin, expected['origin']))
        and size == expected['size']
    )


def _source_map_matches(source: Path, selected: Path) -> bool:
    if source.expanduser().resolve() == selected:
        return True
    # The NUC and development clone may have different HOME prefixes while
    # retaining the repository's maps_waypoints/maps suffix.
    expected_suffix = ('maps_waypoints', 'maps', selected.name)
    return (
        tuple(source.parts[-3:]) == expected_suffix
        and tuple(selected.parts[-3:]) == expected_suffix
    )


def resolve_keepout_selection(map_yaml: str | Path) -> KeepoutSelection:
    """Resolve Keepout for one selected map; enabled errors fail closed."""
    selected = load_map_signature(map_yaml)
    state_path = (
        selected.yaml_path.parent / '.keepout' /
        selected.yaml_path.stem / 'settings.json'
    )
    if not state_path.exists():
        return KeepoutSelection(selected, False, 'settings_absent')
    import json
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise KeepoutConfigurationError(
            f'cannot read Keepout settings {state_path}: {error}'
        ) from error
    if not isinstance(state, dict):
        raise KeepoutConfigurationError('Keepout settings must be a JSON object')
    if not state.get('enabled', False):
        return KeepoutSelection(selected, False, 'settings_disabled', state_path)
    source_text = str(state.get('source_map', '')).strip()
    if not source_text or not _source_map_matches(Path(source_text), selected.yaml_path):
        raise KeepoutConfigurationError(
            'Keepout source_map does not match the selected map'
        )
    if not _metadata_matches(state.get('map_metadata'), selected.metadata()):
        raise KeepoutConfigurationError(
            'Keepout map_metadata does not match the selected map geometry'
        )
    mask_path = state_path.parent / 'mask.yaml'
    mask = load_map_signature(mask_path)
    if not _metadata_matches(mask.metadata(), selected.metadata()):
        raise KeepoutConfigurationError(
            'Keepout mask geometry does not match the selected map'
        )
    return KeepoutSelection(
        selected, True, 'enabled_and_verified', state_path, mask
    )


def params_with_keepout(
    params: dict, filter_info_topic: str
) -> dict:
    """Return a copy with the reference global costmap Keepout plugin enabled."""
    configured = deepcopy(params)
    try:
        block = configured['global_costmap']['global_costmap']['ros__parameters']
    except (KeyError, TypeError) as error:
        raise KeepoutConfigurationError(
            'reference params have no global_costmap parameters'
        ) from error
    filters = block.setdefault('filters', [])
    if not isinstance(filters, list):
        raise KeepoutConfigurationError('global_costmap filters must be a list')
    if 'keepout_filter' not in filters:
        filters.append('keepout_filter')
    block['keepout_filter'] = {
        'plugin': 'nav2_costmap_2d::KeepoutFilter',
        'enabled': True,
        'filter_info_topic': str(filter_info_topic),
    }
    return configured
