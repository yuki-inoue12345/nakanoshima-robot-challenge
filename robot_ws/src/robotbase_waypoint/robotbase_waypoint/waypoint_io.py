"""Read and write the waypoint YAML format inherited from Sirius."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class Waypoint:
    """A planar navigation waypoint."""

    number: int
    x: float
    y: float
    angle_radians: float
    threshold: float = -1.0
    stop: bool = False
    wait_time: float = 0.0
    person_area: bool = False


def _as_waypoint(raw: Dict[str, Any], index: int) -> Waypoint:
    if not isinstance(raw, dict):
        raise ValueError(f'waypoints[{index}] must be a mapping')
    missing = [key for key in ('x', 'y', 'angle_radians') if key not in raw]
    if missing:
        raise ValueError(f'waypoints[{index}] is missing: {", ".join(missing)}')
    unsupported = [
        key for key in ('change_map', 'rotate') if raw.get(key) not in (None, '', 0, 0.0)
    ]
    if unsupported:
        raise ValueError(
            f'waypoints[{index}] uses unsupported fields: '
            f'{", ".join(unsupported)}')
    person_area = raw.get('person_area', False)
    if not isinstance(person_area, bool):
        raise ValueError(f'waypoints[{index}].person_area must be true or false')
    stop = raw.get('stop', False)
    if not isinstance(stop, bool):
        raise ValueError(f'waypoints[{index}].stop must be true or false')
    values = {}
    for key, default in (
            ('x', None), ('y', None), ('angle_radians', None),
            ('threshold', -1.0), ('wait_time', 0.0)):
        value = raw.get(key, default)
        try:
            if isinstance(value, bool):
                raise ValueError('boolean is not a numeric waypoint value')
            values[key] = float(value)
            if not math.isfinite(values[key]):
                raise ValueError('must be finite')
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'waypoints[{index}].{key}: {error}') from error
    number = raw.get('number', index + 1)
    try:
        if isinstance(number, bool) or (isinstance(number, float) and (
                not math.isfinite(number) or not number.is_integer())):
            raise ValueError('must be an integer')
        number = int(number)
        waypoint = Waypoint(
            number=number,
            **values,
            stop=stop,
            person_area=person_area,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'waypoints[{index}].number: {error}') from error
    if waypoint.wait_time < 0.0:
        raise ValueError(f'waypoints[{index}].wait_time must be non-negative')
    return waypoint


def load_waypoints(file_path: str) -> List[Waypoint]:
    """Load and validate a waypoint file."""
    path = Path(os.path.expanduser(file_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Waypoint file not found: {path}')
    with path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get('waypoints'), list):
        raise ValueError("Waypoint YAML must contain a 'waypoints' list")
    waypoints = [_as_waypoint(raw, index) for index, raw in enumerate(data['waypoints'])]
    if not waypoints:
        raise ValueError('Waypoint list is empty')
    return waypoints


def waypoint_to_dict(waypoint: Waypoint) -> Dict[str, Any]:
    """Convert a waypoint while omitting unused optional fields."""
    result = asdict(waypoint)
    if result['threshold'] < 0.0:
        result.pop('threshold')
    if not result['stop']:
        result.pop('stop')
    if result['wait_time'] <= 0.0:
        result.pop('wait_time')
    if not result['person_area']:
        result.pop('person_area')
    return result


def write_waypoints(file_path: str, waypoints: Iterable[Waypoint]) -> None:
    """Atomically write waypoints in the common version 1.0 format."""
    path = Path(os.path.expanduser(file_path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': '1.0',
        'waypoints': [waypoint_to_dict(waypoint) for waypoint in waypoints],
    }
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    os.replace(temporary, path)


def append_waypoint(
        file_path: str, x: float, y: float, yaw: float, *,
        stop: bool = False, wait_time: float = 0.0) -> Waypoint:
    """Append one pose and continue numbering from the existing file."""
    path = Path(os.path.expanduser(file_path)).resolve()
    if path.exists():
        existing = load_waypoints(str(path))
    else:
        existing = []
    number = existing[-1].number + 1 if existing else 1
    waypoint = Waypoint(
        number=number,
        x=float(x),
        y=float(y),
        angle_radians=float(yaw),
        stop=bool(stop),
        wait_time=float(wait_time),
    )
    write_waypoints(str(path), [*existing, waypoint])
    return waypoint
