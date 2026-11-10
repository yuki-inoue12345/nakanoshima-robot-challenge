"""Load an ordered waypoint horizon without fabricating a geometric path."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import yaml


@dataclass(frozen=True)
class Waypoint:
    """One validated goal from the existing robotbase waypoint format."""

    number: int
    x: float
    y: float
    yaw: float
    stop: bool = False
    wait_time: float = 0.0
    person_area: bool = False

    @property
    def task_boundary(self) -> bool:
        """Return whether planning must not proceed beyond this waypoint."""
        return self.stop or self.wait_time > 0.0 or self.person_area


@dataclass(frozen=True)
class Route:
    """A route and a stable identity of its exact YAML bytes."""

    path: Path
    sha256: str
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class Horizon:
    """Ordered Nav2 goals; this object is not itself a path."""

    goals: tuple[Waypoint, ...]
    approximate_length_m: float
    boundary_reason: str

    @property
    def waypoint_numbers(self) -> tuple[int, ...]:
        """Return the exact ordered goal identifiers sent to Nav2."""
        return tuple(item.number for item in self.goals)


def _finite_float(raw, label: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f'{label} must be numeric')
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return value


def load_route(path_value: str) -> Route:
    """Load and validate the subset of waypoint fields used by the planner."""
    path = Path(path_value).expanduser().resolve()
    payload = path.read_bytes()
    document = yaml.safe_load(payload)
    if not isinstance(document, dict) or not isinstance(document.get('waypoints'), list):
        raise ValueError('Waypoint YAML must contain a waypoints list')
    if not document['waypoints']:
        raise ValueError('Waypoint YAML contains no waypoints')

    waypoints = []
    seen_numbers = set()
    for index, raw in enumerate(document['waypoints']):
        if not isinstance(raw, dict):
            raise ValueError(f'waypoints[{index}] must be a mapping')
        try:
            number = int(raw.get('number', index + 1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'waypoints[{index}].number must be an integer') from exc
        if number in seen_numbers:
            raise ValueError(f'duplicate waypoint number: {number}')
        seen_numbers.add(number)
        flags = {}
        for name in ('stop', 'person_area'):
            value = raw.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f'waypoints[{index}].{name} must be true or false')
            flags[name] = value
        waypoint = Waypoint(
            number=number,
            x=_finite_float(raw.get('x'), f'waypoints[{index}].x'),
            y=_finite_float(raw.get('y'), f'waypoints[{index}].y'),
            yaw=_finite_float(
                raw.get('angle_radians'), f'waypoints[{index}].angle_radians'
            ),
            wait_time=_finite_float(
                raw.get('wait_time', 0.0), f'waypoints[{index}].wait_time'
            ),
            **flags,
        )
        if waypoint.wait_time < 0.0:
            raise ValueError(f'waypoints[{index}].wait_time must be non-negative')
        waypoints.append(waypoint)
    return Route(path, hashlib.sha256(payload).hexdigest(), tuple(waypoints))


def _distance(left: Waypoint, right: Waypoint) -> float:
    return math.hypot(right.x - left.x, right.y - left.y)


def _first_hairpin_end(waypoints: Sequence[Waypoint], start: int, end: int) -> int:
    """Stop at the vertex of the first turn of at least 90 degrees."""
    for center in range(max(1, start), end):
        before, vertex, after = waypoints[center - 1:center + 2]
        incoming = (vertex.x - before.x, vertex.y - before.y)
        outgoing = (after.x - vertex.x, after.y - vertex.y)
        product = math.hypot(*incoming) * math.hypot(*outgoing)
        dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        if product > 1.0e-9 and dot <= 0.0:
            return center
    return end


def select_horizon(route: Route, start_number: int, target_length_m: float) -> Horizon:
    """
    Select ordered goals up to a task or foldback boundary.

    The polyline distance is used only to select Nav2 goals. It is never
    published as a Path. The planner result is the sole reference path.
    """
    if not math.isfinite(target_length_m) or target_length_m <= 0.0:
        raise ValueError('target_length_m must be finite and positive')
    matches = [
        index for index, waypoint in enumerate(route.waypoints)
        if waypoint.number == start_number
    ]
    if not matches:
        raise ValueError(f'start waypoint number not found: {start_number}')
    start = matches[0]
    end = len(route.waypoints) - 1
    reason = 'route_end'
    for index in range(start, end + 1):
        if route.waypoints[index].task_boundary:
            end = index
            reason = 'task_boundary'
            break
    hairpin_end = _first_hairpin_end(route.waypoints, start, end)
    if hairpin_end < end:
        end = hairpin_end
        reason = 'foldback_boundary'

    return _select_horizon_range(route, start, end, target_length_m, reason)


def select_horizon_by_index(
        route: Route, start_index: int, safe_end_index: int,
        target_length_m: float) -> Horizon:
    """Select goals inside the follower-provided inclusive safety boundary."""
    if not math.isfinite(target_length_m) or target_length_m <= 0.0:
        raise ValueError('target_length_m must be finite and positive')
    if not 0 <= start_index < len(route.waypoints):
        raise ValueError(f'start waypoint index out of range: {start_index}')
    if not start_index <= safe_end_index < len(route.waypoints):
        raise ValueError(
            f'safe end index must be in [{start_index}, '
            f'{len(route.waypoints) - 1}]: {safe_end_index}'
        )
    end = _first_hairpin_end(route.waypoints, start_index, safe_end_index)
    reason = 'foldback_boundary' if end < safe_end_index else 'safe_section_end'
    return _select_horizon_range(
        route, start_index, end, target_length_m, reason
    )


def _select_horizon_range(
        route: Route, start: int, end: int, target_length_m: float,
        boundary_reason: str) -> Horizon:
    """Select an ordered range; only Nav2 may turn these goals into a Path."""
    selected = []
    length = 0.0
    previous = route.waypoints[start - 1] if start > 0 else None
    for index in range(start, end + 1):
        waypoint = route.waypoints[index]
        if previous is not None:
            length += _distance(previous, waypoint)
        selected.append(waypoint)
        previous = waypoint
        if length >= target_length_m:
            boundary_reason = 'target_reached'
            break
    return Horizon(tuple(selected), length, boundary_reason)


def planar_path_length(poses: Sequence) -> float:
    """Return planar arc length for objects exposing pose.position x/y."""
    total = 0.0
    for left, right in zip(poses, poses[1:]):
        total += math.hypot(
            right.pose.position.x - left.pose.position.x,
            right.pose.position.y - left.pose.position.y,
        )
    return total


def prefix_pose_count(poses: Sequence, maximum_length_m: float) -> int:
    """Keep only existing Nav2 samples in the requested arc-length prefix."""
    if not poses:
        return 0
    total = 0.0
    count = 1
    for left, right in zip(poses, poses[1:]):
        segment = math.hypot(
            right.pose.position.x - left.pose.position.x,
            right.pose.position.y - left.pose.position.y,
        )
        if total + segment > maximum_length_m:
            break
        total += segment
        count += 1
    return count
