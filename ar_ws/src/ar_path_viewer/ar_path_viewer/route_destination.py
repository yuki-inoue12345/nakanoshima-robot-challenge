"""Validate typed route-destination metadata for a direct-distance label."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RouteDestination:
    route_sha256: str
    waypoint_count: int
    waypoint_number: int
    x: float
    y: float


def destination_from_reference_state(
    message, expected_frame: str = 'map'
) -> RouteDestination | None:
    """Accept metadata produced from the Planner's hash-validated Route."""
    if not bool(message.final_waypoint_valid):
        return None
    if str(message.header.frame_id) != str(expected_frame):
        raise ValueError('final_waypoint_frame_mismatch')
    digest = str(message.route_sha256)
    count = int(message.waypoint_count)
    number = int(message.final_waypoint_number)
    x = float(message.final_waypoint_position.x)
    y = float(message.final_waypoint_position.y)
    if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise ValueError('final_waypoint_route_hash_invalid')
    if count <= 0:
        raise ValueError('final_waypoint_count_invalid')
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('final_waypoint_position_invalid')
    return RouteDestination(digest, count, number, x, y)


def direct_distance_m(
    destination: RouteDestination | None,
    route_sha256: str,
    waypoint_count: int,
    current_x: float,
    current_y: float,
) -> float | None:
    """Return map-plane straight-line distance, never a Nav2 path length."""
    if destination is None:
        return None
    if (
        destination.route_sha256 != str(route_sha256)
        or destination.waypoint_count != int(waypoint_count)
    ):
        return None
    values = (float(current_x), float(current_y))
    if not all(math.isfinite(value) for value in values):
        return None
    return math.hypot(destination.x - values[0], destination.y - values[1])
