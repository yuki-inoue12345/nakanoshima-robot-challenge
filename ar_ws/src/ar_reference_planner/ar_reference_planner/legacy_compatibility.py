"""
Fail-closed observations for the pre-reference-state waypoint follower.

The legacy follower exposes its route and navigation mode as parameters, but
does not expose its current waypoint.  A current index is therefore accepted
only when the endpoint of a *new* Nav2 plan uniquely matches a waypoint from
the exact route file.  Continuous navigation additionally requires feedback
from the same Nav2 action goal UUID.

This module never sends an action goal and never publishes velocity commands.
"""

from dataclasses import dataclass
import math

from .route import Route


@dataclass(frozen=True)
class LegacyProgress:
    """Verified subset of the state missing from the legacy follower."""

    next_index: int
    safe_end_index: int
    endpoint_index: int


def unique_waypoint_index(
    route: Route,
    x: float,
    y: float,
    *,
    tolerance_m: float,
    separation_m: float,
) -> int:
    """Return the only route waypoint compatible with a Nav2 plan endpoint."""
    if (
        not math.isfinite(x)
        or not math.isfinite(y)
        or not math.isfinite(tolerance_m)
        or tolerance_m <= 0.0
        or not math.isfinite(separation_m)
        or separation_m < 0.0
    ):
        raise ValueError('invalid_endpoint_match_input')
    distances = sorted(
        (math.hypot(waypoint.x - x, waypoint.y - y), index)
        for index, waypoint in enumerate(route.waypoints)
    )
    if not distances or distances[0][0] > tolerance_m:
        raise ValueError(
            f'plan_endpoint_not_on_route nearest_m='
            f'{distances[0][0] if distances else math.inf:.3f}'
        )
    if len(distances) > 1 and distances[1][0] - distances[0][0] < separation_m:
        # A fresh Nav2 plan endpoint is a stronger identity signal when it
        # coincides with precisely one route point at centimeter precision.
        # The 20-cm conservative margin remains for every non-exact case.
        # Never accept a duplicate or nearly coincident (< 5cm) endpoint.
        nearest_m, second_m = distances[0][0], distances[1][0]
        exact_unique_goal = (
            nearest_m <= 0.01 and second_m - nearest_m >= 0.05
        )
        if not exact_unique_goal:
            raise ValueError(
                'plan_endpoint_waypoint_ambiguous '
                f'nearest_m={nearest_m:.3f} second_m={second_m:.3f}'
            )
    return distances[0][1]


def conservative_safe_end_index(route: Route, start_index: int) -> int:
    """Stop at the first route task boundary, otherwise at the route end."""
    if not 0 <= start_index < len(route.waypoints):
        raise ValueError('legacy_next_waypoint_index_out_of_range')
    for index in range(start_index, len(route.waypoints)):
        if route.waypoints[index].task_boundary:
            return index
    return len(route.waypoints) - 1


def resolve_legacy_progress(
    route: Route,
    navigation_mode: str,
    endpoint_x: float,
    endpoint_y: float,
    *,
    poses_remaining: int | None = None,
    tolerance_m: float = 0.40,
    separation_m: float = 0.20,
) -> LegacyProgress:
    """Resolve progress only from a unique route endpoint and bound feedback."""
    endpoint_index = unique_waypoint_index(
        route,
        endpoint_x,
        endpoint_y,
        tolerance_m=tolerance_m,
        separation_m=separation_m,
    )
    if navigation_mode == 'to_pose':
        next_index = endpoint_index
    elif navigation_mode == 'through_poses':
        if poses_remaining is None or isinstance(poses_remaining, bool):
            raise ValueError('through_poses_feedback_required')
        remaining = int(poses_remaining)
        if remaining <= 0:
            raise ValueError('through_poses_remaining_not_positive')
        next_index = endpoint_index + 1 - remaining
        if not 0 <= next_index <= endpoint_index:
            raise ValueError('through_poses_progress_out_of_range')
    else:
        raise ValueError(f'invalid_navigation_mode={navigation_mode!r}')
    return LegacyProgress(
        next_index=next_index,
        safe_end_index=conservative_safe_end_index(route, next_index),
        endpoint_index=endpoint_index,
    )
