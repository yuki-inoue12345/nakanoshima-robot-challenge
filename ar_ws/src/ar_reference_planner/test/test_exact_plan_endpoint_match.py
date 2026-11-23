"""Live Nav2 endpoint identity: exact-match-only exception is fail-closed."""
from types import SimpleNamespace

from ar_reference_planner.legacy_compatibility import (
    resolve_legacy_progress,
    unique_waypoint_index,
)
import pytest


def route_with_points(*xy):
    return SimpleNamespace(waypoints=[
        SimpleNamespace(x=x, y=y, task_boundary=False)
        for x, y in xy
    ])


def test_exact_endpoint_92mm_second_matches_waypoint_zero():
    route = route_with_points((1., 0.), (1.092, 0.), (9., 0.))
    assert unique_waypoint_index(route, 1., 0., tolerance_m=.4,
                                 separation_m=.2) == 0


def test_exact_endpoint_115mm_second_matches_correct_index():
    route = route_with_points((1., 0.), (1.115, 0.), (9., 0.))
    progress = resolve_legacy_progress(route, 'to_pose', 1.115, 0.,
                                       tolerance_m=.4, separation_m=.2)
    assert progress.next_index == 1
    assert progress.endpoint_index == 1


def test_fuzzy_near_match_still_fails_closed():
    route = route_with_points((1., 0.), (1.092, 0.), (9., 0.))
    with pytest.raises(ValueError, match='ambiguous'):
        unique_waypoint_index(route, 1.03, 0., tolerance_m=.4,
                              separation_m=.2)


def test_duplicate_waypoint_coordinates_always_fail_closed():
    route = route_with_points((1., 0.), (1., 0.), (9., 0.))
    with pytest.raises(ValueError, match='ambiguous'):
        unique_waypoint_index(route, 1., 0., tolerance_m=.4,
                              separation_m=.2)


def test_less_than_5cm_second_point_remains_ambiguous():
    route = route_with_points((1., 0.), (1.03, 0.), (9., 0.))
    with pytest.raises(ValueError, match='ambiguous'):
        unique_waypoint_index(route, 1., 0., tolerance_m=.4,
                              separation_m=.2)


def test_off_route_endpoint_still_rejected():
    route = route_with_points((1., 0.), (1.092, 0.))
    with pytest.raises(ValueError, match='not_on_route'):
        unique_waypoint_index(route, 12., 0., tolerance_m=.4,
                              separation_m=.2)
