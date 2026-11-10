"""Tests for route horizon selection and Nav2-result prefixing."""

from pathlib import Path
from types import SimpleNamespace

from ar_reference_planner.route import (
    load_route,
    planar_path_length,
    prefix_pose_count,
    select_horizon,
)
import pytest
import yaml


def _write_route(tmp_path: Path, points) -> Path:
    path = tmp_path / 'route.yaml'
    path.write_text(yaml.safe_dump({'waypoints': points}), encoding='utf-8')
    return path


def _point(x, y):
    return SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0))
    )


def test_selects_target_horizon_but_does_not_generate_a_path(tmp_path):
    """Select goals by distance without pretending they form a Path."""
    route = load_route(str(_write_route(tmp_path, [
        {'number': index + 1, 'x': float(index * 2), 'y': 0.0,
         'angle_radians': 0.0}
        for index in range(10)
    ])))
    horizon = select_horizon(route, 2, 11.0)

    assert horizon.waypoint_numbers == (2, 3, 4, 5, 6, 7)
    assert horizon.approximate_length_m == pytest.approx(12.0)
    assert not hasattr(horizon, 'poses')


def test_task_boundary_is_included_but_never_crossed(tmp_path):
    """End at a person task rather than planning beyond it."""
    points = [
        {'number': index + 1, 'x': float(index * 2), 'y': 0.0,
         'angle_radians': 0.0}
        for index in range(8)
    ]
    points[3]['person_area'] = True
    route = load_route(str(_write_route(tmp_path, points)))

    horizon = select_horizon(route, 2, 11.0)

    assert horizon.waypoint_numbers == (2, 3, 4)
    assert horizon.boundary_reason == 'task_boundary'


def test_foldback_boundary_is_not_crossed(tmp_path):
    """Stop before a future branch reverses the route direction."""
    route = load_route(str(_write_route(tmp_path, [
        {'number': 1, 'x': 0.0, 'y': 0.0, 'angle_radians': 0.0},
        {'number': 2, 'x': 2.0, 'y': 0.0, 'angle_radians': 0.0},
        {'number': 3, 'x': 4.0, 'y': 0.0, 'angle_radians': 3.14},
        {'number': 4, 'x': 2.0, 'y': 0.0, 'angle_radians': 3.14},
        {'number': 5, 'x': 0.0, 'y': 0.0, 'angle_radians': 3.14},
    ])))

    horizon = select_horizon(route, 2, 20.0)

    assert horizon.waypoint_numbers == (2, 3)
    assert horizon.boundary_reason == 'foldback_boundary'


def test_prefix_uses_only_existing_nav2_samples():
    """Truncate by selecting original planner samples only."""
    poses = [_point(0.0, 0.0), _point(4.0, 0.0), _point(9.0, 0.0), _point(16.0, 0.0)]

    count = prefix_pose_count(poses, 15.0)

    assert count == 3
    assert planar_path_length(poses[:count]) == pytest.approx(9.0)


def test_rejects_duplicate_waypoint_numbers(tmp_path):
    """Reject an ambiguous manually selected route index."""
    path = _write_route(tmp_path, [
        {'number': 1, 'x': 0.0, 'y': 0.0, 'angle_radians': 0.0},
        {'number': 1, 'x': 1.0, 'y': 0.0, 'angle_radians': 0.0},
    ])

    with pytest.raises(ValueError, match='duplicate'):
        load_route(str(path))
