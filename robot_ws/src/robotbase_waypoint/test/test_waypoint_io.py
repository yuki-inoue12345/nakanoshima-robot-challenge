"""Unit tests for the waypoint YAML format."""

from pathlib import Path

import pytest

from robotbase_waypoint.waypoint_io import append_waypoint, load_waypoints


def test_append_and_load(tmp_path: Path):
    waypoint_file = tmp_path / 'route.yaml'
    first = append_waypoint(str(waypoint_file), 1.0, 2.0, 0.5)
    second = append_waypoint(str(waypoint_file), 3.0, 4.0, -0.5)

    assert first.number == 1
    assert second.number == 2
    loaded = load_waypoints(str(waypoint_file))
    assert [waypoint.number for waypoint in loaded] == [1, 2]
    assert loaded[1].angle_radians == -0.5


def test_append_pause_waypoint(tmp_path: Path):
    waypoint_file = tmp_path / 'pause-route.yaml'

    append_waypoint(
        str(waypoint_file), 1.0, 2.0, 0.5,
        stop=True, wait_time=10.0)

    loaded = load_waypoints(str(waypoint_file))
    assert loaded[0].stop is True
    assert loaded[0].wait_time == 10.0
    text = waypoint_file.read_text(encoding='utf-8')
    assert 'stop: true' in text
    assert 'wait_time: 10.0' in text


def test_rejects_missing_pose_field(tmp_path: Path):
    waypoint_file = tmp_path / 'invalid.yaml'
    waypoint_file.write_text(
        'waypoints:\n  - number: 1\n    x: 1.0\n    y: 2.0\n',
        encoding='utf-8')

    with pytest.raises(ValueError, match='angle_radians'):
        load_waypoints(str(waypoint_file))


def test_rejects_sirius_map_switch(tmp_path: Path):
    waypoint_file = tmp_path / 'map-switch.yaml'
    waypoint_file.write_text(
        'waypoints:\n'
        '  - x: 1.0\n'
        '    y: 2.0\n'
        '    angle_radians: 0.0\n'
        '    change_map: second-floor\n',
        encoding='utf-8')

    with pytest.raises(ValueError, match='change_map'):
        load_waypoints(str(waypoint_file))
