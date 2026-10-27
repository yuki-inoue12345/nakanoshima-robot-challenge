import math
from pathlib import Path

import numpy as np
import pytest

from ar_path_viewer.route_io import load_waypoint_records
from ar_path_viewer.simulation import (
    build_frame_schedule,
    interpolate_route,
    segment_headings,
    wrapped_differences,
)


def test_route_interpolation_visits_every_waypoint_and_finishes():
    waypoints = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    route, indices = interpolate_route(waypoints, spacing=0.2)
    assert np.allclose(route[indices], waypoints)
    assert np.allclose(route[-1], waypoints[-1])


def test_turn_signs_are_left_positive_right_negative():
    left = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    right = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, -1.0)))
    assert wrapped_differences(segment_headings(left))[0] == pytest.approx(math.pi / 2)
    assert wrapped_differences(segment_headings(right))[0] == pytest.approx(-math.pi / 2)


def test_stop_and_wait_fields_add_dwell_frames():
    data = Path(__file__).parents[1] / "test_data" / "test_waypoints_b.yaml"
    records = load_waypoint_records(data)
    _, indices = interpolate_route(
        np.array([(record.x, record.y) for record in records]), spacing=0.5
    )
    schedule, dwell = build_frame_schedule(
        indices[-1] + 1, indices, records, fps=10.0, stop_duration_sec=5.0
    )
    assert len(schedule) == indices[-1] + 1 + 4
    assert dwell == pytest.approx(0.4)
