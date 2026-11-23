"""Session-distance safety: actual odometry is never route-gated."""
import math
from ar_path_viewer.odometry_distance import SessionOdomTravelDistanceMeter


def test_session_counts_odom_without_any_follower_state():
    meter = SessionOdomTravelDistanceMeter()
    for i in range(11):
        meter.observe(0.03 * i, 0.0, 1.0 + 0.1 * i)
    assert abs(meter.total_m() - 0.3) < 1e-10
    assert meter.diagnostics()['accepted_steps'] == 10


def test_session_counts_backtracking_as_real_travel_not_net_displacement():
    meter = SessionOdomTravelDistanceMeter()
    for i, x in enumerate([0., .02, .04, .02, .00]):
        meter.observe(x, 0., 1.0 + i * 0.05)
    assert abs(meter.total_m() - 0.08) < 1e-10


def test_session_gaps_and_jumps_are_dropped_and_reported():
    meter = SessionOdomTravelDistanceMeter()
    for x, t in [(0., 1.0), (.05, 1.1), (.45, 2.0),
                 (20., 2.1), (20.05, 2.2)]:
        meter.observe(x, 0., t)
    data = meter.diagnostics()
    assert abs(data['observed_m'] - .10) < 1e-9
    assert data['rejected_gap'] == 1
    assert data['rejected_jump'] == 1


def test_session_out_of_order_and_invalid_input_never_add_motion():
    meter = SessionOdomTravelDistanceMeter()
    meter.observe(0., 0., 1.0)
    meter.observe(.02, 0., 1.1)
    meter.observe(.04, 0., 1.1)
    meter.observe(math.inf, 0., 1.2)
    meter.observe(.06, 0., 1.2)
    assert abs(meter.total_m() - .04) < 1e-10
    assert meter.diagnostics()['rejected_bad_time'] == 1


def test_session_30hz_replay_matches_independent_polyline_integral():
    meter = SessionOdomTravelDistanceMeter()
    positions = [(i * .01, .3 * math.sin(i / 200.0)) for i in range(5800)]
    for i, (x, y) in enumerate(positions):
        meter.observe(x, y, 100.0 + i / 30.0)
    expected = sum(math.hypot(x2 - x1, y2 - y1)
                   for (x1, y1), (x2, y2) in zip(positions, positions[1:]))
    assert abs(meter.total_m() - expected) < 1e-8
    assert meter.diagnostics()['accepted_steps'] == 5799
    assert meter.diagnostics()['rejected_gap'] == 0
