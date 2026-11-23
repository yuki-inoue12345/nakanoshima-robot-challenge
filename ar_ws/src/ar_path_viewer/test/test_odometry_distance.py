"""Regression for measured odometry distance; no ROS required."""
from ar_path_viewer.odometry_distance import OdomTravelDistanceMeter


def test_follower_scoped_odometry_and_no_phantom_jump():
    m = OdomTravelDistanceMeter()
    m.observe(0.0, 0.0, 1.0)
    assert m.total_m() is None
    m.set_follower('a' * 64, 30, 1)
    for i in range(11):
        m.observe(0.05 * i, 0.0, 1.0 + 0.1 * i)
    assert abs(m.total_m() - 0.5) < 1e-6
    m.observe(20.0, 0.0, 2.1)
    m.observe(20.05, 0.0, 2.2)
    assert abs(m.total_m() - 0.55) < 1e-6
    m.set_follower('a' * 64, 30, 8)
    m.observe(21.0, 0.0, 2.3)
    assert abs(m.total_m() - 0.55) < 1e-6


def test_route_change_resets_and_pause_does_not_bridge_motion():
    m = OdomTravelDistanceMeter()
    m.set_follower('a' * 64, 30, 1)
    for i in range(5):
        m.observe(i * 0.03, 0, 1 + i * 0.1)
    assert m.total_m() is not None
    m.set_follower('a' * 64, 30, 4)
    m.observe(50, 0, 2.0)
    m.set_follower('a' * 64, 30, 1)
    m.observe(50, 0, 2.1)
    m.observe(50.03, 0, 2.2)
    assert abs(m.total_m() - 0.15) < 1e-6
    m.set_follower('b' * 64, 30, 1)
    assert m.total_m() is None


def test_missing_and_out_of_order_odom_fail_closed():
    m = OdomTravelDistanceMeter()
    m.set_follower('a' * 64, 30, 1)
    m.observe(0, 0, 1)
    m.observe(0.03, 0, 1.1)
    m.observe(0.08, 0, 1.1)
    m.observe(0.1, 0, 1.2)
    # Skip a missing 1 second interval rather than invent its path.
    m.observe(10, 0, 2.2)
    assert abs(m._total - 0.05) < 1e-6


def test_route_updating_continues_measuring_actual_odom():
    meter = OdomTravelDistanceMeter()
    meter.set_follower('c' * 64, 30, 1)
    for i in range(5):
        meter.observe(0.03 * i, 0.0, 1.0 + i * 0.1)
    meter.set_follower('c' * 64, 30, 2)
    for i in range(5, 11):
        meter.observe(0.03 * i, 0.0, 1.0 + i * 0.1)
    meter.set_follower('c' * 64, 30, 1)
    meter.observe(0.33, 0.0, 2.1)
    assert abs(meter.total_m() - 0.33) < 1e-9
    meter.set_follower('c' * 64, 30, 8)
    meter.observe(0.36, 0.0, 2.2)
    assert abs(meter.total_m() - 0.33) < 1e-9
