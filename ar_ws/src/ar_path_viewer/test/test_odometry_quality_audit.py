"""Validate distance quality reporting while preserving measured distances."""

from ar_path_viewer.odometry_distance import OdomTravelDistanceMeter


def test_rejections_reported_without_inventing_missing_motion():
    meter = OdomTravelDistanceMeter()
    meter.set_follower('a' * 64, 30, 1)
    meter.observe(0.00, 0.0, 1.00)
    meter.observe(0.05, 0.0, 1.10)
    meter.observe(0.07, 0.0, 1.10)  # Duplicate timestamp.
    meter.observe(0.40, 0.0, 2.00)  # Gap is never interpolated.
    meter.observe(5.00, 0.0, 2.10)  # Physically implausible jump.
    meter.observe(5.05, 0.0, 2.20)
    result = meter.diagnostics()
    assert result['accepted_steps'] == 2
    assert abs(result['observed_m'] - 0.10) < 1e-8
    assert result['rejected_bad_time'] == 1
    assert result['rejected_gap'] == 1
    assert result['rejected_jump'] == 1


def test_new_route_clears_audit_counters():
    meter = OdomTravelDistanceMeter()
    meter.set_follower('a' * 64, 30, 1)
    meter.observe(0.0, 0.0, 1.0)
    meter.observe(10.0, 0.0, 2.0)
    assert meter.diagnostics()['rejected_gap'] == 1
    meter.set_follower('b' * 64, 30, 1)
    assert meter.total_m() is None
    assert meter.diagnostics()['rejected_gap'] == 0
    assert meter.diagnostics()['accepted_steps'] == 0


def test_transition_keeps_distance_scope_and_no_phantom_step():
    meter = OdomTravelDistanceMeter()
    meter.set_follower('c' * 64, 30, 1)
    for index in range(5):
        meter.observe(index * 0.03, 0.0, 1.0 + 0.1 * index)
    before = meter.total_m()
    meter.set_follower('c' * 64, 30, 4)
    meter.observe(10.0, 0.0, 2.0)
    meter.set_follower('c' * 64, 30, 1)
    meter.observe(10.0, 0.0, 2.1)
    meter.observe(10.03, 0.0, 2.2)
    assert abs(meter.total_m() - (before + 0.03)) < 1e-8
