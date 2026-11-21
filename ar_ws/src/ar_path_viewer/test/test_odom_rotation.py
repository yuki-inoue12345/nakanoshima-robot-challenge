"""Odometry evidence must only show real, fresh in-place turns."""

from ar_path_viewer.rotation_motion import OdomRotationMotionGate


def test_sustained_measured_left_and_right_turn():
    gate = OdomRotationMotionGate()
    results = [gate.update(0.002, 0.55, i * 0.04) for i in range(15)]
    assert 'ROTATE LEFT' in results
    assert gate.direction(0.57) == 'ROTATE LEFT'
    gate.clear()
    results = [gate.update(0.002, -0.55, i * 0.04) for i in range(15)]
    assert 'ROTATE RIGHT' in results


def test_forward_motion_is_not_in_place_rotation():
    gate = OdomRotationMotionGate()
    assert all(gate.update(0.25, 0.55, i * 0.04) is None
               for i in range(30))


def test_small_yaw_and_noisy_sign_changes_fail_closed():
    gate = OdomRotationMotionGate()
    assert all(gate.update(0.0, 0.10, i * 0.04) is None
               for i in range(20))
    assert all(gate.update(0.0, (-1) ** i * 0.65, i * 0.04) is None
               for i in range(20))


def test_stale_odom_cannot_continue_to_display():
    gate = OdomRotationMotionGate()
    for i in range(15):
        gate.update(0.001, 0.55, i * 0.04)
    assert gate.direction(0.57) == 'ROTATE LEFT'
    assert gate.direction(0.90) is None


def test_clock_jump_and_missing_measurement_reset():
    gate = OdomRotationMotionGate()
    for i in range(12):
        gate.update(0.001, 0.55, i * 0.04)
    assert gate.update(0.001, 0.55, 3.0) is None
    assert gate.update(None, 0.55, 3.04) is None
