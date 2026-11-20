"""Verify ROTATE hints are based on measured stationary turns only."""
import math

from ar_path_viewer.rotation_motion import RotationMotionGate


def test_stable_in_place_left_and_right_rotation():
    gate = RotationMotionGate()
    values = [gate.update((0.0, 0.0, math.radians(45.0 * t)), t)
              for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
    assert "ROTATE LEFT" in values
    gate.clear()
    values = [gate.update((0.0, 0.0, math.radians(-45.0 * t)), t)
              for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
    assert "ROTATE RIGHT" in values


def test_moving_turn_and_small_oscillation_never_count():
    gate = RotationMotionGate()
    for i in range(6):
        t = i * 0.1
        assert gate.update((t * 0.35, 0, t * 0.8), t) is None
    gate.clear()
    for i in range(6):
        t = i * 0.1
        assert gate.update((0, 0, math.radians(2) * (-1)**i), t) is None


def test_invalid_localization_and_clock_gap_clear_evidence():
    gate = RotationMotionGate()
    for t in (0.0, 0.1, 0.2, 0.3):
        gate.update((0, 0, t), t)
    assert gate.update(None, 0.4) is None
    assert gate.update((0, 0, 0.6), 0.5) is None
    assert gate.update((0, 0, 0.7), 2.0) is None


def test_heading_wraparound_preserves_direction():
    gate = RotationMotionGate()
    for t in (0.0, 0.1, 0.2, 0.3, 0.4):
        angle = math.radians(170 + 42 * t)
        result = gate.update((0, 0, angle), t)
    assert result == "ROTATE LEFT"


def test_single_amcl_heading_jump_is_not_rotation():
    gate = RotationMotionGate()
    results = []
    for i in range(7):
        yaw = 0.0 if i < 2 else math.radians(20.0)
        results.append(gate.update((0, 0, yaw), i * 0.08))
    assert all(value is None for value in results)
