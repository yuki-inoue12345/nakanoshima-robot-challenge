"""Vectorized guidance must match original geometry, not approximate routes."""
import math
import numpy as np
import pytest

from ar_path_viewer.offscreen_guidance import (
    _closest_path_anchor, _remaining_polyline,
    measure_path_progress, decide_offscreen_guidance,
)


def legacy_closest(points_xy):
    best = None
    for index, (start, end) in enumerate(zip(points_xy[:-1], points_xy[1:])):
        delta = end - start
        squared_length = float(np.dot(delta, delta))
        fraction = 0.0
        if squared_length > 1e-12:
            fraction = float(np.clip(
                -np.dot(start, delta) / squared_length, 0.0, 1.0
            ))
        point = start + fraction * delta
        distance = float(np.linalg.norm(point))
        candidate = (distance, index, fraction, point)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(
            "guidance requires at least one non-empty Path segment"
        )
    distance, index, fraction, point = best
    return index, fraction, point, distance


def legacy_remaining(points_xy, index, anchor):
    candidates = [anchor]
    candidates.extend(points_xy[index + 1:])
    filtered = [np.asarray(candidates[0], dtype=float)]
    for point in candidates[1:]:
        point = np.asarray(point, dtype=float)
        if np.linalg.norm(point - filtered[-1]) > 1e-9:
            filtered.append(point)
    return np.asarray(filtered)


def assert_equivalent(points_xy):
    expected = legacy_closest(points_xy)
    actual = _closest_path_anchor(points_xy)
    assert expected[0] == actual[0]
    assert actual[1] == pytest.approx(expected[1], abs=1e-12)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0, atol=1e-12)
    assert actual[3] == pytest.approx(expected[3], abs=1e-12)
    old_tail = legacy_remaining(points_xy, expected[0], expected[2])
    new_tail = _remaining_polyline(points_xy, actual[0], actual[2])
    np.testing.assert_allclose(new_tail, old_tail, rtol=0, atol=1e-12)
    assert new_tail.shape == old_tail.shape


@pytest.mark.parametrize('n', (2, 3, 20, 150, 600, 1200))
def test_dense_reference_path_matches_legacy(n):
    x = np.linspace(-0.2, 19.8, n)
    y = 0.4 * np.sin(x / 2)
    assert_equivalent(np.column_stack((x, y)))


def test_hairpin_self_crossing_and_equidistant_tie_match_legacy():
    paths = [
        np.array([[2., 0.2], [1., 0.2], [1., -0.2], [2., -0.2]]),
        np.array([[1., 0.], [0., 0.], [-1., 0.], [0., 0.], [1., 0.]]),
        np.array([[-4., -4.], [-3., -4.], [-3., -5.]]),
    ]
    for path in paths:
        assert_equivalent(path)


def test_near_duplicate_points_preserve_last_accepted_semantics():
    # Adjacent gaps are all tiny, but cumulative movement eventually exceeds
    # the threshold. A naïve np.diff mask would silently lose this geometry.
    path = np.array([
        [0., 0.], [0.6e-9, 0.], [1.2e-9, 0.],
        [1.8e-9, 0.], [1.0, 0.], [2.0, 0.],
    ])
    assert_equivalent(path)
    anchor = np.array([0., 0.])
    np.testing.assert_allclose(
        _remaining_polyline(path, 0, anchor),
        legacy_remaining(path, 0, anchor), rtol=0, atol=1e-12,
    )


def test_real_guidance_stays_truthful_and_valid():
    x = np.linspace(-0.15, 18.0, 600)
    path = np.column_stack((x, 0.3 * np.sin(x / 2), np.zeros_like(x)))
    decision = decide_offscreen_guidance(path)
    progress = measure_path_progress(path)
    assert progress is not None
    assert math.isfinite(progress.remaining_path_m)
    assert math.isfinite(progress.anchor_distance_m)
    if decision.guidance is not None:
        assert decision.guidance.remaining_path_m == pytest.approx(
            progress.remaining_path_m, abs=1e-8,
        )
