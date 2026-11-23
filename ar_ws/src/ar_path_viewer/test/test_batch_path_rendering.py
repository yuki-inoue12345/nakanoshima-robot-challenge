"""Exact-image regressions for the bounded path render fast path."""

import cv2
import numpy as np

from ar_path_viewer.rendering import (
    _draw_segment_batch,
    direction_arrow_segments,
)


def _reference_arrows(segments, spacing_px=90.0, minimum_path_px=18.0,
                      arrow_length_px=18.0):
    import math
    segment_list = list(segments)
    chains, current = [], []
    for start, end in segment_list:
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 1.0:
            continue
        if current and math.hypot(
            start[0] - current[-1][1][0], start[1] - current[-1][1][1]
        ) > 2.0:
            chains.append(current)
            current = []
        current.append((start, end))
    if current:
        chains.append(current)
    arrows = []
    for chain in chains:
        lengths = [math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in chain]
        total = sum(lengths)
        if total < minimum_path_px:
            continue
        targets = list(np.arange(spacing_px, total + 1e-9, spacing_px))
        if not targets:
            targets = [total]
        elif total - targets[-1] >= spacing_px * 0.65:
            targets.append(total)
        for target_distance in targets:
            walked = 0.0
            for (start, end), length in zip(chain, lengths):
                if walked + length + 1e-9 < target_distance:
                    walked += length
                    continue
                fraction = min(1.0, max(0.0, (target_distance - walked) / length))
                target = np.array(start, dtype=float) + fraction * (
                    np.array(end, dtype=float) - np.array(start, dtype=float)
                )
                unit = (np.array(end, dtype=float) - np.array(start, dtype=float)) / length
                origin = target - unit * min(arrow_length_px, target_distance)
                arrows.append((tuple(np.rint(origin).astype(int)),
                               tuple(np.rint(target).astype(int))))
                break
    return arrows


def test_batch_segments_bitwise_equal_to_cv2_line():
    for seed in range(24):
        rng = np.random.default_rng(seed)
        segments = rng.integers(-70, 700, size=(seed * 11, 2, 2),
                                dtype=np.int32)
        # Includes empty, overlapping, clipped, out-of-bounds, and zero-length.
        source = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        for thickness in (1, 3, 6):
            old = source.copy()
            new = source.copy()
            for start, end in segments:
                cv2.line(old, tuple(map(int, start)), tuple(map(int, end)),
                         (255, 255, 0), thickness, cv2.LINE_AA)
            _draw_segment_batch(new, segments, (255, 255, 0), thickness)
            assert np.array_equal(old, new)


def test_dashed_halo_exact_pixel_parity():
    segs = [((20, 40), (180, 110)), ((180, 110), (360, 240)),
            ((180, 70), (180, 70)), ((-10, -10), (19, 40))]
    old = np.full((480, 640, 3), 240, np.uint8)
    new = old.copy()
    for start, end in segs:
        cv2.line(old, start, end, (18, 18, 18), 6, cv2.LINE_AA)
    for start, end in segs:
        cv2.line(old, start, end, (255, 255, 0), 3, cv2.LINE_AA)
    _draw_segment_batch(new, segs, (18, 18, 18), 6)
    _draw_segment_batch(new, segs, (255, 255, 0), 3)
    assert np.array_equal(old, new)


def test_direction_arrows_match_previous_code():
    rng = np.random.default_rng(942)
    for length in (0, 1, 12, 150, 800):
        for spacing in (30.0, 90.0, 170.0):
            coords = np.cumsum(
                rng.integers(-18, 19, size=(length + 1, 2)), axis=0
            )
            segments = [
                (tuple(map(int, coords[i])), tuple(map(int, coords[i + 1])))
                for i in range(length)
            ]
            # Add a deliberate discontinuity to test independent chains.
            if len(segments) > 10:
                segments.insert(10, ((1000, -800), (1030, -800)))
            result = direction_arrow_segments(segments, spacing_px=spacing)
            assert result == _reference_arrows(segments, spacing_px=spacing)
