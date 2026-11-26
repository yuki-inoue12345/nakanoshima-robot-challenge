"""Regression: completion labels outside small frames must not crash."""
import numpy as np

from ar_path_viewer.display_window import _paint_centered_mask, draw_localized_label
from ar_path_viewer.rendering import render_frame


def test_outside_text_rect_is_noop():
    canvas = np.zeros((100, 160, 3), dtype=np.uint8)
    before = canvas.copy()
    _paint_centered_mask(canvas, (18, 95, 132, 138),
                         np.full((12, 106), 255, dtype=np.uint8),
                         (200, 255, 100))
    assert np.array_equal(canvas, before)


def test_partly_outside_bottom_is_clipped_safely():
    canvas = np.zeros((100, 160, 3), dtype=np.uint8)
    _paint_centered_mask(canvas, (18, 80, 132, 110),
                         np.full((12, 106), 255, dtype=np.uint8),
                         (200, 255, 100))
    assert np.count_nonzero(canvas) > 0
    assert np.all(canvas[:80] == 0)


def test_partly_outside_top_left_is_clipped_safely():
    canvas = np.zeros((100, 160, 3), dtype=np.uint8)
    _paint_centered_mask(canvas, (-20, -20, 50, 16),
                         np.full((12, 60), 255, dtype=np.uint8),
                         (255, 100, 200))
    assert np.count_nonzero(canvas) > 0
    assert np.all(canvas[4:] == 0)
    assert np.all(canvas[:, 45:] == 0)


def test_localized_text_label_completely_outside_is_noop():
    canvas = np.zeros((100, 160, 3), dtype=np.uint8)
    draw_localized_label(canvas, (18, 110, 132, 160),
                         '走行終了', 'RUN FINISHED', (255, 255, 255), 22)
    assert not np.count_nonzero(canvas)


def test_small_completion_and_mapping_cards_do_not_crash():
    tiny = np.zeros((100, 160, 3), dtype=np.uint8)
    autonomous = render_frame(
        tiny.copy(), [], 'NAVIGATION HOLD',
        navigation_status='ROUTE COMPLETE',
        travel_distance_status='SESSION ODOM TRAVELED: ~15.3 m',
        travel_distance_status_ja='走行距離：約 15.3 m',
    )
    manual = render_frame(
        tiny.copy(), [], 'MAPPING SESSION ENDED',
        travel_distance_status='ODOM TRAVELED: ~15.3 m',
        travel_distance_status_ja='走行距離：約 15.3 m',
    )
    assert autonomous.shape == tiny.shape
    assert manual.shape == tiny.shape
