"""No-robot manual mapping runtime regression tests."""
import threading
from types import SimpleNamespace

import numpy as np

from ar_path_viewer.odometry_distance import SessionOdomTravelDistanceMeter
from ar_path_viewer.rendering import render_frame
from ar_path_viewer.viewer_node import ArPathViewerNode, _travel_distance_labels


def test_manual_mapping_freezes_distance_and_restarts_without_motion():
    meter = SessionOdomTravelDistanceMeter()
    meter.observe(0.0, 0.0, 1.0)
    meter.observe(0.1, 0.0, 1.1)
    meter.observe(0.2, 0.0, 1.2)
    messages = []
    stub = SimpleNamespace(
        _operation_mode='manual_mapping',
        _travel_session_lock=threading.RLock(),
        _mapping_finished=None,
        _session_distance_meter=meter,
        _log_info=messages.append,
    )
    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(stub)
    assert phase == 'ended' and snapshot is not None
    assert snapshot.available and 0.19 <= snapshot.observed_m <= 0.21
    meter.observe(2.0, 0.0, 1.3)
    assert meter.freeze() is snapshot
    assert stub._mapping_finished is snapshot
    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(stub)
    assert phase == 'running' and snapshot is None
    assert stub._mapping_finished is None
    assert meter.frozen_snapshot() is None
    assert meter.total_m() is None
    meter.observe(0.0, 0.0, 2.0)
    meter.observe(0.1, 0.0, 2.1)
    meter.observe(0.2, 0.0, 2.2)
    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(stub)
    assert phase == 'ended' and snapshot.available


def test_mapping_rendering_never_invents_nav_path_and_draws_end_card():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    moving = render_frame(image.copy(), [], 'MANUAL MAPPING',
                          debug_overlay=False)
    ended = render_frame(image.copy(), [], 'MAPPING SESSION ENDED',
                         travel_distance_status='ODOM TRAVELED: ~15.3 m',
                         travel_distance_status_ja='走行距離：約 15.3 m',
                         debug_overlay=False)
    assert moving.shape == image.shape
    assert ended.shape == image.shape
    assert not np.array_equal(moving, image)
    assert not np.array_equal(ended, moving)
