"""No-robot manual mapping runtime regression tests."""
import threading
from types import SimpleNamespace

import numpy as np

from ar_path_viewer.odometry_distance import SessionOdomTravelDistanceMeter
from ar_path_viewer.rendering import render_frame
from ar_path_viewer.viewer_node import ArPathViewerNode, _travel_distance_labels


def test_manual_mapping_ready_start_finish_restart_without_motion():
    meter = SessionOdomTravelDistanceMeter()
    messages = []
    stub = SimpleNamespace(
        _operation_mode='manual_mapping',
        _mapping_phase='ready',
        _mapping_start_stamp=float('inf'),
        _ros_now_sec=lambda: 1.0,
        _travel_session_lock=threading.RLock(),
        _mapping_finished=None,
        _session_distance_meter=meter,
        _log_info=messages.append,
    )
    assert stub._mapping_phase == 'ready'
    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(stub)
    assert phase == 'running' and snapshot is None
    assert meter.total_m() is None
    meter.observe(0.0, 0.0, 1.0)
    meter.observe(0.1, 0.0, 1.1)
    meter.observe(0.2, 0.0, 1.2)
    live = meter.live_snapshot()
    assert live.available and 0.19 <= live.observed_m <= 0.21
    assert meter.frozen_snapshot() is None
    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(stub)
    assert phase == 'ended' and snapshot is not None
    assert snapshot.available and 0.19 <= snapshot.observed_m <= 0.21
    meter.observe(2.0, 0.0, 1.3)
    assert meter.live_snapshot() is snapshot
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
    assert len([m for m in messages if 'session_started' in m]) == 2


def test_mapping_rendering_never_invents_nav_path_and_draws_end_card():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ready = render_frame(image.copy(), [], 'MAPPING READY',
                         debug_overlay=False)
    moving = render_frame(image.copy(), [], 'MANUAL MAPPING',
                          travel_distance_status='ODOM TRAVELED: ~2.4 m',
                          travel_distance_status_ja='走行距離：約 2.4 m',
                          debug_overlay=False)
    ended = render_frame(image.copy(), [], 'MAPPING SESSION ENDED',
                         travel_distance_status='ODOM TRAVELED: ~15.3 m',
                         travel_distance_status_ja='走行距離：約 15.3 m',
                         debug_overlay=False)
    assert ready.shape == image.shape
    assert moving.shape == image.shape
    assert ended.shape == image.shape
    assert not np.array_equal(ready, image)
    assert not np.array_equal(moving, ready)
    assert not np.array_equal(ended, moving)


def test_odom_is_ignored_until_r_and_after_finish():
    """Camera and SLAM can run before R; stale queued odom stays excluded."""
    from nav_msgs.msg import Odometry

    meter = SessionOdomTravelDistanceMeter()
    fake = SimpleNamespace(
        _operation_mode='manual_mapping', _mapping_phase='ready',
        _mapping_start_stamp=float('inf'),
        _session_distance_meter=meter,
        _travel_session_lock=threading.RLock(),
        _mapping_finished=None,
        _ros_now_sec=lambda: 10.0,
        _str=lambda _name: 'odom',
        _odom_rotation_gate=SimpleNamespace(
            clear=lambda: None, update=lambda *_args: None,
        ),
        _last_reported_rotation=None,
        _log_info=lambda _message: None,
    )

    def odom(stamp, x):
        msg = Odometry()
        msg.header.frame_id = 'odom'
        msg.header.stamp.sec = int(stamp)
        msg.header.stamp.nanosec = int(round((stamp % 1) * 1e9))
        msg.pose.pose.position.x = x
        return msg

    ArPathViewerNode._on_rotation_odom(fake, odom(9.6, 0.0))
    ArPathViewerNode._on_rotation_odom(fake, odom(9.7, 0.1))
    assert meter.live_snapshot().accepted_steps == 0

    phase, _ = ArPathViewerNode._toggle_manual_mapping(fake)
    assert phase == 'running' and fake._mapping_start_stamp == 10.0
    fake._ros_now_sec = lambda: 100.0  # skip unrelated live-rotation gate
    ArPathViewerNode._on_rotation_odom(fake, odom(9.8, 0.2))  # queued pre-R sample
    ArPathViewerNode._on_rotation_odom(fake, odom(10.1, 0.3))
    ArPathViewerNode._on_rotation_odom(fake, odom(10.2, 0.4))
    ArPathViewerNode._on_rotation_odom(fake, odom(10.3, 0.5))
    assert abs(meter.live_snapshot().observed_m - 0.2) < 1e-6
    assert meter.frozen_snapshot() is None

    phase, snapshot = ArPathViewerNode._toggle_manual_mapping(fake)
    assert phase == 'ended'
    ArPathViewerNode._on_rotation_odom(fake, odom(10.4, 0.9))
    assert meter.live_snapshot() is snapshot
    assert abs(snapshot.observed_m - 0.2) < 1e-6
    fake._ros_now_sec = lambda: 11.0
    phase, _ = ArPathViewerNode._toggle_manual_mapping(fake)
    assert phase == 'running'
    assert meter.live_snapshot().observed_m == 0.0
    assert fake._mapping_start_stamp == 11.0
    ArPathViewerNode._on_rotation_odom(fake, odom(10.9, 1.0))
    ArPathViewerNode._on_rotation_odom(fake, odom(11.1, 1.1))
    assert meter.live_snapshot().accepted_steps == 0
