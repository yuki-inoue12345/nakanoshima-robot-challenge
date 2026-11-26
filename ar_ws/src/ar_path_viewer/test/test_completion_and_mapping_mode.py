"""Verify the strict route-completion latch and performance isolation."""

from pathlib import Path
import threading
from types import SimpleNamespace

from ar_path_viewer.odometry_distance import SessionOdomTravelDistanceMeter
from ar_path_viewer.viewer_node import (
    ArPathViewerNode,
    _travel_distance_labels,
)
from robotbase_waypoint_interfaces.msg import WaypointFollowerState


def _finished_message():
    message = WaypointFollowerState()
    message.route_sha256 = 'a' * 64
    message.goal_generation = 10
    message.next_waypoint_number = 10
    message.next_waypoint_index = 10
    message.waypoint_count = 10
    message.activity = WaypointFollowerState.ACTIVITY_FINISHED
    return message


def test_verified_finished_state_freezes_and_survives_later_odom():
    meter = SessionOdomTravelDistanceMeter()
    for index in range(101):
        meter.observe(index * .01, 0.0, 1.0 + index * .03)
    logs = []
    fake = SimpleNamespace(
        _last_verified_final_goal=('a' * 64, 10, 10, 10, 90.0),
        _waypoint_progress_gate=SimpleNamespace(
            navigation_hold_status=lambda _now: 'ROUTE COMPLETE'
        ),
        _travel_session_lock=threading.RLock(),
        _session_distance_meter=meter,
        _route_completion=None,
        _ros_now_sec=lambda: 100.0,
        _log_info=logs.append,
    )
    ArPathViewerNode._accept_verified_route_completion(
        fake, _finished_message()
    )
    frozen = fake._route_completion['distance']
    meter.observe(10.0, 0.0, 10.0)
    assert abs(frozen.observed_m - 1.0) < 1e-10
    assert meter.freeze() == frozen
    assert 'route_completion_latched' in logs[-1]


def test_cancel_stop_or_wrong_goal_cannot_latch_viewer_completion():
    for activity, generation in (
        (WaypointFollowerState.ACTIVITY_HOLDING_STOP, 10),
        (WaypointFollowerState.ACTIVITY_FINISHED, 11),
    ):
        message = _finished_message()
        message.activity = activity
        message.goal_generation = generation
        fake = SimpleNamespace(
            _last_verified_final_goal=('a' * 64, 10, 10, 10, 90.0),
            _waypoint_progress_gate=SimpleNamespace(
                navigation_hold_status=lambda _now: (
                    'ROUTE COMPLETE' if activity == 9 else 'STOPPED'
                )
            ),
            _travel_session_lock=threading.RLock(),
            _session_distance_meter=SessionOdomTravelDistanceMeter(),
            _route_completion=None,
            _ros_now_sec=lambda: 100.0,
            _log_info=lambda _text: None,
        )
        ArPathViewerNode._accept_verified_route_completion(fake, message)
        assert fake._route_completion is None


def test_distance_labels_never_fabricate_missing_measurement():
    empty = SessionOdomTravelDistanceMeter().freeze()
    assert _travel_distance_labels(empty) == (
        'ODOM DISTANCE: UNAVAILABLE', '走行距離：計測不可'
    )
    meter = SessionOdomTravelDistanceMeter()
    meter.observe(0.0, 0.0, 1.0)
    meter.observe(0.1, 0.0, 1.1)
    meter.observe(0.2, 0.0, 1.2)
    english, japanese = _travel_distance_labels(meter.freeze())
    assert '0.2 m' in english
    assert '走行距離' in japanese


def test_viewer_uses_three_workers_without_racing_costmap_state():
    source = (
        Path(__file__).parents[1] / 'ar_path_viewer' / 'viewer_node.py'
    ).read_text()
    assert 'MultiThreadedExecutor(num_threads=3)' in source
    assert '_perception_callback_group' not in source
    assert 'CostmapStore/TransformQueue are intentionally single-owner' in source
