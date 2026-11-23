"""Reference-Planner warm process preserves safety on heartbeat gaps."""
from types import SimpleNamespace


from ar_reference_planner.auto_supervisor import ReferencePlannerAutoSupervisor
from ar_reference_planner.runtime_selection import SelectionStatus
import pytest


class FakeProcess:
    def __init__(self):
        self.running = True
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.running = False


def fake_supervisor():
    map_selection = SimpleNamespace(identity=('map', 'sha', '20261123'))
    route = SimpleNamespace(sha256='a' * 64, waypoints=(1, 2, 3))
    active = SimpleNamespace(
        map_selection=map_selection,
        waypoint_selection=SimpleNamespace(route=route),
    )
    events = []
    fake = SimpleNamespace(
        _map_reader=SimpleNamespace(
            read=lambda: SelectionStatus(map_selection, 'map_verified'),
        ),
        _follower=SimpleNamespace(route_sha256='a' * 64, waypoint_count=3),
        _follower_received_at=10.0,
        _follower_timeout=2.5,
        _follower_process_grace=5.0,
        _process=FakeProcess(),
        _active_selection=active,
        _waypoint_status=SelectionStatus(
            active.waypoint_selection, 'waypoint_verified',
        ),
        _invalidate=lambda reason, force=False: events.append(
            ('invalidate', reason, force)
        ),
        _request_waypoint_parameter=lambda now: None,
        _guard_map_route_transition=lambda selection: (
            selection, 'selection_verified'
        ),
    )

    def clear_follower(reason):
        fake._follower = None
        fake._follower_received_at = None
        fake._waypoint_status = SelectionStatus(None, reason)
        fake._process.stop()
        fake._active_selection = None

    fake._clear_follower = clear_follower
    return fake, active, events


def test_short_heartbeat_gap_invalidates_path_without_restarting_planner():
    fake, original, events = fake_supervisor()
    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=12.641
    )
    assert desired is None
    assert reason.startswith('follower_state_stale_grace age_sec=')
    ReferencePlannerAutoSupervisor._reconcile(
        fake, desired, reason, now=12.641
    )
    assert events == [('invalidate', reason, False)]
    assert fake._process.running
    assert fake._process.stop_calls == 0
    fake._follower_received_at = 12.75
    restored, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=12.80
    )
    assert (
        restored.map_selection.identity
        == original.map_selection.identity
    )
    assert (
        restored.waypoint_selection.route.sha256
        == original.waypoint_selection.route.sha256
    )
    assert reason == 'selection_verified'


def test_long_heartbeat_loss_stops_owned_child():
    fake, _, _ = fake_supervisor()
    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=17.60
    )
    assert desired is None
    assert reason.startswith('follower_state_stale')
    assert fake._process.stop_calls == 1
    assert not fake._process.running


def test_changed_map_stops_child_even_during_short_heartbeat_gap():
    fake, _, _ = fake_supervisor()
    fake._map_reader = SimpleNamespace(read=lambda: SelectionStatus(
        SimpleNamespace(identity=('DIFFERENT',)), 'map_verified'
    ))
    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=12.65
    )
    assert desired is None
    assert reason.startswith('follower_state_stale')
    assert fake._process.stop_calls == 1


def test_changed_follower_route_never_uses_warm_planner():
    fake, _, _ = fake_supervisor()
    fake._follower.route_sha256 = 'b' * 64
    desired, reason = ReferencePlannerAutoSupervisor._desired_selection(
        fake, now=12.65
    )
    assert desired is None
    assert reason.startswith('follower_state_stale')
    assert fake._process.stop_calls == 1


def test_activation_watchdog_stops_only_stuck_reference_child():
    fake, _, events = fake_supervisor()
    fake._planner_started_at = 10.0
    fake._planner_active_seen = False
    fake._planner_activation_timeout = 20.0
    fake._planner_state_future = None
    fake._planner_epoch = 2
    fake._process_retry = 3.0
    ReferencePlannerAutoSupervisor._check_planner_activation(fake, 30.01)
    assert fake._process.stop_calls == 1
    assert fake._active_selection is None
    assert fake._next_process_retry_at == pytest.approx(33.01)
    assert events == [
        ('invalidate', 'reference_planner_activation_timeout', False)
    ]


def test_active_reference_planner_never_restarts_from_startup_watchdog():
    fake, _, events = fake_supervisor()
    fake._planner_started_at = 10.0
    fake._planner_active_seen = True
    fake._planner_activation_timeout = 20.0
    ReferencePlannerAutoSupervisor._check_planner_activation(fake, 100.0)
    assert fake._process.stop_calls == 0
    assert events == []
