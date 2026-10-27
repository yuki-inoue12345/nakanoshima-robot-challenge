from types import SimpleNamespace

import numpy as np
import pytest

from ar_path_viewer.path_adapter import PathStore, path_message_to_array


def test_new_path_atomically_replaces_old_path():
    clock = [10.0]
    store = PathStore(clock=lambda: clock[0])
    old = store.replace(np.array(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))), "map")
    new = store.replace(np.array(((9.0, 1.0, 0.0),)), "odom")
    assert new.generation == old.generation + 1
    assert store.snapshot().frame_id == "odom"
    assert np.array_equal(store.snapshot().points, ((9.0, 1.0, 0.0),))
    assert not store.snapshot().points.flags.writeable


@pytest.mark.parametrize("intervals", ((1.0, 1.0, 1.0), (3.0, 3.0), (0.2, 2.7, 1.1, 4.0)))
def test_update_intervals_refresh_monotonic_receive_time(intervals):
    clock = [0.0]
    store = PathStore(timeout_sec=6.5, clock=lambda: clock[0])
    generation = 0
    for interval in intervals:
        clock[0] += interval
        generation = store.replace(np.array(((1.0, 0.0, 0.0),)), "map").generation
        assert store.state()[0] == "ACTIVE"
    assert generation == len(intervals)


def test_timeout_boundary_is_active_then_stale_after_6_5_seconds():
    clock = [100.0]
    store = PathStore(timeout_sec=6.5, clock=lambda: clock[0])
    store.replace(np.array(((1.0, 0.0, 0.0),)), "map")
    clock[0] = 106.5
    assert store.state()[0] == "ACTIVE"
    clock[0] = 106.500001
    assert store.state()[0] == "PATH STALE"


def test_empty_path_clears_current_path():
    store = PathStore()
    store.replace(np.array(((1.0, 0.0, 0.0),)), "map")
    store.replace(np.empty((0, 3)), "map")
    assert store.state()[0] == "NO PATH"


def test_explicit_clear_removes_previous_path_immediately():
    store = PathStore()
    store.replace(np.array(((1.0, 0.0, 0.0),)), "map")
    store.clear()
    assert store.state()[0] == "NO PATH"
    assert store.snapshot() is None


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_nonfinite_path_is_rejected_without_replacing_valid_path(bad):
    store = PathStore()
    valid = store.replace(np.array(((1.0, 0.0, 0.0),)), "map")
    with pytest.raises(ValueError):
        store.replace(np.array(((bad, 0.0, 0.0),)), "map")
    assert store.snapshot().generation == valid.generation


@pytest.mark.parametrize("frame", ("", "/map", "map frame", "map:bad"))
def test_invalid_frame_id_is_rejected(frame):
    store = PathStore()
    with pytest.raises(ValueError):
        store.replace(np.empty((0, 3)), frame)


def test_path_message_conversion_preserves_frame_and_xyz():
    position = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        poses=[SimpleNamespace(pose=SimpleNamespace(position=position))],
    )
    points, frame = path_message_to_array(message)
    assert frame == "map"
    assert np.array_equal(points, ((1.0, 2.0, 3.0),))
