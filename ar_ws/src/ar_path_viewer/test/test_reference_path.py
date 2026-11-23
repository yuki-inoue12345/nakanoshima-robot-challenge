from types import SimpleNamespace

import numpy as np

from ar_path_viewer.reference_path import ReferencePathGate


ROUTE = "a" * 64


def stamp(value):
    seconds = int(value)
    return SimpleNamespace(
        sec=seconds,
        nanosec=int(round((value - seconds) * 1e9)),
    )


def follower(
    *, now=100.0, route=ROUTE, index=2, generation=7,
    activity=1, allowed=True, mode="to_pose",
):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map", stamp=stamp(now)),
        schema_version=1,
        route_sha256=route,
        next_waypoint_index=index,
        next_waypoint_number=12 + index,
        navigation_mode=mode,
        goal_generation=generation,
        activity=activity,
        reference_allowed=allowed,
        safe_section_end_index=8,
        waypoint_count=10,
    )


def path_message(points, *, frame="map", now=100.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame, stamp=stamp(now)),
        poses=[
            SimpleNamespace(
                header=SimpleNamespace(frame_id=frame),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y, z=z)
                )
            )
            for x, y, z in points
        ],
    )


def reference(
    *, now=100.0, route=ROUTE, index=2, generation=7,
    points=((1.0, 0.0, 0.0), (11.0, 0.0, 0.0)), valid=True,
    reason="",
):
    length = sum(
        np.linalg.norm(np.asarray(b[:2]) - np.asarray(a[:2]))
        for a, b in zip(points, points[1:])
    )
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map", stamp=stamp(now)),
        schema_version=2,
        valid=valid,
        invalid_reason=reason,
        route_sha256=route,
        next_waypoint_index=index,
        next_waypoint_number=12 + index,
        navigation_mode="to_pose",
        goal_generation=generation,
        safe_section_end_index=8,
        waypoint_count=10,
        planner_generation=3,
        follower_state_stamp=stamp(now),
        path_length_m=length,
        path=path_message(points, now=now),
    )


def test_atomic_reference_is_accepted_only_after_matching_follower_state():
    now = [10.0]
    gate = ReferencePathGate(clock=lambda: now[0])

    assert not gate.update_reference(reference(), 100.0)
    assert gate.reason == "no_current_follower_state"
    assert gate.update_follower(follower(), 100.0)
    assert gate.update_reference(reference(), 100.0)
    state, snapshot, reason = gate.state(100.0, now[0])

    assert state == "ACTIVE"
    assert reason == "valid"
    assert snapshot is not None
    assert len(snapshot.points) == 2
    assert gate.metadata.declared_length_m == 10.0


def test_old_reference_schema_is_rejected_without_affecting_follower_schema():
    gate = ReferencePathGate()
    assert gate.update_follower(follower(), 100.0)
    old_reference = reference()
    old_reference.schema_version = 1
    assert not gate.update_reference(old_reference, 100.0)
    assert gate.reason == 'unsupported_reference_schema=1'


def test_route_index_and_goal_generation_mismatches_fail_closed():
    for update in (
        reference(route="b" * 64),
        reference(index=3),
        reference(generation=8),
    ):
        gate = ReferencePathGate()
        assert gate.update_follower(follower(), 100.0)
        assert not gate.update_reference(update, 100.0)
        assert gate.reason == "reference_identity_mismatch"
        assert gate.state(100.0)[0] == "INVALID"


def test_pause_person_search_and_route_change_clear_old_reference_immediately():
    for activity in (3, 4, 6):
        gate = ReferencePathGate()
        assert gate.update_follower(follower(), 100.0)
        assert gate.update_reference(reference(), 100.0)
        assert not gate.update_follower(
            follower(now=101.0, activity=activity, allowed=False), 101.0
        )
        assert gate.state(101.0)[0] == "INVALID"
        assert "reference_disallowed" in gate.reason

    gate = ReferencePathGate()
    assert gate.update_follower(follower(), 100.0)
    assert gate.update_reference(reference(), 100.0)
    assert gate.update_follower(follower(now=101.0, route="c" * 64), 101.0)
    assert gate.state(101.0)[0] == "INVALID"
    assert gate.reason == "reference_identity_changed"


def test_empty_invalid_and_stale_reference_never_remain_visible():
    now = [10.0]
    gate = ReferencePathGate(timeout_sec=6.5, clock=lambda: now[0])
    assert gate.update_follower(follower(), 100.0)
    assert gate.update_reference(reference(), 100.0)

    invalid = reference(valid=False, reason="reference_planner_not_active")
    assert not gate.update_reference(invalid, 100.1)
    assert gate.state(100.1, now[0])[0] == "INVALID"
    assert gate.reason == "reference_planner_not_active"

    assert gate.update_follower(follower(now=101.0), 101.0)
    assert gate.update_reference(reference(now=101.0), 101.0)
    now[0] = 16.6
    assert gate.state(101.5, now[0])[0] == "INVALID"
    assert gate.reason == "reference_path_expired"


def test_stale_follower_bad_frame_and_length_mismatch_are_rejected():
    gate = ReferencePathGate()
    assert not gate.update_follower(follower(now=90.0), 100.0)
    assert "follower_state_stale" in gate.reason

    bad_follower = follower()
    bad_follower.header.frame_id = "odom"
    assert not gate.update_follower(bad_follower, 100.0)
    assert "follower_frame" in gate.reason

    assert gate.update_follower(follower(), 100.0)
    bad_length = reference()
    bad_length.path_length_m = 99.0
    assert not gate.update_reference(bad_length, 100.0)
    assert "reference_path_length_mismatch" in gate.reason


def test_through_poses_reference_identity_is_supported():
    gate = ReferencePathGate()
    active = follower(mode="through_poses")
    envelope = reference()
    envelope.navigation_mode = "through_poses"
    assert gate.update_follower(active, 100.0)
    assert gate.update_reference(envelope, 100.0)
    assert gate.state(100.0)[0] == "ACTIVE"


def test_reference_waits_for_same_identity_follower_delivery_reordering():
    gate = ReferencePathGate(clock=lambda: 10.0)
    assert gate.update_follower(follower(now=100.0), 100.0)
    assert gate.update_reference(reference(now=100.0), 100.0)
    previous = gate.path_store.snapshot()

    # The planner result is based on a state that DDS has not delivered to
    # this subscriber yet.  The currently validated route remains visible.
    assert not gate.update_reference(reference(now=100.5), 100.5)
    assert gate.reason == "waiting_for_matching_follower_state"
    assert gate.state(100.5)[1] is previous

    # Matching fresh state activates the pending message after full
    # revalidation; no timeout relaxation or guessed identity is used.
    assert gate.update_follower(follower(now=100.6), 100.6)
    state, current, reason = gate.state(100.6)
    assert state == "ACTIVE"
    assert reason == "valid"
    assert current is not previous


def test_periodic_same_goal_heartbeats_keep_reference_visible_but_gap_expires():
    gate = ReferencePathGate(follower_timeout_sec=2.5, clock=lambda: 10.0)
    assert gate.update_follower(follower(now=100.0), 100.0)
    assert gate.update_reference(reference(now=100.0), 100.0)

    for now in (100.8, 101.6, 102.4, 103.2):
        assert gate.update_follower(follower(now=now), now)
        assert gate.state(now)[0] == 'ACTIVE'

    assert gate.state(105.71)[0] == 'INVALID'
    assert 'follower_state_stale' in gate.reason
