from types import SimpleNamespace

from ar_path_viewer.waypoint_progress import WaypointProgressGate


def follower_state(
    *,
    stamp=100.0,
    route="a" * 64,
    index=2,
    number=12,
    mode="to_pose",
    generation=4,
    activity=1,
    allowed=True,
):
    seconds = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="map",
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=int((stamp - seconds) * 1e9),
            ),
        ),
        schema_version=1,
        route_sha256=route,
        next_waypoint_index=index,
        next_waypoint_number=number,
        navigation_mode=mode,
        goal_generation=generation,
        activity=activity,
        reference_allowed=allowed,
        waypoint_count=20,
    )


def label(gate, *, now=100.1, path_generation=2, remaining=2.34, anchor=0.1):
    return gate.label(now, path_generation, remaining, anchor, 1.0)


def test_to_pose_uses_only_plan_received_after_follower_goal_state():
    gate = WaypointProgressGate()
    assert gate.update(follower_state(), 100.0, path_generation=1)
    assert label(gate, path_generation=1) is None
    assert gate.reason == "waiting_for_plan_after_goal_change"
    assert label(gate, path_generation=2) == "PLAN LEFT: 2.3 m"


def test_near_waypoint_is_formatted_in_centimetres():
    gate = WaypointProgressGate()
    gate.update(follower_state(), 100.0, path_generation=3)
    assert label(gate, path_generation=4, remaining=0.824) == (
        "PLAN LEFT: 82 cm"
    )


def test_through_poses_is_labeled_only_as_plan_remaining_distance():
    gate = WaypointProgressGate()
    gate.update(follower_state(mode="through_poses"), 100.0, path_generation=1)
    assert label(gate, path_generation=2) == "PLAN LEFT: 2.3 m"
    assert gate.reason == "valid"


def test_goal_change_invalidates_old_plan_until_new_generation_arrives():
    gate = WaypointProgressGate()
    gate.update(follower_state(index=2, generation=4), 100.0, path_generation=3)
    assert label(gate, path_generation=4) is not None

    gate.update(
        follower_state(index=3, number=13, generation=5, stamp=100.2),
        100.2,
        path_generation=4,
    )
    assert label(gate, now=100.3, path_generation=4) is None
    assert label(gate, now=100.3, path_generation=5) == (
        "PLAN LEFT: 2.3 m"
    )


def test_pause_stale_state_and_distant_path_fail_closed():
    gate = WaypointProgressGate()
    assert not gate.update(
        follower_state(activity=4, allowed=False), 100.0, path_generation=1
    )
    assert label(gate, path_generation=2) is None
    assert gate.navigation_hold_status(100.1) == "PAUSED"
    assert gate.navigation_hold_status(103.0) is None

    gate.update(follower_state(stamp=100.0), 100.0, path_generation=2)
    assert label(gate, now=103.0, path_generation=3) is None
    assert "stale" in gate.reason

    gate.update(follower_state(stamp=104.0), 104.0, path_generation=3)
    assert label(
        gate, now=104.1, path_generation=4, anchor=1.01
    ) is None
    assert gate.reason == "path_progress_not_reliable"


def test_guidance_authorization_uses_same_goal_and_freshness_gate():
    gate = WaypointProgressGate()
    gate.update(follower_state(), 100.0, path_generation=7)
    assert not gate.authorizes_plan(100.1, 7)
    assert gate.authorizes_plan(100.1, 8)
    assert not gate.authorizes_plan(103.0, 8)


def test_person_search_and_resume_expose_truthful_hold_status():
    gate = WaypointProgressGate()
    assert not gate.update(
        follower_state(activity=6, allowed=False), 100.0, path_generation=8
    )
    assert gate.navigation_hold_status(100.1) == "PERSON SEARCH"
    assert not gate.authorizes_plan(100.1, 9)

    assert gate.update(
        follower_state(activity=1, allowed=True, stamp=100.2),
        100.2,
        path_generation=9,
    )
    assert gate.navigation_hold_status(100.3) is None
    assert not gate.authorizes_plan(100.3, 9)
    assert gate.authorizes_plan(100.3, 10)
