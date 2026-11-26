"""Completion needs one final goal and three ordered independent facts."""

from action_msgs.msg import GoalStatus

from ar_reference_planner.legacy_completion import LegacyCompletionGate


ROUTE = 'a' * 64
GOAL = '01' * 16


def _bound_gate():
    gate = LegacyCompletionGate('koko_waypoint_follower', 2.0)
    gate.bind_final_goal(
        route_sha256=ROUTE, goal_generation=10, goal_uuid=GOAL,
        final_index=9, final_number=10, waypoint_count=10,
    )
    return gate


def test_real_wp10_event_order_verifies_completion():
    gate = _bound_gate()
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 100.000)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 100.009)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 100.019)

    result = gate.completion(100.2)
    assert result is not None
    assert result.goal_uuid == GOAL
    assert result.final_number == 10
    # A completion is an intentional latch until a verified new goal resets it.
    assert gate.completion(300.0) == result


def test_stopped_or_success_alone_never_completes():
    gate = _bound_gate()
    assert gate.completion(1.0) is None
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 2.0)
    assert gate.completion(2.1) is None


def test_other_goal_route_or_waypoint_cannot_complete():
    gate = _bound_gate()
    gate.observe_status('02' * 16, GoalStatus.STATUS_SUCCEEDED, 10.0)
    gate.observe_log('other_follower', '[WP:10] reached by Nav2', 10.1)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:9] reached by Nav2', 10.2)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 10.3)
    assert gate.completion(10.4) is None


def test_cancel_abort_and_out_of_order_logs_fail_closed():
    for status in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
        gate = _bound_gate()
        gate.observe_status(GOAL, status, 20.0)
        gate.observe_log('koko_waypoint_follower',
                         '[WP:10] reached by Nav2', 20.1)
        gate.observe_log('koko_waypoint_follower',
                         'All waypoints completed', 20.2)
        assert gate.completion(20.3) is None

    gate = _bound_gate()
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 30.0)
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 30.1)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 30.2)
    assert gate.completion(30.3) is None


def test_evidence_outside_window_is_rejected():
    gate = _bound_gate()
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 40.0)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 43.0)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 43.1)
    assert gate.completion(43.2) is None


def test_duplicate_success_status_and_logs_do_not_erase_completion():
    """The same SUCCEEDED can be re-announced after follower completion."""
    gate = _bound_gate()
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 100.000)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 100.009)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 100.019)
    completed = gate.completion(100.020)
    assert completed is not None

    # Repeat the SAME action status after the original completion.  It must
    # not overwrite the first success timestamp and destroy the result.
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 100.220)
    assert gate.completion(100.230) == completed

    # Duplicate messages from the same Follower must be harmless too.
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 100.300)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 100.320)
    assert gate.completion(100.400) == completed

    # A genuinely contradictory terminal status must still invalidate it.
    gate.observe_status(GOAL, GoalStatus.STATUS_ABORTED, 100.500)
    assert gate.completion(100.510) is None


def test_repeated_success_earliest_evidence_kept_before_logs():
    gate = _bound_gate()
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 200.000)
    gate.observe_status(GOAL, GoalStatus.STATUS_SUCCEEDED, 200.100)
    gate.observe_log('koko_waypoint_follower',
                     '[WP:10] reached by Nav2', 200.110)
    gate.observe_log('koko_waypoint_follower',
                     'All waypoints completed', 200.120)
    assert gate.completion(200.130) is not None
