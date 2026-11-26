"""Strict, read-only completion evidence for the legacy Follower."""

from dataclasses import dataclass
import re

from action_msgs.msg import GoalStatus


_REACHED_BY_NAV2 = re.compile(r'\[WP:(\d+)\] reached by Nav2')


@dataclass(frozen=True)
class LegacyCompletion:
    """Identity of one independently verified completed route."""

    route_sha256: str
    goal_generation: int
    goal_uuid: str
    final_index: int
    final_number: int
    waypoint_count: int


class LegacyCompletionGate:
    """
    Join final-goal, Nav2 success, and ordered Follower log evidence.

    No single STOPPED state, old action status, or log line can complete a
    route.  Evidence is bound to the currently verified final goal UUID and
    must arrive within a short monotonic-time window.
    """

    def __init__(self, expected_node_name: str, evidence_window_sec: float = 2.0):
        self.expected_node_name = str(expected_node_name).lstrip('/')
        self.evidence_window_sec = float(evidence_window_sec)
        if not self.expected_node_name or not 0.25 <= self.evidence_window_sec <= 10.0:
            raise ValueError('invalid legacy completion gate configuration')
        self.reset('waiting_for_final_goal')

    def reset(self, reason: str) -> None:
        self._binding = None
        self._succeeded_at = None
        self._reached_at = None
        self._all_complete_at = None
        self.reason = str(reason)

    def bind_final_goal(
        self, *, route_sha256: str, goal_generation: int, goal_uuid: str,
        final_index: int, final_number: int, waypoint_count: int,
    ) -> None:
        binding = LegacyCompletion(
            str(route_sha256), int(goal_generation), str(goal_uuid),
            int(final_index), int(final_number), int(waypoint_count),
        )
        if (
            len(binding.route_sha256) != 64
            or len(binding.goal_uuid) != 32
            or binding.waypoint_count < 1
            or binding.final_index != binding.waypoint_count - 1
            or binding.goal_generation < 1
        ):
            self.reset('invalid_final_goal_binding')
            return
        if binding != self._binding:
            self.reset('waiting_for_final_goal_evidence')
            self._binding = binding

    def observe_status(self, goal_uuid: str, status: int, now: float) -> None:
        if self._binding is None or str(goal_uuid) != self._binding.goal_uuid:
            return
        status = int(status)
        if status == GoalStatus.STATUS_SUCCEEDED:
            # Action status is a snapshot and may be published more than
            # once. Never replace the FIRST observed success time,
            # especially after follower evidence completed the route.
            if self._succeeded_at is None:
                self._succeeded_at = float(now)
            self.reason = 'waiting_for_follower_completion_log'
        elif status in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
            self.reset('final_goal_not_succeeded')

    def observe_log(self, node_name: str, message: str, now: float) -> None:
        if self._binding is None:
            return
        if str(node_name).lstrip('/') != self.expected_node_name:
            return
        text = str(message).strip()
        reached = _REACHED_BY_NAV2.fullmatch(text)
        if reached is not None:
            if int(reached.group(1)) != self._binding.final_number:
                self.reset('follower_reached_nonfinal_waypoint')
                return
            # Preserve the first final-WP evidence; a duplicate log
            # must not invalidate a previously proven completion.
            if self._reached_at is None:
                self._reached_at = float(now)
            self.reason = 'waiting_for_all_waypoints_completed'
        elif text == 'All waypoints completed':
            # The explicit final-WP log must precede this line.  This rejects
            # stale/other-route completion text even if its node name matches.
            if self._reached_at is None:
                self.reason = 'all_waypoints_without_final_reached'
                return
            if self._all_complete_at is None:
                self._all_complete_at = float(now)

    def completion(self, now: float) -> LegacyCompletion | None:
        if self._binding is None:
            return None
        evidence = (
            self._succeeded_at, self._reached_at, self._all_complete_at,
        )
        if any(value is None for value in evidence):
            return None
        first, last = min(evidence), max(evidence)
        if (
            not self._succeeded_at <= self._reached_at <= self._all_complete_at
            or last - first > self.evidence_window_sec
            or float(now) < last
        ):
            self.reset('completion_evidence_expired_or_out_of_order')
            return None
        self.reason = 'legacy_route_completion_verified'
        return self._binding
