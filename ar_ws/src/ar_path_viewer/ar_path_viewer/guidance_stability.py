"""Fail-closed stabilization for UI guidance derived from the latest Path."""

from dataclasses import dataclass
import math

from .offscreen_guidance import OffscreenGuidance


@dataclass(frozen=True)
class StableGuidance:
    """One render decision; a missing guide never reuses an old direction."""

    guidance: OffscreenGuidance | None
    status: str


class GuidanceStabilityGate:
    """Suppress a changed direction briefly instead of displaying stale advice.

    The gate never smooths Path coordinates and never keeps the former arrow
    after a new Path requests another direction.  A changed direction must
    remain the latest decision for ``hold_sec`` before it is rendered.
    """

    def __init__(self, hold_sec: float = 0.35):
        if not math.isfinite(hold_sec) or hold_sec < 0.0:
            raise ValueError('guidance hold time must be finite and non-negative')
        self.hold_sec = float(hold_sec)
        self._active_direction: str | None = None
        self._pending_direction: str | None = None
        self._pending_since: float | None = None

    def clear(self) -> None:
        self._active_direction = None
        self._pending_direction = None
        self._pending_since = None

    def update(
        self, guidance: OffscreenGuidance | None, now_sec: float
    ) -> StableGuidance:
        """Return only a current direction that has passed the hold interval."""
        if guidance is None or not math.isfinite(now_sec):
            self.clear()
            return StableGuidance(None, '')

        direction = str(guidance.direction)
        if direction == self._active_direction:
            self._pending_direction = None
            self._pending_since = None
            return StableGuidance(guidance, '')

        if direction != self._pending_direction:
            # The former arrow is deliberately removed immediately.
            self._pending_direction = direction
            self._pending_since = float(now_sec)
            if self.hold_sec > 0.0:
                return StableGuidance(None, 'DIRECTION CHECK')

        elapsed = float(now_sec) - float(self._pending_since)
        if elapsed < 0.0 or elapsed + 1e-9 < self.hold_sec:
            return StableGuidance(None, 'DIRECTION CHECK')

        self._active_direction = direction
        self._pending_direction = None
        self._pending_since = None
        return StableGuidance(guidance, '')
