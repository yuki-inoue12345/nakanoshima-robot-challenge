"""Bounded, nonblocking camera-frame delay for exact-time Costmap projection.

The Viewer must wait for TF for the image exposure instant rather than asking
for the latest transform.  Only opt-in perception sessions use this queue.
Neither missing TF nor stalled callbacks can extend a frame's finite lifetime.
"""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DelayedFrame:
    image: object
    timing: object


class PerceptionFrameDelay:
    """Release captured frames in order once a small TF-arrival margin passes.

    This queue never sleeps or looks ahead in time. If a frame expires before
    being displayed, it is discarded, rather than assigned a newer timestamp.
    """

    def __init__(self, delay_sec=0.12, max_age_sec=0.30, max_frames=8):
        if (not math.isfinite(delay_sec) or
                not math.isfinite(max_age_sec) or
                not 0.0 <= delay_sec < max_age_sec <= 0.40 or
                not isinstance(max_frames, int) or max_frames < 2):
            raise ValueError('invalid perception frame delay limits')
        self.delay_sec = float(delay_sec)
        self.max_age_sec = float(max_age_sec)
        self.max_frames = max_frames
        self._pending = deque()
        self.discarded = 0

    def clear(self):
        self._pending.clear()

    @property
    def pending_count(self):
        return len(self._pending)

    def add(self, image, timing):
        if image is None or not getattr(timing, 'usable', False):
            self.clear()
            return
        captured = float(timing.monotonic_finished_sec)
        if not math.isfinite(captured):
            self.clear()
            return
        if (self._pending and captured <=
                self._pending[-1].timing.monotonic_finished_sec):
            # Monotonic reset/discontinuity: no mixing clock generations.
            self.clear()
        self._pending.append(DelayedFrame(image=image, timing=timing))
        while len(self._pending) > self.max_frames:
            self._pending.popleft()
            self.discarded += 1

    def peek(self, now_monotonic):
        """Return the oldest eligible frame without consuming it.

        The Viewer keeps this exact image/timestamp pair while an exact-time
        TF is still arriving.  Expiration remains finite and is evaluated on
        every call.
        """
        if not math.isfinite(now_monotonic):
            self.clear()
            return None
        while self._pending:
            item = self._pending[0]
            age = now_monotonic - item.timing.monotonic_finished_sec
            if age < 0.0:
                self.clear()
                return None
            if age > self.max_age_sec:
                self._pending.popleft()
                self.discarded += 1
                continue
            if age >= self.delay_sec:
                return item
            break
        return None

    def consume(self, item):
        """Remove exactly the frame returned by ``peek``."""
        if not self._pending or self._pending[0] is not item:
            return False
        self._pending.popleft()
        return True

    def take(self, now_monotonic):
        """Compatibility helper for callers that do not require TF retry."""
        item = self.peek(now_monotonic)
        if item is not None:
            self.consume(item)
        return item
