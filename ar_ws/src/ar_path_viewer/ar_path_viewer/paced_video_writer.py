"""Constant-rate MP4 pacing for a variable-rate Viewer callback."""

import math
import time


class PacedVideoWriter:
    """Keep MP4 duration close to monotonic session time at configured FPS.

    OpenCV's VideoWriter has no per-frame timestamps. Writing once per late
    callback therefore makes playback run faster than the real session. This
    wrapper fills missed slots with the last frame actually rendered; it never
    invents navigation geometry or changes the current frame.
    """

    def __init__(self, writer, fps: float, clock=time.monotonic) -> None:
        self.writer = writer
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("video FPS must be finite and positive")
        self.clock = clock
        self.started_at = None
        self.frames_written = 0
        self.real_frames_written = 0
        self.duplicate_frames_written = 0
        self._last_frame = None

    def write(self, frame) -> None:
        now = float(self.clock())
        if self.started_at is None:
            self.started_at = now
            self.writer.write(frame)
            self.frames_written = 1
            self.real_frames_written = 1
            self._last_frame = frame.copy()
            return
        target_count = max(
            1,
            int(math.floor(max(0.0, now - self.started_at) * self.fps)) + 1,
        )
        missing = target_count - self.frames_written
        if missing > 0:
            for _ in range(max(0, missing - 1)):
                self.writer.write(self._last_frame)
                self.duplicate_frames_written += 1
            self.writer.write(frame)
            self.frames_written = target_count
            self.real_frames_written += 1
        # Retain the newest real rendering even when this callback arrived
        # before the next output slot.
        self._last_frame = frame.copy()

    def fill_until_now(self) -> None:
        """Fill elapsed fixed-rate slots with the last real rendering."""
        if self.started_at is None or self._last_frame is None:
            return
        now = float(self.clock())
        target_count = max(
            1,
            int(math.floor(max(0.0, now - self.started_at) * self.fps)) + 1,
        )
        while self.frames_written < target_count:
            self.writer.write(self._last_frame)
            self.frames_written += 1
            self.duplicate_frames_written += 1

    def release(self) -> None:
        self.writer.release()
