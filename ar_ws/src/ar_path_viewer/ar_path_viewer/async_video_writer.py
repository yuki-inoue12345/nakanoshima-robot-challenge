"""Bounded asynchronous final-frame MP4 recording."""

import math
import threading
import time
from collections import deque

from .paced_video_writer import PacedVideoWriter


class AsyncPacedVideoWriter:
    """Keep only the newest submitted rendering while encoding off-thread.

    The queue is intentionally one frame.  If the encoder or disk stalls, the
    camera/AR callback remains nonblocking and intermediate renderings are
    counted as replaced.  PacedVideoWriter preserves wall-time duration with
    explicitly counted duplicate output slots when encoding recovers.
    """

    def __init__(self, writer, fps, *, clock=time.monotonic):
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("video FPS must be finite and positive")
        self._paced = PacedVideoWriter(writer, self.fps, clock=clock)
        self._condition = threading.Condition()
        self._latest = None
        self._submitted_generation = 0
        self._consumed_generation = 0
        self._stop = False
        self._released = False
        self._submitted_frames = 0
        self._replaced_frames = 0
        self.error_reason = ""
        self._encode_durations = deque(maxlen=200)
        self._metrics_lock = threading.Lock()
        self._frames_written = 0
        self._real_frames_written = 0
        self._duplicate_frames_written = 0
        self._thread = threading.Thread(
            target=self._run, name="ar-video-writer", daemon=True
        )
        self._thread.start()

    def write(self, frame):
        copied = frame.copy()
        with self._condition:
            if self._stop or self.error_reason:
                return False
            if self._submitted_generation > self._consumed_generation:
                self._replaced_frames += 1
            self._submitted_frames += 1
            self._submitted_generation += 1
            self._latest = copied
            self._condition.notify_all()
            return True

    def _encode(self, frame):
        started = time.monotonic()
        self._paced.write(frame)
        with self._metrics_lock:
            self._update_paced_snapshot()
            self._encode_durations.append(time.monotonic() - started)

    def _update_paced_snapshot(self):
        """Copy writer-thread-only counters for nonblocking diagnostics."""
        self._frames_written = self._paced.frames_written
        self._real_frames_written = self._paced.real_frames_written
        self._duplicate_frames_written = self._paced.duplicate_frames_written

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop or (
                            self._submitted_generation
                            > self._consumed_generation
                        )
                    )
                    if (
                        self._submitted_generation
                        > self._consumed_generation
                    ):
                        generation = self._submitted_generation
                        frame = self._latest
                        self._consumed_generation = generation
                    else:
                        frame = None
                    stop = self._stop
                if frame is not None:
                    self._encode(frame)
                if stop:
                    # Include the elapsed tail without inventing a new AR
                    # rendering. Missing fixed-rate MP4 slots are explicit
                    # duplicates of the last completed rendering.
                    started = time.monotonic()
                    self._paced.fill_until_now()
                    with self._metrics_lock:
                        self._update_paced_snapshot()
                        self._encode_durations.append(
                            time.monotonic() - started
                        )
                    break
        except BaseException as exc:
            with self._condition:
                self.error_reason = (
                    f"video_writer_exception_{type(exc).__name__}"
                )
        finally:
            try:
                self._paced.release()
            except BaseException as exc:
                with self._condition:
                    if not self.error_reason:
                        self.error_reason = (
                            f"video_release_exception_{type(exc).__name__}"
                        )

    @property
    def frames_written(self):
        with self._metrics_lock:
            return self._frames_written

    @property
    def real_frames_written(self):
        with self._metrics_lock:
            return self._real_frames_written

    @property
    def duplicate_frames_written(self):
        with self._metrics_lock:
            return self._duplicate_frames_written

    @property
    def submitted_frames(self):
        with self._condition:
            return self._submitted_frames

    @property
    def replaced_frames(self):
        with self._condition:
            return self._replaced_frames

    @property
    def encode_duration_sec(self):
        with self._metrics_lock:
            values = tuple(self._encode_durations)
        return 0.0 if not values else values[-1]

    @property
    def max_encode_duration_sec(self):
        with self._metrics_lock:
            values = tuple(self._encode_durations)
        return 0.0 if not values else max(values)

    @property
    def worker_alive(self):
        return self._thread.is_alive()

    def release(self):
        with self._condition:
            if self._released:
                return
            self._released = True
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive() and not self.error_reason:
            self.error_reason = "video_writer_shutdown_timeout"
