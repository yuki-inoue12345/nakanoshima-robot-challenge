"""Conservative, read-only actual-in-place-rotation hint from stable TF poses.

Unlike TURN LEFT/RIGHT (where the live Nav2 Path points), ROTATE LEFT/RIGHT
only reports a sustained measured robot yaw change with little translation.
It cannot command rotation or predict what Nav2 intends to do next.
"""

from collections import deque
import math


class RotationMotionGate:
    def __init__(
        self,
        window_sec: float = 0.55,
        minimum_angle_deg: float = 11.0,
        maximum_translation_m: float = 0.065,
        minimum_duration_sec: float = 0.28,
        maximum_rate_deg_sec: float = 150.0,
    ):
        self.window_sec = float(window_sec)
        self.minimum_angle_deg = float(minimum_angle_deg)
        self.maximum_translation_m = float(maximum_translation_m)
        self.minimum_duration_sec = float(minimum_duration_sec)
        self.maximum_rate_deg_sec = float(maximum_rate_deg_sec)
        if (
            not all(math.isfinite(v) for v in (
                self.window_sec, self.minimum_angle_deg,
                self.maximum_translation_m, self.minimum_duration_sec,
                self.maximum_rate_deg_sec,
            ))
            or self.window_sec <= self.minimum_duration_sec
            or self.minimum_duration_sec <= 0.0
            or self.minimum_angle_deg <= 0.0
            or self.maximum_translation_m <= 0.0
            or self.maximum_rate_deg_sec <= 0.0
        ):
            raise ValueError("invalid rotation observation thresholds")
        self.clear()

    def clear(self):
        self._samples = deque()

    def update(self, pose, now_sec: float):
        """Return ROTATE LEFT/RIGHT only for verified observable motion."""
        if pose is None:
            self.clear()
            return None
        try:
            x, y, yaw = (float(value) for value in pose)
            now = float(now_sec)
        except (ValueError, TypeError):
            self.clear()
            return None
        if not all(math.isfinite(v) for v in (x, y, yaw, now)):
            self.clear()
            return None
        if self._samples and (
            now <= self._samples[-1][0]
            or now - self._samples[-1][0] > 0.20
        ):
            self.clear()
        self._samples.append((now, x, y, yaw))
        # Keep the oldest observation in approximately one complete window.
        while (
            len(self._samples) > 2
            and now - self._samples[1][0] >= self.window_sec
        ):
            self._samples.popleft()
        start = self._samples[0]
        elapsed = now - start[0]
        if elapsed < self.minimum_duration_sec:
            return None
        translation = math.hypot(x - start[1], y - start[2])
        angle = math.degrees(math.atan2(
            math.sin(yaw - start[3]), math.cos(yaw - start[3]),
        ))
        rate = abs(angle) / elapsed
        increments = [
            math.degrees(math.atan2(
                math.sin(second[3] - first[3]),
                math.cos(second[3] - first[3]),
            ))
            for first, second in zip(
                list(self._samples)[:-1], list(self._samples)[1:]
            )
        ]
        # Reject AMCL/TF heading jumps and sign-changing yaw noise. An
        # actual rotation needs several consistent small observations.
        signed_evidence = [
            delta for delta in increments if abs(delta) >= 0.15
        ]
        if (
            len(signed_evidence) < 3
            or any(abs(delta) > 8.0 for delta in increments)
            or abs(sum(signed_evidence)) < 0.85 * sum(
                abs(delta) for delta in signed_evidence
            )
        ):
            return None
        if (
            translation > self.maximum_translation_m
            or abs(angle) < self.minimum_angle_deg
            or rate > self.maximum_rate_deg_sec
        ):
            return None
        return "ROTATE LEFT" if angle > 0.0 else "ROTATE RIGHT"


class OdomRotationMotionGate:
    """Measured filtered-odometry in-place yaw, never commanded velocity.

    Reset on linear motion, time gaps, invalid values or reversed yaw. A recent
    continuous sample run is required; no extrapolation from old odometry.
    """

    def __init__(
        self,
        minimum_duration_sec: float = 0.18,
        minimum_angle_deg: float = 6.0,
        minimum_samples: int = 4,
    ):
        self.minimum_duration_sec = float(minimum_duration_sec)
        self.minimum_angle_deg = float(minimum_angle_deg)
        self.minimum_samples = int(minimum_samples)
        if (
            not math.isfinite(self.minimum_duration_sec)
            or not math.isfinite(self.minimum_angle_deg)
            or self.minimum_duration_sec <= 0.0
            or self.minimum_angle_deg <= 0.0
            or self.minimum_samples < 3
        ):
            raise ValueError('invalid odometry rotation thresholds')
        self.clear()

    def clear(self):
        self._samples = deque()
        self._direction = None

    def update(self, linear_m_s, angular_rad_s, stamp_sec):
        try:
            v, w, now = (float(linear_m_s), float(angular_rad_s),
                         float(stamp_sec))
        except (TypeError, ValueError):
            self.clear()
            return None
        if (not all(math.isfinite(x) for x in (v, w, now))
                or abs(v) >= 0.12 or abs(w) < 0.20 or abs(w) > 1.6):
            self.clear()
            return None
        if self._samples:
            prev_t, prev_w = self._samples[-1]
            if (now <= prev_t or now - prev_t > 0.20
                    or (w > 0) != (prev_w > 0)):
                self.clear()
        self._samples.append((now, w))
        while len(self._samples) > 2 and now - self._samples[1][0] >= 0.80:
            self._samples.popleft()
        elapsed = now - self._samples[0][0]
        if (
            len(self._samples) < self.minimum_samples
            or elapsed < self.minimum_duration_sec
        ):
            return None
        signed_angle = sum(
            (t1 - t0) * (w0 + w1) / 2.0
            for (t0, w0), (t1, w1) in zip(
                list(self._samples)[:-1], list(self._samples)[1:]
            )
        )
        if abs(signed_angle) < math.radians(self.minimum_angle_deg):
            return None
        self._direction = 'ROTATE LEFT' if w > 0 else 'ROTATE RIGHT'
        return self._direction

    def direction(self, now_sec):
        if not self._samples:
            return None
        age = float(now_sec) - self._samples[-1][0]
        if not math.isfinite(age) or age < -0.05 or age > 0.22:
            self.clear()
            return None
        return self._direction
