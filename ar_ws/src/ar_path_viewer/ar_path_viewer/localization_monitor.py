"""ROS-independent AMCL quality gating with conservative recovery hysteresis."""

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np


LOCALIZATION_OK = "LOCALIZATION OK"
LOCALIZING = "LOCALIZING"
LOCALIZATION_UNSTABLE = "LOCALIZATION UNSTABLE"


@dataclass(frozen=True)
class LocalizationThresholds:
    expected_frame: str = "map"
    pose_timeout_sec: float = 3.0
    stamp_max_age_sec: float = 3.0
    future_tolerance_sec: float = 1.5
    tf_max_age_sec: float = 3.0
    max_position_stddev_m: float = 0.75
    max_yaw_stddev_rad: float = 0.75
    jump_distance_m: float = 1.5
    jump_yaw_rad: float = 1.2
    max_linear_speed_mps: float = 1.5
    max_angular_speed_radps: float = 2.0
    stationary_hold_distance_m: float = 0.5
    stationary_hold_yaw_rad: float = 0.5
    recovery_samples: int = 1
    recovery_sec: float = 2.0

    def validate(self) -> None:
        if not self.expected_frame or self.expected_frame.startswith("/"):
            raise ValueError("localization_global_frame must be a relative frame name")
        positive = {
            "localization_pose_timeout_sec": self.pose_timeout_sec,
            "localization_stamp_max_age_sec": self.stamp_max_age_sec,
            "localization_future_tolerance_sec": self.future_tolerance_sec,
            "localization_tf_max_age_sec": self.tf_max_age_sec,
            "localization_max_position_stddev_m": self.max_position_stddev_m,
            "localization_max_yaw_stddev_rad": self.max_yaw_stddev_rad,
            "localization_jump_distance_m": self.jump_distance_m,
            "localization_jump_yaw_rad": self.jump_yaw_rad,
            "localization_max_linear_speed_mps": self.max_linear_speed_mps,
            "localization_max_angular_speed_radps": self.max_angular_speed_radps,
            "localization_stationary_hold_distance_m": self.stationary_hold_distance_m,
            "localization_stationary_hold_yaw_rad": self.stationary_hold_yaw_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.recovery_samples < 1:
            raise ValueError("localization_recovery_samples must be at least 1")
        if not math.isfinite(self.recovery_sec) or self.recovery_sec < 0.0:
            raise ValueError("localization_recovery_sec must be finite and non-negative")


@dataclass(frozen=True)
class AmclObservation:
    x: float
    y: float
    yaw: float
    covariance: tuple[float, ...]
    frame_id: str
    stamp_sec: float


@dataclass(frozen=True)
class LocalizationStatus:
    state: str
    detail: str
    stable: bool
    receive_count: int


def amcl_message_to_observation(message) -> AmclObservation:
    """Convert a PoseWithCovarianceStamped-like object without importing ROS."""
    header = message.header
    pose_with_covariance = message.pose
    pose = pose_with_covariance.pose
    position = pose.position
    orientation = pose.orientation
    quaternion = np.asarray(
        (orientation.x, orientation.y, orientation.z, orientation.w), dtype=np.float64
    )
    if not np.isfinite(quaternion).all():
        raise ValueError("AMCL orientation contains NaN or Inf")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9 or abs(norm - 1.0) > 0.05:
        raise ValueError(f"AMCL quaternion norm is invalid: {norm:.6f}")
    xq, yq, zq, wq = quaternion / norm
    yaw = math.atan2(
        2.0 * (wq * zq + xq * yq),
        1.0 - 2.0 * (yq * yq + zq * zq),
    )
    stamp = header.stamp
    nanosec = int(stamp.nanosec)
    if nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("AMCL timestamp nanosec is outside [0, 1e9)")
    return AmclObservation(
        x=float(position.x),
        y=float(position.y),
        yaw=float(yaw),
        covariance=tuple(float(value) for value in pose_with_covariance.covariance),
        frame_id=str(header.frame_id).strip(),
        stamp_sec=float(stamp.sec) + nanosec * 1e-9,
    )


class LocalizationMonitor:
    """Gate AR drawing until AMCL quality remains acceptable for a short period."""

    def __init__(
        self,
        thresholds: LocalizationThresholds = LocalizationThresholds(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self._clock = clock
        self.receive_count = 0
        self._last_received_at: float | None = None
        self._baseline: tuple[float, float, float, float] | None = None
        self._last_pose: tuple[float, float, float] | None = None
        self._good_samples = 0
        self._good_since: float | None = None
        self._last_error = ""
        self._stable = False
        self._ever_stable = False
        self._last_metrics = ""

    def reject(self, reason: str, received_at: float | None = None) -> LocalizationStatus:
        now = self._now(received_at)
        self.receive_count += 1
        self._last_received_at = now
        self._mark_bad(str(reason))
        return self.status(now)

    def observe(
        self,
        observation: AmclObservation,
        ros_now_sec: float,
        received_at: float | None = None,
    ) -> LocalizationStatus:
        now = self._now(received_at)
        self.receive_count += 1
        self._last_received_at = now
        error, metrics = self._validate_observation(observation, float(ros_now_sec), now)
        if error:
            self._mark_bad(error)
            return self.status(now)

        self._last_error = ""
        self._last_metrics = metrics
        self._last_pose = (observation.x, observation.y, observation.yaw)
        if self._good_samples == 0:
            self._good_since = now
        self._good_samples += 1
        stable_for = max(0.0, now - float(self._good_since))
        if (
            self._good_samples >= self.thresholds.recovery_samples
            and stable_for >= self.thresholds.recovery_sec
        ):
            self._stable = True
            self._ever_stable = True
        else:
            self._stable = False
        return self.status(now)

    def status(
        self,
        now: float | None = None,
        freshness_confirmed: bool = False,
        tf_pose: tuple[float, float, float] | None = None,
    ) -> LocalizationStatus:
        timestamp = self._now(now)
        if self._last_received_at is None:
            return LocalizationStatus(
                LOCALIZING, "no /amcl_pose received since startup", False, self.receive_count
            )
        age = max(0.0, timestamp - self._last_received_at)
        held_by_fresh_tf = self._can_hold_with_tf(freshness_confirmed, tf_pose)
        if age > self.thresholds.pose_timeout_sec and not held_by_fresh_tf:
            if not self._last_error.startswith("AMCL reception stale"):
                self._mark_bad(
                    f"AMCL reception stale: age={age:.3f}s "
                    f"timeout={self.thresholds.pose_timeout_sec:.3f}s"
                )
            return LocalizationStatus(
                LOCALIZATION_UNSTABLE, self._last_error, False, self.receive_count
            )
        if not self._stable and not self._last_error and self._good_since is not None:
            stable_for = max(0.0, timestamp - self._good_since)
            if (
                self._good_samples >= self.thresholds.recovery_samples
                and stable_for >= self.thresholds.recovery_sec
            ):
                self._stable = True
                self._ever_stable = True
        if self._stable:
            freshness = "AMCL pose held with fresh TF; " if held_by_fresh_tf else ""
            return LocalizationStatus(
                LOCALIZATION_OK,
                f"{freshness}stable samples={self._good_samples}; {self._last_metrics}",
                True,
                self.receive_count,
            )
        if self._last_error:
            return LocalizationStatus(
                LOCALIZATION_UNSTABLE, self._last_error, False, self.receive_count
            )
        stable_for = max(0.0, timestamp - float(self._good_since))
        detail = (
            f"AMCL recovery samples={self._good_samples}/"
            f"{self.thresholds.recovery_samples} stable_for={stable_for:.3f}/"
            f"{self.thresholds.recovery_sec:.3f}s"
        )
        state = LOCALIZATION_UNSTABLE if self._ever_stable else LOCALIZING
        return LocalizationStatus(state, detail, False, self.receive_count)

    def _validate_observation(
        self, observation: AmclObservation, ros_now_sec: float, received_at: float
    ) -> tuple[str, str]:
        if observation.frame_id != self.thresholds.expected_frame:
            return (
                f"AMCL frame mismatch: got={observation.frame_id!r} "
                f"expected={self.thresholds.expected_frame!r}",
                "",
            )
        numeric = (observation.x, observation.y, observation.yaw, observation.stamp_sec)
        if not all(math.isfinite(value) for value in numeric):
            return "AMCL pose or timestamp contains NaN or Inf", ""
        if not math.isfinite(ros_now_sec):
            return "ROS clock is not finite", ""
        if observation.stamp_sec <= 0.0:
            return "AMCL timestamp is zero or negative", ""
        stamp_age = ros_now_sec - observation.stamp_sec
        if stamp_age > self.thresholds.stamp_max_age_sec:
            return (
                f"AMCL timestamp stale: age={stamp_age:.3f}s "
                f"limit={self.thresholds.stamp_max_age_sec:.3f}s",
                "",
            )
        if stamp_age < -self.thresholds.future_tolerance_sec:
            return (
                f"AMCL timestamp is in the future: offset={-stamp_age:.3f}s "
                f"limit={self.thresholds.future_tolerance_sec:.3f}s",
                "",
            )
        covariance = np.asarray(observation.covariance, dtype=np.float64)
        if covariance.shape != (36,) or not np.isfinite(covariance).all():
            return "AMCL covariance must contain 36 finite values", ""
        xy_covariance = np.array(
            ((covariance[0], covariance[1]), (covariance[6], covariance[7])),
            dtype=np.float64,
        )
        xy_covariance = 0.5 * (xy_covariance + xy_covariance.T)
        eigenvalues = np.linalg.eigvalsh(xy_covariance)
        yaw_variance = float(covariance[35])
        if float(eigenvalues[0]) < -1e-9 or yaw_variance < 0.0:
            return "AMCL covariance is not positive semidefinite", ""
        position_sigma = math.sqrt(max(0.0, float(eigenvalues[-1])))
        yaw_sigma = math.sqrt(yaw_variance)
        if position_sigma > self.thresholds.max_position_stddev_m:
            return (
                f"AMCL position uncertainty too high: sigma={position_sigma:.3f}m "
                f"limit={self.thresholds.max_position_stddev_m:.3f}m",
                "",
            )
        if yaw_sigma > self.thresholds.max_yaw_stddev_rad:
            return (
                f"AMCL yaw uncertainty too high: sigma={yaw_sigma:.3f}rad "
                f"limit={self.thresholds.max_yaw_stddev_rad:.3f}rad",
                "",
            )

        previous = self._baseline
        self._baseline = (observation.x, observation.y, observation.yaw, received_at)
        if previous is not None:
            px, py, pyaw, previous_at = previous
            delta_time = max(0.0, received_at - previous_at)
            if delta_time <= self.thresholds.pose_timeout_sec:
                distance = math.hypot(observation.x - px, observation.y - py)
                yaw_delta = abs(
                    math.atan2(
                        math.sin(observation.yaw - pyaw),
                        math.cos(observation.yaw - pyaw),
                    )
                )
                distance_limit = (
                    self.thresholds.jump_distance_m
                    + self.thresholds.max_linear_speed_mps * delta_time
                )
                yaw_limit = (
                    self.thresholds.jump_yaw_rad
                    + self.thresholds.max_angular_speed_radps * delta_time
                )
                if distance > distance_limit:
                    return (
                        f"AMCL position jump: distance={distance:.3f}m "
                        f"limit={distance_limit:.3f}m dt={delta_time:.3f}s",
                        "",
                    )
                if yaw_delta > yaw_limit:
                    return (
                        f"AMCL yaw jump: delta={yaw_delta:.3f}rad "
                        f"limit={yaw_limit:.3f}rad dt={delta_time:.3f}s",
                        "",
                    )
        return (
            "",
            f"position_sigma={position_sigma:.3f}m yaw_sigma={yaw_sigma:.3f}rad "
            f"stamp_age={stamp_age:.3f}s",
        )

    def _mark_bad(self, reason: str) -> None:
        self._stable = False
        self._good_samples = 0
        self._good_since = None
        self._last_error = reason

    def _can_hold_with_tf(
        self,
        freshness_confirmed: bool,
        tf_pose: tuple[float, float, float] | None,
    ) -> bool:
        if (
            not freshness_confirmed
            or self._last_error
            or self._good_samples == 0
            or self._last_pose is None
            or tf_pose is None
        ):
            return False
        tx, ty, tyaw = (float(value) for value in tf_pose)
        if not all(math.isfinite(value) for value in (tx, ty, tyaw)):
            return False
        px, py, pyaw = self._last_pose
        distance = math.hypot(tx - px, ty - py)
        yaw_delta = abs(math.atan2(math.sin(tyaw - pyaw), math.cos(tyaw - pyaw)))
        return (
            distance <= self.thresholds.stationary_hold_distance_m
            and yaw_delta <= self.thresholds.stationary_hold_yaw_rad
        )

    def _now(self, value: float | None) -> float:
        timestamp = self._clock() if value is None else float(value)
        if not math.isfinite(timestamp):
            raise ValueError("monotonic timestamp must be finite")
        return timestamp
