from types import SimpleNamespace

import numpy as np
import pytest

from ar_path_viewer.localization_monitor import (
    AmclObservation,
    LOCALIZATION_OK,
    LOCALIZATION_UNSTABLE,
    LOCALIZING,
    LocalizationMonitor,
    LocalizationThresholds,
    amcl_message_to_observation,
)


def covariance(position_variance=0.04, yaw_variance=0.01):
    values = np.zeros(36, dtype=np.float64)
    values[0] = position_variance
    values[7] = position_variance
    values[35] = yaw_variance
    return tuple(values)


def observation(
    x=0.0,
    y=0.0,
    yaw=0.0,
    stamp=100.0,
    frame="map",
    covariance_values=None,
):
    return AmclObservation(
        x=x,
        y=y,
        yaw=yaw,
        covariance=covariance_values or covariance(),
        frame_id=frame,
        stamp_sec=stamp,
    )


def thresholds(**overrides):
    values = {
        "recovery_samples": 3,
        "recovery_sec": 1.0,
        "pose_timeout_sec": 3.0,
    }
    values.update(overrides)
    return LocalizationThresholds(**values)


def test_no_amcl_pose_is_localizing():
    monitor = LocalizationMonitor(thresholds(), clock=lambda: 10.0)
    status = monitor.status()
    assert status.state == LOCALIZING
    assert status.stable is False
    assert "no /amcl_pose" in status.detail


def test_consecutive_good_samples_transition_to_stable():
    now = [0.0]
    monitor = LocalizationMonitor(thresholds(), clock=lambda: now[0])
    status = monitor.observe(observation(stamp=100.0), 100.0)
    assert status.state == LOCALIZING
    now[0] = 0.5
    status = monitor.observe(observation(x=0.1, stamp=100.5), 100.5)
    assert status.state == LOCALIZING
    now[0] = 1.0
    status = monitor.observe(observation(x=0.2, stamp=101.0), 101.0)
    assert status.state == LOCALIZATION_OK
    assert status.stable is True


def test_covariance_degradation_hides_path_until_recovery_is_sustained():
    now = [0.0]
    monitor = LocalizationMonitor(
        thresholds(recovery_samples=2, recovery_sec=0.5), clock=lambda: now[0]
    )
    monitor.observe(observation(stamp=100.0), 100.0)
    now[0] = 0.5
    assert monitor.observe(observation(stamp=100.5), 100.5).stable

    now[0] = 0.6
    bad = covariance(position_variance=1.0)
    status = monitor.observe(
        observation(stamp=100.6, covariance_values=bad), 100.6
    )
    assert status.state == LOCALIZATION_UNSTABLE
    assert "uncertainty too high" in status.detail

    now[0] = 1.0
    status = monitor.observe(observation(x=0.1, stamp=101.0), 101.0)
    assert status.state == LOCALIZATION_UNSTABLE
    now[0] = 1.5
    status = monitor.observe(observation(x=0.2, stamp=101.5), 101.5)
    assert status.stable is True


def test_stale_reception_and_old_or_future_header_stamps_are_rejected():
    now = [0.0]
    monitor = LocalizationMonitor(
        thresholds(recovery_samples=1, recovery_sec=0.0), clock=lambda: now[0]
    )
    assert monitor.observe(observation(stamp=100.0), 100.0).stable
    now[0] = 3.01
    status = monitor.status()
    assert status.state == LOCALIZATION_UNSTABLE
    assert "reception stale" in status.detail

    now[0] = 4.0
    status = monitor.observe(observation(stamp=90.0), 104.0)
    assert "timestamp stale" in status.detail
    now[0] = 5.0
    status = monitor.observe(observation(stamp=107.0), 105.0)
    assert "in the future" in status.detail


def test_fresh_tf_holds_last_good_pose_during_stationary_amcl_behavior():
    now = [0.0]
    monitor = LocalizationMonitor(
        thresholds(recovery_samples=1, recovery_sec=2.0), clock=lambda: now[0]
    )
    assert not monitor.observe(observation(stamp=100.0), 100.0).stable
    now[0] = 2.0
    assert monitor.status(freshness_confirmed=True, tf_pose=(0.1, 0.0, 0.0)).stable
    now[0] = 20.0
    status = monitor.status(freshness_confirmed=True, tf_pose=(0.1, 0.0, 0.0))
    assert status.stable
    assert "held with fresh TF" in status.detail

    status = monitor.status(freshness_confirmed=True, tf_pose=(2.0, 0.0, 0.0))
    assert not status.stable
    assert "reception stale" in status.detail


def test_frame_mismatch_and_non_positive_covariance_are_rejected():
    monitor = LocalizationMonitor(thresholds(), clock=lambda: 0.0)
    status = monitor.observe(observation(frame="odom"), 100.0)
    assert status.state == LOCALIZATION_UNSTABLE
    assert "frame mismatch" in status.detail

    bad = list(covariance())
    bad[0] = -1.0
    status = monitor.observe(
        observation(covariance_values=tuple(bad)), 100.0, received_at=0.1
    )
    assert "positive semidefinite" in status.detail


def test_implausible_jump_is_rejected_but_normal_motion_is_not():
    now = [0.0]
    monitor = LocalizationMonitor(
        thresholds(recovery_samples=1, recovery_sec=0.0), clock=lambda: now[0]
    )
    assert monitor.observe(observation(stamp=100.0), 100.0).stable
    now[0] = 1.0
    normal = monitor.observe(observation(x=2.0, stamp=101.0), 101.0)
    assert normal.stable
    now[0] = 1.1
    jumped = monitor.observe(observation(x=8.0, stamp=101.1), 101.1)
    assert jumped.state == LOCALIZATION_UNSTABLE
    assert "position jump" in jumped.detail


def test_jump_uses_new_consistent_pose_as_recovery_baseline():
    now = [0.0]
    monitor = LocalizationMonitor(
        thresholds(recovery_samples=2, recovery_sec=0.5), clock=lambda: now[0]
    )
    monitor.observe(observation(stamp=100.0), 100.0)
    now[0] = 0.5
    assert monitor.observe(observation(stamp=100.5), 100.5).stable
    now[0] = 0.6
    assert not monitor.observe(observation(x=10.0, stamp=100.6), 100.6).stable
    now[0] = 1.1
    assert not monitor.observe(observation(x=10.1, stamp=101.1), 101.1).stable
    now[0] = 1.6
    assert monitor.observe(observation(x=10.2, stamp=101.6), 101.6).stable


def test_pose_message_conversion_validates_quaternion_and_timestamp():
    message = SimpleNamespace(
        header=SimpleNamespace(
            frame_id="map", stamp=SimpleNamespace(sec=100, nanosec=250_000_000)
        ),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            covariance=covariance(),
        ),
    )
    converted = amcl_message_to_observation(message)
    assert converted.frame_id == "map"
    assert converted.stamp_sec == pytest.approx(100.25)
    assert converted.yaw == pytest.approx(0.0)

    message.pose.pose.orientation.w = 0.0
    with pytest.raises(ValueError, match="quaternion norm"):
        amcl_message_to_observation(message)
