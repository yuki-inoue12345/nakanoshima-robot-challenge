"""ROS-independent Local Costmap perception overlay tests."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ar_path_viewer.costmap_overlay import (
    CostmapCameraSyncSettings,
    CostmapOverlaySettings,
    CostmapStore,
    CostmapTransformQueue,
    HIGH_COST_CLASS,
    LETHAL_COST_CLASS,
    UNKNOWN_CLASS,
    ProjectedCostmap,
    align_costmap_to_image_time,
    estimate_image_frame_timing,
    occupancy_grid_to_snapshot,
    project_costmap_points,
)
from ar_path_viewer.projection import CameraIntrinsics, CameraMount
from ar_path_viewer.rendering import draw_costmap_overlay


def _grid(
    data,
    *,
    width=2,
    height=2,
    resolution=1.0,
    stamp=10.0,
    frame="odom",
    origin=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    seconds = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame,
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=int(round((stamp - seconds) * 1e9)),
            ),
        ),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(
                    x=origin[0], y=origin[1], z=origin[2]
                ),
                orientation=SimpleNamespace(
                    x=quaternion[0],
                    y=quaternion[1],
                    z=quaternion[2],
                    w=quaternion[3],
                ),
            ),
        ),
        data=list(data),
    )


def _transform(stamp, *, translation=(0.0, 0.0, 0.0), yaw_deg=0.0):
    yaw = np.deg2rad(float(yaw_deg))
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=int(stamp),
                nanosec=int(round((stamp - int(stamp)) * 1e9)),
            )
        ),
        transform=SimpleNamespace(
            translation=SimpleNamespace(
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
            ),
            rotation=SimpleNamespace(
                x=0.0,
                y=0.0,
                z=float(np.sin(yaw / 2.0)),
                w=float(np.cos(yaw / 2.0)),
            ),
        ),
    )


def _timing(stamp, *, source="camera_timestamped", duration=0.01, previous=None):
    settings = CostmapCameraSyncSettings()
    return estimate_image_frame_timing(
        ros_started_sec=stamp - duration,
        ros_finished_sec=stamp,
        monotonic_started_sec=20.0,
        monotonic_finished_sec=20.0 + duration,
        source_kind=source,
        settings=settings,
        previous_stamp_sec=previous,
        source_timestamp_monotonic_sec=20.0 + duration,
        timestamp_source="v4l2_monotonic_soe",
    )


def _single_cell_snapshot(stamp=10.0):
    return occupancy_grid_to_snapshot(
        _grid((100,), width=1, height=1, resolution=1.0, stamp=stamp),
        CostmapOverlaySettings(),
        ros_now_sec=stamp,
        received_at=20.0,
    )


def test_grid_cells_keep_unknown_high_and_lethal_semantics():
    settings = CostmapOverlaySettings(
        high_cost_min=60, lethal_cost_min=100, plane_height_m=0.02
    )

    snapshot = occupancy_grid_to_snapshot(
        _grid((-1, 0, 60, 100)),
        settings,
        ros_now_sec=10.1,
        received_at=20.0,
    )

    assert snapshot.unknown_cells == 1
    assert snapshot.high_cost_cells == 1
    assert snapshot.lethal_cost_cells == 1
    assert set(snapshot.classes) == {
        UNKNOWN_CLASS,
        HIGH_COST_CLASS,
        LETHAL_COST_CLASS,
    }
    np.testing.assert_allclose(
        snapshot.points,
        ((0.5, 0.5, 0.02), (0.5, 1.5, 0.02), (1.5, 1.5, 0.02)),
    )
    assert snapshot.analysis_points.shape == (4, 3)
    np.testing.assert_array_equal(snapshot.analysis_costs, (-1, 0, 60, 100))


def test_camera_time_alignment_accepts_stationary_small_delay():
    snapshot = _single_cell_snapshot(10.0)
    aligned = align_costmap_to_image_time(
        snapshot,
        image_timing=_timing(10.05),
        transform_at_costmap=_transform(10.0),
        transform_at_image=_transform(10.05),
        settings=CostmapCameraSyncSettings(),
    )

    assert aligned.costmap_age_sec == pytest.approx(0.05)
    assert aligned.translation_m == pytest.approx(0.0)
    assert aligned.rotation_deg == pytest.approx(0.0)
    assert np.allclose(aligned.points_in_base, snapshot.points)


@pytest.mark.parametrize("delay", (0.5, 1.5))
def test_delayed_ninety_degree_turn_is_rejected(delay):
    snapshot = _single_cell_snapshot(10.0)

    with pytest.raises(ValueError, match="costmap_camera_age_exceeded"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.0 + delay),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.0 + delay, yaw_deg=-90.0),
            settings=CostmapCameraSyncSettings(),
        )


def test_ninety_degree_turn_never_leaves_old_point_in_front():
    snapshot = _single_cell_snapshot(10.0)
    old_front = np.array(((2.0, 0.0, 0.02),), dtype=np.float64)
    snapshot = snapshot.__class__(
        **{
            **snapshot.__dict__,
            "points": old_front,
            "analysis_points": old_front.copy(),
            "classes": np.array((LETHAL_COST_CLASS,), dtype=np.uint8),
            "analysis_costs": np.array((100,), dtype=np.int16),
        }
    )
    relaxed = CostmapCameraSyncSettings(
        max_costmap_age_sec=1.0,
        max_rotation_deg=100.0,
    )
    aligned = align_costmap_to_image_time(
        snapshot,
        image_timing=_timing(10.5),
        transform_at_costmap=_transform(10.0),
        transform_at_image=_transform(10.5, yaw_deg=-90.0),
        settings=relaxed,
    )

    assert aligned.rotation_deg == pytest.approx(90.0)
    assert aligned.points_in_base[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert aligned.points_in_base[0, 1] == pytest.approx(-2.0)
    assert not np.allclose(aligned.points_in_base, old_front)

    with pytest.raises(ValueError, match="costmap_camera_rotation_exceeded"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.1),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.1, yaw_deg=-90.0),
            settings=CostmapCameraSyncSettings(max_costmap_age_sec=1.0),
        )


def test_forward_motion_uses_image_time_robot_pose():
    snapshot = _single_cell_snapshot(10.0)
    aligned = align_costmap_to_image_time(
        snapshot,
        image_timing=_timing(10.1),
        transform_at_costmap=_transform(10.0),
        # Robot moved +0.10 m in the source frame, so source appears -0.10 m
        # in the new base frame.
        transform_at_image=_transform(10.1, translation=(-0.1, 0.0, 0.0)),
        settings=CostmapCameraSyncSettings(),
    )

    assert aligned.translation_m == pytest.approx(0.1)
    assert np.allclose(
        aligned.points_in_base[:, 0], snapshot.points[:, 0] - 0.1
    )


def test_untrusted_image_timestamp_suppresses_alignment():
    snapshot = _single_cell_snapshot(10.0)
    timing = _timing(10.05, source="video")
    assert not timing.usable
    assert "untrusted_source" in timing.reason

    with pytest.raises(ValueError, match="untrusted_source"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=timing,
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.05),
            settings=CostmapCameraSyncSettings(),
        )


def test_image_time_tf_must_exist_at_the_requested_timestamp():
    snapshot = _single_cell_snapshot(10.0)

    with pytest.raises(ValueError, match="image_tf_not_exact"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.1),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.0),
            settings=CostmapCameraSyncSettings(),
        )


def test_translation_over_limit_suppresses_alignment():
    snapshot = _single_cell_snapshot(10.0)

    with pytest.raises(ValueError, match="translation_exceeded"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.1),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(
                10.1, translation=(-0.21, 0.0, 0.0)
            ),
            settings=CostmapCameraSyncSettings(),
        )


def test_ros_clock_reset_makes_image_timestamp_unusable():
    timing = _timing(50.0, previous=100.0)

    assert not timing.usable
    assert timing.reason == "ros_clock_reset_between_frames"


def test_new_image_recomputes_points_instead_of_reusing_old_base_coordinates():
    snapshot = _single_cell_snapshot(10.0)
    first = align_costmap_to_image_time(
        snapshot,
        image_timing=_timing(10.05),
        transform_at_costmap=_transform(10.0),
        transform_at_image=_transform(10.05, translation=(-0.02, 0.0, 0.0)),
        settings=CostmapCameraSyncSettings(),
    )
    second = align_costmap_to_image_time(
        snapshot,
        image_timing=_timing(10.10),
        transform_at_costmap=_transform(10.0),
        transform_at_image=_transform(10.10, translation=(-0.08, 0.0, 0.0)),
        settings=CostmapCameraSyncSettings(),
    )

    assert np.allclose(
        second.points_in_base[:, 0], first.points_in_base[:, 0] - 0.06
    )


def test_grid_origin_yaw_is_applied_to_cell_centers():
    half = np.sqrt(0.5)
    snapshot = occupancy_grid_to_snapshot(
        _grid(
            (100, 0),
            width=2,
            height=1,
            origin=(2.0, 3.0, 0.0),
            quaternion=(0.0, 0.0, half, half),
        ),
        CostmapOverlaySettings(),
        ros_now_sec=10.0,
        received_at=10.0,
    )

    np.testing.assert_allclose(snapshot.points[0, :2], (1.5, 3.5), atol=1e-9)


def test_sampling_is_bounded_per_class_without_changing_counts():
    values = [-1] * 100 + [70] * 100 + [100] * 100
    snapshot = occupancy_grid_to_snapshot(
        _grid(values, width=30, height=10),
        CostmapOverlaySettings(max_points_per_class=12),
        ros_now_sec=10.0,
        received_at=10.0,
    )

    assert snapshot.sampled_cells == 36
    assert snapshot.analysis_points.shape == (300, 3)
    assert snapshot.analysis_costs.shape == (300,)
    assert snapshot.unknown_cells == 100
    assert snapshot.high_cost_cells == 100
    assert snapshot.lethal_cost_cells == 100


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (_grid((0, 101, 0, 0)), "within"),
        (_grid((0, 0, 0)), "length mismatch"),
        (_grid((0, 0, 0, 0), frame=""), "frame_id"),
        (_grid((0, 0, 0, 0), stamp=0.0), "timestamp"),
    ],
)
def test_invalid_grid_is_rejected(message, match):
    with pytest.raises(ValueError, match=match):
        occupancy_grid_to_snapshot(
            message,
            CostmapOverlaySettings(),
            ros_now_sec=10.0,
            received_at=10.0,
        )


def test_store_clears_invalid_and_stale_data_then_recovers():
    now = [10.0]
    store = CostmapStore(
        CostmapOverlaySettings(timeout_sec=1.0, stamp_max_age_sec=1.0),
        clock=lambda: now[0],
    )
    assert store.update(_grid((0, 100, 0, 0)), 10.0)
    assert store.state(10.5)[0] == "ACTIVE"

    now[0] = 11.1
    state, snapshot, reason = store.state(11.1)
    assert state == "STALE"
    assert snapshot is None
    assert "receive_stale" in reason

    now[0] = 12.0
    assert store.update(_grid((0, 100, 0, 0), stamp=12.0), 12.0)
    assert not store.update(_grid((0, 101, 0, 0), stamp=12.0), 12.0)
    assert store.snapshot is None
    assert "invalid_local_costmap" in store.reason


def test_store_ignores_out_of_order_without_discarding_newer_snapshot():
    store = CostmapStore(CostmapOverlaySettings(), clock=lambda: 20.0)
    assert store.update(_grid((0, 100, 0, 0), stamp=20.0), 20.0)
    newest = store.snapshot

    assert not store.update(_grid((100, 0, 0, 0), stamp=19.9), 20.0)

    assert store.snapshot is newest
    assert store.state(20.0)[0] == "ACTIVE"
    assert store.reason.startswith("out_of_order_local_costmap_ignored")



def _costmap_pipeline_for_frame_order():
    """Use the same store-then-queue order as the Viewer callback."""
    now = [100.3]
    settings = CostmapOverlaySettings(
        timeout_sec=1.5, stamp_max_age_sec=1.5,
        tf_wait_timeout_sec=0.25,
    )
    store = CostmapStore(settings, clock=lambda: now[0])
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def deliver(frame, stamp, *, tf_available=True):
        accepted = store.update(
            _grid((100, 0, 0, 0), stamp=stamp, frame=frame), now[0]
        )
        if accepted:
            assert queue.submit(store.snapshot)
        def transform(item):
            if not tf_available:
                raise RuntimeError("exact TF is not available")
            return item.points.copy(), item.analysis_points.copy()
        state, selected, _ = queue.resolve(now[0], transform)
        identity = None if selected is None else (
            selected.snapshot.frame_id, selected.snapshot.stamp_sec
        )
        return accepted, state, identity

    return now, store, queue, deliver


@pytest.mark.parametrize("late_stamp", (100.1, 100.2, 100.25))
def test_costmap_pipeline_rejects_retired_frame_even_if_stamp_is_newer(
    late_stamp,
):
    """A→B→A must not revive an old frame, including forward-stamped A."""
    _, store, queue, deliver = _costmap_pipeline_for_frame_order()
    assert deliver("odom", 100.0) == (True, "ACTIVE", ("odom", 100.0))
    assert deliver("map", 100.2) == (True, "ACTIVE", ("map", 100.2))
    assert deliver("odom", late_stamp) == (
        False, "ACTIVE", ("map", 100.2)
    )
    assert store.snapshot.frame_id == "map"
    assert queue.pending_count == 0
    assert "out_of_order_local_costmap_ignored" in store.reason


def test_costmap_pipeline_rejects_old_frame_while_new_frame_tf_pending():
    """A must not be promoted while newer frame B awaits exact-time TF."""
    _, store, queue, deliver = _costmap_pipeline_for_frame_order()
    assert deliver("odom", 100.0)[1] == "ACTIVE"
    assert deliver("map", 100.2, tf_available=False) == (
        True, "WAITING_TF", None
    )
    assert deliver("odom", 100.25, tf_available=False) == (
        False, "WAITING_TF", None
    )
    assert store.snapshot.frame_id == "map"
    assert queue.pending_count == 1
    assert deliver("map", 100.3) == (True, "ACTIVE", ("map", 100.3))


def test_costmap_pipeline_allows_fresh_third_frame_but_no_retired_reuse():
    _, store, queue, deliver = _costmap_pipeline_for_frame_order()
    assert deliver("odom", 100.0)[0]
    assert deliver("map", 100.2)[0]
    assert deliver("base_map", 100.25) == (
        True, "ACTIVE", ("base_map", 100.25)
    )
    assert deliver("map", 100.27)[0] is False
    assert store.snapshot.frame_id == "base_map"
    assert queue.resolve(
        100.3,
        lambda item: (item.points.copy(), item.analysis_points.copy()),
    )[1].snapshot.frame_id == "base_map"


def test_costmap_pipeline_expiry_or_clear_preserves_cross_frame_watermark():
    now, store, queue, deliver = _costmap_pipeline_for_frame_order()
    assert deliver("odom", 100.0)[0]
    assert deliver("map", 100.2, tf_available=False)[1] == "WAITING_TF"
    now[0] = 100.7
    assert queue.resolve(
        now[0],
        lambda _: (_ for _ in ()).throw(RuntimeError("no TF")),
    )[0] == "UNAVAILABLE"
    store.clear("invalid_local_costmap")
    queue.clear("invalid_local_costmap")
    assert not store.update(_grid((100, 0, 0, 0), stamp=100.1), 100.7)
    assert "out_of_order" in store.reason
    assert not store.update(
        _grid((100, 0, 0, 0), stamp=100.6, frame="odom"), 100.7
    )
    assert "retired_frame" in store.reason
    assert store.update(
        _grid((100, 0, 0, 0), stamp=100.7, frame="map"), 100.7
    )
    assert queue.submit(store.snapshot)


def test_queue_standalone_rejects_old_frame_and_cross_frame_rewind():
    now = [100.3]
    settings = CostmapOverlaySettings()
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def snapshot(frame, stamp):
        return occupancy_grid_to_snapshot(
            _grid((100, 0, 0, 0), stamp=stamp, frame=frame),
            settings, ros_now_sec=now[0], received_at=now[0],
        )

    assert queue.submit(snapshot("odom", 100.0))
    assert not queue.submit(snapshot("map", 99.9))
    assert queue.pending_count == 1
    assert queue.submit(snapshot("map", 100.2))
    assert queue.pending_count == 1
    assert not queue.submit(snapshot("odom", 100.25))
    assert queue.pending_count == 1
    assert "retired_frame" in queue.reason
    assert queue.resolve(
        now[0], lambda item: (item.points.copy(), item.analysis_points.copy())
    )[1].snapshot.frame_id == "map"


def test_clock_reset_requires_explicit_reset_for_both_store_and_queue():
    now, store, queue, deliver = _costmap_pipeline_for_frame_order()
    assert deliver("odom", 100.0)[0]
    assert deliver("map", 100.2)[0]
    # An older stamp or retired frame can never self-authorize a new epoch.
    assert not store.update(
        _grid((100, 0, 0, 0), stamp=90.0, frame="odom"), 90.0
    )
    assert not queue.submit(occupancy_grid_to_snapshot(
        _grid((100, 0, 0, 0), stamp=90.0, frame="odom"),
        store.settings, ros_now_sec=90.0, received_at=now[0],
    ))
    # A verified reset can be handled by restarting the Viewer. Neither
    # store nor queue accepts the rewound stamp automatically in this process.
    now[0] = 90.0
    new_store = CostmapStore(store.settings, clock=lambda: now[0])
    new_queue = CostmapTransformQueue(store.settings, clock=lambda: now[0])
    assert new_store.update(
        _grid((100, 0, 0, 0), stamp=90.0, frame="odom"), 90.0
    )
    assert new_queue.submit(new_store.snapshot)
    assert new_queue.resolve(
        90.0,
        lambda item: (item.points.copy(), item.analysis_points.copy()),
    )[1].snapshot.stamp_sec == 90.0


def test_projection_preserves_ros_left_right_sign():
    points = np.asarray(((4.0, 1.0, 0.0), (4.0, -1.0, 0.0)))
    projection = project_costmap_points(
        points,
        np.asarray((LETHAL_COST_CLASS, HIGH_COST_CLASS), dtype=np.uint8),
        CameraMount(x=0.0, y=0.0, z=0.0),
        CameraIntrinsics(),
        0.1,
    )

    # Robot-left (+y) is optical-left (smaller u).
    assert projection.lethal_cost_pixels[0, 0] < 319
    assert projection.high_cost_pixels[0, 0] > 319
    assert projection.visible_points == 2


def test_sparse_rendering_keeps_cost_classes_visually_distinct():
    projection = ProjectedCostmap(
        unknown_pixels=np.asarray(((40, 40),), dtype=np.int32),
        high_cost_pixels=np.asarray(((80, 40),), dtype=np.int32),
        lethal_cost_pixels=np.asarray(((120, 40),), dtype=np.int32),
        input_points=3,
        visible_points=3,
    )
    image = np.zeros((100, 160, 3), dtype=np.uint8)

    frame = draw_costmap_overlay(image, projection, alpha=1.0, radius_px=2)

    assert np.any(frame[35:46, 35:46] != 0)
    assert np.any(frame[35:46, 75:86] != 0)
    assert np.any(frame[35:46, 115:126] != 0)
    assert not np.array_equal(frame[40, 40], frame[40, 80])
    assert cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 10


def test_exact_tf_queue_waits_recovers_and_never_uses_untransformed_data():
    now = [20.0]
    settings = CostmapOverlaySettings(
        timeout_sec=1.0,
        stamp_max_age_sec=1.0,
        tf_wait_timeout_sec=0.5,
        max_pending_messages=3,
    )
    snapshot = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=20.0), settings,
        ros_now_sec=20.0, received_at=20.0,
    )
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])
    queue.submit(snapshot)
    available = [False]

    def transform(item):
        assert item.stamp_sec == 20.0
        if not available[0]:
            raise RuntimeError("future extrapolation")
        return item.points + 1.0, item.analysis_points + 1.0

    state, synchronized, reason = queue.resolve(20.0, transform)
    assert state == "WAITING_TF"
    assert synchronized is None
    assert reason.startswith("waiting_for_exact_tf")

    available[0] = True
    now[0] = 20.02
    state, synchronized, reason = queue.resolve(20.02, transform)
    assert state == "ACTIVE"
    assert reason == "exact_tf_ready"
    np.testing.assert_allclose(synchronized.points_in_base, snapshot.points + 1.0)


def test_exact_tf_queue_holds_only_fresh_exact_pair_and_expires_bounded_wait():
    now = [30.0]
    settings = CostmapOverlaySettings(
        timeout_sec=0.4,
        stamp_max_age_sec=0.4,
        tf_wait_timeout_sec=0.3,
        max_pending_messages=2,
    )
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])
    first = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=30.0), settings,
        ros_now_sec=30.0, received_at=30.0,
    )
    queue.submit(first)
    transform = lambda item: (item.points.copy(), item.analysis_points.copy())
    assert queue.resolve(30.0, transform)[0] == "ACTIVE"

    second = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=30.1), settings,
        ros_now_sec=30.1, received_at=30.1,
    )
    queue.submit(second)

    def unavailable(_item):
        raise RuntimeError("not ready")

    now[0] = 30.2
    state, synchronized, reason = queue.resolve(30.2, unavailable)
    assert state == "ACTIVE"
    assert synchronized.snapshot.stamp_sec == 30.0
    assert reason == "using_previous_exact_sync_while_waiting"

    now[0] = 30.51
    state, synchronized, reason = queue.resolve(30.51, unavailable)
    assert state == "UNAVAILABLE"
    assert synchronized is None
    assert reason.startswith("exact_tf_wait_expired")


def test_exact_tf_queue_never_promotes_old_pending_after_newer_arrives():
    """P1-02: A's late TF must not beat a newer still-waiting B."""
    now = [99.1]
    settings = CostmapOverlaySettings(tf_wait_timeout_sec=1.0)
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])
    first = occupancy_grid_to_snapshot(
        _grid((100, 0, 0, 0), stamp=99.0), settings,
        ros_now_sec=99.0, received_at=99.0,
    )
    second = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=99.1), settings,
        ros_now_sec=99.1, received_at=99.1,
    )
    assert queue.submit(first)
    assert queue.submit(second)
    attempted = []

    def only_old_tf_available(item):
        attempted.append(item.stamp_sec)
        if item.stamp_sec == first.stamp_sec:
            return item.points.copy(), item.analysis_points.copy()
        raise RuntimeError("latest TF has not arrived")

    state, synchronized, reason = queue.resolve(99.1, only_old_tf_available)
    assert state == "WAITING_TF"
    assert synchronized is None
    assert "waiting_for_exact_tf" in reason
    assert first.stamp_sec not in attempted

    state, synchronized, reason = queue.resolve(
        99.1,
        lambda item: (item.points.copy(), item.analysis_points.copy()),
    )
    assert state == "ACTIVE"
    assert synchronized.snapshot.stamp_sec == second.stamp_sec
    assert queue.resolved_count == 1


def test_exact_tf_queue_keeps_only_previously_ready_while_newest_waits():
    """P1-02: a previously displayed R may persist; a new A may not."""
    now = [20.0]
    settings = CostmapOverlaySettings(tf_wait_timeout_sec=1.0)
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def sample(stamp):
        return occupancy_grid_to_snapshot(
            _grid((100, 0, 0, 0), stamp=stamp), settings,
            ros_now_sec=stamp, received_at=stamp,
        )

    ready = sample(20.0)
    assert queue.submit(ready)
    assert queue.resolve(
        20.0, lambda item: (item.points.copy(), item.analysis_points.copy())
    )[1].snapshot.stamp_sec == 20.0

    now[0] = 20.2
    assert queue.submit(sample(20.1))
    assert queue.submit(sample(20.2))

    attempted = []

    def only_superseded_tf_available(item):
        attempted.append(item.stamp_sec)
        if item.stamp_sec == 20.1:
            return item.points.copy(), item.analysis_points.copy()
        raise RuntimeError("newest TF not available")

    state, selected, reason = queue.resolve(20.2, only_superseded_tf_available)
    assert state == "ACTIVE"
    assert selected.snapshot.stamp_sec == 20.0
    assert reason == "using_previous_exact_sync_while_waiting"
    assert 20.1 not in attempted

    state, selected, reason = queue.resolve(
        20.2,
        lambda item: (item.points.copy(), item.analysis_points.copy()),
    )
    assert state == "ACTIVE"
    assert selected.snapshot.stamp_sec == 20.2
    assert reason == "exact_tf_ready"


def test_exact_tf_queue_rejects_superseded_stamp_after_latest_expires():
    """A missing TF must not be revived after newer B timed out."""
    now = [40.0]
    settings = CostmapOverlaySettings(
        timeout_sec=1.0, stamp_max_age_sec=1.0, tf_wait_timeout_sec=0.1,
    )
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def sample(stamp, received_at=None):
        return occupancy_grid_to_snapshot(
            _grid((100, 0, 0, 0), stamp=stamp), settings,
            ros_now_sec=stamp,
            received_at=stamp if received_at is None else received_at,
        )

    assert queue.submit(sample(40.0))
    assert queue.submit(sample(40.1))
    now[0] = 40.3
    state, selected, _ = queue.resolve(
        40.3, lambda item: (_ for _ in ()).throw(RuntimeError("TF late"))
    )
    assert state == "UNAVAILABLE"
    assert selected is None
    # Older message delivered late after the newest one expired.
    assert not queue.submit(sample(40.0, received_at=40.3))
    assert queue.ignored_out_of_order_count == 1
    state, selected, _ = queue.resolve(
        40.3, lambda item: (item.points.copy(), item.analysis_points.copy())
    )
    assert state == "UNAVAILABLE"
    assert selected is None


def test_exact_tf_queue_invalid_clear_does_not_reenable_old_stamp():
    """Invalid input clears visuals without resetting the stamp watermark."""
    now = [60.2]
    settings = CostmapOverlaySettings()
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def sample(stamp, frame="odom"):
        return occupancy_grid_to_snapshot(
            _grid((0, 100, 0, 0), stamp=stamp, frame=frame), settings,
            ros_now_sec=stamp, received_at=now[0],
        )

    assert queue.submit(sample(60.2))
    queue.clear("invalid_local_costmap")
    assert queue.pending_count == 0
    assert not queue.submit(sample(60.1))
    assert queue.ignored_out_of_order_count == 1
    assert queue.submit(sample(60.3))
    assert queue.pending_count == 1


def test_exact_tf_queue_frame_change_preserves_ordering_watermark():
    """Frame switch clears geometry but requires a strictly newer stamp."""
    now = [70.0]
    settings = CostmapOverlaySettings()
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def sample(stamp, frame):
        return occupancy_grid_to_snapshot(
            _grid((0, 100, 0, 0), stamp=stamp, frame=frame), settings,
            ros_now_sec=stamp, received_at=now[0],
        )

    assert queue.submit(sample(70.0, "odom"))
    ready = queue.resolve(
        70.0, lambda item: (item.points.copy(), item.analysis_points.copy())
    )[1]
    assert ready.snapshot.frame_id == "odom"
    assert queue.submit(sample(70.1, "map"))
    assert queue.pending_count == 1

    def no_map_tf(_):
        raise RuntimeError("map TF absent")

    state, selected, _ = queue.resolve(70.1, no_map_tf)
    assert state == "WAITING_TF"
    assert selected is None


def test_exact_tf_queue_frame_change_after_expiry_keeps_stamp_order():
    """Keep a global watermark even when pending and ready are gone."""
    now = [80.0]
    settings = CostmapOverlaySettings(tf_wait_timeout_sec=0.1)
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])

    def sample(stamp, frame):
        return occupancy_grid_to_snapshot(
            _grid((0, 100, 0, 0), stamp=stamp, frame=frame), settings,
            ros_now_sec=stamp, received_at=now[0],
        )

    assert queue.submit(sample(80.0, "odom"))
    now[0] = 80.3
    state, selected, _ = queue.resolve(80.3, lambda _: (_ for _ in ()).throw(
        RuntimeError("TF still unavailable")
    ))
    assert state == "UNAVAILABLE"
    assert selected is None
    assert queue.pending_count == 0
    # A frame change cannot implicitly rewind the message stamp epoch.
    assert not queue.submit(sample(70.0, "map"))
    assert queue.submit(sample(80.4, "map"))
    assert queue.pending_count == 1
    assert queue.resolve(
        80.4,
        lambda item: (item.points.copy(), item.analysis_points.copy()),
    )[1].snapshot.frame_id == "map"


def test_exact_tf_queue_rejects_out_of_order_snapshot_after_newer_ready():
    now = [40.0]
    settings = CostmapOverlaySettings()
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])
    newer = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=40.0), settings,
        ros_now_sec=40.0, received_at=40.0,
    )
    assert queue.submit(newer)
    transform = lambda item: (item.points.copy(), item.analysis_points.copy())
    assert queue.resolve(40.0, transform)[1].snapshot.stamp_sec == 40.0

    older = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=39.8), settings,
        ros_now_sec=40.0, received_at=40.0,
    )
    assert not queue.submit(older)
    assert queue.reason == "out_of_order_costmap_ignored"
    assert queue.ignored_out_of_order_count == 1


def test_exact_tf_queue_is_bounded_and_frame_change_discards_old_ready_data():
    now = [50.0]
    settings = CostmapOverlaySettings(max_pending_messages=2)
    queue = CostmapTransformQueue(settings, clock=lambda: now[0])
    for stamp in (50.0, 50.1, 50.2):
        snapshot = occupancy_grid_to_snapshot(
            _grid((0, 100, 0, 0), stamp=stamp), settings,
            ros_now_sec=stamp, received_at=stamp,
        )
        assert queue.submit(snapshot)
    # Superseded samples are discarded, not held for later TF fallback.
    assert queue.pending_count == 1
    assert queue.superseded_count == 2
    assert queue.expired_count == 0

    transform = lambda item: (item.points.copy(), item.analysis_points.copy())
    state, synchronized, _ = queue.resolve(50.2, transform)
    assert state == "ACTIVE"
    assert synchronized.snapshot.stamp_sec == 50.2

    changed = occupancy_grid_to_snapshot(
        _grid((0, 100, 0, 0), stamp=50.3, frame="map"), settings,
        ros_now_sec=50.3, received_at=50.3,
    )
    assert queue.submit(changed)

    def unavailable(_item):
        raise RuntimeError("new frame TF not ready")

    state, synchronized, reason = queue.resolve(50.3, unavailable)
    assert state == "WAITING_TF"
    assert synchronized is None
    assert reason.startswith("waiting_for_exact_tf")


def test_camera_sync_rejects_roll_and_pitch_even_with_zero_yaw():
    """Yaw-only motion checks can incorrectly admit severe camera tilt."""
    snapshot = _single_cell_snapshot(10.0)
    old_tf = _transform(10.0)
    for axis in ("x", "y"):
        half_angle = np.deg2rad(45.0) / 2.0
        component = np.sin(half_angle)
        new_tf = _transform(10.1)
        new_tf.transform.rotation.x = float(component if axis == "x" else 0.0)
        new_tf.transform.rotation.y = float(component if axis == "y" else 0.0)
        new_tf.transform.rotation.z = 0.0
        new_tf.transform.rotation.w = float(np.cos(half_angle))

        with pytest.raises(ValueError, match="costmap_camera_rotation_exceeded"):
            align_costmap_to_image_time(
                snapshot,
                image_timing=_timing(10.1),
                transform_at_costmap=old_tf,
                transform_at_image=new_tf,
                settings=CostmapCameraSyncSettings(),
            )


def test_camera_sync_rejects_vertical_displacement():
    snapshot = _single_cell_snapshot(10.0)
    with pytest.raises(ValueError, match="costmap_camera_translation_exceeded"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.1),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.1, translation=(0.0, 0.0, 0.25)),
            settings=CostmapCameraSyncSettings(),
        )


def test_camera_sync_requires_near_exact_tf_stamps():
    snapshot = _single_cell_snapshot(10.0)
    # Previously, a TF from 30 ms before the image timestamp passed the
    # 50 ms tolerance despite the "exact-time" contract.
    with pytest.raises(ValueError, match="image_tf_not_exact"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.10),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.07),
            settings=CostmapCameraSyncSettings(),
        )


def test_camera_sync_rejects_small_clock_reversal_during_capture():
    settings = CostmapCameraSyncSettings()
    timing = estimate_image_frame_timing(
        ros_started_sec=10.0,
        ros_finished_sec=9.98,  # only 20 ms backwards
        monotonic_started_sec=2.0,
        monotonic_finished_sec=2.02,
        source_kind="camera",
        settings=settings,
    )
    assert not timing.usable
    assert timing.reason == "ros_clock_reversed_during_capture"


def test_camera_sync_rejects_small_backward_jump_between_frames():
    timing = _timing(10.0, previous=10.02)
    assert not timing.usable
    assert timing.reason == "ros_clock_reset_between_frames"


def test_camera_sync_rejects_uninitialized_ros_clock():
    timing = estimate_image_frame_timing(
        ros_started_sec=0.0,
        ros_finished_sec=0.01,
        monotonic_started_sec=20.0,
        monotonic_finished_sec=20.01,
        source_kind="camera",
        settings=CostmapCameraSyncSettings(),
    )
    assert not timing.usable
    assert timing.reason == "ros_clock_uninitialized"


def test_camera_sync_capture_duration_fail_closed():
    timing = _timing(10.2, duration=0.125)
    assert not timing.usable
    assert "image_capture_too_slow" in timing.reason


def test_camera_sync_rejects_current_tf_missing_instead_of_using_old_tf():
    snapshot = _single_cell_snapshot(10.0)
    # The real Viewer requires this lookup to succeed.  An integration
    # failure must not fall back to the old base-frame points.
    with pytest.raises(ValueError, match="image_tf_not_exact"):
        align_costmap_to_image_time(
            snapshot,
            image_timing=_timing(10.15),
            transform_at_costmap=_transform(10.0),
            transform_at_image=_transform(10.0),
            settings=CostmapCameraSyncSettings(),
        )


def test_costmap_grid_size_limits_reject_before_numpy_conversion():
    class GuardedData:
        def __len__(self):
            return 10000000000

        def __array__(self, dtype=None):
            raise AssertionError("oversized grid must not be converted to numpy")

    message = _grid((0,))
    message.info.width = 100000
    message.info.height = 100000
    message.data = GuardedData()
    with pytest.raises(ValueError, match="costmap grid (width|size|cells) limit"):
        occupancy_grid_to_snapshot(
            message, CostmapOverlaySettings(),
            ros_now_sec=10.0, received_at=20.0,
        )


@pytest.mark.parametrize("width,height", [(513, 1), (1, 513), (257, 256)])
def test_costmap_grid_rejects_oversized_dimensions_or_cells(width, height):
    message = _grid([0] * (width * height), width=width, height=height)
    with pytest.raises(ValueError, match="costmap grid (width|height|cells) limit"):
        occupancy_grid_to_snapshot(
            message, CostmapOverlaySettings(),
            ros_now_sec=10.0, received_at=20.0,
        )


def test_costmap_grid_at_maximum_cell_budget_is_valid():
    # This is deliberately above the observed 120x120=14,400 local costmap.
    width, height = 256, 256
    message = _grid([0] * (width * height), width=width, height=height)
    snapshot = occupancy_grid_to_snapshot(
        message, CostmapOverlaySettings(),
        ros_now_sec=10.0, received_at=20.0,
    )
    assert snapshot.analysis_points.shape == (width * height, 3)
    assert snapshot.lethal_cost_cells == 0


def test_costmap_invalid_grid_fails_closed_and_recovers():
    store = CostmapStore(CostmapOverlaySettings(), clock=lambda: 20.0)
    assert store.update(_grid((0, 100, 0, 0)), 10.0)
    oversized = _grid((0,))
    oversized.info.width = 100000
    oversized.info.height = 100000
    assert not store.update(oversized, 10.1)
    assert "limit" in store.reason
    assert store.snapshot is None
    assert store.update(_grid((0, 100, 0, 0), stamp=10.2), 10.2)


@pytest.mark.parametrize("name", [
    "max_grid_width_cells", "max_grid_height_cells", "max_grid_cells"
])
def test_costmap_settings_reject_nonpositive_grid_limits(name):
    with pytest.raises(ValueError, match=name):
        CostmapOverlaySettings(**{name: 0}).validate()
