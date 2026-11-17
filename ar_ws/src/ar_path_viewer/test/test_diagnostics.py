from types import SimpleNamespace

from rclpy.logging import get_logger

from ar_path_viewer.diagnostics import DiagnosticReporter


class RecordingLogger:
    def __init__(self):
        self.records = []

    def info(self, message):
        self.records.append(("info", message))

    def warning(self, message):
        self.records.append(("warning", message))


def test_receive_counter_timestamp_frame_and_points_are_logged_without_spam():
    now = [10.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 5.0, clock=lambda: now[0])
    reporter.path_received("map", 12, 123.25)
    now[0] = 11.0
    reporter.path_received("map", 13, 124.25)
    assert len(logger.records) == 1
    assert "count=1" in logger.records[0][1]
    assert "received_monotonic=10.000000" in logger.records[0][1]
    assert "header_stamp=123.250000000" in logger.records[0][1]
    assert "frame_id='map' points=12" in logger.records[0][1]
    now[0] = 15.1
    reporter.path_received("map", 14, 128.25)
    assert len(logger.records) == 2
    assert "count=3" in logger.records[-1][1]


def test_path_outcomes_and_state_changes_include_no_path_cause():
    now = [1.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, clock=lambda: now[0])
    reporter.state_changed("NO PATH", reporter.no_path_detail())
    reporter.state_changed("NO PATH", "must not repeat")
    reporter.path_received("map", 0, 0.0)
    reporter.path_discarded("empty Path; current route cleared")
    reporter.state_changed("ACTIVE", "generation=2")
    reporter.state_changed("NO PATH", reporter.no_path_detail())
    text = "\n".join(message for _, message in logger.records)
    assert "no Path received since startup" in text
    assert "plan_discarded" in text
    assert "empty Path" in text
    assert text.count("current='NO PATH'") == 2


def test_tf_failure_and_out_of_view_are_rate_limited_with_reasons():
    now = [20.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 5.0, clock=lambda: now[0])
    reporter.tf_failure("map", "robot/base_footprint", RuntimeError("lookup timeout"))
    reporter.tf_failure("map", "robot/base_footprint", RuntimeError("lookup timeout"))
    reporter.path_out_of_view("map", 99)
    reporter.path_out_of_view("map", 99)
    assert len(logger.records) == 2
    text = "\n".join(message for _, message in logger.records)
    assert "lookup timeout" in text
    assert "no clipped segment intersects the camera frustum" in text


def test_diagnostics_can_be_disabled():
    logger = RecordingLogger()
    reporter = DiagnosticReporter(False, logger)
    reporter.path_received("map", 1, 0.0)
    reporter.path_discarded("bad")
    reporter.state_changed("NO PATH", "disabled")
    assert reporter.path_receive_count == 1
    assert logger.records == []


def test_localization_diagnostics_are_rate_limited_but_log_transitions():
    now = [10.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 5.0, clock=lambda: now[0])
    reporter.localization_update("LOCALIZING", "first valid sample", 1)
    now[0] = 11.0
    reporter.localization_update("LOCALIZING", "same state", 2)
    reporter.localization_update("LOCALIZATION OK", "stable", 2)
    assert len(logger.records) == 2
    assert "state='LOCALIZING'" in logger.records[0][1]
    assert "state='LOCALIZATION OK'" in logger.records[1][1]


def test_real_rclpy_logger_allows_warning_info_warning_transitions():
    reporter = DiagnosticReporter(
        True,
        get_logger("ar_path_viewer_diagnostics_severity_regression"),
    )
    reporter.state_changed("NO PATH", "waiting for /plan")
    reporter.path_received("map", 2, 1.0)
    reporter.state_changed("PATH STALE", "test transition")
    assert reporter.path_receive_count == 1
    assert reporter.current_state == "PATH STALE"


def test_projection_diagnostics_are_rate_limited_per_visibility_state():
    now = [10.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 5.0, clock=lambda: now[0])
    detail = "transformed_points=119 front_points=80 image_points=20 drawable_segments=19"
    reporter.projection_update("map", detail)
    reporter.projection_update("map", detail)
    reporter.projection_update("map", detail, "all forward Path points are below the image")
    reporter.projection_update("map", detail, "all forward Path points are below the image")
    assert len(logger.records) == 2
    now[0] = 15.1
    reporter.projection_update("map", detail)
    assert len(logger.records) == 3
    assert "drawable_segments=19" in logger.records[-1][1]


def test_effective_camera_configuration_is_logged_once_with_projection_values():
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger)
    reporter.camera_configuration(
        "width=640 height=480 fx=525.000 fy=525.000 "
        "camera_y=0.0940 camera_pitch_up_deg=7.500"
    )
    assert logger.records == [
        (
            "info",
            "camera_configuration width=640 height=480 "
            "fx=525.000 fy=525.000 camera_y=0.0940 "
            "camera_pitch_up_deg=7.500",
        )
    ]


def test_camera_pipeline_counts_reads_sync_queue_rendering_and_recording():
    now = [10.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 5.0, clock=lambda: now[0])
    reporter.camera_pipeline(
        read_success=True,
        timestamp_usable=False,
        timestamp_reason="camera_timestamp_missing",
        mode="preview_sync_wait",
        pending_frames=0,
        discarded_frames=2,
        recorded=True,
        source_metrics=SimpleNamespace(
            transport_age_sec=0.035,
            read_duration_sec=0.008,
            dequeue_duration_sec=0.001,
            decode_duration_sec=0.004,
            frame_timestamp_monotonic_sec=123.4,
            dequeue_monotonic_sec=123.435,
            sequence=42,
            sequence_gap=3,
            dequeued_buffers=4,
            discarded_buffers=3,
            total_discarded_buffers=9,
            error_reason="",
        ),
        tick_interval_sec=0.05,
        tick_duration_sec=0.012,
    )
    now[0] = 10.1
    reporter.camera_pipeline(
        read_success=True,
        timestamp_usable=True,
        timestamp_reason="v4l2_monotonic_soe",
        mode="synchronized",
        pending_frames=3,
        discarded_frames=2,
        recorded=True,
    )
    text = "\n".join(message for _, message in logger.records)
    assert "camera_pipeline reads=1 read_success=1" in text
    assert "timestamp_usable=0" in text
    assert "rendered=1" in text
    assert "recorded=1 pending=0 discarded=2" in text
    assert "transport_age_ms=p50:35.0,p95:35.0,max:35.0" in text
    assert "read_ms=p50:8.0,p95:8.0,max:8.0" in text
    assert "dequeue_ms=p50:1.0,p95:1.0,max:1.0" in text
    assert "decode_ms=p50:4.0,p95:4.0,max:4.0" in text
    assert "tick_interval_ms=p50:50.0,p95:50.0,max:50.0" in text
    assert "v4l2_sequence=42" in text
    assert "sequence_gap=3 dequeued=4" in text
    assert "v4l2_stamp_mono=123.4 dq_mono=123.435" in text
    assert "source_discarded=3 source_discarded_total=9" in text
    assert reporter.camera_read_count == 2
    assert reporter.camera_read_success_count == 2
    assert reporter.camera_timestamp_usable_count == 1
    assert reporter.camera_display_frame_count == 2
    assert reporter.camera_synchronized_frame_count == 1
    assert reporter.camera_recorded_frame_count == 2


def test_costmap_visual_diagnostics_separate_live_hold_and_hidden_counts():
    now = [1.0]
    logger = RecordingLogger()
    reporter = DiagnosticReporter(True, logger, 0.01, clock=lambda: now[0])
    reporter.costmap_visual_display("live", 0.0, "exact_time_synchronized")
    now[0] += 0.02
    reporter.costmap_visual_display(
        "hold", 0.18, "last_exact_time_projection_visual_only"
    )
    now[0] += 0.02
    reporter.costmap_visual_display(
        "hidden", 0.0, "costmap_history_stale_for_image"
    )

    text = "\n".join(message for _, message in logger.records)
    assert "mode=live" in text
    assert "mode=hold hold_age_sec=0.180" in text
    assert "mode=hidden" in text
    assert reporter.costmap_live_display_count == 1
    assert reporter.costmap_hold_display_count == 1
    assert reporter.costmap_hidden_display_count == 1


def test_emitted_diagnostics_are_forwarded_to_session_sink():
    logger = RecordingLogger()
    saved = []
    reporter = DiagnosticReporter(
        True, logger, record_sink=lambda level, message: saved.append((level, message))
    )
    reporter.path_received("map", 119, 123.0)
    reporter.projection_update(
        "map", "path_length_m=8.0 visible_path_length_m=3.0"
    )
    assert saved[0][0] == "info"
    assert "plan_rx" in saved[0][1]
    assert "path_projection" in saved[1][1]
    assert "path_length_m=8.0" in saved[1][1]
