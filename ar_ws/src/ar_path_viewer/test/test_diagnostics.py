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
