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
