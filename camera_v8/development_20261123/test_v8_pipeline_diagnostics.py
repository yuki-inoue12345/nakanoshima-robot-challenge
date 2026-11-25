from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from sensor_msgs.msg import Image

import analyze_ar_v8_diagnostics as analyzer
from v8_camera_source import SharedCameraSource
from v8_shared_camera import RosFrameFeed, V8ModelStageDiagnostics


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.callback = None
        self.timer_callback = None
        self.logger = _Logger()

    def create_subscription(self, _type, _topic, callback, _qos):
        self.callback = callback
        return object()

    def create_timer(self, _period, callback):
        self.timer_callback = callback
        return object()

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))


def _message(value, stamp=99.98):
    msg = Image()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
    msg.height = 2
    msg.width = 3
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = 9
    msg.data = np.full((2, 3, 3), value, dtype=np.uint8).tobytes()
    return msg


def test_v8_latest_only_feed_counts_overwrite_and_consumes_matching_pixels():
    node = _Node()
    feed = RosFrameFeed(node, pipeline_diagnostics=True, process_every=5)
    reader = feed.new_reader()
    node.callback(_message(10, 99.96))
    node.callback(_message(20, 99.98))

    ok, frame = reader.read()
    data = feed._metrics.snapshot(reset=False)

    assert ok and np.all(frame == 20)
    assert data['counters']['received'] == 2
    assert data['counters']['accepted'] == 2
    assert data['counters']['overwritten'] == 1
    assert data['counters']['consumed'] == 1
    assert data['counters']['reader_skipped'] == 1


def test_synthetic_30hz_load_never_builds_an_unbounded_frame_queue():
    node = _Node()
    feed = RosFrameFeed(node, pipeline_diagnostics=True)
    reader = feed.new_reader()
    consumed = []
    for sequence in range(300):  # ten seconds of synthetic 30 Hz capture
        node.callback(_message(sequence % 256, 90.0 + sequence / 30.0))
        if sequence % 3 == 2:  # a deliberately slower 10 Hz consumer
            ok, frame = reader.read()
            assert ok
            consumed.append(int(frame[0, 0, 0]))
    counts = feed._metrics.snapshot(reset=False)['counters']
    assert len(consumed) == 100
    assert consumed[-1] == 299 % 256
    assert counts['accepted'] == 300
    assert counts['consumed'] == 100
    assert counts['overwritten'] == 200
    assert counts['reader_skipped'] == 200
    assert not hasattr(feed, '_frame_queue')


def test_v8_model_stage_wrapper_records_without_changing_results_and_restores():
    node = _Node()
    feed = RosFrameFeed(node)

    class Base:
        @staticmethod
        def detect_persons_with_tracking(value):
            return value + 1

    module = SimpleNamespace(
        base=Base,
        call_person_vlm_for_candidate=lambda value: value * 2,
        call_front_photo_vlm=lambda value: value * 3,
    )
    original_yolo = module.base.detect_persons_with_tracking
    original_vlm = module.call_person_vlm_for_candidate

    with V8ModelStageDiagnostics(module, feed):
        assert module.base.detect_persons_with_tracking(2) == 3
        assert module.call_person_vlm_for_candidate(3) == 6
        assert module.call_front_photo_vlm(4) == 12

    assert module.base.detect_persons_with_tracking is original_yolo
    assert module.call_person_vlm_for_candidate is original_vlm
    counts = feed._metrics.snapshot(reset=False)['counters']
    assert counts['yolo_calls'] == 1
    assert counts['vlm_calls'] == 2


def test_v8_periodic_log_distinguishes_bookkeeping_fps_from_measurement():
    node = _Node()
    feed = RosFrameFeed(node, process_every=5)
    node.callback(_message(1))
    feed.set_search_active(True)
    feed.record_model_stage('yolo', 0.01)
    node.timer_callback()
    text = '\n'.join(node.logger.messages)
    assert 'camera_fps_bookkeeping=15.0' in text
    assert 'camera_fps_is_measured=false' in text
    assert 'process_every=5' in text
    assert 'yolo_calls=1' in text


def test_diagnostic_logger_failure_cannot_stop_owner_or_v8_pipeline():
    class BrokenLogger:
        def info(self, _message):
            raise OSError('synthetic log failure')

        def warning(self, _message):
            raise OSError('synthetic log failure')

    owner = SimpleNamespace(
        _maybe_log_pipeline=lambda: (_ for _ in ()).throw(OSError('failed')),
        get_logger=lambda: BrokenLogger(),
    )
    SharedCameraSource._safe_log_pipeline(owner)

    node = _Node()
    feed = RosFrameFeed(node)
    node.logger = BrokenLogger()
    feed._log_pipeline = lambda: (_ for _ in ()).throw(OSError('failed'))
    feed._safe_log_pipeline()


def test_field_collector_uses_real_process_patterns_and_never_records_images():
    script = (Path(__file__).parent / 'collect_ar_v8_diagnostics.sh').read_text()
    assert '/v8_camera_source[.]py' in script
    assert '/ar_path_viewer/lib/ar_path_viewer/ar_path_viewer' in script
    assert '/daikoudai_v8[.]py' in script
    assert 'pidstat -H -h -t -u -r -w' in script
    assert '/koko_v8_camera_source' in script
    assert '/ar_path_viewer' in script
    assert '/camera_all_in_one_person_search' in script
    assert 'analyze_this_run.sh' in script
    assert '/koko/c920/raw_image' not in script.split('ros2 bag record', 1)[1]
    assert '/koko/person_search/raw_image' not in script.split('ros2 bag record', 1)[1]


def _write_synthetic_run(
    tmp_path,
    *,
    owner_rates=(30.0, 30.0, 30.0),
    callback_rates=(30.0, 30.0, 30.0),
    gui_rates=(30.0, 30.0, 30.0),
    display_ages=(80.0, 80.0, 80.0),
    v8_receive=30.0,
    v8_consume=30.0,
    include_v8=True,
    include_post=True,
):
    owner_path = tmp_path / 'owner.log'
    viewer_path = tmp_path / 'viewer.log'
    v8_path = tmp_path / 'v8.log'
    owner_lines = []
    viewer_lines = []
    v8_lines = []
    epoch = 1_800_000_000.0
    phases = ((epoch + 1.0, 0, 'false'), (epoch + 6.0, 1, 'true'))
    if include_post:
        phases += ((epoch + 11.0, 2, 'false'),)
    for stamp, index, active in phases:
        owner_lines.append(
            f'[INFO] [{stamp:.3f}] camera_owner_pipeline '
            f'ar_publish_fps={owner_rates[index]:.1f} '
            f'v8_publish_fps={30.0 if active == "true" else 0.0:.1f} '
            f'v8_active={active} cpu_pct=20.0 rss_mib=100.0')
        iso = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
        viewer_lines.append(
            f'{iso} [INFO] camera_pipeline '
            f'ros_callback_fps_est={callback_rates[index]:.1f} '
            f'gui_new_fps_est={gui_rates[index]:.1f} '
            f'display_age_ms=mean:60.0,p50:55.0,'
            f'p95:{display_ages[index]:.1f},max:{display_ages[index]:.1f} '
            f'process_cpu_pct=35.0 mode=synchronized timestamp_reason=ok')
        if include_v8:
            v8_lines.append(
                f'[INFO] [{stamp + 0.2:.3f}] v8_image_pipeline '
                f'receive_fps={v8_receive if active == "true" else 0.0:.1f} '
                f'consume_fps={v8_consume if active == "true" else 0.0:.1f} '
                f'yolo_fps={2.0 if active == "true" else 0.0:.1f} '
                f'cpu_pct=40.0 rss_mib=500.0')
    owner_lines.insert(
        1,
        f'[INFO] [{epoch + 5.0:.3f}] V8 raw frames enabled by person_area arrival',
    )
    if include_post:
        owner_lines.insert(
            -1,
            f'[INFO] [{epoch + 10.0:.3f}] V8 raw frames stopped at terminal done; sent=150',
        )
    owner_path.write_text('\n'.join(owner_lines) + '\n')
    viewer_path.write_text('\n'.join(viewer_lines) + '\n')
    v8_path.write_text('\n'.join(v8_lines) + '\n')
    return owner_path, viewer_path, v8_path


def test_analyzer_reads_iso_viewer_time_and_compound_display_age(tmp_path):
    paths = _write_synthetic_run(tmp_path)
    report = analyzer.build_report(*paths)
    assert report['phase_source'] == 'exact owner transitions'
    assert report['samples'] == {'owner': 3, 'viewer': 3, 'v8': 3}
    assert report['phases']['normal']['viewer_display_age_ms_p95']['p50'] == 80.0


def test_analyzer_collection_window_excludes_an_older_v8_server_session(tmp_path):
    paths = _write_synthetic_run(tmp_path)
    old_line = (
        '[INFO] [1700000000.000] v8_image_pipeline receive_fps=1.0 '
        'consume_fps=1.0 yolo_fps=0.0 cpu_pct=99.0\n'
    )
    paths[2].write_text(old_line + paths[2].read_text())
    report = analyzer.build_report(*paths, start_epoch=1_800_000_000.0)
    assert report['samples']['v8'] == 3
    assert report['phases']['search']['v8_receive_fps']['p50'] == 30.0


def test_phase_fallback_handles_collection_started_during_search():
    owner = [{
        'time': 100.0,
        'fields': {'v8_active': 'true'},
    }]
    events = [(105.0, 'post_search')]
    assert analyzer.nearest_phase(101.0, owner, events) == 'search'
    assert analyzer.nearest_phase(106.0, owner, events) == 'post_search'


def test_analyzer_finds_owner_to_ar_delivery_drop(tmp_path):
    paths = _write_synthetic_run(tmp_path, callback_rates=(10.0, 10.0, 10.0),
                                 gui_rates=(10.0, 10.0, 10.0))
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith(
        'FIRST: owner-to-AR DDS delivery')


def test_analyzer_finds_viewer_stage_when_callback_is_30_but_gui_is_10(tmp_path):
    paths = _write_synthetic_run(tmp_path, gui_rates=(10.0, 10.0, 10.0))
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith('FIRST: Viewer render')


def test_analyzer_finds_search_only_contention(tmp_path):
    paths = _write_synthetic_run(
        tmp_path,
        gui_rates=(30.0, 10.0, 30.0),
        v8_receive=30.0,
        v8_consume=30.0,
    )
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'] == (
        'FIRST: search-correlated system contention')


def test_analyzer_finds_v8_consumption_when_delivery_is_healthy(tmp_path):
    paths = _write_synthetic_run(
        tmp_path, v8_receive=30.0, v8_consume=8.0)
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith(
        'FIRST: V8 frame consumption')


def test_analyzer_distinguishes_low_fps_from_low_latency(tmp_path):
    paths = _write_synthetic_run(
        tmp_path,
        owner_rates=(15.0, 15.0, 15.0),
        callback_rates=(15.0, 15.0, 15.0),
        gui_rates=(10.0, 10.0, 10.0),
        display_ages=(70.0, 75.0, 80.0),
    )
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith('FIRST: camera owner')
    assert report['latency_assessment'].startswith('LATENCY BOUNDED')


def test_analyzer_detects_latency_accumulation(tmp_path):
    paths = _write_synthetic_run(
        tmp_path, display_ages=(70.0, 190.0, 420.0))
    report = analyzer.build_report(*paths)
    assert report['latency_assessment'].startswith('LATENCY ACCUMULATING')
    assert report['recommended_first_branch'].startswith('FIRST: latency/jitter')


def test_analyzer_refuses_conclusion_when_v8_log_is_missing(tmp_path):
    paths = _write_synthetic_run(tmp_path, include_v8=False)
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith('INSUFFICIENT DATA')


def test_analyzer_refuses_conclusion_when_post_phase_is_missing(tmp_path):
    paths = _write_synthetic_run(tmp_path, include_post=False)
    report = analyzer.build_report(*paths)
    assert report['recommended_first_branch'].startswith('INSUFFICIENT DATA')


def test_analyzer_recognizes_exact_comma_separated_viewer_log_format():
    raw = (
        "camera_pipeline ros_callback_fps_est=30.0 "
        "display_age_ms=mean:70.0,p50:65.0,p95:240.0,max:260.0 "
        "gui_new_fps_est=10.0"
    )
    row = {"fields": analyzer.parse_fields(raw)}
    assert analyzer.compound_value(row, "display_age_ms", "p50") == 65.0
    assert analyzer.compound_value(row, "display_age_ms", "p95") == 240.0
