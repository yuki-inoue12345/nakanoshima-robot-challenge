from datetime import datetime, timezone

from ar_path_viewer.log_gui import LogGui
from ar_path_viewer.session_log import SessionLog, resolve_session_paths


FIXED_TIME = datetime(2026, 11, 6, 18, 22, 6, tzinfo=timezone.utc)


def test_recording_and_log_names_are_paired_and_never_overwrite(tmp_path):
    requested = tmp_path / "kokochan_navi_20261106_182206.mp4"
    requested.write_bytes(b"existing recording")
    paths = resolve_session_paths(str(requested), "", str(tmp_path), FIXED_TIME)
    assert paths.video == tmp_path / "kokochan_navi_20261106_182206_001.mp4"
    assert paths.log == tmp_path / "kokochan_navi_20261106_182206_001.log"
    assert requested.read_bytes() == b"existing recording"

    paths.log.write_text("existing log", encoding="utf-8")
    next_paths = resolve_session_paths(str(paths.video), "", str(tmp_path), FIXED_TIME)
    assert next_paths.video == tmp_path / "kokochan_navi_20261106_182206_001_001.mp4"
    assert next_paths.log == tmp_path / "kokochan_navi_20261106_182206_001_001.log"


def test_no_video_uses_timestamped_default_log_name(tmp_path):
    paths = resolve_session_paths("", "", str(tmp_path), FIXED_TIME)
    assert paths.video is None
    assert paths.log == tmp_path / "kokochan_navi_20261106_182206.log"


def test_session_log_streams_to_disk_and_keeps_selectable_text(tmp_path):
    timestamps = iter((FIXED_TIME,) * 10)
    log = SessionLog(
        tmp_path / "session.log",
        max_entries=100,
        now=lambda: next(timestamps),
    )
    log.record("INFO", "camera_configuration camera_pitch_up_deg=7.500")
    log.record("WARNING", "viewer_state current='PATH OUT OF VIEW'")
    assert "camera_configuration" in log.text()
    assert "PATH OUT OF VIEW" in log.text()
    log.close()
    saved = (tmp_path / "session.log").read_text(encoding="utf-8")
    assert "session_start" in saved
    assert "camera_configuration" in saved
    assert "session_end" in saved


def test_copy_text_prioritizes_latest_projection_configuration_and_errors(tmp_path):
    log = SessionLog(tmp_path / "copy.log", max_entries=200)
    log.record("INFO", "camera_configuration fx=525 camera_pitch_up_deg=7.5")
    for index in range(150):
        log.record("INFO", f"routine filler index={index} " + "x" * 40)
    log.record("ERROR", "camera exception test failure")
    log.record(
        "INFO",
        "path_projection path_length_m=8.2 endpoint_distance_m=7.9 "
        "camera_forward_path_length_m=7.5 visible_path_length_m=2.1 "
        "nonvisible_forward_path_length_m=5.4",
    )
    copied = log.copy_text(3000)
    assert len(copied) <= 3000
    assert "log truncated for clipboard" in copied
    assert "camera_configuration" in copied
    assert "camera exception test failure" in copied
    assert "path_projection" in copied
    assert "path_length_m=8.2" in copied


def test_file_failure_keeps_in_memory_log_available(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file", encoding="utf-8")
    log = SessionLog(blocker / "session.log")
    log.record("ERROR", "still available")
    assert log.file_error
    assert "still available" in log.text()
    log.close()


class FakeClipboardRoot:
    def __init__(self):
        self.clipboard = ""
        self.destroyed = False

    def clipboard_clear(self):
        self.clipboard = ""

    def clipboard_append(self, content):
        self.clipboard += content

    def update_idletasks(self):
        pass

    def update(self):
        pass

    def destroy(self):
        self.destroyed = True


def test_clipboard_success_and_failure_are_nonfatal():
    gui = LogGui()
    root = FakeClipboardRoot()
    gui._root = root
    success, message = gui.copy("path_projection test")
    assert success
    assert root.clipboard == "path_projection test"
    assert "コピーしました" in message
    gui.close()
    assert root.destroyed

    failing = LogGui()
    failing._ensure_root = lambda: (_ for _ in ()).throw(RuntimeError("no display"))
    success, message = failing.copy("test")
    assert not success
    assert "no display" in message
    success, message = failing.show("test")
    assert not success
    assert "no display" in message
