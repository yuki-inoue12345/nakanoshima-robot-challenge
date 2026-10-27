"""Regression tests for persistent Camera V7 UI diagnostic logs."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"
MS2_PATH = ROOT / "person_search_vlm_milestone2_v7.py"


def load_ui_module():
    loader = importlib.machinery.SourceFileLoader("v7_log_diagnostics_ui", str(UI_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def read_prompt(name: str) -> str:
    tree = ast.parse(MS2_PATH.read_text(encoding="utf-8"), filename=str(MS2_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            value = value.func.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value.strip()
    raise AssertionError(f"prompt not found: {name}")


V7 = load_ui_module()


class LogDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_append_is_kept_in_memory_and_flushed_to_file(self):
        path = V7.create_ui_log_file(self.root)
        store = V7.UiLogStore(path)
        self.assertTrue(store.append("first line"))
        self.assertEqual(["first line"], store.history)
        self.assertEqual("first line\n", path.read_text(encoding="utf-8"))

    def test_multiple_lines_preserve_order(self):
        path = V7.create_ui_log_file(self.root)
        store = V7.UiLogStore(path)
        store.append("one")
        store.append("two\nthree")
        self.assertEqual(["one", "two", "three"], store.history)
        self.assertEqual(["one", "two", "three"], path.read_text(encoding="utf-8").splitlines())

    def test_background_queue_persists_before_ui_drain(self):
        path = V7.create_ui_log_file(self.root)
        store = V7.UiLogStore(path)
        log_queue = V7.PersistingLogQueue(store.append)
        log_queue.put("subprocess stdout")
        self.assertEqual("subprocess stdout\n", path.read_text(encoding="utf-8"))
        self.assertEqual(("subprocess stdout", True), log_queue.get_nowait())

    def test_error_classification(self):
        self.assertIn(V7.LOG_CATEGORY_ERROR, V7.classify_log_line("Traceback: ERROR failed"))

    def test_ros_classification(self):
        self.assertIn(V7.LOG_CATEGORY_ROS, V7.classify_log_line("ARRIVAL_ACK_PUBLISH_SUCCEEDED"))

    def test_recording_classification(self):
        self.assertIn(V7.LOG_CATEGORY_RECORDING, V7.classify_log_line("recording completed frame_count=623"))

    def test_shutdown_classification(self):
        self.assertIn(V7.LOG_CATEGORY_SHUTDOWN, V7.classify_log_line("[SHUTDOWN_ORIGIN] signal=SIGTERM"))

    def test_person_search_classification(self):
        self.assertIn(V7.LOG_CATEGORY_SEARCH, V7.classify_log_line("[SEARCH_STAGE] WORKER_ENTERED"))

    def test_important_is_union_of_key_events(self):
        for line in (
            "ERROR camera failed",
            "WORKER_ENTERED",
            "ARRIVAL_CALLBACK_ENTERED",
            "TERMINAL_DONE_PUBLISH_SUCCEEDED",
            "recording completed",
            "saved: /tmp/image.jpg",
        ):
            with self.subTest(line=line):
                self.assertIn(V7.LOG_CATEGORY_IMPORTANT, V7.classify_log_line(line))

    def test_search_and_category_filters_are_combined(self):
        lines = [
            "ARRIVAL_ACK_PUBLISH_SUCCEEDED",
            "ARRIVAL_CALLBACK_ENTERED",
            "recording ACK marker",
            "ordinary line",
        ]
        self.assertEqual(
            ["ARRIVAL_ACK_PUBLISH_SUCCEEDED"],
            V7.filter_log_lines(lines, V7.LOG_CATEGORY_ROS, "ACK_PUBLISH"),
        )

    def test_shutdown_origin_sigterm_has_highest_priority(self):
        summary = V7.summarize_ui_logs([
            "$ python server.py --person-search-server",
            "[SHUTDOWN_ORIGIN] signal=SIGTERM signum=15",
            "[終了] returncode=-15",
        ])
        self.assertEqual("SIGTERM", summary["exit_reason"])
        self.assertEqual("-15", summary["exit_code"])

    def test_keyboard_stop_is_summarized_as_preview_stop(self):
        summary = V7.summarize_ui_logs([
            "$ python server.py --capture-server",
            "keyboard stop requested from camera preview",
        ])
        self.assertEqual("F10/プレビュー停止", summary["exit_reason"])

    def test_nonzero_returncode_is_abnormal_exit(self):
        summary = V7.summarize_ui_logs(["$ python worker.py", "[終了] returncode=7"])
        self.assertEqual("subprocess異常終了", summary["exit_reason"])
        self.assertEqual("7", summary["exit_code"])

    def test_recording_summary_uses_only_logged_facts(self):
        summary = V7.summarize_ui_logs([
            "$ python server.py --capture-server",
            "[CAMERA_RECORDING] recording started",
            "[CAMERA_RECORDING] frame_count=623",
            "[CAMERA_RECORDING] recording completed",
        ])
        self.assertEqual("completed", summary["recording_state"])
        self.assertEqual("確認済み", summary["recording_started"])
        self.assertEqual("623", summary["recording_frame_count"])

    def test_log_save_failure_never_raises_into_camera_flow(self):
        store = V7.UiLogStore(self.root)  # A directory cannot be opened as a file.
        self.assertFalse(store.append("must remain in memory"))
        self.assertEqual(["must remain in memory"], store.history)
        self.assertIn("IsADirectoryError", store.last_error)

    def test_log_filename_is_timestamped_unique_and_latest_is_not_overwritten(self):
        now = dt.datetime(2026, 9, 30, 14, 15, 30).astimezone()
        first = V7.create_ui_log_file(self.root, started_at=now)
        second = V7.create_ui_log_file(self.root, started_at=now)
        self.assertEqual("camera_v7_20260930_141530.log", first.name)
        self.assertEqual("camera_v7_20260930_141530_001.log", second.name)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_latest_execution_summary_does_not_reuse_old_success(self):
        old_detail = json.dumps({"status": "FOUND_FRONT_PHOTO", "success": True})
        summary = V7.summarize_ui_logs([
            "$ python first.py --person-search-server",
            "mission_detail:" + old_detail,
            "$ python second.py --person-search-server",
            "ARRIVAL_CALLBACK_ENTERED",
        ])
        self.assertEqual("未実行", summary["search_result"])
        self.assertEqual("received", summary["ros_arrival"])

    def test_existing_person_search_result_contract_is_unchanged(self):
        outcome = V7.classify_area_search_results({
            "target_A": {
                "status": "FOUND_FRONT_PHOTO",
                "ms2_success": True,
                "final_score": 0.9,
            }
        })
        self.assertEqual("FOUND_FRONT_PHOTO", outcome["status"])
        self.assertTrue(outcome["success"])
        self.assertTrue(outcome["safe_to_advance"])

    def test_prompt_sha_is_unchanged(self):
        expected = {
            "MS2_PERSON_PROMPT": "94abc619e21424b420f6faf19542e4b149303f36c9428ddca17f8c5bc332d7de",
            "FRONT_PHOTO_PROMPT": "482706fa783943c0b443e81e8b3a508be9490bff5214105ec72bf7f7c3f6ee3e",
        }
        for name, digest in expected.items():
            self.assertEqual(digest, hashlib.sha256(read_prompt(name).encode()).hexdigest())

    def test_ros_interface_and_qos_source_contract_is_unchanged(self):
        source = inspect.getsource(V7.run_person_search_server_mode)
        for topic in (
            "/person_area_arrived",
            "/person_search_done",
            "/person_search_done_detail",
            "/person_search/arrival_received",
        ):
            self.assertIn(topic, UI_PATH.read_text(encoding="utf-8"))
        self.assertIn("ReliabilityPolicy.RELIABLE", source)
        self.assertIn("DurabilityPolicy.TRANSIENT_LOCAL", source)
        self.assertIn("depth=1", source)
        self.assertIn("DurabilityPolicy.VOLATILE", source)
        self.assertIn("depth=10", source)
        self.assertIn("person_search_done_safe_to_advance_v1", UI_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
