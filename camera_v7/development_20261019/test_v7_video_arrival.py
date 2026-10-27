"""Focused tests for Camera V7 symmetric offline-video search windows."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"
LOADER = importlib.machinery.SourceFileLoader("daikoudai_v7_video_arrival_test", str(UI_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
V7 = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = V7
LOADER.exec_module(V7)


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class VideoArrivalTests(unittest.TestCase):
    def configure_arrival(self, video_mode, area_index, area1, area2):
        search_args = argparse.Namespace()
        V7.configure_search_area_arrival_sec(
            search_args, video_mode, area_index, area1, area2
        )
        return search_args.area_arrival_sec

    def command_ui(self, config, *, include_runtime_tuning=False):
        ui = object.__new__(V7.Milestone2TargetUI)
        ui.current_config = lambda: config
        ui.local_camera_devices = lambda: ["/dev/video-test"]
        if not include_runtime_tuning:
            ui.ms2_runtime_tuning_args = lambda: []
        return ui

    def unwrap_ros_command(self, command):
        if command[:2] == ["bash", "-lc"]:
            return shlex.split(command[2].split("&& exec ", 1)[1])
        return command

    def assert_single_flag_value(self, command, flag, expected):
        self.assertEqual(1, command.count(flag), f"duplicate/missing flag: {flag}")
        index = command.index(flag)
        self.assertEqual(expected, command[index + 1])
        if index + 2 < len(command):
            self.assertTrue(command[index + 2].startswith("--"), command[index:index + 3])

    def base_config(self, input_mode="video"):
        config = {
            "input_mode": input_mode,
            "video": "/tmp/arrival.mp4",
            "video_area1_center_sec": "405.0",
            "video_area2_center_sec": "1055.0",
            "video_search_half_window_sec": "5.0",
            "camera_index": "0",
            "camera_device": "",
            "person_arrival_topic": "/person_area_arrived",
            "person_done_topic": "/person_search_done",
            "person_detail_topic": "/person_search_done_detail",
            "targets": [{"target_id": "target_A"}, {"target_id": "target_B"}],
            "record_run_video": False,
            "search_duration_sec": "20",
            "max_total_vlm_calls": "6",
            "decision_accept": "candidate",
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
        }
        config.update({
            key: value
            for key, value in V7.DEFAULT_TARGET_PROFILES["target_A"].items()
            if key in V7.TARGET_FIELD_KEYS
        })
        return config

    def config_ui(self):
        ui = object.__new__(V7.Milestone2TargetUI)
        values = {
            "video": "",
            "video_area1_center_sec": "0.0",
            "video_area2_center_sec": "0.0",
            "video_search_half_window_sec": "5.0",
            "camera_index": "0",
            "camera_device": "",
            "search_target_mode": "both",
            "search_duration_sec": "20",
            "max_total_vlm_calls": "6",
            "decision_accept": "candidate",
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            "capture_request_topic": "/camera_capture_request",
            "capture_done_topic": "/camera_capture_done",
            "capture_out_dir": "outputs/topic_camera_capture",
            "person_arrival_topic": "/person_area_arrived",
            "person_done_topic": "/person_search_done",
            "person_detail_topic": "/person_search_done_detail",
        }
        ui.vars = {key: FakeVar(value) for key, value in values.items()}
        ui.input_mode = FakeVar("camera")
        ui.record_run_video_var = FakeVar(False)
        ui.quick_text_target_A = FakeVar("")
        ui.quick_text_target_B = FakeVar("")
        ui.value_from_combo = lambda key, _options: ui.vars[key].get()
        ui.set_combo_value = lambda key, _options, value: ui.vars[key].set(value)
        ui.target_profile_from_ui = lambda target_id: dict(V7.DEFAULT_TARGET_PROFILES[target_id])
        ui.apply_target_profile_to_ui = lambda _target_id, _profile: None
        ui.log = lambda _message: None
        return ui

    def test_01_area1_center_and_half_window(self):
        self.assertEqual(
            (400.0, 10.0),
            V7.resolve_video_area_search_window(1, "405", "1055", "5"),
        )

    def test_02_area2_center_and_half_window(self):
        self.assertEqual(
            (1050.0, 10.0),
            V7.resolve_video_area_search_window(2, "405", "1055", "5"),
        )

    def test_03_direct_video_command_uses_derived_window(self):
        command = self.command_ui(self.base_config()).direct_command()
        self.assert_single_flag_value(command, "--area_arrival_sec", "400.0")
        self.assert_single_flag_value(command, "--search_duration_sec", "10.0")
        self.assertNotIn("--video_area1_arrival_sec", command)
        self.assertNotIn("--video_area2_arrival_sec", command)

    def test_04_ros_video_command_uses_both_derived_windows(self):
        command = self.unwrap_ros_command(self.command_ui(self.base_config()).ros_command())
        self.assert_single_flag_value(command, "--video_area1_arrival_sec", "400.0")
        self.assert_single_flag_value(command, "--video_area2_arrival_sec", "1050.0")
        self.assert_single_flag_value(command, "--search_duration_sec", "10.0")

    def test_05_generated_flags_have_one_value_each(self):
        direct = self.command_ui(self.base_config()).direct_command()
        server = self.unwrap_ros_command(self.command_ui(self.base_config()).ros_command())
        for command, expected in [
            (direct, {"--area_arrival_sec": "400.0", "--search_duration_sec": "10.0"}),
            (server, {
                "--video_area1_arrival_sec": "400.0",
                "--video_area2_arrival_sec": "1050.0",
                "--search_duration_sec": "10.0",
            }),
        ]:
            for flag, value in expected.items():
                self.assert_single_flag_value(command, flag, value)

    def test_06_camera_command_ignores_video_window_and_keeps_duration(self):
        config = self.base_config("camera")
        config.update(
            video_area1_center_sec="not-a-number",
            video_area2_center_sec="nan",
            video_search_half_window_sec="-1",
            search_duration_sec="20",
        )
        command = self.command_ui(config).direct_command()
        self.assertNotIn("--video", command)
        self.assertNotIn("--area_arrival_sec", command)
        self.assert_single_flag_value(command, "--search_duration_sec", "20")
        self.assertEqual(0.0, self.configure_arrival(False, 1, "invalid", "invalid"))

    def test_07_center_before_half_window_clips_start_to_zero(self):
        self.assertEqual((0.0, 10.0), V7.resolve_video_search_window("3", "5"))

    def test_08_negative_center_is_rejected(self):
        with self.assertRaises(ValueError):
            V7.resolve_video_search_window("-0.1", "5")

    def test_09_nan_center_is_rejected(self):
        with self.assertRaises(ValueError):
            V7.resolve_video_search_window("nan", "5")

    def test_10_infinite_center_is_rejected(self):
        with self.assertRaises(ValueError):
            V7.resolve_video_search_window("inf", "5")

    def test_11_nonpositive_half_window_is_rejected(self):
        for value in ("0", "-0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                V7.resolve_video_search_window("405", value)

    def test_12_nonfinite_half_window_is_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                V7.resolve_video_search_window("405", value)

    def test_13_one_person_mode_selects_target_a_only(self):
        ui = object.__new__(V7.Milestone2TargetUI)
        ui.value_from_combo = lambda _key, _options: "target_A"
        self.assertEqual(["target_A"], ui.selected_target_ids())

    def test_14_two_person_mode_selects_a_and_b(self):
        ui = object.__new__(V7.Milestone2TargetUI)
        ui.value_from_combo = lambda _key, _options: "both"
        self.assertEqual(["target_A", "target_B"], ui.selected_target_ids())

    def test_15_target_b_only_mode_does_not_exist(self):
        mode_values = [value for _label, value in V7.SEARCH_TARGET_MODE_OPTIONS]
        self.assertEqual(["target_A", "both"], mode_values)

    def test_16_area_occurrence_maps_area1_to_a_and_area2_to_b(self):
        profiles = [
            dict(V7.DEFAULT_TARGET_PROFILES["target_A"]),
            dict(V7.DEFAULT_TARGET_PROFILES["target_B"]),
        ]
        states = {"target_A": {"ms2_success": False}, "target_B": {"ms2_success": False}}
        self.assertEqual("target_A", V7.select_area_target_profile(profiles, states, 1)["target_id"])
        self.assertEqual("target_B", V7.select_area_target_profile(profiles, states, 2)["target_id"])

    def test_17_non_numeric_center_and_half_window_are_rejected(self):
        with self.assertRaises(ValueError):
            V7.resolve_video_search_window("invalid", "5")
        with self.assertRaises(ValueError):
            V7.resolve_video_search_window("405", "invalid")

    def test_18_config_save_and_load_preserves_video_window(self):
        with tempfile.TemporaryDirectory(prefix="v7_video_window_config_", dir="/tmp") as temp:
            original_path = V7.LAST_CONFIG_PATH
            V7.LAST_CONFIG_PATH = Path(temp) / "config.json"
            self.addCleanup(setattr, V7, "LAST_CONFIG_PATH", original_path)
            source = self.config_ui()
            source.input_mode.set("video")
            source.vars["video_area1_center_sec"].set("405.0")
            source.vars["video_area2_center_sec"].set("1055.0")
            source.vars["video_search_half_window_sec"].set("5.0")
            source.save_config(silent=True)
            saved = json.loads(V7.LAST_CONFIG_PATH.read_text(encoding="utf-8"))
            self.assertEqual("405.0", saved["video_area1_center_sec"])
            self.assertEqual("1055.0", saved["video_area2_center_sec"])
            self.assertEqual("5.0", saved["video_search_half_window_sec"])
            self.assertNotIn("video_area1_arrival_sec", saved)
            self.assertNotIn("video_area2_arrival_sec", saved)

            restored = self.config_ui()
            restored._load_last_config(silent=True)
            self.assertEqual("405.0", restored.vars["video_area1_center_sec"].get())
            self.assertEqual("1055.0", restored.vars["video_area2_center_sec"].get())
            self.assertEqual("5.0", restored.vars["video_search_half_window_sec"].get())

    def test_19_server_low_level_arrival_helper_remains_compatible(self):
        self.assertEqual(400.0, self.configure_arrival(True, 1, "400", "1050"))
        self.assertEqual(1050.0, self.configure_arrival(True, 2, "400", "1050"))
        self.assertEqual(0.0, self.configure_arrival(False, 1, "invalid", "invalid"))

    def test_20_unsupported_video_area_is_rejected(self):
        with self.assertRaises(RuntimeError):
            V7.resolve_video_area_search_window(3, "405", "1055", "5")

    def test_21_generated_commands_use_384_person_max_tokens_once(self):
        ui = self.command_ui(self.base_config(), include_runtime_tuning=True)
        direct = ui.direct_command()
        server = self.unwrap_ros_command(ui.ros_command())
        self.assert_single_flag_value(direct, "--person_max_tokens", "384")
        self.assert_single_flag_value(server, "--person_max_tokens", "384")


if __name__ == "__main__":
    unittest.main()
