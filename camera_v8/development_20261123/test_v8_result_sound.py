"""Hardware-free tests for V7-compatible sound restore in Camera V8."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import v8_result_sound as sound

HERE = Path(__file__).resolve().parent


class V8ResultSoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.found = root / "zundamon_found.wav"
        self.not_found = root / "zundamon_not_found.wav"
        self.player = root / "cvlc"
        for f in (self.found, self.not_found):
            f.write_bytes(b"RIFFdummy")
        self.player.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.player.chmod(0o755)
        for name, value in (
            ("PERSON_SEARCH_FOUND_SOUND", self.found),
            ("PERSON_SEARCH_NOT_FOUND_SOUND", self.not_found),
            ("PERSON_SEARCH_NOTIFICATION_PLAYER", self.player),
        ):
            p = patch.object(sound, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.logger = Mock()

    def test_found_plays_original_found_file(self):
        with patch("v8_result_sound.subprocess.Popen") as popen:
            self.assertTrue(sound.play_person_search_notification("FOUND_FRONT_PHOTO", self.logger))
            popen.assert_called_once()
            self.assertEqual(popen.call_args.args[0][-1], str(self.found))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertTrue(popen.call_args.kwargs["close_fds"])

    def test_not_found_plays_original_not_found_file(self):
        with patch("v8_result_sound.subprocess.Popen") as popen:
            self.assertTrue(sound.play_person_search_notification("NOT_FOUND_OR_NOT_FRONT", self.logger))
            popen.assert_called_once()
            self.assertEqual(popen.call_args.args[0][-1], str(self.not_found))

    def test_errors_and_diagnostics_never_play(self):
        with patch("v8_result_sound.subprocess.Popen") as popen:
            for status in ("ERROR", "ERROR_BUSY_CONFLICT", "BUSY", "WORKER_SHUTDOWN", "", None):
                self.assertFalse(sound.play_person_search_notification(status, self.logger))
            popen.assert_not_called()

    def test_missing_found_file_does_not_play(self):
        self.found.unlink()
        with patch("v8_result_sound.subprocess.Popen") as popen:
            self.assertFalse(sound.play_person_search_notification("FOUND_FRONT_PHOTO", self.logger))
            popen.assert_not_called()
            self.logger.warning.assert_called()

    def test_missing_not_found_file_does_not_play(self):
        self.not_found.unlink()
        with patch("v8_result_sound.subprocess.Popen") as popen:
            self.assertFalse(sound.play_person_search_notification("NOT_FOUND_OR_NOT_FRONT", self.logger))
            popen.assert_not_called()

    def test_missing_player_does_not_play(self):
        self.player.unlink()
        with patch("v8_result_sound.subprocess.Popen") as popen:
            self.assertFalse(sound.play_person_search_notification("FOUND_FRONT_PHOTO", self.logger))
            popen.assert_not_called()
            self.logger.warning.assert_called()

    def test_player_failure_does_not_break_result(self):
        with patch("v8_result_sound.subprocess.Popen", side_effect=OSError("audio unavailable")):
            self.assertFalse(sound.play_person_search_notification("NOT_FOUND_OR_NOT_FRONT", self.logger))
            self.logger.warning.assert_called()

    def test_engine_calls_audio_only_after_successful_terminal_publish(self):
        script = (HERE / "daikoudai_v8.py").read_text(encoding="utf-8")
        ast.parse(script)
        publish = 'self.publish_result(bool(detail["safe_to_advance"]), detail)'
        assign = 'notification_status = str(detail.get("status", ""))'
        call = 'play_person_search_notification(\n                        notification_status,\n                        self.get_logger(),\n                    )'
        self.assertEqual(script.count(publish), 1)
        self.assertEqual(script.count(assign), 1)
        self.assertEqual(script.count('if notification_status:'), 1)
        self.assertEqual(script.count(call), 1)
        self.assertLess(script.index(publish), script.index(assign))
        self.assertLess(script.index(assign), script.index(call))

    def test_audio_module_never_opens_camera_or_publishes_navigation(self):
        script = (HERE / "v8_result_sound.py").read_text(encoding="utf-8")
        ast.parse(script)
        for forbidden in ("VideoCapture(", "VideoWriter(", "/cmd_vel", "/navigate_to_pose", "rclpy"):
            self.assertNotIn(forbidden, script)


if __name__ == "__main__":
    unittest.main()
