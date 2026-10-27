"""Regression tests for structured person-search result audio notifications."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"


def load_ui_module():
    loader = importlib.machinery.SourceFileLoader(
        "v7_voice_notification_ui", str(UI_PATH)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


UI = load_ui_module()


class PersonSearchVoiceNotificationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.player = self.base / "cvlc"
        self.player.write_bytes(b"player")
        self.player.chmod(0o755)
        self.found_sound = self.base / "found.wav"
        self.found_sound.write_bytes(b"found")
        self.not_found_sound = self.base / "not_found.wav"
        self.not_found_sound.write_bytes(b"not-found")
        self.logger = mock.Mock()
        self.path_patches = (
            mock.patch.object(UI, "PERSON_SEARCH_NOTIFICATION_PLAYER", self.player),
            mock.patch.object(UI, "PERSON_SEARCH_FOUND_SOUND", self.found_sound),
            mock.patch.object(
                UI, "PERSON_SEARCH_NOT_FOUND_SOUND", self.not_found_sound
            ),
        )
        for patcher in self.path_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_async_launch(self, status, expected_sound):
        with mock.patch.object(UI.subprocess, "Popen") as popen:
            self.assertTrue(UI.play_person_search_notification(status, self.logger))
        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(str(self.player), args[0][0])
        self.assertEqual(str(expected_sound), args[0][-1])
        self.assertIn("--play-and-exit", args[0])
        self.assertIs(UI.subprocess.DEVNULL, kwargs["stdout"])
        self.assertIs(UI.subprocess.DEVNULL, kwargs["stderr"])
        self.assertTrue(kwargs["start_new_session"])

    def test_found_starts_found_sound_once(self):
        self.assert_async_launch("FOUND_FRONT_PHOTO", self.found_sound)

    def test_not_found_starts_not_found_sound_once(self):
        self.assert_async_launch(
            "NOT_FOUND_OR_NOT_FRONT", self.not_found_sound
        )

    def test_error_statuses_are_silent(self):
        with mock.patch.object(UI.subprocess, "Popen") as popen:
            for status in ("ERROR", "ERROR_BUSY_CONFLICT", "ERROR_ANYTHING"):
                with self.subTest(status=status):
                    self.assertFalse(
                        UI.play_person_search_notification(status, self.logger)
                    )
        popen.assert_not_called()

    def test_missing_sound_warns_without_raising(self):
        missing = self.base / "missing.wav"
        with mock.patch.object(UI, "PERSON_SEARCH_FOUND_SOUND", missing), mock.patch.object(
            UI.subprocess, "Popen"
        ) as popen:
            self.assertFalse(
                UI.play_person_search_notification(
                    "FOUND_FRONT_PHOTO", self.logger
                )
            )
        popen.assert_not_called()
        self.logger.warning.assert_called_once()

    def test_popen_failure_warns_without_raising(self):
        with mock.patch.object(
            UI.subprocess, "Popen", side_effect=OSError("injected failure")
        ) as popen:
            self.assertFalse(
                UI.play_person_search_notification(
                    "FOUND_FRONT_PHOTO", self.logger
                )
            )
        popen.assert_called_once()
        self.logger.warning.assert_called_once()

    def test_server_worker_has_one_structured_notification_call(self):
        source = inspect.getsource(UI.run_person_search_server_mode)
        self.assertEqual(1, source.count("play_person_search_notification("))
        self.assertNotIn("FOUND_FRONT_PHOTO in line", source)


if __name__ == "__main__":
    unittest.main()
