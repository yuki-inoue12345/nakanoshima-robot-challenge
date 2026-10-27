"""Regression tests for the persistent Camera owner and latest-frame broker."""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import camera_video_recording as recording

# The Camera ownership tests never initialize YOLO. Keep this unit test
# independent of the production venv while exercising the real MS2 source.
if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics
import person_search_vlm_milestone2_v7 as ms2


class FakeCv2:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_FRAME_WIDTH = 4
    CAP_PROP_FRAME_HEIGHT = 5

    @staticmethod
    def VideoWriter_fourcc(*_letters):
        return 0


class FakePreviewManager:
    display_available = False

    def activate(self, *_args, **_kwargs):
        return False

    def show(self, *_args, **_kwargs):
        return None

    def deactivate(self, *_args, **_kwargs):
        return True


class ControlledCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.release_count = 0
        self.settings = {}

    def push(self, value):
        frame = np.full((2, 3, 3), int(value), dtype=np.uint8)
        self.frames.put(frame)

    def read(self):
        try:
            return True, self.frames.get(timeout=0.02)
        except queue.Empty:
            return False, None

    def set(self, property_id, value):
        self.settings[property_id] = value
        return True

    def get(self, property_id):
        return {
            FakeCv2.CAP_PROP_FPS: 15.0,
            FakeCv2.CAP_PROP_FRAME_WIDTH: 3.0,
            FakeCv2.CAP_PROP_FRAME_HEIGHT: 2.0,
        }.get(property_id, 0.0)

    def release(self):
        self.release_count += 1


class AlwaysFailCapture(ControlledCapture):
    def read(self):
        time.sleep(0.005)
        return False, None


class FakeWriter:
    def __init__(self, *, fail_write=False):
        self.frames = []
        self.release_count = 0
        self.fail_write = fail_write

    def isOpened(self):
        return True

    def write(self, frame):
        if self.fail_write:
            raise OSError("injected writer failure")
        self.frames.append(frame.copy())
        return True

    def release(self):
        self.release_count += 1


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class PersistentCameraOwnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def make_recorder(
        self,
        capture,
        *,
        read_fail_tolerance=1000,
        writer=None,
        opener_exception=None,
    ):
        opener_calls = []
        writers = []

        def opener():
            opener_calls.append(True)
            if opener_exception is not None:
                raise opener_exception
            return capture, "/dev/video-test", ["/dev/video-test(V4L2)"]

        def writer_factory(*_args, **_kwargs):
            created = writer if writer is not None else FakeWriter()
            writers.append(created)
            return created

        manifest = recording.CameraVideoManifest(
            self.root / "camera_video_manifest.json"
        )
        recorder = recording.DriveRecorder(
            video_dir=self.root,
            manifest=manifest,
            capture_opener=opener,
            cv2_module=FakeCv2,
            preview_manager=FakePreviewManager(),
            requested_fps=15.0,
            camera_source=0,
            camera_width=3,
            camera_height=2,
            read_fail_tolerance=read_fail_tolerance,
            preview=False,
            writer_factory=writer_factory,
        )
        return recorder, manifest, opener_calls, writers

    def test_one_capture_one_mission_writer_and_continuous_frames(self):
        capture = ControlledCapture()
        recorder, manifest, opener_calls, writers = self.make_recorder(capture)
        shutdown_event = threading.Event()
        owner = recording.CameraOwnershipCoordinator(
            recorder, shutdown_event=shutdown_event
        )

        self.assertTrue(owner.start_drive())
        self.assertTrue(recorder.opened_event.wait(1.0))
        capture.push(1)
        self.assertTrue(wait_until(lambda: writers and len(writers[0].frames) == 1))

        # Area 1: detaching the search reader cannot release the physical Camera.
        self.assertTrue(owner.pause_for_search())
        reader1 = owner.create_search_reader()
        capture.push(2)
        ok, consumer_frame = reader1.read()
        self.assertTrue(ok)
        self.assertEqual(2, int(consumer_frame[0, 0, 0]))
        consumer_frame[:, :, :] = 99
        # No next sequence means no duplicate delivery.
        self.assertEqual((False, None), reader1.read())
        reader1.release()
        self.assertEqual(0, capture.release_count)
        self.assertTrue(owner.resume_after_search())

        # Area 2/VLM stall: producer/writer continue while the reader is idle.
        self.assertTrue(owner.pause_for_search())
        reader2 = owner.create_search_reader()
        frame_count_before_stall = len(writers[0].frames)
        for value in range(3, 11):
            capture.push(value)
        self.assertTrue(
            wait_until(
                lambda: (
                    len(writers[0].frames) >= frame_count_before_stall + 8
                    and recorder.latest_frame_sequence >= 10
                )
            )
        )
        ok, latest = reader2.read()
        self.assertTrue(ok)
        self.assertEqual(10, int(latest[0, 0, 0]))
        reader2.release()
        self.assertEqual(0, capture.release_count)
        self.assertTrue(owner.resume_after_search())

        self.assertTrue(owner.shutdown())
        self.assertEqual(1, len(opener_calls))
        self.assertEqual(1, len(writers))
        self.assertEqual(1, writers[0].release_count)
        self.assertEqual(1, capture.release_count)
        self.assertEqual(
            list(range(1, 11)),
            [int(frame[0, 0, 0]) for frame in writers[0].frames],
        )
        # The consumer mutation did not alter the mission writer's frame 2.
        self.assertEqual(2, int(writers[0].frames[1][0, 0, 0]))
        snapshot = manifest.snapshot()
        self.assertEqual(1, snapshot["segment_count"])
        self.assertEqual("mission", snapshot["segments"][0]["kind"])
        self.assertTrue(
            Path(snapshot["segments"][0]["video_path"]).name.startswith("mission_")
        )

    def test_read_failure_wakes_waiting_reader(self):
        capture = AlwaysFailCapture()
        recorder, _manifest, _opener_calls, writers = self.make_recorder(
            capture, read_fail_tolerance=3
        )
        self.assertTrue(recorder.start())
        self.assertTrue(recorder.opened_event.wait(1.0))
        reader = recorder.create_search_reader(wait_timeout_sec=2.0)
        started = time.monotonic()
        self.assertEqual((False, None), reader.read())
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(recorder.released_event.wait(1.0))
        self.assertFalse(recorder.is_running())
        self.assertIn("camera_read_failure", recorder.camera_error)
        self.assertEqual([], writers)

    def test_camera_open_failure_leaves_no_thread_or_writer(self):
        recorder, _manifest, opener_calls, writers = self.make_recorder(
            None, opener_exception=OSError("injected open failure")
        )
        self.assertTrue(recorder.start())
        self.assertTrue(recorder.released_event.wait(1.0))
        self.assertFalse(recorder.is_running())
        self.assertEqual(1, len(opener_calls))
        self.assertEqual([], writers)
        self.assertTrue(recorder.stop())

    def test_writer_failure_is_fail_soft_for_shared_reader(self):
        capture = ControlledCapture()
        writer = FakeWriter(fail_write=True)
        recorder, _manifest, _opener_calls, _writers = self.make_recorder(
            capture, writer=writer
        )
        self.assertTrue(recorder.start())
        self.assertTrue(recorder.opened_event.wait(1.0))
        reader = recorder.create_search_reader()
        capture.push(7)
        ok, frame = reader.read()
        self.assertTrue(ok)
        self.assertEqual(7, int(frame[0, 0, 0]))
        self.assertTrue(recorder.is_running())
        reader.release()
        self.assertTrue(recorder.stop())


class InputPathCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def args(self, *extra):
        return ms2.build_parser().parse_args(
            ["--out_dir", str(self.root / f"run_{time.time_ns()}"), *extra]
        )

    @staticmethod
    def capture():
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.0
        capture.set.return_value = True
        return capture

    def test_recording_off_keeps_direct_camera_open_and_release(self):
        capture = self.capture()
        args = self.args("--camera_index", "0")
        with (
            mock.patch.object(
                ms2,
                "open_camera_capture_with_fallback",
                return_value=(capture, "0(V4L2)", ["0(V4L2)"]),
            ) as opener,
            mock.patch.object(ms2.base, "YOLO", side_effect=RuntimeError("stop")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                ms2.run_milestone2_search(args)
        opener.assert_called_once_with(0)
        capture.release.assert_called_once_with()

    def test_video_input_keeps_video_capture_path(self):
        capture = self.capture()
        video_path = str(self.root / "input.mp4")
        args = self.args("--video", video_path)
        with (
            mock.patch.object(ms2.cv2, "VideoCapture", return_value=capture) as video_open,
            mock.patch.object(ms2, "open_camera_capture_with_fallback") as camera_open,
            mock.patch.object(ms2.base, "YOLO", side_effect=RuntimeError("stop")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                ms2.run_milestone2_search(args)
        video_open.assert_called_once_with(video_path)
        camera_open.assert_not_called()
        capture.release.assert_called_once_with()

    def test_external_reader_bypasses_physical_camera_open(self):
        capture = self.capture()
        capture.source_label = "/dev/video-shared"
        args = self.args("--camera_index", "0")
        with (
            mock.patch.object(ms2, "open_camera_capture_with_fallback") as opener,
            mock.patch.object(ms2.base, "YOLO", side_effect=RuntimeError("stop")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                ms2.run_milestone2_search(args, external_capture=capture)
        opener.assert_not_called()
        capture.set.assert_not_called()
        capture.release.assert_called_once_with()

    def test_dev_server_passes_shared_reader_without_touching_normal_ui(self):
        dev_source = (Path(__file__).resolve().parent / "daikoudai_v7.py.new").read_text(
            encoding="utf-8"
        )
        self.assertIn("create_search_reader()", dev_source)
        self.assertIn("external_capture=shared_capture", dev_source)
        normal_source = (Path(__file__).resolve().parent / "daikoudai_v7.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("external_capture=shared_capture", normal_source)


if __name__ == "__main__":
    unittest.main()
