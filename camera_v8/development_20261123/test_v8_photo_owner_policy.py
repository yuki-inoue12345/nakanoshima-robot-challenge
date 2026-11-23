"""Hardware-free guardrails for V8 standalone/shared photo source ownership."""
import ast
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from v8_photo_session import require_camera_free, stop_owned_process

HERE = Path(__file__).resolve().parent


class PhotoOwnerPolicyTests(TestCase):
    def test_v8_ui_never_dispatches_v7_direct_camera_capture(self):
        source = (HERE / 'ui_v7_compat.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn("str(HERE / 'v8_photo_session.py')", source)
        self.assertNotIn('return super().topic_capture_command()', source)
        self.assertNotIn('return super().check_camera_devices()', source)
        self.assertNotIn('return v7.run_capture_server_mode', source)
        self.assertNotIn('VideoCapture(', source)

    def test_photo_session_has_no_direct_device_open(self):
        source = (HERE / 'v8_photo_session.py').read_text(encoding='utf-8')
        ast.parse(source)
        for bad in ('VideoCapture(', 'VideoWriter(', '/cmd_vel', '/navigate_to_pose'):
            self.assertNotIn(bad, source)
        self.assertIn('v8_camera_source.py', source)
        self.assertIn('v8_shared_photo_preview.py', source)
        self.assertIn('start_new_session=True', source)

    def test_fuser_claimed_camera_must_be_rejected(self):
        with patch('v8_photo_session.subprocess.run', return_value=Mock(returncode=0)):
            with self.assertRaises(RuntimeError):
                require_camera_free(Path('/dev/null'))

    def test_fuser_unknown_status_fails_closed(self):
        with patch('v8_photo_session.subprocess.run', return_value=Mock(returncode=2)):
            with self.assertRaises(RuntimeError):
                require_camera_free(Path('/dev/null'))

    def test_only_owned_running_process_is_terminated(self):
        proc = Mock()
        proc.poll.return_value = None
        stop_owned_process(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()
        proc.kill.assert_not_called()
        stopped = Mock()
        stopped.poll.return_value = 0
        stop_owned_process(stopped)
        stopped.terminate.assert_not_called()

    def test_original_v7_and_ar_are_not_modified(self):
        script = (HERE / 'ui_v7_compat.py').read_text(encoding='utf-8')
        self.assertIn('EXPECTED_V7_SHA256', script)
        self.assertIn('V8_ENGINE', script)
        self.assertIn("return v7.ros_wrapped_command(command)", script)
