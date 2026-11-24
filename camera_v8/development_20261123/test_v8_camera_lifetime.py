"""Hardware-free C920 parent lifetime and cross-domain ownership tests."""
from __future__ import annotations

import ast
import errno
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import v8_parent_death_guard as guard

HERE = Path(__file__).resolve().parent


class V8CameraLifetimeTests(TestCase):
    def test_rejects_invalid_parent_pid(self):
        for value in (-1, 0, 1):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                guard.arm_parent_death_guard(value)

    def test_rejects_orphan_before_prctl(self):
        with patch.object(guard.os, 'getppid', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'already gone'):
                guard.arm_parent_death_guard(1234)

    def test_prctl_registers_sigterm_and_rechecks_parent(self):
        import signal
        fake_prctl = Mock(return_value=0)
        fake_libc = Mock(prctl=fake_prctl)
        with patch.object(guard.os, 'getppid', return_value=1234), \
             patch.object(guard.ctypes, 'CDLL', return_value=fake_libc), \
             patch.object(guard.platform, 'system', return_value='Linux'):
            guard.arm_parent_death_guard(1234)
        fake_prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_rejects_parent_died_during_prctl(self):
        fake_libc = Mock(prctl=Mock(return_value=0))
        with patch.object(guard.os, 'getppid', side_effect=[1234, 1]), \
             patch.object(guard.ctypes, 'CDLL', return_value=fake_libc):
            with self.assertRaisesRegex(RuntimeError, 'during guard setup'):
                guard.arm_parent_death_guard(1234)

    def test_prctl_failure_is_fail_closed(self):
        fake_libc = Mock(prctl=Mock(return_value=-1))
        with patch.object(guard.os, 'getppid', return_value=1234), \
             patch.object(guard.ctypes, 'CDLL', return_value=fake_libc), \
             patch.object(guard.ctypes, 'get_errno', return_value=errno.EPERM):
            with self.assertRaises(OSError):
                guard.arm_parent_death_guard(1234)

    def test_both_camera_launchers_pass_expected_parent(self):
        photo = (HERE / 'v8_photo_session.py').read_text()
        ar = (HERE / 'run_ar_with_v8_share.sh').read_text()
        camera = (HERE / 'v8_camera_source.py').read_text()
        ast.parse(photo)
        ast.parse(camera)
        self.assertIn("'--expected-parent-pid', str(os.getpid())", photo)
        self.assertIn('--expected-parent-pid "$$"', ar)
        self.assertIn('arm_parent_death_guard(args.expected_parent_pid)', camera)

    def test_photo_session_handles_terminal_hangup(self):
        photo = (HERE / 'v8_photo_session.py').read_text()
        self.assertIn('signal.signal(signal.SIGHUP, handle_exit)', photo)
        self.assertIn('stop_owned_process(owner)', photo)
        self.assertIn('start_new_session=True', photo)

    def test_ar_wrapper_uses_cross_domain_physical_camera_check(self):
        ar = (HERE / 'run_ar_with_v8_share.sh').read_text()
        self.assertIn('fuser "$REAL_DEVICE"', ar)
        self.assertIn('Camera already occupied', ar)
        self.assertIn('flock -n 8', ar)
        self.assertIn('kokochan-v8-camera-', ar)
        self.assertIn('trap cleanup EXIT', ar)
        self.assertIn("trap 'exit 129' HUP", ar)
        self.assertNotIn('pkill', ar)
        self.assertNotIn('killall', ar)

    def test_no_navigation_or_drive_commands_in_guard(self):
        guard_src = (HERE / 'v8_parent_death_guard.py').read_text()
        for forbidden in ('/cmd_vel', '/navigate_to_pose', 'VideoCapture(', 'VideoWriter('):
            self.assertNotIn(forbidden, guard_src)
