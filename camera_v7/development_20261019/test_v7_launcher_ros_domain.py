"""Side-effect-free tests for the production launcher ROS domain contract."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "run_daikoudai_v7_ui.sh"


class LauncherRosDomainTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Path(self.tempdir.name)
        self.launcher = self.workspace / LAUNCHER.name
        shutil.copy2(LAUNCHER, self.launcher)
        (self.workspace / "v7_startup_preflight.py").write_text(
            "from pathlib import Path\nPath('preflight_ran').write_text('yes')\n",
            encoding="utf-8",
        )
        (self.workspace / "daikoudai_v7.py").write_text(
            "import os\nprint('APP_STUB_ROS_DOMAIN_ID=' + os.environ.get('ROS_DOMAIN_ID', ''))\n",
            encoding="utf-8",
        )

    def run_launcher(self, robot_env: Path) -> tuple[subprocess.CompletedProcess, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "ROBOT_ENV": str(robot_env),
                "DISPLAY": "",
                "WAYLAND_DISPLAY": "",
            }
        )
        completed = subprocess.run(
            ["bash", str(self.launcher)],
            cwd=self.workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        log_path = self.workspace / "outputs" / "daikoudai_v7_desktop.log"
        log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        return completed, log

    def write_robot_env(self, content: str) -> Path:
        path = self.workspace / "robot.env"
        path.write_text(content, encoding="utf-8")
        return path

    def assert_domain_passes(self, value: str) -> None:
        robot_env = self.write_robot_env(
            f'ROBOTBASE_ROS_DOMAIN_ID="{value}"\n'
        )
        completed, log = self.run_launcher(robot_env)
        self.assertEqual(0, completed.returncode, completed.stderr + log)
        self.assertIn(f"ROS_DOMAIN_ID: {value} (from {robot_env})", log)
        self.assertIn(f"APP_STUB_ROS_DOMAIN_ID={value}", log)
        self.assertTrue((self.workspace / "preflight_ran").is_file())

    def assert_domain_fails(self, content: str) -> None:
        robot_env = self.write_robot_env(content)
        completed, log = self.run_launcher(robot_env)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("INVALID_ROBOT_DOMAIN", completed.stderr + log)
        self.assertNotIn("APP_STUB_ROS_DOMAIN_ID=", log)
        self.assertFalse((self.workspace / "preflight_ran").exists())

    def test_domain_54_passes_and_is_exported(self):
        self.assert_domain_passes("54")

    def test_domain_57_passes_and_is_exported(self):
        self.assert_domain_passes("57")

    def test_missing_key_fails_closed(self):
        self.assert_domain_fails('ROBOT_NAME="renya"\n')

    def test_blank_value_fails_closed(self):
        self.assert_domain_fails('ROBOTBASE_ROS_DOMAIN_ID=""\n')

    def test_non_numeric_value_fails_closed(self):
        self.assert_domain_fails('ROBOTBASE_ROS_DOMAIN_ID="abc"\n')

    def test_negative_value_fails_closed(self):
        self.assert_domain_fails('ROBOTBASE_ROS_DOMAIN_ID="-1"\n')

    def test_value_above_dds_range_fails_closed(self):
        self.assert_domain_fails('ROBOTBASE_ROS_DOMAIN_ID="233"\n')

    def test_missing_robot_env_fails_before_preflight_or_app(self):
        missing = self.workspace / "missing.env"
        completed, log = self.run_launcher(missing)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("ROBOT_ENV_NOT_FOUND", completed.stderr + log)
        self.assertNotIn("APP_STUB_ROS_DOMAIN_ID=", log)
        self.assertFalse((self.workspace / "preflight_ran").exists())

    def test_unreadable_robot_env_fails_before_preflight_or_app(self):
        robot_env = self.write_robot_env('ROBOTBASE_ROS_DOMAIN_ID="54"\n')
        robot_env.chmod(0)
        if os.access(robot_env, os.R_OK):
            self.skipTest("current test user can read mode-000 files")
        completed, log = self.run_launcher(robot_env)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("ROBOT_ENV_NOT_FOUND", completed.stderr + log)
        self.assertNotIn("APP_STUB_ROS_DOMAIN_ID=", log)
        self.assertFalse((self.workspace / "preflight_ran").exists())

    def test_source_has_no_numeric_domain_fallback(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'ROBOT_ENV="${ROBOT_ENV:-$HOME/robotbase_ws_renya/robot.env}"',
            source,
        )
        self.assertIn('export ROS_DOMAIN_ID="$ROBOT_DOMAIN"', source)
        self.assertNotIn("export ROS_DOMAIN_ID=57", source)
        self.assertNotIn("export ROS_DOMAIN_ID=54", source)


if __name__ == "__main__":
    unittest.main()
