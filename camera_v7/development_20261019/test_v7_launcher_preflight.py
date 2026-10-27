"""V7 desktop dependency launcher and standalone LM API preflight contracts."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import v7_startup_preflight as preflight


ROOT = Path(__file__).resolve().parent
EXTERNAL_VENV = "/home/koko-chan-nuc/venvs/person_search"
MODEL = "gemma-4-e2b-it-qat"


class LauncherPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="f07_launcher_", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "camera workspace"
        self.workspace.mkdir()
        self.external = self.root / "external venv"
        self.trace = self.root / "interpreters.txt"
        self.launched = self.root / "ui_started"
        self.robot_env = self.root / "robot.env"
        self.robot_env.write_text('ROBOTBASE_ROS_DOMAIN_ID="54"\n')
        self.requests = []
        self.models_status = 200
        self.models_body = {"data": [{"id": MODEL}]}
        self.chat_status = 200
        self.chat_body = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        self.chat_delay = 0
        self.models_delay = 0
        self.stubs = self.root / "stubs"
        self.stubs.mkdir()
        # No real ROS, camera, ML package, UI, or LM Studio is started.
        for name in ("tkinter", "cv2", "numpy", "ultralytics", "rclpy"):
            (self.stubs / (name + ".py")).write_text("")
        (self.stubs / "std_msgs").mkdir()
        (self.stubs / "std_msgs" / "__init__.py").write_text("")
        (self.stubs / "std_msgs" / "msg.py").write_text("")
        (self.workspace / "person_search_vlm_milestone2_v7.py").write_text("")
        (self.workspace / "daikoudai_v7.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "Path(os.environ['UI_MARKER']).write_text('started')\n"
        )
        script = (ROOT / "run_daikoudai_v7_ui.sh").read_text()
        # Substitute only the hardware path in the copied production launcher.
        script = script.replace(EXTERNAL_VENV, str(self.external))
        (self.workspace / "run_daikoudai_v7_ui.sh").write_text(script)
        helper = ROOT / "v7_startup_preflight.py"
        if helper.exists():
            shutil.copyfile(helper, self.workspace / helper.name)
        self.system = self.root / "system_bin"
        self.system.mkdir()
        self.make_python(self.system / "python3", "system")
        self.env = dict(os.environ, PATH=f"{self.system}:/usr/bin:/bin",
                        PYTHONPATH=str(self.stubs), PYTHONDONTWRITEBYTECODE="1",
                        INTERPRETER_TRACE=str(self.trace), UI_MARKER=str(self.launched),
                        ROBOT_ENV=str(self.robot_env), DISPLAY="", WAYLAND_DISPLAY="")
        for name in ("VIRTUAL_ENV", "PYTHONHOME"):
            self.env.pop(name, None)

    def make_python(self, path, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' {shlex.quote(label)} >> \"$INTERPRETER_TRACE\"\n"
            f"exec {shlex.quote(sys.executable)} -B \"$@\"\n"
        )
        path.chmod(0o755)

    def venv(self, directory, label):
        for executable in ("python", "python3"):
            self.make_python(directory / "bin" / executable, label)
        (directory / "bin" / "activate").write_text(
            f"export PATH={shlex.quote(str(directory / 'bin'))}:\"$PATH\"\n"
        )

    def config(self, url):
        (self.workspace / ".daikoudai_v7_target_ui_last.json").write_text(
            json.dumps({"lmstudio_base_url": url})
        )

    def api(self):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def respond(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected when testing request timeout.

            def do_GET(self):
                fixture.requests.append(("GET", self.path, None))
                if fixture.models_delay:
                    time.sleep(fixture.models_delay)
                self.respond(fixture.models_status, fixture.models_body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append(("POST", self.path, body))
                if fixture.chat_delay:
                    time.sleep(fixture.chat_delay)
                self.respond(fixture.chat_status, fixture.chat_body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/v1"
        self.config(url)
        return url

    def run_launcher(self):
        return subprocess.run(["bash", str(self.workspace / "run_daikoudai_v7_ui.sh")],
                              env=self.env, capture_output=True, text=True, timeout=20)

    def run_preflight(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(self.workspace / "v7_startup_preflight.py"), *arguments],
            cwd=self.workspace, env=self.env, capture_output=True, text=True, timeout=10,
        )

    def assert_allowed(self, label):
        result = self.run_launcher()
        log = self.workspace / "outputs" / "daikoudai_v7_desktop.log"
        self.assertEqual(0, result.returncode, result.stderr + log.read_text())
        self.assertTrue(self.launched.exists())
        self.assertEqual([label, label], self.trace.read_text().splitlines())
        self.assertEqual([], self.requests, "desktop dependency preflight must not call LM API")

    def assert_denied(self, code):
        result = self.run_launcher()
        self.assertNotEqual(0, result.returncode)
        self.assertFalse(self.launched.exists(), "UI must not start on failed preflight")
        self.assertIn(code, result.stderr, "desktop failure must also be visible outside the log")
        self.assertIn(code, (self.workspace / "outputs" / "daikoudai_v7_desktop.log").read_text())
        return result

    def assert_preflight_denied(self, code, *arguments):
        result = self.run_preflight(*arguments)
        self.assertNotEqual(0, result.returncode)
        self.assertIn(code, result.stderr)
        self.assertFalse(self.launched.exists(), "standalone preflight must not launch the UI")
        return result

    def assert_api_denied(self, code):
        url = self.api()
        return self.assert_preflight_denied(code, "--base-url", url)

    def test_external_venv_has_priority_over_both_local_venvs(self):
        self.venv(self.external, "external")
        self.venv(self.workspace / "venv", "workspace")
        self.venv(self.workspace / ".venv", "dotvenv")
        self.assert_allowed("external")

    def test_workspace_venv_fallback(self):
        self.venv(self.workspace / "venv", "workspace")
        self.venv(self.workspace / ".venv", "dotvenv")
        self.assert_allowed("workspace")

    def test_dotvenv_fallback(self):
        self.venv(self.workspace / ".venv", "dotvenv")
        self.assert_allowed("dotvenv")

    def test_system_python_fallback_allows_launch_with_lm_studio_stopped(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            self.config(f"http://127.0.0.1:{reserved.getsockname()[1]}/v1")
            self.assert_allowed("system")

    def test_unloaded_target_model_does_not_block_desktop_launch(self):
        self.models_body = {"data": [{"id": "another-model"}]}
        self.api()
        self.assert_allowed("system")

    def test_nonexecutable_external_python_falls_back(self):
        self.venv(self.external, "external")
        (self.external / "bin" / "python").chmod(0o644)
        self.venv(self.workspace / "venv", "workspace")
        self.assert_allowed("workspace")

    def test_selected_external_with_missing_dependency_does_not_silently_fallback(self):
        self.venv(self.external, "external")
        self.venv(self.workspace / "venv", "workspace")
        (self.stubs / "cv2.py").write_text("raise ImportError('injected missing opencv')\n")
        self.assert_denied("DEPENDENCY_ERROR")
        self.assertEqual(["external"], self.trace.read_text().splitlines())
        self.assertEqual([], self.requests)

    def test_system_missing_dependency_denies_launch(self):
        (self.stubs / "ultralytics.py").write_text("raise ImportError('injected missing YOLO')\n")
        self.assert_denied("DEPENDENCY_ERROR")
        self.assertEqual([], self.requests)

    def test_transitive_search_import_failure_denies_launch(self):
        (self.workspace / "person_search_vlm_milestone2_v7.py").write_text(
            "raise ImportError('injected broken search dependency')\n"
        )
        self.assert_denied("DEPENDENCY_ERROR")

    def test_api_connection_refused_denies_standalone_preflight_and_reports_port(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))  # Bound but not listening: cannot hit a real server.
            port = reserved.getsockname()[1]
            result = self.assert_preflight_denied(
                "API_CONNECTION_REFUSED", "--base-url", f"http://127.0.0.1:{port}/v1"
            )
        self.assertIn(str(port), result.stderr)
        self.assertIn("lms server status", result.stderr)
        self.assertIn(f"lms server start --port {port}", result.stderr)

    def test_missing_model_denies_standalone_preflight_before_chat(self):
        self.models_body = {"data": [{"id": "another-model"}]}
        self.assert_api_denied("MODEL_UNAVAILABLE")
        self.assertEqual(["GET"], [r[0] for r in self.requests])

    def test_model_identifier_must_match_exactly(self):
        self.models_body = {"data": [{"id": MODEL + "-other"}]}
        self.assert_api_denied("MODEL_UNAVAILABLE")
        self.assertEqual(["GET"], [r[0] for r in self.requests], "no model fallback is allowed")

    def test_models_http_error_is_distinct_from_missing_model(self):
        self.models_status = 404
        self.models_body = {"error": "wrong API endpoint"}
        result = self.assert_api_denied("MODELS_HTTP_ERROR")
        self.assertIn("404", result.stderr)

    def test_models_invalid_json_denies_standalone_preflight(self):
        self.models_body = "not json"
        self.assert_api_denied("MODELS_INVALID_RESPONSE")

    def test_models_error_body_with_http_200_denies_standalone_preflight(self):
        self.models_body = {"error": "backend not ready"}
        self.assert_api_denied("MODELS_INVALID_RESPONSE")

    def test_chat_http_error_denies_standalone_preflight(self):
        self.chat_status = 503
        self.chat_body = {"error": "model cannot be loaded"}
        self.assert_api_denied("CHAT_HTTP_ERROR")

    def test_chat_invalid_json_denies_standalone_preflight(self):
        self.chat_body = "not json"
        self.assert_api_denied("CHAT_INVALID_RESPONSE")

    def test_chat_empty_content_denies_standalone_preflight(self):
        self.chat_body = {"choices": [{"message": {"content": " "}}]}
        self.assert_api_denied("CHAT_INVALID_RESPONSE")

    def test_chat_error_body_with_http_200_denies_standalone_preflight(self):
        self.chat_body = {"error": "inference failed"}
        self.assert_api_denied("CHAT_INVALID_RESPONSE")

    def test_chat_nonassistant_response_denies_standalone_preflight(self):
        self.chat_body = {"choices": [{"message": {"role": "user", "content": "OK"}}]}
        self.assert_api_denied("CHAT_INVALID_RESPONSE")

    def test_desktop_error_dialog_includes_preflight_failure(self):
        dialog = self.root / "dialog.txt"
        zenity = self.system / "zenity"
        zenity.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$DIALOG_MARKER\"\n")
        zenity.chmod(0o755)
        self.env.update(DISPLAY=":fake", DIALOG_MARKER=str(dialog))
        (self.stubs / "cv2.py").write_text("raise ImportError('injected missing opencv')\n")
        self.assert_denied("DEPENDENCY_ERROR")
        self.assertIn("dependency preflight", dialog.read_text())

    def test_missing_saved_config_has_expected_api_default(self):
        import v7_startup_preflight as preflight
        self.assertEqual("http://127.0.0.1:1234/v1", preflight.base_url_from_config(
            self.workspace / "missing_config.json", None,
        ))

    def test_models_and_short_chat_allow_standalone_preflight_with_saved_url(self):
        self.api()
        result = self.run_preflight("--config", str(self.workspace / ".daikoudai_v7_target_ui_last.json"))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(self.launched.exists())
        self.assertEqual(["/v1/models", "/v1/chat/completions"], [r[1] for r in self.requests])
        body = self.requests[1][2]
        self.assertEqual(MODEL, body["model"])
        self.assertFalse(body["stream"])
        self.assertGreater(body["max_tokens"], 0)
        self.assertLessEqual(body["max_tokens"], 64)
        self.assertEqual(0, body["temperature"])

    def test_invalid_saved_config_denies_standalone_preflight_without_api(self):
        config_path = self.workspace / ".daikoudai_v7_target_ui_last.json"
        config_path.write_text("not json")
        self.assert_preflight_denied("CONFIG_ERROR", "--config", str(config_path))
        self.assertEqual([], self.requests)

    def test_invalid_url_denies_standalone_preflight_without_api(self):
        self.assert_preflight_denied("CONFIG_ERROR", "--base-url", "ftp://127.0.0.1:1234/v1")
        self.assertEqual([], self.requests)

    def test_chat_timeout_denies_preflight(self):
        url = self.api()
        self.chat_delay = 0.2
        result = subprocess.run(
            [sys.executable, "-B", str(self.workspace / "v7_startup_preflight.py"),
             "--base-url", url, "--chat-timeout", "0.05"],
            env=self.env, capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("CHAT_TIMEOUT", result.stderr)
        self.assertFalse(self.launched.exists())

    def test_models_timeout_denies_preflight_before_chat(self):
        url = self.api()
        self.models_delay = 0.2
        result = subprocess.run(
            [sys.executable, "-B", str(self.workspace / "v7_startup_preflight.py"),
             "--base-url", url, "--models-timeout", "0.05"],
            env=self.env, capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("MODELS_TIMEOUT", result.stderr)
        self.assertEqual(["GET"], [r[0] for r in self.requests])
        self.assertFalse(self.launched.exists())

    def test_prepare_ready_api_uses_fast_path_without_desktop_or_lms(self):
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[MODEL]),
            mock.patch.object(preflight, "check_api") as check_api,
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(preflight, "discover_lm_studio_desktop") as discover,
            mock.patch.object(preflight, "wait_for_models") as wait_models,
            mock.patch.object(preflight, "wait_for_local_target_model") as wait_catalog,
            mock.patch.object(
                preflight, "minimize_lm_studio_windows", return_value=False
            ) as minimize,
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        check_api.assert_called_once_with(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        run_lms.assert_not_called()
        discover.assert_not_called()
        wait_models.assert_not_called()
        wait_catalog.assert_not_called()
        minimize.assert_not_called()

    def test_prepare_api_down_direct_server_start_succeeds(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(preflight, "wait_for_models", return_value=[MODEL]),
            mock.patch.object(preflight, "check_api") as check_api,
            mock.patch.object(preflight, "discover_lm_studio_desktop") as discover,
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual("server", run_lms.call_args.args[0][0])
        check_api.assert_called_once()
        discover.assert_not_called()

    def test_cold_start_automatic_restore_avoids_exact_model_load(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        output = io.StringIO()
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(
                preflight,
                "wait_for_models",
                side_effect=[[], [MODEL]],
            ) as wait_models,
            mock.patch.object(preflight, "wait_for_local_target_model") as wait_catalog,
            mock.patch.object(preflight, "check_api") as check_api,
            redirect_stdout(output),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(1, run_lms.call_count)
        self.assertEqual("server", run_lms.call_args.args[0][0])
        self.assertEqual(False, wait_models.call_args_list[0].kwargs["require_target"])
        self.assertEqual(True, wait_models.call_args_list[1].kwargs["require_target"])
        wait_catalog.assert_not_called()
        check_api.assert_called_once()
        self.assertIn("waiting for automatic restore", output.getvalue())
        self.assertIn("became visible during restore wait", output.getvalue())

    def test_restore_wait_polls_until_exact_model_appears(self):
        clock = [0.0]

        def monotonic():
            value = clock[0]
            clock[0] += 0.1
            return value

        with (
            mock.patch.object(
                preflight,
                "available_model_ids",
                side_effect=[[], [], [MODEL]],
            ) as available,
            mock.patch.object(preflight.time, "monotonic", side_effect=monotonic),
            mock.patch.object(preflight.time, "sleep"),
        ):
            result = preflight.wait_for_models(
                preflight.DEFAULT_BASE_URL,
                require_target=True,
                timeout=2.0,
            )
        self.assertEqual([MODEL], result)
        self.assertEqual(3, available.call_count)

    def test_cold_start_restore_timeout_waits_for_catalog_then_loads_exact_model(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        restore_timeout = preflight.StartupError("LM_PREPARE_TIMEOUT", "not restored")
        with (
            mock.patch.object(
                preflight, "available_model_ids", side_effect=[unavailable, []]
            ),
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(
                preflight,
                "wait_for_models",
                side_effect=[[], restore_timeout, [MODEL]],
            ),
            mock.patch.object(preflight, "wait_for_local_target_model") as wait_catalog,
            mock.patch.object(preflight, "check_api"),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        wait_catalog.assert_called_once_with()
        self.assertEqual(2, run_lms.call_count)
        self.assertEqual("server", run_lms.call_args_list[0].args[0][0])
        self.assertEqual(
            ["load", MODEL, "-y", "--identifier", MODEL],
            run_lms.call_args_list[1].args[0],
        )

    def test_preload_api_recheck_sees_restore_and_skips_load(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        restore_timeout = preflight.StartupError("LM_PREPARE_TIMEOUT", "not restored")
        output = io.StringIO()
        with (
            mock.patch.object(
                preflight,
                "available_model_ids",
                side_effect=[unavailable, [MODEL]],
            ),
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(
                preflight,
                "wait_for_models",
                side_effect=[[], restore_timeout],
            ),
            mock.patch.object(preflight, "wait_for_local_target_model"),
            mock.patch.object(preflight, "check_api") as check_api,
            redirect_stdout(output),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(1, run_lms.call_count)
        self.assertEqual("server", run_lms.call_args.args[0][0])
        check_api.assert_called_once_with(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertIn("became visible before load; load skipped", output.getvalue())

    def test_identifier_collision_waits_for_exact_model_then_continues(self):
        collision = preflight.StartupError(
            "LMS_COMMAND_FAILED",
            f"Error: A model with identifier {MODEL} already exists.",
        )
        output = io.StringIO()
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[]),
            mock.patch.object(preflight, "run_lms", side_effect=collision) as run_lms,
            mock.patch.object(preflight, "wait_for_local_target_model"),
            mock.patch.object(preflight, "wait_for_models", return_value=[MODEL]) as wait,
            mock.patch.object(preflight, "check_api") as check_api,
            redirect_stdout(output),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(["load", MODEL, "-y", "--identifier", MODEL],
                         run_lms.call_args.args[0])
        wait.assert_called_once_with(
            preflight.DEFAULT_BASE_URL,
            require_target=True,
            timeout=preflight.IDENTIFIER_COLLISION_TIMEOUT_SEC,
        )
        check_api.assert_called_once_with(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertIn("checking concurrent restore", output.getvalue())
        self.assertIn("became visible after identifier collision", output.getvalue())

    def test_identifier_collision_without_exact_model_fails_closed(self):
        collision = preflight.StartupError(
            "LMS_COMMAND_FAILED",
            f"A model with identifier '{MODEL}' already exists",
        )
        timed_out = preflight.StartupError("LM_PREPARE_TIMEOUT", "exact absent")
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[]),
            mock.patch.object(preflight, "run_lms", side_effect=collision),
            mock.patch.object(preflight, "wait_for_local_target_model"),
            mock.patch.object(preflight, "wait_for_models", side_effect=timed_out),
            mock.patch.object(preflight, "check_api") as check_api,
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(
            "TARGET_MODEL_IDENTIFIER_COLLISION_TIMEOUT", caught.exception.code
        )
        self.assertIn("競合文字列だけでは成功扱いしません", str(caught.exception))
        check_api.assert_not_called()

    def test_identifier_collision_with_only_e4b_visible_fails_closed(self):
        collision = preflight.StartupError(
            "LMS_COMMAND_FAILED",
            f"A model with identifier {MODEL} already exists",
        )
        timed_out = preflight.StartupError(
            "LM_PREPARE_TIMEOUT", "available=['gemma-4-e4b-it-qat']"
        )
        with (
            mock.patch.object(
                preflight,
                "available_model_ids",
                return_value=["gemma-4-e4b-it-qat"],
            ),
            mock.patch.object(preflight, "run_lms", side_effect=collision) as run_lms,
            mock.patch.object(preflight, "wait_for_local_target_model"),
            mock.patch.object(preflight, "wait_for_models", side_effect=timed_out),
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(
            "TARGET_MODEL_IDENTIFIER_COLLISION_TIMEOUT", caught.exception.code
        )
        self.assertEqual(1, run_lms.call_count)
        self.assertEqual(["load", MODEL, "-y", "--identifier", MODEL],
                         run_lms.call_args.args[0])

    def test_prepare_direct_failure_launches_desktop_then_retries_server(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        daemon_missing = preflight.StartupError(
            "LMS_COMMAND_FAILED",
            "LM Studio daemon is not running and no valid installation could be found",
        )
        process = mock.Mock()
        process.poll.return_value = None
        output = io.StringIO()
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms", side_effect=[daemon_missing, None]) as run_lms,
            mock.patch.object(
                preflight,
                "discover_lm_studio_desktop",
                return_value=(["/tmp/LM Studio.AppImage"], "appimage:test"),
            ),
            mock.patch.object(preflight, "desktop_process_running", return_value=False),
            mock.patch.object(
                preflight, "launch_lm_studio_desktop", return_value=process
            ) as launch,
            mock.patch.object(preflight, "wait_for_desktop_readiness") as readiness,
            mock.patch.object(preflight, "minimize_lm_studio_windows") as minimize,
            mock.patch.object(preflight, "wait_for_models", return_value=[MODEL]),
            mock.patch.object(preflight, "check_api") as check_api,
            redirect_stdout(output),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(2, run_lms.call_count)
        launch.assert_called_once_with(["/tmp/LM Studio.AppImage", "--no-sandbox"])
        readiness.assert_called_once_with(process)
        minimize.assert_called_once_with()
        check_api.assert_called_once()
        log = output.getvalue()
        for marker in (
            "direct start attempted",
            "direct start failed",
            "executable discovered",
            "launch attempted",
            "retry succeeded",
            "final success",
        ):
            self.assertIn(marker, log)

    def test_prepare_existing_desktop_process_is_not_duplicated(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        daemon_missing = preflight.StartupError(
            "LMS_COMMAND_FAILED", "LM Studio daemon is not running"
        )
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms", side_effect=[daemon_missing, None]),
            mock.patch.object(
                preflight,
                "discover_lm_studio_desktop",
                return_value=(["/tmp/LM-Studio.AppImage"], "appimage:test"),
            ),
            mock.patch.object(preflight, "desktop_process_running", return_value=True),
            mock.patch.object(preflight, "launch_lm_studio_desktop") as launch,
            mock.patch.object(preflight, "wait_for_desktop_readiness") as readiness,
            mock.patch.object(preflight, "minimize_lm_studio_windows") as minimize,
            mock.patch.object(preflight, "wait_for_models", return_value=[MODEL]),
            mock.patch.object(preflight, "check_api"),
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        launch.assert_not_called()
        readiness.assert_called_once_with(None)
        minimize.assert_called_once_with()

    def test_prepare_without_desktop_or_appimage_fails_closed(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        daemon_missing = preflight.StartupError(
            "LMS_COMMAND_FAILED", "no valid installation could be found"
        )
        not_found = preflight.StartupError("LM_DESKTOP_NOT_FOUND", "none")
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms", side_effect=daemon_missing),
            mock.patch.object(
                preflight, "discover_lm_studio_desktop", side_effect=not_found
            ),
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertEqual("LM_DESKTOP_NOT_FOUND", caught.exception.code)

    def test_prepare_desktop_launch_failure_fails_closed(self):
        unavailable = preflight.StartupError("API_CONNECTION_REFUSED", "down")
        daemon_missing = preflight.StartupError(
            "LMS_COMMAND_FAILED", "LM Studio daemon is not running"
        )
        launch_failed = preflight.StartupError("LM_DESKTOP_LAUNCH_FAILED", "boom")
        with (
            mock.patch.object(preflight, "available_model_ids", side_effect=unavailable),
            mock.patch.object(preflight, "run_lms", side_effect=daemon_missing),
            mock.patch.object(
                preflight,
                "discover_lm_studio_desktop",
                return_value=(["/tmp/LM-Studio.AppImage"], "test"),
            ),
            mock.patch.object(preflight, "desktop_process_running", return_value=False),
            mock.patch.object(
                preflight, "launch_lm_studio_desktop", side_effect=launch_failed
            ),
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertEqual("LM_DESKTOP_LAUNCH_FAILED", caught.exception.code)

    def test_desktop_early_exit_fails_closed(self):
        process = mock.Mock(returncode=7)
        process.poll.return_value = 7
        with self.assertRaises(preflight.StartupError) as caught:
            preflight.wait_for_desktop_readiness(
                process,
                timeout=1.0,
                probe=lambda: False,
                monotonic_fn=lambda: 0.0,
                sleep_fn=lambda _seconds: None,
            )
        self.assertEqual("LM_DESKTOP_EARLY_EXIT", caught.exception.code)

    def test_desktop_readiness_timeout_fails_closed(self):
        state = [0.0]

        def monotonic():
            state[0] += 0.6
            return state[0]

        with self.assertRaises(preflight.StartupError) as caught:
            preflight.wait_for_desktop_readiness(
                None,
                timeout=1.0,
                probe=lambda: False,
                monotonic_fn=monotonic,
                sleep_fn=lambda _seconds: None,
            )
        self.assertEqual("LM_DESKTOP_READINESS_TIMEOUT", caught.exception.code)

    def test_exact_target_is_the_only_model_loaded(self):
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[]),
            mock.patch.object(preflight, "run_lms") as run_lms,
            mock.patch.object(preflight, "wait_for_local_target_model") as wait_catalog,
            mock.patch.object(preflight, "wait_for_models", return_value=[MODEL]),
            mock.patch.object(preflight, "check_api") as check_api,
        ):
            preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)

        self.assertEqual(1, run_lms.call_count)
        arguments = run_lms.call_args.args[0]
        self.assertEqual(
            ["load", MODEL, "-y", "--identifier", MODEL], arguments
        )
        self.assertNotIn("fallback", " ".join(arguments).lower())
        wait_catalog.assert_called_once_with()
        check_api.assert_called_once()

    def test_exact_model_load_failure_does_not_fallback_or_download(self):
        failure = preflight.StartupError("LMS_COMMAND_FAILED", "local model missing")
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[]),
            mock.patch.object(preflight, "run_lms", side_effect=failure) as run_lms,
            mock.patch.object(preflight, "wait_for_local_target_model"),
            mock.patch.object(preflight, "wait_for_models") as wait_models,
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertEqual("TARGET_MODEL_LOAD_FAILED", caught.exception.code)
        self.assertEqual(["load", MODEL, "-y", "--identifier", MODEL],
                         run_lms.call_args.args[0])
        wait_models.assert_not_called()

    def test_catalog_not_ready_then_exact_model_appears(self):
        transient = preflight.StartupError("LMS_CATALOG_UNAVAILABLE", "starting")
        responses = iter((set(), transient, {MODEL}))
        calls = []
        clock = [0.0]

        def probe():
            value = next(responses)
            calls.append(value)
            if isinstance(value, Exception):
                raise value
            return value

        def monotonic():
            value = clock[0]
            clock[0] += 0.1
            return value

        preflight.wait_for_local_target_model(
            timeout=2.0,
            probe=probe,
            monotonic_fn=monotonic,
            sleep_fn=lambda _seconds: None,
        )
        self.assertEqual(3, len(calls))

    def test_missing_exact_local_model_fails_closed_before_load(self):
        missing = preflight.StartupError(
            "TARGET_MODEL_NOT_IN_LOCAL_CATALOG", "exact E2B absent"
        )
        with (
            mock.patch.object(preflight, "available_model_ids", return_value=[]),
            mock.patch.object(
                preflight, "wait_for_local_target_model", side_effect=missing
            ),
            mock.patch.object(preflight, "run_lms") as run_lms,
        ):
            with self.assertRaises(preflight.StartupError) as caught:
                preflight.prepare_lm(preflight.DEFAULT_BASE_URL, 5.0, 15.0)
        self.assertEqual("TARGET_MODEL_NOT_IN_LOCAL_CATALOG", caught.exception.code)
        run_lms.assert_not_called()

    def test_e4b_only_catalog_never_falls_back_from_exact_e2b(self):
        clock = [0.0]

        def monotonic():
            value = clock[0]
            clock[0] += 0.6
            return value

        with self.assertRaises(preflight.StartupError) as caught:
            preflight.wait_for_local_target_model(
                timeout=1.0,
                probe=lambda: {"gemma-4-e4b-it-qat"},
                monotonic_fn=monotonic,
                sleep_fn=lambda _seconds: None,
            )
        self.assertEqual("TARGET_MODEL_NOT_IN_LOCAL_CATALOG", caught.exception.code)
        self.assertIn("gemma-4-e4b-it-qat", str(caught.exception))

    def test_local_catalog_parses_only_llm_model_keys(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "models": [
                        {"type": "llm", "modelKey": MODEL},
                        {"type": "embedding", "modelKey": "not-an-llm"},
                    ]
                }
            ),
        )
        with (
            mock.patch.object(preflight, "find_lms_executable", return_value="/tmp/lms"),
            mock.patch.object(preflight.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual({MODEL}, preflight.local_llm_model_keys())
        self.assertEqual(["/tmp/lms", "ls", "--json"], run.call_args.args[0])

    def test_wmctrl_matching_lm_studio_window_is_minimized(self):
        listing = mock.Mock(
            returncode=0,
            stdout=(
                "0x04000001  0 terminal.Terminal host Terminal\n"
                "0x05000004  0 lm studio.LM Studio host LM Studio\n"
            ),
        )
        minimized = mock.Mock(returncode=0, stdout="")
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/wmctrl"),
            mock.patch.object(
                preflight.subprocess, "run", side_effect=[listing, minimized]
            ) as run,
        ):
            self.assertTrue(preflight.minimize_lm_studio_windows())
        self.assertEqual(["/usr/bin/wmctrl", "-lx"], run.call_args_list[0].args[0])
        self.assertEqual(
            ["/usr/bin/wmctrl", "-ir", "0x05000004", "-b", "add,hidden"],
            run.call_args_list[1].args[0],
        )

    def test_missing_wmctrl_is_nonfatal(self):
        with (
            mock.patch.object(preflight.shutil, "which", return_value=None),
            mock.patch.object(preflight.subprocess, "run") as run,
        ):
            self.assertFalse(preflight.minimize_lm_studio_windows())
        run.assert_not_called()

    def test_wmctrl_minimize_failure_is_nonfatal(self):
        listing = mock.Mock(
            returncode=0,
            stdout="0x05000004  0 lm studio.LM Studio host LM Studio\n",
        )
        failed = mock.Mock(returncode=1, stdout="denied")
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/wmctrl"),
            mock.patch.object(preflight.subprocess, "run", side_effect=[listing, failed]),
        ):
            self.assertFalse(preflight.minimize_lm_studio_windows())

    def test_desktop_exec_parser_handles_quotes_options_and_field_codes(self):
        executable = self.root / "LM Studio AppImage"
        executable.write_text("stub")
        executable.chmod(0o755)
        desktop = self.root / "lm-studio.desktop"
        desktop.write_text(
            "[Desktop Entry]\n"
            f'Exec="{executable}" --no-sandbox %U\n'
        )
        self.assertEqual(
            [str(executable), "--no-sandbox"],
            preflight.parse_desktop_exec(desktop),
        )

    def test_appimage_discovery_is_user_independent(self):
        home = self.root / "arbitrary-user-home"
        downloads = home / "Downloads"
        downloads.mkdir(parents=True)
        appimage = downloads / "LM-Studio-0.4.21-2-x64.AppImage"
        appimage.write_text("stub")
        appimage.chmod(0o755)

        command, source = preflight.discover_lm_studio_desktop(
            environ={}, home=home
        )

        self.assertEqual([str(appimage), "--no-sandbox"], command)
        self.assertEqual(f"appimage:{appimage}", source)

    def test_explicit_desktop_executable_has_highest_discovery_priority(self):
        home = self.root / "another-home"
        (home / "Downloads").mkdir(parents=True)
        explicit = self.root / "explicit LM Studio.AppImage"
        explicit.write_text("stub")
        explicit.chmod(0o755)
        fallback = home / "Downloads" / "LM-Studio-fallback.AppImage"
        fallback.write_text("stub")
        fallback.chmod(0o755)

        command, source = preflight.discover_lm_studio_desktop(
            environ={preflight.LM_STUDIO_DESKTOP_ENV: str(explicit)},
            home=home,
        )

        self.assertEqual([str(explicit), "--no-sandbox"], command)
        self.assertEqual(
            f"environment:{preflight.LM_STUDIO_DESKTOP_ENV}", source
        )

    def test_desktop_entry_discovery_precedes_download_appimage(self):
        home = self.root / "home"
        desktop_dir = home / "Desktop"
        downloads = home / "Downloads"
        desktop_dir.mkdir(parents=True)
        downloads.mkdir(parents=True)
        executable = self.root / "LM Studio executable"
        executable.write_text("stub")
        executable.chmod(0o755)
        desktop = desktop_dir / "lm-studio.desktop"
        desktop.write_text(f'[Desktop Entry]\nExec="{executable}" --flag\n')
        appimage = downloads / "LM-Studio-new.AppImage"
        appimage.write_text("stub")
        appimage.chmod(0o755)

        command, source = preflight.discover_lm_studio_desktop(
            environ={}, home=home
        )

        self.assertEqual([str(executable), "--flag"], command)
        self.assertEqual(f"desktop:{desktop}", source)

    def test_desktop_launch_is_detached_and_does_not_leave_pipes(self):
        process = mock.Mock()
        log_path = self.root / "logs" / "desktop.log"
        with mock.patch.object(preflight.subprocess, "Popen", return_value=process) as popen:
            result = preflight.launch_lm_studio_desktop(
                ["/tmp/LM-Studio.AppImage"], log_path=log_path
            )

        self.assertIs(process, result)
        kwargs = popen.call_args.kwargs
        self.assertIs(preflight.subprocess.DEVNULL, kwargs["stdin"])
        self.assertIs(preflight.subprocess.STDOUT, kwargs["stderr"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertNotEqual(preflight.subprocess.PIPE, kwargs["stdout"])
        self.assertEqual(
            ["/tmp/LM-Studio.AppImage", "--no-sandbox"],
            popen.call_args.args[0],
        )

    def test_appimage_launch_adds_no_sandbox_and_preserves_arguments(self):
        self.assertEqual(
            ["/tmp/LM Studio.AppImage", "--no-sandbox", "--flag", "value"],
            preflight.appimage_launch_command(
                ["/tmp/LM Studio.AppImage", "--flag", "value"]
            ),
        )

    def test_appimage_launch_does_not_duplicate_no_sandbox(self):
        command = ["/tmp/LM-Studio.AppImage", "--flag", "--no-sandbox"]
        self.assertEqual(command, preflight.appimage_launch_command(command))
        self.assertEqual(1, preflight.appimage_launch_command(command).count("--no-sandbox"))

    def test_non_appimage_launch_does_not_add_no_sandbox(self):
        command = ["/opt/lm-studio/lm-studio", "--flag"]
        self.assertEqual(command, preflight.appimage_launch_command(command))

    def test_lms_desktop_ready_requires_daemon_running_json(self):
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps({"status": "running", "isDaemon": False}),
        )
        with (
            mock.patch.object(preflight, "find_lms_executable", return_value="/tmp/lms"),
            mock.patch.object(preflight.subprocess, "run", return_value=completed) as run,
        ):
            self.assertTrue(preflight.lms_desktop_ready())
        self.assertEqual(
            ["/tmp/lms", "daemon", "status", "--json"],
            run.call_args.args[0],
        )

    def test_lms_desktop_ready_rejects_rc0_when_daemon_is_not_running(self):
        for payload in (
            {"status": "stopped"},
            {"running": False},
            {"status": "starting"},
        ):
            with self.subTest(payload=payload):
                with (
                    mock.patch.object(preflight, "find_lms_executable", return_value="/tmp/lms"),
                    mock.patch.object(
                        preflight.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=0, stdout=json.dumps(payload)),
                    ),
                ):
                    self.assertFalse(preflight.lms_desktop_ready())

    def test_lms_desktop_ready_rejects_invalid_json_and_command_failure(self):
        results = (
            mock.Mock(returncode=0, stdout="not-json"),
            mock.Mock(returncode=1, stdout='{"status":"running"}'),
        )
        for completed in results:
            with self.subTest(returncode=completed.returncode):
                with (
                    mock.patch.object(preflight, "find_lms_executable", return_value="/tmp/lms"),
                    mock.patch.object(preflight.subprocess, "run", return_value=completed),
                ):
                    self.assertFalse(preflight.lms_desktop_ready())

    def test_desktop_readiness_poll_accepts_only_later_success(self):
        probes = iter((False, False, True))
        clock = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        preflight.wait_for_desktop_readiness(
            None,
            timeout=2.0,
            probe=lambda: next(probes),
            monotonic_fn=lambda: next(clock),
            sleep_fn=lambda _seconds: None,
        )

    def test_prepare_source_contains_no_install_or_download_commands(self):
        source = (ROOT / "v7_startup_preflight.py").read_text(encoding="utf-8")
        for forbidden in ("lms get", "install.sh", "apt install", "pip install"):
            self.assertNotIn(forbidden, source)

    def test_ui_prepare_timeout_is_finite_and_covers_cold_start_budget(self):
        source = (ROOT / "daikoudai_v7.py.new").read_text(encoding="utf-8")
        self.assertIn("LM_STUDIO_PREPARE_TIMEOUT_SEC = 420.0", source)
        self.assertIn("timeout=LM_STUDIO_PREPARE_TIMEOUT_SEC", source)


if __name__ == "__main__":
    unittest.main()
