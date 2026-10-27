#!/usr/bin/env python3
"""Check V7 desktop dependencies and optionally prepare LM Studio.

Dependency-only mode never starts LM Studio. --prepare-lm may start the local
LM Studio API and prepare the fixed production model. It never initializes ROS,
opens a camera, or runs recognition.
"""

from __future__ import annotations

import argparse
import errno
import importlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
TARGET_MODEL = "gemma-4-e2b-it-qat"
MAX_RESPONSE_BYTES = 1024 * 1024
LM_STUDIO_DESKTOP_ENV = "LM_STUDIO_DESKTOP_EXECUTABLE"
DESKTOP_READINESS_TIMEOUT_SEC = 45.0
DESKTOP_READINESS_POLL_SEC = 0.5
SERVER_COMMAND_TIMEOUT_SEC = 30.0
SERVER_API_TIMEOUT_SEC = 30.0
MODEL_LOAD_TIMEOUT_SEC = 120.0
MODEL_READY_TIMEOUT_SEC = 120.0
MODEL_RESTORE_TIMEOUT_SEC = 15.0
LOCAL_MODEL_CATALOG_TIMEOUT_SEC = 15.0
MODEL_DISCOVERY_POLL_SEC = 0.5
IDENTIFIER_COLLISION_TIMEOUT_SEC = 10.0
WMCTRL_TIMEOUT_SEC = 3.0


class StartupError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def base_url_from_config(config: Path | None, explicit_url: str | None) -> str:
    url = DEFAULT_BASE_URL
    if explicit_url is not None:
        url = explicit_url
    elif config is not None and config.exists():
        try:
            saved = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("saved config must be a JSON object")
            url = saved.get("lmstudio_base_url", DEFAULT_BASE_URL)
        except (OSError, ValueError) as exc:
            raise StartupError("CONFIG_ERROR", f"UI設定 {config} を読めません: {exc}") from exc
    try:
        if not isinstance(url, str):
            raise ValueError("LM Studio URL must be a string")
        url = url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("http(s) URLとhostが必要です")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("query/fragment/credentials付きURLは利用できません")
        _ = parsed.port  # Validate malformed/out-of-range ports.
    except ValueError as exc:
        raise StartupError("CONFIG_ERROR", f"LM Studio URLが不正です: {exc}") from exc
    return url


def check_dependencies() -> None:
    failures = []
    for module in ("tkinter", "cv2", "numpy", "ultralytics", "rclpy", "std_msgs.msg",
                   "person_search_vlm_milestone2_v7"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise StartupError("DEPENDENCY_ERROR", f"選択Python {sys.executable} の依存関係不足/不整合: "
                           + "; ".join(failures))


def request_json(base_url: str, stage: str, timeout: float, payload: dict | None = None) -> dict:
    endpoint = base_url + ("/models" if stage == "MODELS" else "/chat/completions")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    # The local API must be reached directly, regardless of desktop proxy env.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        hint = "APIのportと /v1/models を確認してください。" if stage == "MODELS" else (
            f"{TARGET_MODEL} のload状態とLM Studio server logを確認してください。"
        )
        raise StartupError(f"{stage}_HTTP_ERROR", f"{endpoint}: HTTP {exc.code}. {hint}") from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, (socket.timeout, TimeoutError)):
            code = f"{stage}_TIMEOUT"
        elif isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
            code = "API_CONNECTION_REFUSED" if stage == "MODELS" else "CHAT_CONNECTION_REFUSED"
        else:
            code = "API_UNREACHABLE" if stage == "MODELS" else "CHAT_UNREACHABLE"
        parsed = urllib.parse.urlsplit(base_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        raise StartupError(code, f"{endpoint}: {reason}. GUI起動だけではAPI readyになりません。"
                           f"server停止/接続先port不一致を切り分けるため、LM Studio側で "
                           f"lms server status とlisten portを確認してください。"
                           f"この接続先portは {port}。server停止なら lms server start --port {port}。"
                           "portが違う場合はUI設定のLM Studio URLかserver設定を合わせてください。") from exc
    try:
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("API response too large")
        result = json.loads(raw)
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("API response is not a successful JSON object")
    except (ValueError, UnicodeError) as exc:
        raise StartupError(f"{stage}_INVALID_RESPONSE", f"{endpoint}: {exc}") from exc
    return result


def check_api(base_url: str, models_timeout: float, chat_timeout: float) -> None:
    models = request_json(base_url, "MODELS", models_timeout)
    entries = models.get("data")
    if not isinstance(entries, list) or any(not isinstance(item, dict) or
                                           not isinstance(item.get("id"), str) for item in entries):
        raise StartupError("MODELS_INVALID_RESPONSE", f"{base_url}/models: data/id形式が不正です")
    available = [item["id"] for item in entries]
    if TARGET_MODEL not in available:
        raise StartupError("MODEL_UNAVAILABLE", f"{TARGET_MODEL} が {base_url}/models にありません。"
                           f"available={available}。LM Studioで対象modelをload/availableにしてください。")
    print(f"models: OK ({TARGET_MODEL})", flush=True)
    chat = request_json(base_url, "CHAT", chat_timeout, {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": "Reply only OK."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    })
    try:
        message = chat["choices"][0]["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("chat completion must contain an assistant message")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty/non-text chat completion")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise StartupError("CHAT_INVALID_RESPONSE", f"{base_url}/chat/completions: {exc}") from exc
    print("short chat: OK", flush=True)


def available_model_ids(base_url: str, timeout: float) -> list[str]:
    models = request_json(base_url, "MODELS", timeout)
    entries = models.get("data")
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str)
        for item in entries
    ):
        raise StartupError(
            "MODELS_INVALID_RESPONSE",
            f"{base_url}/models: data/id形式が不正です",
        )
    return [item["id"] for item in entries]


def find_lms_executable() -> str:
    executable = shutil.which("lms")
    if executable:
        return executable

    candidate = Path.home() / ".lmstudio" / "bin" / "lms"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)

    raise StartupError(
        "LMS_NOT_FOUND",
        "lms CLI が PATH / ~/.lmstudio/bin にありません。",
    )


def parse_desktop_exec(desktop_file: Path) -> list[str]:
    """Parse a Desktop Entry Exec line without treating options as a path."""
    try:
        lines = desktop_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StartupError(
            "LM_DESKTOP_ENTRY_ERROR",
            f"desktop launcherを読めません: {desktop_file}: {exc}",
        ) from exc

    exec_value = next(
        (line.split("=", 1)[1].strip() for line in lines if line.startswith("Exec=")),
        "",
    )
    if not exec_value:
        raise StartupError(
            "LM_DESKTOP_ENTRY_ERROR",
            f"desktop launcherにExecがありません: {desktop_file}",
        )
    try:
        tokens = shlex.split(exec_value, posix=True)
    except ValueError as exc:
        raise StartupError(
            "LM_DESKTOP_ENTRY_ERROR",
            f"desktop launcherのExecを解析できません: {desktop_file}: {exc}",
        ) from exc

    # Desktop field codes (%U, %F, etc.) are launch placeholders, not paths.
    command = [token for token in tokens if not token.startswith("%")]
    if not command:
        raise StartupError(
            "LM_DESKTOP_ENTRY_ERROR",
            f"desktop launcherのExecが空です: {desktop_file}",
        )
    return command


def _validated_desktop_command(command: list[str]) -> list[str] | None:
    if not command:
        return None
    executable = Path(command[0]).expanduser()
    if executable.is_absolute() or "/" in command[0]:
        resolved = executable.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            return None
        return [str(resolved), *command[1:]]
    resolved_name = shutil.which(command[0])
    if resolved_name is None:
        return None
    return [resolved_name, *command[1:]]


def appimage_launch_command(command: list[str]) -> list[str]:
    """Add the required sandbox override only to an AppImage executable."""
    normalized = list(command)
    if not normalized:
        return normalized
    if Path(normalized[0]).name.lower().endswith(".appimage"):
        if "--no-sandbox" not in normalized[1:]:
            normalized.insert(1, "--no-sandbox")
    return normalized


def discover_lm_studio_desktop(
    *,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> tuple[list[str], str]:
    """Find an already-installed LM Studio desktop executable in safe order."""
    environment = os.environ if environ is None else environ
    home_dir = Path.home() if home is None else Path(home)

    explicit = environment.get(LM_STUDIO_DESKTOP_ENV, "").strip()
    if explicit:
        command = _validated_desktop_command([str(Path(explicit).expanduser())])
        if command is None:
            raise StartupError(
                "LM_DESKTOP_EXPLICIT_INVALID",
                f"{LM_STUDIO_DESKTOP_ENV} の既存実行ファイルを利用できません: {explicit}",
            )
        return appimage_launch_command(command), f"environment:{LM_STUDIO_DESKTOP_ENV}"

    desktop_files = (
        home_dir / "Desktop" / "lm-studio.desktop",
        home_dir / ".local" / "share" / "applications" / "lm-studio.desktop",
    )
    for desktop_file in desktop_files:
        if not desktop_file.is_file():
            continue
        try:
            command = _validated_desktop_command(parse_desktop_exec(desktop_file))
        except StartupError as exc:
            print(f"LM desktop: skipped invalid entry {exc}", flush=True)
            continue
        if command is not None:
            return appimage_launch_command(command), f"desktop:{desktop_file}"

    for appimage in sorted(
            (home_dir / "Downloads").glob("LM-Studio-*.AppImage"), reverse=True):
        command = _validated_desktop_command([str(appimage)])
        if command is not None:
            return appimage_launch_command(command), f"appimage:{appimage}"

    raise StartupError(
        "LM_DESKTOP_NOT_FOUND",
        "既存のLM Studio desktop/AppImageが見つかりません。"
        f"{LM_STUDIO_DESKTOP_ENV}、desktop launcher、Downloadsを確認してください。"
        "自動download/installは行いません。",
    )


def desktop_process_running(command: list[str]) -> bool:
    """Best-effort check that avoids launching a duplicate desktop process."""
    if not command:
        return False
    target = str(Path(command[0]).expanduser().resolve())
    proc_root = Path("/proc")
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError:
        return False
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            executable = str((process_dir / "exe").resolve())
            cmdline = (process_dir / "cmdline").read_bytes().split(b"\0", 1)[0]
            first_argument = os.fsdecode(cmdline)
        except (OSError, UnicodeError):
            continue
        if executable == target or first_argument == target:
            return True
    return False


def launch_lm_studio_desktop(
    command: list[str],
    *,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """Start an installed desktop detached from this finite preflight process."""
    if log_path is None:
        log_path = (
            Path.home() / ".local" / "state" / "camera_v7" / "lm_studio_desktop.log"
        )
    try:
        launch_command = appimage_launch_command(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_stream:
            process = subprocess.Popen(
                launch_command,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        process._camera_v7_log_path = str(log_path)
        return process
    except OSError as exc:
        raise StartupError(
            "LM_DESKTOP_LAUNCH_FAILED",
            f"LM Studio desktopを起動できません: {command[0]}: {exc}",
        ) from exc


def lms_desktop_ready() -> bool:
    """Return whether the installed lms CLI can communicate with its backend."""
    try:
        completed = subprocess.run(
            [find_lms_executable(), "daemon", "status", "--json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, StartupError):
        return False
    if completed.returncode != 0:
        return False
    try:
        status = json.loads(completed.stdout or "")
    except (TypeError, ValueError):
        return False
    return isinstance(status, dict) and status.get("status") == "running"


def wait_for_desktop_readiness(
    process: subprocess.Popen | None,
    *,
    timeout: float = DESKTOP_READINESS_TIMEOUT_SEC,
    probe=None,
    monotonic_fn=None,
    sleep_fn=None,
) -> None:
    """Poll desktop/backend readiness with early-exit and finite-time checks."""
    readiness_probe = lms_desktop_ready if probe is None else probe
    now = time.monotonic if monotonic_fn is None else monotonic_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    deadline = now() + timeout
    while now() < deadline:
        if process is not None and process.poll() is not None:
            log_path = getattr(process, "_camera_v7_log_path", "")
            log_hint = f" desktop_log={log_path}" if log_path else ""
            raise StartupError(
                "LM_DESKTOP_EARLY_EXIT",
                f"LM Studio desktopがready前に終了しました: "
                f"returncode={process.returncode}.{log_hint}",
            )
        if readiness_probe():
            print("LM desktop: readiness confirmed", flush=True)
            return
        sleep(min(DESKTOP_READINESS_POLL_SEC, max(0.0, deadline - now())))
    log_path = getattr(process, "_camera_v7_log_path", "") if process else ""
    log_hint = f" desktop_log={log_path}" if log_path else ""
    raise StartupError(
        "LM_DESKTOP_READINESS_TIMEOUT",
        f"LM Studio desktop/backendが{timeout:.0f}秒以内にreadyになりませんでした。"
        f"{log_hint}",
    )


def minimize_lm_studio_windows() -> bool:
    """Best-effort minimize of current LM Studio windows; never fail preflight."""
    executable = shutil.which("wmctrl")
    if executable is None:
        print("LM desktop: minimize skipped (wmctrl unavailable)", flush=True)
        return False
    try:
        listed = subprocess.run(
            [executable, "-lx"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=WMCTRL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"LM desktop: minimize failed (non-fatal: {exc})", flush=True)
        return False
    if listed.returncode != 0:
        print(
            f"LM desktop: minimize failed (non-fatal: wmctrl -lx rc={listed.returncode})",
            flush=True,
        )
        return False

    window_ids = []
    for line in (listed.stdout or "").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        window_id, _desktop, details = parts
        if not window_id.lower().startswith("0x"):
            continue
        normalized = details.lower().replace("-", " ").replace("_", " ")
        if "lm studio.lm studio" in normalized:
            window_ids.append(window_id)
    if not window_ids:
        print("LM desktop: minimize skipped (LM Studio window not found)", flush=True)
        return False

    print("LM desktop: minimize attempted", flush=True)
    minimized = False
    for window_id in window_ids:
        try:
            completed = subprocess.run(
                [executable, "-ir", window_id, "-b", "add,hidden"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WMCTRL_TIMEOUT_SEC,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"LM desktop: minimize failed (non-fatal: {exc})", flush=True)
            continue
        if completed.returncode == 0:
            minimized = True
        else:
            print(
                f"LM desktop: minimize failed "
                f"(non-fatal: window={window_id} rc={completed.returncode})",
                flush=True,
            )
    if minimized:
        print("LM desktop: minimized", flush=True)
    return minimized


def run_lms(arguments: list[str], timeout: float) -> None:
    executable = find_lms_executable()
    print("LM CLI: " + " ".join(arguments), flush=True)
    try:
        subprocess.run(
            [executable, *arguments],
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise StartupError(
            "LMS_COMMAND_TIMEOUT",
            f"lms command が {timeout:.0f} 秒以内に終了しません。",
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = str(exc.stdout or exc.output or "").strip().replace("\n", " ")
        if len(output) > 2000:
            output = output[-2000:]
        detail = f" output={output}" if output else ""
        raise StartupError(
            "LMS_COMMAND_FAILED",
            f"lms command に失敗しました: returncode={exc.returncode}.{detail}",
        ) from exc
    except OSError as exc:
        raise StartupError(
            "LMS_COMMAND_FAILED",
            f"lms command を実行できません: {exc}",
        ) from exc


def local_llm_model_keys() -> set[str]:
    """Return exact local LLM model keys reported by the existing lms catalog."""
    try:
        completed = subprocess.run(
            [find_lms_executable(), "ls", "--json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, StartupError) as exc:
        raise StartupError(
            "LMS_CATALOG_UNAVAILABLE",
            f"lms local model catalogを確認できません: {exc}",
        ) from exc
    if completed.returncode != 0:
        raise StartupError(
            "LMS_CATALOG_UNAVAILABLE",
            f"lms ls --json に失敗しました: returncode={completed.returncode}",
        )
    try:
        payload = json.loads(completed.stdout or "")
    except (TypeError, ValueError) as exc:
        raise StartupError(
            "LMS_CATALOG_INVALID_RESPONSE",
            f"lms ls --json のJSONが不正です: {exc}",
        ) from exc

    model_keys: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            model_type = str(value.get("type", "")).strip().lower()
            model_key = value.get("modelKey")
            if model_type == "llm" and isinstance(model_key, str) and model_key:
                model_keys.add(model_key)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return model_keys


def wait_for_local_target_model(
    *,
    timeout: float = LOCAL_MODEL_CATALOG_TIMEOUT_SEC,
    probe=None,
    monotonic_fn=None,
    sleep_fn=None,
) -> None:
    """Wait finitely for the exact production model to enter the local catalog."""
    catalog_probe = local_llm_model_keys if probe is None else probe
    now = time.monotonic if monotonic_fn is None else monotonic_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    deadline = now() + timeout
    last_error: StartupError | None = None
    last_keys: set[str] = set()
    while now() < deadline:
        try:
            last_keys = set(catalog_probe())
            if TARGET_MODEL in last_keys:
                print(
                    f"LM model: exact target visible in local catalog ({TARGET_MODEL})",
                    flush=True,
                )
                return
        except StartupError as exc:
            last_error = exc
        sleep(min(MODEL_DISCOVERY_POLL_SEC, max(0.0, deadline - now())))

    suffix = f" available={sorted(last_keys)}"
    if last_error is not None:
        suffix += f" last={last_error.code}: {last_error}"
    raise StartupError(
        "TARGET_MODEL_NOT_IN_LOCAL_CATALOG",
        f"ローカルcatalogに本番Gemma {TARGET_MODEL} が期限内に現れませんでした。"
        f"{suffix}。自動downloadや別モデルへのfallbackは行いません。",
    )


def is_target_identifier_collision(error: StartupError) -> bool:
    """Recognize only the exact target identifier's concurrent-registration race."""
    if error.code != "LMS_COMMAND_FAILED":
        return False
    normalized = " ".join(str(error).lower().split())
    for quote in ('"', "'", "`"):
        normalized = normalized.replace(quote, "")
    marker = f"a model with identifier {TARGET_MODEL.lower()} already exists"
    return marker in normalized


def server_start_requires_desktop(error: StartupError) -> bool:
    """Recognize the local CLI failure that requires an installed desktop daemon."""
    if error.code == "LMS_NOT_FOUND":
        return True
    if error.code != "LMS_COMMAND_FAILED":
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "daemon is not running",
            "no valid installation could be found",
            "lm studio daemon",
            "desktop app is not running",
        )
    )


def wait_for_models(
    base_url: str,
    *,
    require_target: bool,
    timeout: float,
) -> list[str]:
    deadline = time.monotonic() + timeout
    last_error: StartupError | None = None

    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            available = available_model_ids(
                base_url,
                min(1.5, remaining),
            )
            if not require_target or TARGET_MODEL in available:
                return available
        except StartupError as exc:
            last_error = exc

        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    suffix = ""
    if last_error is not None:
        suffix = f" last={last_error.code}: {last_error}"

    raise StartupError(
        "LM_PREPARE_TIMEOUT",
        "LM Studio API/model の準備が期限内に完了しませんでした。" + suffix,
    )


def prepare_lm(
    base_url: str,
    models_timeout: float,
    chat_timeout: float,
) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname

    if parsed.scheme != "http" or host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise StartupError(
            "LM_PREPARE_LOCAL_ONLY",
            "UIからの自動起動はlocalhostのLM Studioだけ対応します。"
            "現在のURL: " + base_url,
        )

    port = parsed.port or 80
    server_arguments = [
        "server",
        "start",
        "-p",
        str(port),
        "--bind",
        "127.0.0.1",
    ]

    cold_start = False

    # 1. Local LM Studio APIを準備する。
    try:
        available = available_model_ids(
            base_url,
            min(models_timeout, 1.5),
        )
        print("LM API: already running", flush=True)
    except StartupError as exc:
        if exc.code not in {
            "API_CONNECTION_REFUSED",
            "API_UNREACHABLE",
            "MODELS_TIMEOUT",
        }:
            raise

        cold_start = True
        print("LM server: direct start attempted", flush=True)
        try:
            run_lms(server_arguments, SERVER_COMMAND_TIMEOUT_SEC)
        except StartupError as direct_error:
            print(
                f"LM server: direct start failed ({direct_error.code}: {direct_error})",
                flush=True,
            )
            if not server_start_requires_desktop(direct_error):
                raise StartupError(
                    "LM_SERVER_DIRECT_START_FAILED",
                    f"LM Studio serverの直接起動に失敗しました: {direct_error}",
                ) from direct_error

            command, source = discover_lm_studio_desktop()
            # Keep this normalization here as well as in discovery/launch so
            # custom discovery hooks cannot accidentally omit the AppImage
            # sandbox argument required on the production desktop.
            command = appimage_launch_command(command)
            print(
                f"LM desktop: executable discovered source={source} "
                f"executable={command[0]}",
                flush=True,
            )
            desktop_process = None
            if desktop_process_running(command):
                print("LM desktop: existing process detected; launch skipped", flush=True)
            else:
                print("LM desktop: launch attempted", flush=True)
                desktop_process = launch_lm_studio_desktop(command)

            wait_for_desktop_readiness(desktop_process)
            minimize_lm_studio_windows()
            print("LM server: retry attempted", flush=True)
            try:
                run_lms(server_arguments, SERVER_COMMAND_TIMEOUT_SEC)
            except StartupError as retry_error:
                print(
                    f"LM server: retry failed ({retry_error.code}: {retry_error})",
                    flush=True,
                )
                raise StartupError(
                    "LM_SERVER_RETRY_FAILED",
                    f"desktop ready後もLM Studio serverを起動できません: {retry_error}",
                ) from retry_error
            print("LM server: retry succeeded", flush=True)
        else:
            print("LM server: direct start succeeded", flush=True)

        try:
            available = wait_for_models(
                base_url,
                require_target=False,
                timeout=SERVER_API_TIMEOUT_SEC,
            )
        except StartupError as api_error:
            raise StartupError(
                "LM_SERVER_API_TIMEOUT",
                f"server開始後もLM Studio APIがreadyになりません: {api_error}",
            ) from api_error

    # 2. 本番Gemmaだけを許可する。
    #    mobile版や別モデルへのfallbackは絶対に行わない。
    target_visible = TARGET_MODEL in available
    if cold_start and not target_visible:
        print(
            f"LM model: waiting for automatic restore ({MODEL_RESTORE_TIMEOUT_SEC:.0f}s)",
            flush=True,
        )
        try:
            available = wait_for_models(
                base_url,
                require_target=True,
                timeout=MODEL_RESTORE_TIMEOUT_SEC,
            )
        except StartupError as restore_error:
            print(
                f"LM model: automatic restore wait ended without exact target "
                f"({restore_error.code})",
                flush=True,
            )
        else:
            target_visible = TARGET_MODEL in available
            if target_visible:
                print(
                    f"LM model: exact target became visible during restore wait "
                    f"({TARGET_MODEL})",
                    flush=True,
                )
    if target_visible:
        print(f"LM model: exact target already visible ({TARGET_MODEL})", flush=True)
    else:
        print(
            f"LM model: exact target not visible ({TARGET_MODEL})",
            flush=True,
        )

    # 3. すでにchatまで通るなら何もしない。
    if target_visible:
        try:
            check_api(base_url, models_timeout, chat_timeout)
        except StartupError as exc:
            if exc.code == "MODEL_UNAVAILABLE":
                raise StartupError(
                    "TARGET_MODEL_NOT_VISIBLE",
                    f"本番Gemma {TARGET_MODEL} が準備中に見えなくなりました。",
                ) from exc

            # modelは見えるが、まだロードされていない等のchat側エラーだけ
            # exact targetのlms loadを試す。
            if not exc.code.startswith("CHAT_"):
                raise

            print(
                f"LM model: visible but chat not ready ({exc.code}); "
                f"load started ({TARGET_MODEL})",
                flush=True,
            )
        else:
            print("LM model: already ready", flush=True)
            print("LM API/chat: final success", flush=True)
            print("LM Studio prepare: PASS", flush=True)
            return

    # 4. Local catalogへexact modelが現れた後だけloadする。
    print("LM model: waiting for local catalog readiness", flush=True)
    wait_for_local_target_model()

    # The automatic restore can complete between the earlier restore wait and
    # catalog readiness. Recheck the API immediately before issuing lms load.
    available_before_load = available_model_ids(
        base_url,
        min(models_timeout, 1.5),
    )
    if TARGET_MODEL in available_before_load:
        print(
            "LM model: exact target became visible before load; load skipped",
            flush=True,
        )
        check_api(base_url, models_timeout, chat_timeout)
        print("LM API/chat: final success", flush=True)
        print("LM Studio prepare: PASS", flush=True)
        return

    print(f"LM model: exact load started ({TARGET_MODEL})", flush=True)
    try:
        run_lms(
            [
                "load",
                TARGET_MODEL,
                "-y",
                "--identifier",
                TARGET_MODEL,
            ],
            MODEL_LOAD_TIMEOUT_SEC,
        )
    except StartupError as load_error:
        if not is_target_identifier_collision(load_error):
            raise StartupError(
                "TARGET_MODEL_LOAD_FAILED",
                f"ローカルの本番Gemma {TARGET_MODEL} をloadできません: {load_error}。"
                "自動downloadや別モデルへのfallbackは行いません。",
            ) from load_error

        print(
            "LM model: identifier already exists; checking concurrent restore",
            flush=True,
        )
        try:
            wait_for_models(
                base_url,
                require_target=True,
                timeout=IDENTIFIER_COLLISION_TIMEOUT_SEC,
            )
        except StartupError as collision_error:
            raise StartupError(
                "TARGET_MODEL_IDENTIFIER_COLLISION_TIMEOUT",
                f"identifier競合後も本番Gemma {TARGET_MODEL} が期限内に見えません: "
                f"{collision_error}。競合文字列だけでは成功扱いしません。",
            ) from collision_error
        print(
            "LM model: exact target became visible after identifier collision",
            flush=True,
        )
    else:
        try:
            wait_for_models(
                base_url,
                require_target=True,
                timeout=MODEL_READY_TIMEOUT_SEC,
            )
        except StartupError as ready_error:
            raise StartupError(
                "TARGET_MODEL_LOAD_TIMEOUT",
                f"本番Gemma {TARGET_MODEL} がload後も期限内に見えません: {ready_error}",
            ) from ready_error
        print(f"LM model: load succeeded ({TARGET_MODEL})", flush=True)

    # 5. 最後は必ず実chatで検証。
    check_api(base_url, models_timeout, chat_timeout)
    print("LM API/chat: final success", flush=True)
    print("LM Studio prepare: PASS", flush=True)


def positive_timeout(value: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return timeout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-url", help="Standalone check URL; launcher uses saved UI config")
    parser.add_argument("--models-timeout", type=positive_timeout, default=5.0)
    parser.add_argument("--chat-timeout", type=positive_timeout, default=15.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dependencies-only",
        action="store_true",
        help="UI起動に必要なPython依存関係だけ確認する",
    )
    mode.add_argument(
        "--prepare-lm",
        action="store_true",
        help="localhostのLM Studio APIと対象modelを起動・準備して疎通確認する",
    )
    args = parser.parse_args()

    try:
        check_dependencies()
        print(f"dependencies: OK ({sys.executable})", flush=True)

        if args.dependencies_only:
            print("V7 dependency preflight: PASS", flush=True)
            return 0

        base_url = base_url_from_config(args.config, args.base_url)
        print(f"LM Studio API: {base_url} model={TARGET_MODEL}", flush=True)

        if args.prepare_lm:
            prepare_lm(base_url, args.models_timeout, args.chat_timeout)
        else:
            check_api(base_url, args.models_timeout, args.chat_timeout)

    except StartupError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr, flush=True)
        return 1

    print("V7 startup preflight: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
