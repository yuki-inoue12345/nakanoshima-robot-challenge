#!/usr/bin/env bash
set -eo pipefail
exec 3>&2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR"
APP="daikoudai_v7.py.new"
LOG_DIR="$WS_DIR/outputs"
LOG_FILE="$LOG_DIR/daikoudai_v7_desktop.log"

mkdir -p "$LOG_DIR"
cd "$WS_DIR"

show_launch_error() {
  local message="$1"
  printf '%s\nLog: %s\n' "$message" "$LOG_FILE" >&3
  if command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    zenity --error --title="V7 起動失敗" \
      --text="$message
$(tail -n 12 "$LOG_FILE")
Log: $LOG_FILE" || true
  fi
}

# Standard-library bootstrap only; no ROS, camera, recognition, or GUI startup.
prepare_lm_api() {
  "$PYTHON_BIN" -B - <<'PY'
import errno
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

API_URL = "http://127.0.0.1:1234/v1/models"
MODEL = "gemma-4-e2b-it-qat"
HTTP_SEC = 1.5
API_READY_SEC = 30.0
MODEL_READY_SEC = 120.0
POLL_SEC = 0.5
MAX_BYTES = 1024 * 1024


class LaunchError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Only the fixed local endpoint can establish readiness.


opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def read_models(timeout):
    """None means unhealthy HTTP; an empty set means healthy, no model loaded."""
    try:
        with opener.open(API_URL, timeout=timeout) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return None
        body = json.loads(raw)
        if not isinstance(body, dict) or "error" in body:
            return None
        entries = body.get("data")
        if not isinstance(entries, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("id"), str)
                for item in entries):
            return None
        return {item["id"] for item in entries}
    except (urllib.error.URLError, OSError, ValueError, UnicodeError):
        return None


def models(timeout):
    # urllib's socket timeout alone is not a whole-response deadline: a server
    # can drip bytes indefinitely. Bound the complete read, including headers
    # and body. A late read-only probe is ignored and its daemon thread cannot
    # delay process exit or start any CLI/state-changing operation.
    result = queue.Queue(maxsize=1)

    def probe():
        result.put(read_models(timeout))

    threading.Thread(target=probe, daemon=True).start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        return None


def require_lms():
    executable = shutil.which("lms")
    # Desktop sessions may omit LM Studio's standard CLI directory from PATH.
    if not executable:
        candidate = Path.home() / ".lmstudio" / "bin" / "lms"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            executable = str(candidate)
    if not executable:
        raise LaunchError("LMS_NOT_FOUND: LM API/serverまたはmodelの起動に lms が必要ですが、"
                          "PATH / ~/.lmstudio/bin にありません。")
    return executable


def assert_free_port():
    # Bind is only an availability probe, never an API-ready test. Do not
    # change or terminate an existing owner. A subsequent bind race makes
    # lms fail and startup stays closed; no attempt to take over is made.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 1234))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise LaunchError("127.0.0.1:1234 は使用中ですが LM API が応答していません。"
                              "LM Studio/API Server の状態を確認してください。") from exc
        raise LaunchError(f"PORT_CHECK_FAILED: 1234の空き状態を確認できません: {exc}") from exc


def run_lms(arguments, deadline, timeout_code):
    executable = require_lms()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LaunchError(f"{timeout_code}: LM準備期限を超えました。")
    print("LM CLI: " + " ".join(arguments), flush=True)
    try:
        # Bound this newly started CLI command only. No existing LM Studio,
        # API server, or port-owner process is targeted. stdout/stderr inherit
        # the log, avoiding a pipe held open by a daemon spawned by the CLI.
        subprocess.run([executable, *arguments], check=True, timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise LaunchError(f"{timeout_code}: lms commandが期限内に終了しません。") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchError(f"LMS_COMMAND_FAILED: {exc}. LM Studio serviceの状態を確認してください。") from exc


def wait_ready(deadline, require_model, timeout_code):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LaunchError(f"{timeout_code}: HTTP API/modelの準備が期限内に確認できません。")
        available = models(min(HTTP_SEC, remaining))
        # Do not accept a response completing after the shared CLI/poll budget.
        if time.monotonic() >= deadline:
            raise LaunchError(f"{timeout_code}: HTTP API/modelの準備期限を超えました。")
        if available is not None and (not require_model or MODEL in available):
            return available
        time.sleep(min(POLL_SEC, max(0.0, deadline - time.monotonic())))


try:
    available = models(HTTP_SEC)
    if available is None:
        assert_free_port()
        deadline = time.monotonic() + API_READY_SEC
        run_lms(["server", "start", "-p", "1234", "--bind", "127.0.0.1"],
                deadline, "API_READY_TIMEOUT")
        available = wait_ready(deadline, False, "API_READY_TIMEOUT")
    if MODEL not in available:
        deadline = time.monotonic() + MODEL_READY_SEC
        run_lms(["load", MODEL, "-y", "--identifier", MODEL],
                deadline, "MODEL_READY_TIMEOUT")
        wait_ready(deadline, True, "MODEL_READY_TIMEOUT")
    print(f"LM AUTO READY: {API_URL} ({MODEL})", flush=True)
except LaunchError as exc:
    print(str(exc), file=sys.stderr, flush=True)
    # Keep CLI stdout/stderr attached directly to the logfile rather than a
    # tee pipe that a newly started service could retain. Descriptor 3 is the
    # desktop stderr preserved by this shell; mirror only the bootstrap error.
    with os.fdopen(os.dup(3), "w") as desktop_error:
        print(str(exc), file=desktop_error, flush=True)
    sys.exit(1)
PY
}

{
  echo "==== $(date '+%Y-%m-%d %H:%M:%S') daikoudai_v7 desktop launch ===="
  echo "cwd: $(pwd)"

  # Ubuntuのデスクトップ起動ではnounsetが有効な場合があり、ROS setupが落ちることがある。
  set +u
  if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "ROS: jazzy"
  elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "ROS: humble"
  else
    echo "ROS: setup.bash not found"
  fi
  set -u

  export ROS_DOMAIN_ID=57
  echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

  PYTHON_BIN=""
  VENV_DIR=""
  for candidate in "/home/koko-chan-nuc/venvs/person_search" "$WS_DIR/venv" "$WS_DIR/.venv"; do
    if [ -f "$candidate/bin/python" ] && [ -x "$candidate/bin/python" ]; then
      PYTHON_BIN="$candidate/bin/python"
      VENV_DIR="$candidate"
      break
    fi
  done
  if [ -n "$VENV_DIR" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
      source "$VENV_DIR/bin/activate"
    fi
    echo "venv: $VENV_DIR"
  elif ! PYTHON_BIN="$(type -P python3)"; then
    show_launch_error "PYTHON_NOT_FOUND: Python interpreterがありません。"
    exit 1
  else
    echo "venv: not found, using PATH python3 (dependencies must pass)"
  fi

  echo "python: $PYTHON_BIN"

  # UIそのものはLM Studioなしでも起動可能にする。
  # Python/ROS/Camera側依存関係だけは起動前に必須確認する。
  if ! "$PYTHON_BIN" -B "$WS_DIR/v7_startup_preflight.py" \
      --dependencies-only 2>&1 | tee /dev/fd/3; then
    show_launch_error "V7 dependency preflight失敗。人物探索UIは起動しません。"
    exit 1
  fi

  exec "$PYTHON_BIN" "$WS_DIR/$APP"
} >> "$LOG_FILE" 2>&1
