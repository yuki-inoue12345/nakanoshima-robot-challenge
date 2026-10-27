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


def run_lms(arguments: list[str], timeout: float) -> None:
    executable = find_lms_executable()
    print("LM CLI: " + " ".join(arguments), flush=True)
    try:
        subprocess.run(
            [executable, *arguments],
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise StartupError(
            "LMS_COMMAND_TIMEOUT",
            f"lms command が {timeout:.0f} 秒以内に終了しません。",
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StartupError(
            "LMS_COMMAND_FAILED",
            f"lms command に失敗しました: {exc}",
        ) from exc


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

        print("LM API: starting local server", flush=True)
        run_lms(
            [
                "server",
                "start",
                "-p",
                str(port),
                "--bind",
                "127.0.0.1",
            ],
            30.0,
        )
        available = wait_for_models(
            base_url,
            require_target=False,
            timeout=30.0,
        )

    # 2. 本番Gemmaだけを許可する。
    #    mobile版や別モデルへのfallbackは絶対に行わない。
    if TARGET_MODEL not in available:
        print(
            f"LM model: waiting for production target {TARGET_MODEL}",
            flush=True,
        )
        try:
            available = wait_for_models(
                base_url,
                require_target=True,
                timeout=max(5.0, min(15.0, models_timeout)),
            )
        except StartupError as exc:
            raise StartupError(
                "TARGET_MODEL_NOT_VISIBLE",
                f"本番Gemma {TARGET_MODEL} がLM Studio / LM Linkから見えません。"
                f" available={available}。"
                "先輩PC・LM Link・本番Gemmaの状態を確認してください。"
                "別モデルへのfallbackは行いません。",
            ) from exc

    # 3. すでにchatまで通るなら何もしない。
    try:
        check_api(base_url, models_timeout, chat_timeout)
    except StartupError as exc:
        if exc.code == "MODEL_UNAVAILABLE":
            raise StartupError(
                "TARGET_MODEL_NOT_VISIBLE",
                f"本番Gemma {TARGET_MODEL} が準備中に見えなくなりました。",
            ) from exc

        # modelは見えるが、まだロードされていない等のchat側エラーだけ
        # lms loadを試す。
        if not exc.code.startswith("CHAT_"):
            raise

        print(
            f"LM model: visible but chat not ready ({exc.code}); "
            f"loading {TARGET_MODEL}",
            flush=True,
        )
    else:
        print("LM model: already ready", flush=True)
        print("LM Studio prepare: PASS", flush=True)
        return

    # 4. LM Link / LM Studioから見えている本番モデルだけをロード。
    run_lms(
        [
            "load",
            TARGET_MODEL,
            "-y",
            "--identifier",
            TARGET_MODEL,
        ],
        120.0,
    )

    wait_for_models(
        base_url,
        require_target=True,
        timeout=120.0,
    )

    # 5. 最後は必ず実chatで検証。
    check_api(base_url, models_timeout, chat_timeout)
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
