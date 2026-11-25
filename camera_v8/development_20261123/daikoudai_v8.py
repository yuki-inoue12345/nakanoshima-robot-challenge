#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese target-input UI for the Milestone-2 person search.

This UI is a small launcher. It does not change the recognition logic; it only
turns easy Japanese target input into the arguments used by the existing
Milestone-2 scripts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from camera_preview import HighGuiPreviewManager
from v8_result_sound import play_person_search_notification


APP_DIR = Path(__file__).resolve().parent
V7_DIR = APP_DIR.parents[1] / "camera_v7" / "development_20261019"
if not (V7_DIR / "person_search_vlm_milestone2_v7.py").is_file():
    raise RuntimeError(f"Camera V7 MS2 algorithm missing: {V7_DIR}")
sys.path.insert(0, str(V7_DIR))
THIS_FILE = Path(__file__).resolve()
SIM_SCRIPT = V7_DIR / "person_search_vlm_milestone2_v7.py"
BOOL_NODE_SCRIPT = APP_DIR / "person_search_milestone2_bool_ros_node.py"
LAST_CONFIG_PATH = APP_DIR / ".daikoudai_v8_target_ui_last.json"
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
DEFAULT_REQUEST_TOPIC = "/camera_capture_request"
DEFAULT_DONE_TOPIC = "/camera_capture_done"
DEFAULT_CAPTURE_OUT_DIR = "outputs/topic_camera_capture"
DEFAULT_ARRIVAL_TOPIC = "/person_area_arrived"
DEFAULT_SEARCH_DONE_TOPIC = "/person_search_done"
DEFAULT_SEARCH_DETAIL_TOPIC = "/person_search_done_detail"
DEFAULT_ARRIVAL_ACK_TOPIC = "/person_search/arrival_received"
GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 15.0
ROS_SPIN_POLL_TIMEOUT_SEC = 0.1
SERVER_SHUTDOWN_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})

CV_F10_KEYS = {65479, 0xFFC7, 16777273, 0x01000039}


class GracefulShutdownRunner:
    """Move blocking recorder/ROS cleanup out of the Python signal handler."""

    def __init__(
        self,
        request_shutdown: Callable[[], None],
        shutdown_runtime: Callable[[], None],
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._request_shutdown = request_shutdown
        self._shutdown_runtime = shutdown_runtime
        self._logger = logger or (lambda _message: None)
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None

    def request(self) -> bool:
        with self._lock:
            if self._thread is not None:
                return False
            self._thread = threading.Thread(
                target=self._run,
                name="person-search-graceful-shutdown",
                daemon=False,
            )
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            self._request_shutdown()
        except Exception as exc:
            self._logger(f"shutdown request failed: {type(exc).__name__}: {exc}")
        try:
            self._shutdown_runtime()
        except Exception as exc:
            self._logger(f"runtime shutdown failed: {type(exc).__name__}: {exc}")

    def wait(self, timeout_sec: float = GRACEFUL_SHUTDOWN_TIMEOUT_SEC) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.1, float(timeout_sec)))
        return not thread.is_alive()


def spin_node_with_signal_poll(
    node: Any,
    rclpy_module: Any,
    shutdown_requested: threading.Event,
    *,
    timeout_sec: float = ROS_SPIN_POLL_TIMEOUT_SEC,
) -> None:
    """Spin one callback at a time and observe the shutdown event promptly."""
    poll_timeout = max(0.01, float(timeout_sec))
    while rclpy_module.ok() and not shutdown_requested.is_set():
        rclpy_module.spin_once(node, timeout_sec=poll_timeout)


def block_server_shutdown_signals() -> None:
    """Route process stop signals to sigwait instead of Python's main thread."""
    signal.pthread_sigmask(signal.SIG_BLOCK, SERVER_SHUTDOWN_SIGNALS)


def start_server_shutdown_signal_waiter(
    callback: Callable[[int, Any], None],
) -> threading.Thread:
    """Receive SIGINT/SIGTERM even while the ROS executor is in a C wait."""
    def wait_for_signal() -> None:
        signum = signal.sigwait(SERVER_SHUTDOWN_SIGNALS)
        callback(signum, None)

    thread = threading.Thread(
        target=wait_for_signal,
        name="person-search-signal-waiter",
        daemon=True,
    )
    thread.start()
    return thread


def publish_true_arrival_ack(arrival_value: Any, publisher: Any, message_type: Any) -> bool:
    """Acknowledge only a true arrival callback without changing search state."""
    if not bool(arrival_value):
        return False
    ack = message_type()
    ack.data = True
    publisher.publish(ack)
    return True


def classify_area_search_results(area_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return the terminal area outcome without collapsing target ERRORs."""
    error_target_ids = sorted(
        target_id
        for target_id, result in area_results.items()
        if str(result.get("status", "")).strip().upper() == "ERROR"
    )
    if error_target_ids:
        return {
            "status": "ERROR",
            "success": False,
            "found_target_id": "",
            "safe_to_advance": False,
            "error_target_ids": error_target_ids,
        }

    winners = []
    for target_id, result in area_results.items():
        if bool(result.get("ms2_success", False)):
            winners.append((float(result.get("final_score", 0.0) or 0.0), target_id))
    winners.sort(reverse=True)
    found_target_id = winners[0][1] if winners else ""
    return {
        "status": "FOUND_FRONT_PHOTO" if found_target_id else "NOT_FOUND_OR_NOT_FRONT",
        "success": bool(found_target_id),
        "found_target_id": found_target_id,
        "safe_to_advance": True,
        "error_target_ids": [],
    }


def area_outcome_safe_to_advance(outcome: Dict[str, Any]) -> bool:
    """Return the completion contract consumed by Navigation."""
    return bool(outcome.get("safe_to_advance", False))


def result_detail_with_completion_contract(
    detail: Dict[str, Any],
    *,
    safe_to_advance: bool,
    terminal: bool = True,
) -> Dict[str, Any]:
    """Attach an explicit contract to diagnostic/result JSON."""
    payload = dict(detail)
    payload["safe_to_advance"] = bool(safe_to_advance)
    payload["terminal"] = bool(terminal)
    payload["completion_contract"] = "person_search_done_safe_to_advance_v1"
    return payload


def record_target_search_observation(state: Dict[str, Any], result: Dict[str, Any], area_index: int) -> None:
    """Keep the latest attempt separately from committed FOUND evidence."""
    state["last_search_area_index"] = int(area_index)
    state["last_search_result"] = dict(result)
    state["last_not_found_reason"] = result.get("not_found_reason")
    state["best_candidate_evidence"] = result.get("best_candidate_evidence")


AREA_TARGET_ID_BY_INDEX = {
    1: "target_A",
    2: "target_B",
}


class AreaTargetMappingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def select_area_target_profile(
    target_profiles: List[Dict[str, Any]],
    target_states: Dict[str, Dict[str, Any]],
    area_index: int,
) -> Dict[str, Any]:
    """Map one navigation area to exactly one explicitly named target."""
    expected_target_id = AREA_TARGET_ID_BY_INDEX.get(int(area_index))
    if expected_target_id is None:
        expected_areas = ",".join(str(value) for value in AREA_TARGET_ID_BY_INDEX)
        raise AreaTargetMappingError(
            "AREA_TARGET_INDEX_OUT_OF_RANGE",
            f"area_index={area_index}; expected one of [{expected_areas}]",
        )
    matches = [
        profile
        for profile in target_profiles
        if str(profile.get("target_id", "")) == expected_target_id
    ]
    if not matches:
        raise AreaTargetMappingError(
            "AREA_TARGET_PROFILE_MISSING",
            f"area_index={area_index} requires target_id={expected_target_id}",
        )
    if len(matches) != 1:
        raise AreaTargetMappingError(
            "AREA_TARGET_PROFILE_DUPLICATE",
            f"target_id={expected_target_id} appears {len(matches)} times",
        )
    state = target_states.get(expected_target_id)
    if not isinstance(state, dict):
        raise AreaTargetMappingError(
            "AREA_TARGET_STATE_MISSING",
            f"state for target_id={expected_target_id} is unavailable",
        )
    if bool(state.get("ms2_success", False)):
        raise AreaTargetMappingError(
            "AREA_TARGET_ALREADY_FOUND",
            f"target_id={expected_target_id} was already found and will not be searched again",
        )
    return matches[0]


def run_area_target_search_once(
    target_profiles: List[Dict[str, Any]],
    target_states: Dict[str, Dict[str, Any]],
    area_index: int,
    search_target: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Run and record only the target explicitly assigned to this area."""
    profile = select_area_target_profile(target_profiles, target_states, area_index)
    target_id = str(profile["target_id"])
    result = search_target(profile)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"TARGET_SEARCH_RESULT_INVALID: target_id={target_id} did not return a dict"
        )
    result = dict(result)
    result["target_id"] = target_id
    result["area_index"] = int(area_index)
    record_target_search_observation(target_states[target_id], result, area_index)
    return target_id, {target_id: result}


def is_cv_f10_key(key: int) -> bool:
    try:
        key_int = int(key)
    except (TypeError, ValueError):
        return False
    return key_int in CV_F10_KEYS or (key_int & 0xFFFF) in {65479, 0xFFC7}

TARGET_FIELD_KEYS = [
    "subject_type",
    "target_upper_color",
    "target_upper_secondary_color",
    "target_upper_pattern",
    "target_upper_type",
    "target_lower_color",
    "target_lower_secondary_color",
    "target_lower_pattern",
    "target_lower_type",
    "target_gender",
    "target_item",
]

DEFAULT_TARGET_PROFILES: Dict[str, Dict[str, str]] = {
    "target_A": {
        "target_id": "target_A",
        "label": "Target A",
        "subject_type": "mannequin",
        "target_upper_color": "white",
        "target_upper_secondary_color": "navy",
        "target_upper_pattern": "striped",
        "target_upper_type": "sweater",
        "target_lower_color": "blue",
        "target_lower_secondary_color": "none",
        "target_lower_pattern": "solid",
        "target_lower_type": "skirt",
        "target_gender": "none",
        "target_item": "unspecified",
    },
    "target_B": {
        "target_id": "target_B",
        "label": "Target B",
        "subject_type": "person",
        "target_upper_color": "gray",
        "target_upper_secondary_color": "beige",
        "target_upper_pattern": "plaid",
        "target_upper_type": "shirt",
        "target_lower_color": "black",
        "target_lower_secondary_color": "none",
        "target_lower_pattern": "solid",
        "target_lower_type": "pants",
        "target_gender": "male",
        "target_item": "unspecified",
    },
}


class TargetProfilesError(ValueError):
    pass


def load_target_profiles(targets_json: str = "") -> List[Dict[str, Any]]:
    if not targets_json:
        return [dict(DEFAULT_TARGET_PROFILES["target_A"]), dict(DEFAULT_TARGET_PROFILES["target_B"])]
    try:
        parsed = json.loads(targets_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TargetProfilesError(f"TARGETS_JSON_PARSE_ERROR: {exc}") from exc
    if not isinstance(parsed, list):
        raise TargetProfilesError("TARGETS_JSON_NOT_LIST")
    if not parsed:
        raise TargetProfilesError("TARGETS_JSON_EMPTY")

    profiles: List[Dict[str, Any]] = []
    for idx, profile in enumerate(parsed):
        if not isinstance(profile, dict):
            continue
        target_id = str(profile.get("target_id") or f"target_{idx + 1}")
        defaults = DEFAULT_TARGET_PROFILES.get(target_id, {})
        has_explicit_target_field = any(key in profile for key in TARGET_FIELD_KEYS)
        if not defaults and not has_explicit_target_field:
            continue
        profiles.append({**defaults, **profile, "target_id": target_id})
    if not profiles:
        raise TargetProfilesError("TARGETS_JSON_NO_USABLE_PROFILE")
    return profiles


COLOR_OPTIONS: List[Tuple[str, str]] = [
    ("指定なし", "none"),
    ("白", "white"),
    ("黒", "black"),
    ("グレー", "gray"),
    ("水色", "light_blue"),
    ("青", "blue"),
    ("紺", "navy"),
    ("ベージュ", "beige"),
    ("タン", "tan"),
    ("クリーム", "cream"),
    ("茶色", "brown"),
    ("カーキ", "khaki"),
    ("緑", "green"),
    ("赤", "red"),
    ("ピンク", "pink"),
    ("黄色", "yellow"),
]

SUBJECT_TYPE_OPTIONS: List[Tuple[str, str]] = [
    ("自動", "auto"),
    ("人間", "person"),
    ("マネキン", "mannequin"),
]

PATTERN_OPTIONS: List[Tuple[str, str]] = [
    ("指定なし", "none"),
    ("無地", "solid"),
    ("ボーダー・ストライプ", "striped"),
    ("チェック", "plaid"),
    ("プリント・グラフィック", "graphic"),
    ("花柄", "floral"),
    ("カラーブロック", "color_block"),
]

UPPER_TYPE_OPTIONS: List[Tuple[str, str]] = [
    ("指定なし", "none"),
    ("シャツ", "shirt"),
    ("Tシャツ", "t_shirt"),
    ("パーカー", "hoodie"),
    ("ベスト", "vest"),
    ("スウェット", "sweatshirt"),
    ("ジャケット", "jacket"),
    ("コート", "coat"),
    ("セーター", "sweater"),
    ("カーディガン", "cardigan"),
    ("ポロシャツ", "polo_shirt"),
]

LOWER_TYPE_OPTIONS: List[Tuple[str, str]] = [
    ("指定なし", "none"),
    ("ズボン", "pants"),
    ("半ズボン", "shorts"),
    ("スカート", "skirt"),
]

GENDER_OPTIONS: List[Tuple[str, str]] = [
    ("指定なし", "none"),
    ("男性", "male"),
    ("女性", "female"),
]

DECISION_OPTIONS: List[Tuple[str, str]] = [
    ("候補以上", "candidate"),
    ("強い候補のみ", "strong_candidate"),
    ("弱い候補以上", "weak_candidate"),
]

SEARCH_TARGET_MODE_OPTIONS: List[Tuple[str, str]] = [
    ("A/B両方", "both"),
    ("Target Aだけ", "target_A"),
    ("Target Bだけ", "target_B"),
]


COLOR_WORDS: List[Tuple[str, str]] = [
    (r"ライトブルー|薄青|薄い青|水色", "light_blue"),
    (r"ネイビー|紺", "navy"),
    (r"グレー|灰色|灰", "gray"),
    (r"ベージュ|肌色|クリーム", "beige"),
    (r"タン", "tan"),
    (r"カーキ", "khaki"),
    (r"ブラウン|茶色|茶", "brown"),
    (r"ホワイト|白", "white"),
    (r"ブラック|黒", "black"),
    (r"ブルー|青|デニム", "blue"),
    (r"グリーン|緑", "green"),
    (r"ピンク", "pink"),
    (r"レッド|赤", "red"),
    (r"イエロー|黄色|黄", "yellow"),
]

TYPE_WORDS: List[Tuple[str, str]] = [
    (r"パーカー|フーディ|フード", "hoodie"),
    (r"ベスト|チョッキ", "vest"),
    (r"カーディガン", "cardigan"),
    (r"ポロシャツ|ポロ", "polo_shirt"),
    (r"スウェット|トレーナー", "sweatshirt"),
    (r"Tシャツ|Ｔシャツ|tシャツ|ティーシャツ|半袖T|半袖Ｔ", "t_shirt"),
    (r"シャツ", "shirt"),
    (r"ジャケット", "jacket"),
    (r"コート", "coat"),
    (r"セーター|ニット", "sweater"),
]

PATTERN_WORDS: List[Tuple[str, str]] = [
    (r"ボーダー|横縞|横しま|ストライプ|縞", "striped"),
    (r"チェック|格子柄|格子|プラッド", "plaid"),
    (r"プリント|グラフィック|ロゴ", "graphic"),
    (r"花柄|フローラル", "floral"),
    (r"カラーブロック|色分け|切り替え", "color_block"),
    (r"無地", "solid"),
]


def label_map(options: List[Tuple[str, str]]) -> Dict[str, str]:
    return {label: value for label, value in options}


def reverse_label(options: List[Tuple[str, str]], value: str) -> str:
    for label, v in options:
        if v == value:
            return label
    return options[0][0]


def first_match(patterns: List[Tuple[str, str]], text: str) -> Optional[Tuple[str, re.Match[str]]]:
    for pattern, value in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return value, match
    return None


def nearest_color_before(text: str, index: int, window: int = 12) -> Optional[str]:
    start = max(0, index - window)
    chunk = text[start:index]
    best: Optional[Tuple[int, str]] = None
    for pattern, value in COLOR_WORDS:
        for match in re.finditer(pattern, chunk, flags=re.IGNORECASE):
            pos = match.start()
            if best is None or pos > best[0]:
                best = (pos, value)
    return best[1] if best else None


def parse_japanese_target(text: str) -> Dict[str, str]:
    normalized = re.sub(r"[\s、，。/・]+", "", text)
    result: Dict[str, str] = {}

    if re.search(r"マネキン|人形|展示人形|トルソー", normalized, flags=re.IGNORECASE):
        result["subject_type"] = "mannequin"
    elif re.search(r"人間|人物|人\b", normalized, flags=re.IGNORECASE):
        result["subject_type"] = "person"

    if re.search(r"男性|男の人|男\b|male", normalized, flags=re.IGNORECASE):
        result["target_gender"] = "male"
        result.setdefault("subject_type", "person")
    elif re.search(r"女性|女の人|女\b|female", normalized, flags=re.IGNORECASE):
        result["target_gender"] = "female"
        result.setdefault("subject_type", "person")

    def collect_matches(options: List[Tuple[str, str]]) -> List[Tuple[int, int, str]]:
        collected: List[Tuple[int, int, str]] = []
        for pattern, value in options:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                if any(match.start() < end and match.end() > start for start, end, _ in collected):
                    continue
                collected.append((match.start(), match.end(), value))
        return sorted(collected, key=lambda item: item[0])

    color_matches = collect_matches(COLOR_WORDS)
    pattern_matches = collect_matches(PATTERN_WORDS)
    upper_matches = collect_matches(TYPE_WORDS)
    lower_options = [
        (r"半ズボン|ショートパンツ|短パン", "shorts"),
        (r"スカート", "skirt"),
        (r"ズボン|パンツ", "pants"),
    ]
    lower_matches = collect_matches(lower_options)
    garment_anchors = sorted(
        [(start, end, "upper", value) for start, end, value in upper_matches]
        + [(start, end, "lower", value) for start, end, value in lower_matches],
        key=lambda item: item[0],
    )

    def clause_values(anchor: Tuple[int, int, str, str]) -> Tuple[List[str], List[str]]:
        start = anchor[0]
        previous_end = max((other[1] for other in garment_anchors if other[1] <= start), default=0)
        colors: List[str] = []
        for pos, _end, value in color_matches:
            if previous_end <= pos < start and value not in colors:
                colors.append(value)
        patterns = [value for pos, _end, value in pattern_matches if previous_end <= pos < start]
        return colors[-2:], patterns[-1:]

    selected_upper: Optional[Tuple[int, int, str, str]] = None
    for anchor in garment_anchors:
        if anchor[2] == "upper" and anchor[3] in {"vest", "jacket", "coat", "hoodie"}:
            selected_upper = anchor
            break
    if selected_upper is None:
        selected_upper = next((anchor for anchor in garment_anchors if anchor[2] == "upper"), None)
    selected_lower = next((anchor for anchor in garment_anchors if anchor[2] == "lower"), None)

    if selected_upper is not None:
        result["target_upper_type"] = selected_upper[3]
        colors, patterns = clause_values(selected_upper)
        if colors:
            result["target_upper_color"] = colors[0]
        if len(colors) >= 2:
            result["target_upper_secondary_color"] = colors[1]
        if patterns:
            result["target_upper_pattern"] = patterns[0]

    if selected_lower is not None:
        result["target_lower_type"] = selected_lower[3]
        colors, patterns = clause_values(selected_lower)
        if colors:
            result["target_lower_color"] = colors[0]
        if len(colors) >= 2:
            result["target_lower_secondary_color"] = colors[1]
        if patterns:
            result["target_lower_pattern"] = patterns[0]

    if selected_upper is None:
        if color_matches:
            result["target_upper_color"] = color_matches[0][2]
        if pattern_matches and selected_lower is None:
            result["target_upper_pattern"] = pattern_matches[0][2]

    return result


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def ros_wrapped_command(args: List[str]) -> List[str]:
    if Path(ROS_SETUP).exists():
        return ["bash", "-lc", f"source {shlex.quote(ROS_SETUP)} && exec {shell_join(args)}"]
    return args


def run_person_search_server_mode(argv: Optional[List[str]] = None) -> None:
    # Per-domain singleton prevents a stale GUI/server from duplicating YOLO/VLM work.
    from v8_search_singleton import V8ServerAlreadyRunning, acquire_v8_person_search_lock
    try:
        _v8_search_lock_fd = acquire_v8_person_search_lock()
    except V8ServerAlreadyRunning as exc:
        raise SystemExit(str(exc)) from None
    # Block before cv2/rclpy/MS2 imports can create native threads. Process-wide
    # SIGTERM may otherwise be delivered to an older unblocked thread and take
    # its default action before the dedicated sigwait thread can receive it.
    block_server_shutdown_signals()
    parser = argparse.ArgumentParser(description="UI内蔵のMS2人物探索トピックサーバー")
    parser.add_argument("--arrival_topic", default="/person_area_arrived")
    parser.add_argument("--done_topic", default="/person_search_done")
    parser.add_argument("--detail_topic", default="/person_search_done_detail")
    parser.add_argument("--stop_topic", default="/stop")
    parser.add_argument("--targets_json", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--camera_device", default="")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=float, default=15.0)
    parser.add_argument("--camera_read_fail_tolerance", type=int, default=30)
    parser.add_argument("--raw_image_topic", default="/koko/person_search/raw_image")
    parser.add_argument(
        "--pipeline_diagnostics", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--subject_type", choices=["auto", "person", "mannequin"], default="auto")
    parser.add_argument("--target_upper_color", default="white")
    parser.add_argument("--target_upper_secondary_color", default="none")
    parser.add_argument("--target_upper_pattern", default="none")
    parser.add_argument("--target_upper_type", default="shirt")
    parser.add_argument("--target_lower_color", default="black")
    parser.add_argument("--target_lower_secondary_color", default="none")
    parser.add_argument("--target_lower_pattern", default="none")
    parser.add_argument("--target_lower_type", default="pants")
    parser.add_argument("--target_gender", default="male")
    parser.add_argument("--target_item", default="unspecified")
    parser.add_argument("--search_duration_sec", type=float, default=20.0)
    parser.add_argument("--search_until_found", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_search_runtime_sec", type=float, default=0.0)
    parser.add_argument("--max_total_vlm_calls", type=int, default=6)
    parser.add_argument("--decision_accept", choices=["strong_candidate", "candidate", "weak_candidate"], default="candidate")
    parser.add_argument("--lmstudio_base_url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--accept_front_views", default="front")
    parser.add_argument("--max_front_vlm_calls", type=int, default=4)
    parser.add_argument("--front_retry_interval_sec", type=float, default=1.0)
    parser.add_argument("--require_cv_face_for_front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_min_bbox_aspect", type=float, default=0.28)
    parser.add_argument("--front_min_crop_quality", type=float, default=0.60)
    parser.add_argument("--front_min_area_ratio", type=float, default=0.06)
    parser.add_argument("--front_min_height_ratio", type=float, default=0.35)
    parser.add_argument("--front_crop_margin", type=float, default=0.12)
    parser.add_argument("--front_reject_edge_crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_allow_vertical_edge_with_face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_vertical_edge_face_min_quality", type=float, default=0.60)
    parser.add_argument("--best_photo_hold_sec", type=float, default=2.5)
    parser.add_argument("--lmstudio_timeout", type=int, default=8)
    parser.add_argument("--front_vlm_timeout", type=int, default=2)
    parser.add_argument("--person_max_tokens", type=int, default=260)
    parser.add_argument("--person_vlm_long_side", type=int, default=384)
    parser.add_argument("--person_crop_margin", type=float, default=0.45)
    parser.add_argument("--attribute_crop_margin", type=float, default=0.0)
    parser.add_argument("--use_fast_heuristic_person_match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_heuristic_allow_distinct_targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--front_fast_local_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_vlm_on_fast_gate_miss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--front_fast_min_skin_area_ratio", type=float, default=0.018)
    parser.add_argument("--front_fast_min_skin_width_ratio", type=float, default=0.12)
    parser.add_argument("--front_fast_min_skin_height_ratio", type=float, default=0.10)
    parser.add_argument("--fast_heuristic_accept_score", type=float, default=0.75)
    parser.add_argument("--accept_crop_heuristic_for_distinct_targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm_min_track_frames", type=int, default=1)
    parser.add_argument("--min_crop_quality_for_vlm", type=float, default=0.35)
    parser.add_argument("--target_color_send_threshold", type=float, default=0.05)
    parser.add_argument("--high_quality_send_threshold", type=float, default=0.55)
    parser.add_argument("--max_person_vlm_calls_per_memory", type=int, default=2)
    parser.add_argument("--person_vlm_retry_interval_sec", type=float, default=2.0)
    parser.add_argument("--person_vlm_retry_min_quality_gain", type=float, default=0.08)
    parser.add_argument("--process_every", type=int, default=5)
    parser.add_argument("--search_window_width", type=int, default=960)
    parser.add_argument("--search_window_height", type=int, default=540)
    parser.add_argument("--search_window_x", type=int, default=220)
    parser.add_argument("--search_window_y", type=int, default=120)
    args = parser.parse_args(argv)
    validated_target_profiles = load_target_profiles(args.targets_json)

    import traceback

    import cv2
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Bool, String

    from v8_shared_camera import (
        RosFrameFeed, V8FrameCoordinator, V8ModelStageDiagnostics,
    )
    from person_search_vlm_milestone2_v7 import (
        build_parser,
        open_camera_capture_with_fallback,
        request_stop as request_ms2_stop,
        run_milestone2_search,
    )
    ms2_module = sys.modules[run_milestone2_search.__module__]

    def json_dumps(data: Dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)

    def target_state(profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target_id": str(profile.get("target_id", "")),
            "subject_type": str(profile.get("subject_type", "auto")),
            "target_attributes": {k: profile.get(k) for k in TARGET_FIELD_KEYS if k in profile},
            "found": False,
            "ms2_success": False,
            "found_area_index": None,
            "evidence_path": "",
            "result": None,
            "last_search_area_index": None,
            "last_search_result": None,
            "last_not_found_reason": None,
            "best_candidate_evidence": None,
        }

    def mission_success(states: Dict[str, Dict[str, Any]]) -> bool:
        a = states.get("target_A", {})
        b = states.get("target_B", {})
        return (
            bool(a.get("ms2_success"))
            and bool(b.get("ms2_success"))
            and a.get("found_area_index") is not None
            and b.get("found_area_index") is not None
            and a.get("found_area_index") != b.get("found_area_index")
        )

    def build_search_args(profile: Dict[str, Any], area_index: int, mission_dir: Path):
        search_args = build_parser().parse_args([])
        search_args.video = str(args.video or "")
        search_args.camera_index = int(args.camera_index)
        search_args.camera_device = str(args.camera_device or "")
        search_args.camera_width = int(args.camera_width)
        search_args.camera_height = int(args.camera_height)
        search_args.camera_fps = float(args.camera_fps)
        search_args.camera_read_fail_tolerance = int(args.camera_read_fail_tolerance)
        search_args.record_search_video = False
        search_args.search_video_path = ""
        search_args.search_video_dir = ""
        search_args.record_video_fps = 0.0
        search_args.subject_type = str(profile.get("subject_type", "auto"))
        search_args.target_upper_color = str(profile.get("target_upper_color", "none"))
        search_args.target_upper_secondary_color = str(profile.get("target_upper_secondary_color", "none"))
        search_args.target_upper_pattern = str(profile.get("target_upper_pattern", "none"))
        search_args.target_upper_type = str(profile.get("target_upper_type", "none"))
        search_args.target_lower_color = str(profile.get("target_lower_color", "none"))
        search_args.target_lower_secondary_color = str(profile.get("target_lower_secondary_color", "none"))
        search_args.target_lower_pattern = str(profile.get("target_lower_pattern", "none"))
        search_args.target_lower_type = str(profile.get("target_lower_type", "none"))
        search_args.target_gender = str(profile.get("target_gender", "none"))
        search_args.target_item = str(profile.get("target_item", "unspecified"))
        search_args.area_arrival_sec = 0.0
        search_args.search_duration_sec = float(args.search_duration_sec)
        search_args.search_until_found = bool(args.search_until_found)
        search_args.max_search_runtime_sec = float(args.max_search_runtime_sec)
        search_args.max_total_vlm_calls = int(args.max_total_vlm_calls)
        search_args.decision_accept = args.decision_accept
        search_args.lmstudio_base_url = args.lmstudio_base_url
        search_args.accept_front_views = args.accept_front_views
        search_args.strict_ms2_front = True
        search_args.require_cv_face_for_front = bool(args.require_cv_face_for_front)
        search_args.front_min_bbox_aspect = float(args.front_min_bbox_aspect)
        search_args.front_min_crop_quality = float(args.front_min_crop_quality)
        search_args.front_min_area_ratio = float(args.front_min_area_ratio)
        search_args.front_min_height_ratio = float(args.front_min_height_ratio)
        search_args.front_crop_margin = float(args.front_crop_margin)
        search_args.front_reject_edge_crops = bool(args.front_reject_edge_crops)
        search_args.front_allow_vertical_edge_with_face = bool(args.front_allow_vertical_edge_with_face)
        search_args.front_vertical_edge_face_min_quality = float(args.front_vertical_edge_face_min_quality)
        search_args.best_photo_hold_sec = float(args.best_photo_hold_sec)
        search_args.max_front_vlm_calls = int(args.max_front_vlm_calls)
        search_args.front_retry_interval_sec = float(args.front_retry_interval_sec)
        search_args.vlm_min_track_frames = int(args.vlm_min_track_frames)
        search_args.lmstudio_timeout = int(args.lmstudio_timeout)
        search_args.front_vlm_timeout = int(args.front_vlm_timeout)
        search_args.person_max_tokens = int(args.person_max_tokens)
        search_args.person_vlm_long_side = int(args.person_vlm_long_side)
        search_args.person_crop_margin = float(args.person_crop_margin)
        search_args.attribute_crop_margin = float(args.attribute_crop_margin)
        search_args.use_fast_heuristic_person_match = bool(args.use_fast_heuristic_person_match)
        search_args.fast_heuristic_allow_distinct_targets = bool(args.fast_heuristic_allow_distinct_targets)
        search_args.accept_crop_heuristic_for_distinct_targets = bool(args.accept_crop_heuristic_for_distinct_targets)
        search_args.front_fast_local_gate = bool(args.front_fast_local_gate)
        search_args.front_vlm_on_fast_gate_miss = bool(args.front_vlm_on_fast_gate_miss)
        search_args.front_fast_min_skin_area_ratio = float(args.front_fast_min_skin_area_ratio)
        search_args.front_fast_min_skin_width_ratio = float(args.front_fast_min_skin_width_ratio)
        search_args.front_fast_min_skin_height_ratio = float(args.front_fast_min_skin_height_ratio)
        search_args.fast_heuristic_accept_score = float(args.fast_heuristic_accept_score)
        search_args.min_crop_quality_for_vlm = float(args.min_crop_quality_for_vlm)
        search_args.target_color_send_threshold = float(args.target_color_send_threshold)
        search_args.high_quality_send_threshold = float(args.high_quality_send_threshold)
        search_args.max_person_vlm_calls_per_memory = int(args.max_person_vlm_calls_per_memory)
        search_args.person_vlm_retry_interval_sec = float(args.person_vlm_retry_interval_sec)
        search_args.person_vlm_retry_min_quality_gain = float(args.person_vlm_retry_min_quality_gain)
        search_args.process_every = int(args.process_every)
        search_args.search_window_width = int(args.search_window_width)
        search_args.search_window_height = int(args.search_window_height)
        search_args.search_window_x = int(args.search_window_x)
        search_args.search_window_y = int(args.search_window_y)
        search_args.continue_after_not_front = True
        search_args.ms2_use_feature_person_prompt = True
        search_args.ms2_relaxed_color_feature_type_match = True
        search_args.out_dir = str(mission_dir / f"area_{area_index}" / str(profile.get("target_id", "target")))
        return search_args

    class AllInOnePersonSearchNode(Node):
        def __init__(self) -> None:
            super().__init__("camera_all_in_one_person_search")
            self._busy = False
            self._busy_conflict = False
            self._lock = threading.Lock()
            self._last_arrival_state = False
            self._worker_thread: Optional[threading.Thread] = None
            self._shutdown_requested = threading.Event()
            self.area_index = 0
            self.target_profiles = [dict(profile) for profile in validated_target_profiles]
            self.target_states = {
                str(profile.get("target_id", f"target_{idx + 1}")): target_state(profile)
                for idx, profile in enumerate(self.target_profiles)
            }
            self.mission_dir = Path("outputs") / f"mission_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.mission_dir.mkdir(parents=True, exist_ok=True)
            self._preview_manager = HighGuiPreviewManager(
                cv2_module=cv2,
                logger=self._preview_log,
            )
            self._frame_feed = RosFrameFeed(
                self, args.raw_image_topic,
                pipeline_diagnostics=args.pipeline_diagnostics,
                process_every=args.process_every,
            )
            self._camera_ownership = V8FrameCoordinator(
                self._frame_feed, self._shutdown_requested
            )
            self.done_pub = self.create_publisher(Bool, args.done_topic, 10)
            self.detail_pub = self.create_publisher(String, args.detail_topic, 10)
            arrival_ack_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.arrival_ack_pub = self.create_publisher(
                Bool, DEFAULT_ARRIVAL_ACK_TOPIC, arrival_ack_qos
            )
            arrival_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                Bool, args.arrival_topic, self.on_arrived, arrival_qos
            )
            self.get_logger().info(f"listening: {args.arrival_topic}")
            self.get_logger().info(f"publishing safe-to-advance completion: {args.done_topic}")
            self.get_logger().info(f"publishing detail: {args.detail_topic}")
            self.get_logger().info("Navigation owns /stop for person-search sessions")
            self.get_logger().info(
                f"publishing arrival callback ACK: {DEFAULT_ARRIVAL_ACK_TOPIC}"
            )
            self.get_logger().info(f"mission_dir: {self.mission_dir}")
            self.get_logger().info("registered targets: " + ", ".join(self.target_states.keys()))
            self.get_logger().info("when true arrives, run YOLO + LM Studio MS2 person search")
        def _preview_log(self, message: str) -> None:
            self.get_logger().info(f"[CAMERA_PREVIEW] {message}")

        def _request_shutdown_from_preview(self) -> None:
            self.request_shutdown()

        def publish_result(self, safe_to_advance: bool, detail: Dict[str, Any]) -> None:
            payload = result_detail_with_completion_contract(
                detail,
                safe_to_advance=safe_to_advance,
                terminal=True,
            )
            detail_msg = String()
            detail_msg.data = json_dumps(payload)
            self.detail_pub.publish(detail_msg)
            done = Bool()
            done.data = bool(safe_to_advance)
            self.done_pub.publish(done)

        def publish_diagnostic(self, detail: Dict[str, Any]) -> None:
            """Publish a non-terminal diagnostic without completing an active search."""
            payload = result_detail_with_completion_contract(
                detail,
                safe_to_advance=False,
                terminal=False,
            )
            detail_msg = String()
            detail_msg.data = json_dumps(payload)
            self.detail_pub.publish(detail_msg)

        def mission_status(self) -> str:
            return "MISSION_SUCCESS" if mission_success(self.target_states) else "MISSION_INCOMPLETE"

        def mission_detail(self, *, area_index: int, status: str, found_target_id: str = "", area_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "area_index": int(area_index),
                "status": status,
                "mission_status": self.mission_status(),
                "mission_success": mission_success(self.target_states),
                "found_target_id": found_target_id,
                "mission_dir": str(self.mission_dir),
                "target_A": self.target_states.get("target_A", {}),
                "target_B": self.target_states.get("target_B", {}),
                "area_results": area_results or {},
            }

        def write_mission_summary(self, detail: Dict[str, Any]) -> None:
            path = self.mission_dir / "mission_summary.json"
            for target_id, state in self.target_states.items():
                target_dir = self.mission_dir / target_id
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "target_state.json").write_text(
                    json.dumps(state, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            payload = {
                "mission_success": mission_success(self.target_states),
                "mission_status": self.mission_status(),
                "area_index": self.area_index,
                "target_A": self.target_states.get("target_A", {}),
                "target_B": self.target_states.get("target_B", {}),
                "last_detail": detail,
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        def select_area_winner(self, area_results: Dict[str, Dict[str, Any]]) -> str:
            return str(classify_area_search_results(area_results)["found_target_id"])

        def on_arrived(self, msg: Bool) -> None:
            current = bool(msg.data)
            publish_true_arrival_ack(current, self.arrival_ack_pub, Bool)
            # Edge handling, worker ownership and terminal finalization share
            # one lock. ACK remains callback receipt, not worker admission.
            with self._lock:
                if not current:
                    self._last_arrival_state = False
                    self.get_logger().info("arrival false ignored")
                    return
                if self._shutdown_requested.is_set():
                    self.get_logger().info("arrival ignored during shutdown")
                    return
                if self._last_arrival_state:
                    if self._busy:
                        self._busy_conflict = True
                        self.publish_diagnostic({
                            "status": "BUSY",
                            "area_index": self.area_index,
                            "success": False,
                            "ms1_success": False,
                            "ms2_success": False,
                            "error_message": "duplicate/new arrival received while person search is running",
                        })
                    self.get_logger().info("duplicate arrival true ignored")
                    return
                self._last_arrival_state = True
                if self._busy:
                    self._busy_conflict = True
                    self.publish_diagnostic({
                        "status": "BUSY",
                        "area_index": self.area_index,
                        "success": False,
                        "ms1_success": False,
                        "ms2_success": False,
                        "error_message": "person search is already running",
                    })
                    return
                self._busy = True
                self._busy_conflict = False
                self.area_index += 1
                area_index = self.area_index
                self._worker_thread = threading.Thread(target=self._search_worker, args=(area_index,), daemon=False)
                self._worker_thread.start()

        def request_shutdown(self) -> None:
            # A shutdown ordered before terminal finalization denies release.
            # Do not hold this lock while waiting for recorder resource cleanup.
            with self._lock:
                self._shutdown_requested.set()
            request_ms2_stop()
            self._camera_ownership.shutdown()
            self.get_logger().info("shutdown requested: MS2 search stop flag set")

        def wait_for_search_worker(self, timeout_sec: float = 8.0) -> bool:
            worker = self._worker_thread
            if worker is None or worker is threading.current_thread():
                return True
            worker.join(timeout=max(0.1, float(timeout_sec)))
            return not worker.is_alive()

        def _search_worker(self, area_index: int) -> None:
            area_results: Dict[str, Dict[str, Any]] = {}
            outcome: Optional[Dict[str, Any]] = None
            detail: Optional[Dict[str, Any]] = None
            admission_ready = False

            def worker_error_detail(exc: Exception, reason: str) -> Dict[str, Any]:
                return result_detail_with_completion_contract({
                    "status": "ERROR",
                    "success": False,
                    "person_found": False,
                    "ms1_success": False,
                    "ms2_success": False,
                    "front_photo_saved": False,
                    "area_index": int(area_index),
                    "mission_dir": str(self.mission_dir),
                    "target_A": self.target_states.get("target_A", {}),
                    "target_B": self.target_states.get("target_B", {}),
                    "error_reason": getattr(exc, "code", reason),
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }, safe_to_advance=False, terminal=True)

            try:
                if not self._camera_ownership.pause_for_search():
                    raise RuntimeError(
                        "drive recorder Camera release was not confirmed; search not started"
                    )
                admission_ready = True
                self.get_logger().info(f"person search started: area_index={area_index}")
                self.get_logger().info(
                    'v8_pipeline_configuration '
                    f'process_every={args.process_every} '
                    f'camera_fps_bookkeeping={args.camera_fps:.1f} '
                    'camera_fps_is_measured=false '
                    f'diagnostics={str(args.pipeline_diagnostics).lower()}'
                )

                def search_target(profile: Dict[str, Any]) -> Dict[str, Any]:
                    if self._shutdown_requested.is_set():
                        raise RuntimeError("person search interrupted by shutdown")
                    target_id = str(profile.get("target_id", "target"))
                    self.get_logger().info(f"area {area_index}: searching {target_id}")
                    try:
                        with V8ModelStageDiagnostics(
                            ms2_module, self._frame_feed,
                            enabled=args.pipeline_diagnostics,
                        ):
                            result = run_milestone2_search(
                                build_search_args(profile, area_index, self.mission_dir),
                                external_capture=self._camera_ownership.create_search_reader(),
                                preview_manager=self._preview_manager,
                                preview_stop_callback=self._request_shutdown_from_preview,
                            )
                    except Exception as exc:
                        result = {
                            "status": "ERROR",
                            "found": False,
                            "person_found": False,
                            "ms1_success": False,
                            "ms2_success": False,
                            "front_photo_saved": False,
                            "decision": "error",
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        self.get_logger().error(f"area {area_index}: {target_id} search error: {exc}")
                    return result

                _target_id, area_results = run_area_target_search_once(
                    self.target_profiles,
                    self.target_states,
                    area_index,
                    search_target,
                )

                if self._shutdown_requested.is_set():
                    raise RuntimeError("person search interrupted by shutdown")

                outcome = classify_area_search_results(area_results)
            except Exception as exc:
                detail = worker_error_detail(exc, "WORKER_FATAL")
            finally:
                # Preserve unconditional ownership cleanup, including worker
                # aborts. No terminal can release Robot before this finishes.
                # False is a normal no-op (e.g. recording OFF).
                try:
                    self._camera_ownership.resume_after_search()
                except Exception as exc:
                    admission_ready = False
                    detail = worker_error_detail(exc, "WORKER_CLEANUP_FAILED")

            detail_log = ""
            terminal_error = ""
            notification_status = ""
            with self._lock:
                terminal_started = False
                summary_started = False
                provisional_state = None
                try:
                    if self._shutdown_requested.is_set():
                        admission_ready = False
                        try:
                            raise RuntimeError("person search interrupted by shutdown")
                        except RuntimeError as exc:
                            detail = worker_error_detail(exc, "WORKER_SHUTDOWN")
                    if detail is None:
                        if outcome is None:
                            raise RuntimeError("person search terminal outcome is unavailable")
                        found_target_id = str(outcome["found_target_id"])
                        safe_to_advance = area_outcome_safe_to_advance(outcome)
                        busy_conflict = bool(self._busy_conflict)
                        if found_target_id and not busy_conflict:
                            result = area_results[found_target_id]
                            state = self.target_states[found_target_id]
                            provisional_state = (state, dict(state))
                            state["found"] = True
                            state["ms2_success"] = True
                            state["found_area_index"] = int(area_index)
                            state["evidence_path"] = result.get("image_path") or result.get("crop_path", "")
                            state["result"] = result
                        detail = self.mission_detail(
                            area_index=area_index,
                            status=str(outcome["status"]),
                            found_target_id=found_target_id,
                            area_results=area_results,
                        )
                        detail["success"] = bool(outcome["success"])
                        detail["error_target_ids"] = list(outcome["error_target_ids"])
                        detail["done_topic"] = args.done_topic
                        detail["detail_topic"] = args.detail_topic
                        if busy_conflict:
                            safe_to_advance = False
                            detail["busy_conflict"] = True
                            detail["status_before_busy_conflict"] = detail["status"]
                            detail["status"] = "ERROR_BUSY_CONFLICT"
                            detail["error_message"] = (
                                "arrival ambiguity detected while the worker was active; "
                                "navigation release denied"
                            )
                        detail = result_detail_with_completion_contract(
                            detail, safe_to_advance=safe_to_advance, terminal=True,
                        )
                    summary_started = True
                    self.write_mission_summary(detail)
                    detail_log = json_dumps(detail)
                    # Ready is staged under the admission lock. Next callbacks
                    # cannot observe it through their guard until detail/done
                    # publication finishes and this lock is released.
                    self._busy = not admission_ready
                    if admission_ready:
                        self._busy_conflict = False
                    terminal_started = True
                    self.publish_result(bool(detail["safe_to_advance"]), detail)
                    notification_status = str(detail.get("status", ""))
                except Exception as exc:
                    # Pre-publication failure may send one unsafe terminal.
                    # Once publication starts, partial delivery is uncertain:
                    # never retry and turn one terminal into true->false.
                    self._busy = True
                    admission_ready = False
                    terminal_error = str(exc)
                    if not terminal_started:
                        if provisional_state is not None:
                            state, previous = provisional_state
                            state.clear()
                            state.update(previous)
                        detail = worker_error_detail(exc, "WORKER_FINALIZATION_FAILED")
                        try:
                            if not summary_started:
                                # Persist an error outcome if final-state build
                                # failed before any summary write was attempted.
                                summary_started = True
                                try:
                                    self.write_mission_summary(detail)
                                except Exception as summary_exc:
                                    terminal_error += f"; error summary failed: {summary_exc}"
                            detail_log = json_dumps(detail)
                            terminal_started = True
                            self.publish_result(False, detail)
                        except Exception as publish_exc:
                            terminal_error += f"; unsafe terminal publication failed: {publish_exc}"

            # Mirror V7: play result sound ONLY after successful ROS terminal publication.
            # Audio failure must never change ROS completion or navigation admission.
            if notification_status:
                try:
                    play_person_search_notification(
                        notification_status,
                        self.get_logger(),
                    )
                except Exception as exc:
                    self.get_logger().warning(f"V8 result sound failed: {exc}")

            # Old worker has no shared-state/resource writes beyond this point.
            # Freeze log payload before admission, and never let late logging
            # failures re-enter terminal publication or clear a new busy state.
            try:
                self.get_logger().info("mission_detail:" + detail_log)
                self.get_logger().info(
                    f"area search done: area={area_index} status={detail.get('status')} "
                    f"found_target_id={detail.get('found_target_id', '')} mission={detail.get('mission_status')}"
                )
                if terminal_error:
                    self.get_logger().error(terminal_error)
                elif detail.get("error_message"):
                    self.get_logger().error(str(detail["error_message"]))
                if admission_ready:
                    self.get_logger().info("ready for next area arrival false->true")
            except Exception:
                pass

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = AllInOnePersonSearchNode()

    def shutdown_ros_runtime() -> None:
        if not node.wait_for_search_worker():
            node.get_logger().warning(
                "search worker did not finish before ROS shutdown fallback"
            )
        if not node._preview_manager.shutdown():
            node.get_logger().warning(
                "preview owner did not finish before ROS shutdown fallback"
            )
        if rclpy.ok():
            rclpy.shutdown()

    shutdown_runner = GracefulShutdownRunner(
        node.request_shutdown,
        shutdown_ros_runtime,
        logger=lambda message: node.get_logger().error(message),
    )

    def handle_shutdown(_signum, _frame) -> None:
        shutdown_runner.request()

    start_server_shutdown_signal_waiter(handle_shutdown)
    try:
        spin_node_with_signal_poll(node, rclpy, node._shutdown_requested)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            shutdown_runner.request()
            shutdown_runner.wait()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class Milestone2TargetUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("中チャレ MS2 人物探索 V8（AR映像共有）")
        # 全画面にはせず、起動時に主要UIがすべて見える大きさにする。
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        window_w = min(1220, max(1100, screen_w - 100))
        window_h = min(1000, max(900, screen_h - 60))
        self.geometry(f"{window_w}x{window_h}")
        self.minsize(960, 720)

        self.process: Optional[subprocess.Popen[str]] = None
        self.stop_requested = False
        self.stop_notice_until = 0.0
        self.stop_notice_shown = False
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.vars: Dict[str, tk.StringVar] = {}
        self.capture_notice_after_id: Optional[str] = None
        self.notice_window: Optional[tk.Toplevel] = None
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.log_window: Optional[tk.Toplevel] = None
        self.log_text: Optional[tk.Text] = None
        self.log_history: List[str] = []

        self._build_ui()
        self._load_last_config(silent=True)
        self._install_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._drain_log_queue)

    def _var(self, key: str, default: str = "") -> tk.StringVar:
        var = tk.StringVar(value=default)
        self.vars[key] = var
        return var

    def _install_shortcuts(self) -> None:
        self.bind_all("<F10>", self._shortcut_stop_process)

    def _shortcut_capture_once(self, _event: Optional[tk.Event] = None) -> str:
        self.publish_capture_request_from_ui()
        return "break"

    def _shortcut_capture_server(self, _event: Optional[tk.Event] = None) -> str:
        self.run_topic_capture_node()
        return "break"

    def _shortcut_stop_process(self, _event: Optional[tk.Event] = None) -> str:
        self.stop_process()
        return "break"

    def attach_guide(self, widget: tk.Widget, text: str) -> None:
        def enter(event: tk.Event) -> None:
            if hasattr(self, "guide_var"):
                self.guide_var.set("操作ガイド: " + text)
            self.show_tooltip(widget, text, event)

        def leave(_event: tk.Event) -> None:
            if hasattr(self, "guide_var"):
                self.guide_var.set("操作ガイド: 到着イベント待機中はカメラを直接開きません。F10で停止。")
            self.hide_tooltip()

        widget.bind("<Enter>", enter, add="+")
        widget.bind("<Leave>", leave, add="+")

    def show_tooltip(self, widget: tk.Widget, text: str, event: Optional[tk.Event] = None) -> None:
        self.hide_tooltip()
        tip = tk.Toplevel(self)
        tip.withdraw()
        tip.overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(
            tip,
            text=text,
            background="#fff8c5",
            foreground="#111",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=6,
            justify="left",
            wraplength=420,
            font=("", 10),
        )
        label.pack()
        x = widget.winfo_rootx() + 8
        y = widget.winfo_rooty() + widget.winfo_height() + 8
        if event is not None:
            x = int(event.x_root) + 12
            y = int(event.y_root) + 16
        tip.geometry(f"+{x}+{y}")
        tip.deiconify()
        self.tooltip_window = tip

    def hide_tooltip(self) -> None:
        if self.tooltip_window is not None:
            try:
                self.tooltip_window.destroy()
            except tk.TclError:
                pass
            self.tooltip_window = None

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(7, weight=1)

        title = ttk.Label(root, text="中チャレ MS2 対象入力UI", font=("", 18, "bold"))
        title.grid(row=0, column=0, sticky="w")

        quick = ttk.LabelFrame(root, text="日本語で入力")
        quick.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        quick.columnconfigure(0, weight=1)
        self.quick_text = self._var("quick_text", "白シャツ、黒ズボンの男性")
        ttk.Entry(quick, textvariable=self.quick_text, font=("", 13)).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        quick_apply = ttk.Button(quick, text="文章から反映", command=self.apply_quick_text)
        quick_apply.grid(row=0, column=1, padx=8, pady=8)
        self.attach_guide(quick_apply, "日本語の説明文をTarget A/Bの服装設定に反映します。")

        form = ttk.Frame(root)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        target = ttk.LabelFrame(form, text="探索対象 Target A / Target B")
        target.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        target.columnconfigure(0, weight=1)
        target.columnconfigure(1, weight=1)
        runtime = ttk.LabelFrame(form, text="実行設定")
        runtime.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        target_a = ttk.LabelFrame(target, text="Target A: 白紺ボーダーのマネキン")
        target_a.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        target_b = ttk.LabelFrame(target, text="Target B: チェックシャツの人間")
        target_b.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self._build_target_fields(target_a, "target_A", DEFAULT_TARGET_PROFILES["target_A"])
        self._build_target_fields(target_b, "target_B", DEFAULT_TARGET_PROFILES["target_B"])

        status = ttk.LabelFrame(target, text="走行中の状態")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.area_status_var = self._var("area_status", "現在の人物エリア: 0 / 2")
        self.target_a_status_var = self._var("target_A_status", "Target A: NOT_FOUND / found area: -")
        self.target_b_status_var = self._var("target_B_status", "Target B: NOT_FOUND / found area: -")
        ttk.Label(status, textvariable=self.area_status_var).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(status, textvariable=self.target_a_status_var).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(status, textvariable=self.target_b_status_var).grid(row=2, column=0, sticky="w", padx=8, pady=3)

        self.input_mode = self._var("input_mode", "ros")
        self._var("video", "")
        self._var("camera_index", "-1")
        self._var("camera_device", "")
        ttk.Label(runtime, text="入力: ARのC920生映像（探索中のみ）").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8, pady=5)
        self._combo(runtime, 1, "探索対象", "search_target_mode", SEARCH_TARGET_MODE_OPTIONS, "both")
        self._entry(runtime, 5, "探索時間 秒", "search_duration_sec", "20")
        self._entry(runtime, 6, "VLM最大回数", "max_total_vlm_calls", "6")
        self._combo(runtime, 7, "発見判定", "decision_accept", DECISION_OPTIONS, "candidate")
        self._entry(runtime, 8, "LM Studio URL", "lmstudio_base_url", "http://127.0.0.1:1234/v1")
        self._entry(runtime, 9, "撮影要求トピック", "capture_request_topic", "/camera_capture_request")
        self._entry(runtime, 10, "撮影完了トピック", "capture_done_topic", "/camera_capture_done")
        self._entry(runtime, 11, "撮影保存先", "capture_out_dir", "outputs/topic_camera_capture")
        self._entry(runtime, 12, "探索開始トピック", "person_arrival_topic", DEFAULT_ARRIVAL_TOPIC)
        self._entry(runtime, 13, "探索完了トピック", "person_done_topic", DEFAULT_SEARCH_DONE_TOPIC)
        self._entry(runtime, 14, "探索詳細トピック", "person_detail_topic", DEFAULT_SEARCH_DETAIL_TOPIC)

        ttk.Label(root, text="動画録画はARナビが担当。V8は人物探索時だけROS生映像を受信します。"
                  ).grid(row=3, column=0, sticky="w", pady=(10, 0))

        buttons = ttk.Frame(root)
        buttons.grid(row=4, column=0, sticky="ew", pady=(12, 8))
        person_buttons = ttk.Frame(buttons)
        person_buttons.pack(anchor="w", fill="x")
        server_btn = ttk.Button(person_buttons, text="人物探索サーバー起動", command=self.run_ros_node)
        server_btn.pack(side="left", padx=8)
        self.attach_guide(server_btn, "走行側のtrueを待つ人物探索サーバーを起動します。true後、設定した探索時間内でMS2判定します。")
        arrival_btn = ttk.Button(person_buttons, text="手動で探索開始 false→true", command=self.publish_person_arrival_from_ui)
        arrival_btn.pack(side="left")
        self.attach_guide(arrival_btn, "走行側の代わりに、このUIから/person_area_arrivedへfalse→trueを1回送ります。設定時間だけ探索します。")
        stop_btn = ttk.Button(person_buttons, text="停止", command=self.stop_process)
        stop_btn.pack(side="left", padx=8)
        self.attach_guide(stop_btn, "起動中の人物探索/写真だけサーバーを止めます。UIでもカメラ画面でもF10、ターミナル起動ならCtrl+Cです。停止はエラー扱いにしません。")
        utility_buttons = ttk.Frame(buttons)
        utility_buttons.pack(anchor="w", fill="x", pady=(6, 0))
        show_cmd_btn = ttk.Button(utility_buttons, text="コマンド表示", command=self.show_command)
        show_cmd_btn.pack(side="left", padx=8)
        self.attach_guide(show_cmd_btn, "今のUI設定で実行される人物探索コマンドをログに表示します。")
        log_btn = ttk.Button(utility_buttons, text="ログ表示", command=self.show_log_window)
        log_btn.pack(side="left")
        self.attach_guide(log_btn, "人物探索の出力・停止ログを別ウィンドウで表示します。閉じても処理は停止しません。")
        save_btn = ttk.Button(utility_buttons, text="設定保存", command=self.save_config)
        save_btn.pack(side="right")
        self.attach_guide(save_btn, "現在の対象人物・カメラ・トピック設定を保存します。")
        load_btn = ttk.Button(utility_buttons, text="前回読込", command=lambda: self._load_last_config(silent=False))
        load_btn.pack(side="right", padx=8)
        self.attach_guide(load_btn, "最後に保存/終了した設定をUIへ戻します。")
        reset_btn = ttk.Button(utility_buttons, text="探索状態リセット", command=self.reset_mission_state_ui)
        reset_btn.pack(side="right")
        self.attach_guide(reset_btn, "Target A/Bの発見済み状態とエリア番号をリセットします。")

        self.guide_var = self._var("operation_guide", "V8: 人物探索サーバーを起動し、Followerの人物エリア到着を待機します。F10で停止。")
        guide = ttk.Label(root, textvariable=self.guide_var, foreground="#005a8d", wraplength=980, justify="left")
        guide.grid(row=5, column=0, sticky="ew", pady=(0, 6))

        hint = ttk.Label(
            root,
            text=(
                "人物探索: 上の対象人物指定 + LM Studio URL を使ってMS1/MS2判定します。"
                "手動で探索開始 false→trueを押すと、このUIから探索開始トピックを飛ばし、設定時間内でMS2判定します。"
                "V8は物理カメラを開かず、探索時だけARの元映像を使用します。"
                "状態は別画面に大きく表示します。停止はボタン、UIのF10、カメラ画面のF10です。"
            ),
            foreground="#444",
            wraplength=920,
            justify="left",
        )
        hint.grid(row=6, column=0, sticky="w", pady=(0, 8))

    def _combo(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        options: List[Tuple[str, str]],
        default_value: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        labels = [label for label, _ in options]
        var = self._var(key, reverse_label(options, default_value))
        combo = ttk.Combobox(parent, textvariable=var, values=labels, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        parent.columnconfigure(1, weight=1)

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str, default: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        entry = ttk.Entry(parent, textvariable=self._var(key, default))
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        parent.columnconfigure(1, weight=1)

    def _target_key(self, target_id: str, field: str) -> str:
        return f"{target_id}_{field}"

    def _build_target_fields(self, parent: ttk.Frame, target_id: str, defaults: Dict[str, str]) -> None:
        self._combo(parent, 0, "対象種別", self._target_key(target_id, "subject_type"), SUBJECT_TYPE_OPTIONS, defaults["subject_type"])
        self._combo(parent, 1, "上メイン色", self._target_key(target_id, "target_upper_color"), COLOR_OPTIONS, defaults["target_upper_color"])
        self._combo(parent, 2, "上サブ色", self._target_key(target_id, "target_upper_secondary_color"), COLOR_OPTIONS, defaults["target_upper_secondary_color"])
        self._combo(parent, 3, "上柄", self._target_key(target_id, "target_upper_pattern"), PATTERN_OPTIONS, defaults["target_upper_pattern"])
        self._combo(parent, 4, "上種類", self._target_key(target_id, "target_upper_type"), UPPER_TYPE_OPTIONS, defaults["target_upper_type"])
        self._combo(parent, 5, "下メイン色", self._target_key(target_id, "target_lower_color"), COLOR_OPTIONS, defaults["target_lower_color"])
        self._combo(parent, 6, "下サブ色", self._target_key(target_id, "target_lower_secondary_color"), COLOR_OPTIONS, defaults["target_lower_secondary_color"])
        self._combo(parent, 7, "下柄", self._target_key(target_id, "target_lower_pattern"), PATTERN_OPTIONS, defaults["target_lower_pattern"])
        self._combo(parent, 8, "下種類", self._target_key(target_id, "target_lower_type"), LOWER_TYPE_OPTIONS, defaults["target_lower_type"])
        self._combo(parent, 9, "性別", self._target_key(target_id, "target_gender"), GENDER_OPTIONS, defaults["target_gender"])
        self._entry(parent, 10, "持ち物", self._target_key(target_id, "target_item"), defaults["target_item"])

    def value_from_combo(self, key: str, options: List[Tuple[str, str]]) -> str:
        return label_map(options).get(self.vars[key].get(), "none")

    def set_combo_value(self, key: str, options: List[Tuple[str, str]], value: str) -> None:
        self.vars[key].set(reverse_label(options, value))

    def apply_quick_text(self) -> None:
        parsed = parse_japanese_target(self.quick_text.get())
        target_id = "target_A" if parsed.get("subject_type") == "mannequin" else "target_B"
        self.apply_target_profile_to_ui(target_id, parsed)
        self.log("日本語入力を反映しました: " + json.dumps(parsed, ensure_ascii=False))

    def apply_target_profile_to_ui(self, target_id: str, profile: Dict[str, str]) -> None:
        option_map = {
            "subject_type": SUBJECT_TYPE_OPTIONS,
            "target_upper_color": COLOR_OPTIONS,
            "target_upper_secondary_color": COLOR_OPTIONS,
            "target_upper_pattern": PATTERN_OPTIONS,
            "target_upper_type": UPPER_TYPE_OPTIONS,
            "target_lower_color": COLOR_OPTIONS,
            "target_lower_secondary_color": COLOR_OPTIONS,
            "target_lower_pattern": PATTERN_OPTIONS,
            "target_lower_type": LOWER_TYPE_OPTIONS,
            "target_gender": GENDER_OPTIONS,
        }
        for field, options in option_map.items():
            if field in profile:
                self.set_combo_value(self._target_key(target_id, field), options, str(profile[field]))
        if "target_item" in profile:
            self.vars[self._target_key(target_id, "target_item")].set(str(profile["target_item"]))

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.vars["video"].set(path)
            self.input_mode.set("video")

    def check_camera_devices(self) -> None:
        self.input_mode.set("camera")
        devices = self.local_camera_devices()
        if not devices:
            self.log("カメラ確認: /dev/video* が見つかりません")
            self.show_capture_notice("カメラなし", kind="error", duration_ms=1600)
            return
        self.log("カメラ確認: 見つかったデバイス -> " + ", ".join(devices))
        try:
            import cv2
        except Exception as exc:
            self.log(f"カメラ確認: OpenCV import失敗: {exc}")
            self.show_capture_notice("カメラ確認エラー", kind="error", duration_ms=1600)
            return
        opened: List[str] = []
        for dev in devices:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            ok = bool(cap.isOpened())
            read_ok = False
            if ok:
                read_ok, _frame = cap.read()
            cap.release()
            self.log(f"カメラ確認: {dev} open={ok} read={bool(read_ok)}")
            if ok and read_ok:
                opened.append(dev)
        if opened:
            self.vars["camera_device"].set(opened[0])
            self.log(f"カメラ確認: 使用候補を camera_device に設定しました: {opened[0]}")
            self.show_capture_notice("カメラOK", kind="done", duration_ms=1200)
        else:
            self.log("カメラ確認: デバイスは見えますがOpenCVで開けません。他アプリが使用中の可能性があります。")
            self.show_capture_notice("カメラ開けない", kind="error", duration_ms=1800)

    def selected_target_ids(self) -> List[str]:
        mode = self.value_from_combo("search_target_mode", SEARCH_TARGET_MODE_OPTIONS)
        if mode == "target_A":
            return ["target_A"]
        if mode == "target_B":
            return ["target_B"]
        return ["target_A", "target_B"]

    def current_config(self) -> Dict[str, Any]:
        target_a = self.target_profile_from_ui("target_A")
        target_b = self.target_profile_from_ui("target_B")
        target_map = {"target_A": target_a, "target_B": target_b}
        selected_ids = self.selected_target_ids()
        selected_targets = [target_map[target_id] for target_id in selected_ids]
        direct_target = selected_targets[0]
        return {
            "input_mode": self.input_mode.get(),
            "video": self.vars["video"].get().strip(),
            "camera_index": self.vars["camera_index"].get().strip() or "0",
            "camera_device": self.vars["camera_device"].get().strip(),
            "search_target_mode": self.value_from_combo("search_target_mode", SEARCH_TARGET_MODE_OPTIONS),
            "all_targets": [target_a, target_b],
            "targets": selected_targets,
            "selected_target_ids": selected_ids,
            # 直接実行ボタンは選択中の先頭ターゲットを単体テストする。
            **{k: direct_target[k] for k in TARGET_FIELD_KEYS},
            "search_duration_sec": self.vars["search_duration_sec"].get().strip() or "20",
            "max_total_vlm_calls": self.vars["max_total_vlm_calls"].get().strip() or "6",
            "decision_accept": self.value_from_combo("decision_accept", DECISION_OPTIONS),
            "lmstudio_base_url": self.vars["lmstudio_base_url"].get().strip(),
            "capture_request_topic": self.vars["capture_request_topic"].get().strip() or "/camera_capture_request",
            "capture_done_topic": self.vars["capture_done_topic"].get().strip() or "/camera_capture_done",
            "capture_out_dir": self.vars["capture_out_dir"].get().strip() or "outputs/topic_camera_capture",
            "person_arrival_topic": self.vars["person_arrival_topic"].get().strip() or DEFAULT_ARRIVAL_TOPIC,
            "person_done_topic": self.vars["person_done_topic"].get().strip() or DEFAULT_SEARCH_DONE_TOPIC,
            "person_detail_topic": self.vars["person_detail_topic"].get().strip() or DEFAULT_SEARCH_DETAIL_TOPIC,
        }

    def target_profile_from_ui(self, target_id: str) -> Dict[str, str]:
        return {
            "target_id": target_id,
            "label": DEFAULT_TARGET_PROFILES.get(target_id, {}).get("label", target_id),
            "subject_type": self.value_from_combo(self._target_key(target_id, "subject_type"), SUBJECT_TYPE_OPTIONS),
            "target_upper_color": self.value_from_combo(self._target_key(target_id, "target_upper_color"), COLOR_OPTIONS),
            "target_upper_secondary_color": self.value_from_combo(self._target_key(target_id, "target_upper_secondary_color"), COLOR_OPTIONS),
            "target_upper_pattern": self.value_from_combo(self._target_key(target_id, "target_upper_pattern"), PATTERN_OPTIONS),
            "target_upper_type": self.value_from_combo(self._target_key(target_id, "target_upper_type"), UPPER_TYPE_OPTIONS),
            "target_lower_color": self.value_from_combo(self._target_key(target_id, "target_lower_color"), COLOR_OPTIONS),
            "target_lower_secondary_color": self.value_from_combo(self._target_key(target_id, "target_lower_secondary_color"), COLOR_OPTIONS),
            "target_lower_pattern": self.value_from_combo(self._target_key(target_id, "target_lower_pattern"), PATTERN_OPTIONS),
            "target_lower_type": self.value_from_combo(self._target_key(target_id, "target_lower_type"), LOWER_TYPE_OPTIONS),
            "target_gender": self.value_from_combo(self._target_key(target_id, "target_gender"), GENDER_OPTIONS),
            "target_item": self.vars[self._target_key(target_id, "target_item")].get().strip() or "unspecified",
        }

    def local_camera_devices(self) -> List[str]:
        devices = [str(path) for path in sorted(Path("/dev").glob("video*"))]
        devices.extend(str(path) for path in sorted(Path("/dev/v4l/by-id").glob("*")) if path.exists())
        return devices

    def append_input_args(self, cmd: List[str], cfg: Dict[str, Any]) -> None:
        mode = str(cfg.get("input_mode", "camera"))
        video = str(cfg.get("video", "")).strip()
        camera_device = str(cfg.get("camera_device", "")).strip()
        if mode == "video":
            if not video:
                raise ValueError("動画モードでは動画パスを指定してください。「動画を選ぶ」を押してください。")
            cmd.extend(["--video", video])
            return

        devices = self.local_camera_devices()
        if not devices:
            raise ValueError(
                "このPCでカメラデバイスが見つかりません。USB接続、権限、または他アプリがカメラを使用中でないか確認してください。"
            )
        if camera_device:
            cmd.extend(["--camera_device", camera_device])
        else:
            cmd.extend(["--camera_index", cfg["camera_index"]])

    def selected_camera_sources(self, cfg: Dict[str, Any]) -> List[Any]:
        camera_device = str(cfg.get("camera_device", "")).strip()
        sources: List[Any] = []
        if camera_device:
            sources.append(camera_device)
        else:
            try:
                sources.append(int(str(cfg.get("camera_index", "0")).strip() or "0"))
            except ValueError:
                sources.append(str(cfg.get("camera_index", "0")).strip())
        for dev in self.local_camera_devices():
            if dev not in sources:
                sources.append(dev)
        return sources

    def probe_camera_before_trigger(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        if str(cfg.get("input_mode", "camera")) != "camera":
            return True, "video mode"
        try:
            import cv2
        except Exception as exc:
            return False, f"OpenCV import失敗: {exc}"

        tried: List[str] = []
        for source in self.selected_camera_sources(cfg):
            for api_name, api in (("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)):
                label = f"{source}({api_name})"
                tried.append(label)
                cap = cv2.VideoCapture(source, api)
                opened = bool(cap.isOpened())
                read_ok = False
                if opened:
                    read_ok, _frame = cap.read()
                cap.release()
                if opened and read_ok:
                    return True, f"camera ok: {label}"
        return (
            False,
            "カメラを開けません。写真だけサーバー/カメラだけ起動/Zoom等を閉じるか、"
            f"カメラ確認で使えるデバイスを選んでください。tried={tried}",
        )

    def direct_command(self) -> List[str]:
        raise RuntimeError("V8 direct-camera use disabled; start the ROS person-search server.")

    def ros_command(self) -> List[str]:
        cfg = self.current_config()
        cmd = [
            sys.executable,
            str(THIS_FILE),
            "--person-search-server",
            "--arrival_topic",
            cfg["person_arrival_topic"],
            "--done_topic",
            cfg["person_done_topic"],
            "--detail_topic",
            cfg["person_detail_topic"],
            "--targets_json",
            json.dumps(cfg["targets"], ensure_ascii=False, separators=(",", ":")),
        ]
        self._append_common_args(cmd, cfg)
        cmd.extend(self.ms2_runtime_tuning_args())
        cmd.extend(["--raw_image_topic", "/koko/person_search/raw_image"])
        return ros_wrapped_command(cmd)

    def ms2_runtime_tuning_args(self) -> List[str]:
        # v7通常版は、誤成功を抑えつつ設定時間内でMS2判定する。
        return [
            "--vlm_min_track_frames",
            "1",
            "--camera_width",
            "640",
            "--camera_height",
            "480",
            "--camera_fps",
            "15",
            "--camera_read_fail_tolerance",
            "30",
            "--process_every",
            "5",
            "--search_window_width",
            "960",
            "--search_window_height",
            "540",
            "--search_window_x",
            "220",
            "--search_window_y",
            "120",
            "--lmstudio_timeout",
            "6",
            "--front_vlm_timeout",
            "2",
            "--person_max_tokens",
            "220",
            "--person_vlm_long_side",
            "320",
            "--person_crop_margin",
            "0.45",
            "--attribute_crop_margin",
            "0.0",
            "--use_fast_heuristic_person_match",
            "--front_fast_local_gate",
            "--front_vlm_on_fast_gate_miss",
            "--fast_heuristic_accept_score",
            "0.75",
            "--min_crop_quality_for_vlm",
            "0.35",
            "--target_color_send_threshold",
            "0.05",
            "--high_quality_send_threshold",
            "0.55",
            "--max_person_vlm_calls_per_memory",
            "2",
            "--person_vlm_retry_interval_sec",
            "2.0",
            "--person_vlm_retry_min_quality_gain",
            "0.08",
            "--require_cv_face_for_front",
            "--front_min_bbox_aspect",
            "0.28",
            "--front_min_crop_quality",
            "0.60",
            "--front_min_area_ratio",
            "0.06",
            "--front_min_height_ratio",
            "0.35",
            "--front_crop_margin",
            "0.12",
            "--front_reject_edge_crops",
            "--front_allow_vertical_edge_with_face",
            "--front_vertical_edge_face_min_quality",
            "0.60",
            "--best_photo_hold_sec",
            "2.5",
            "--max_front_vlm_calls",
            "4",
            "--front_retry_interval_sec",
            "1.0",
            "--no-fast_heuristic_allow_distinct_targets",
            "--no-accept_crop_heuristic_for_distinct_targets",
        ]

    def topic_capture_command(self) -> List[str]:
        cfg = self.current_config()
        cmd = [
            sys.executable,
            str(THIS_FILE),
            "--capture-server",
            "--request_topic",
            cfg["capture_request_topic"],
            "--done_topic",
            cfg["capture_done_topic"],
            "--out_dir",
            cfg["capture_out_dir"],
            "--prefix",
            "ui_capture",
            "--width",
            "640",
            "--height",
            "480",
            "--fps",
            "15",
            "--warmup_frames",
            "3",
            "--preview_width",
            "1440",
            "--preview_height",
            "810",
            "--preview",
            "--topmost",
        ]
        if cfg["camera_device"]:
            cmd.extend(["--camera_device", cfg["camera_device"]])
        else:
            cmd.extend(["--camera_index", cfg["camera_index"]])
        return ros_wrapped_command(cmd)

    def _append_common_args(self, cmd: List[str], cfg: Dict[str, Any]) -> None:
        for key in [
            "subject_type",
            "target_upper_color",
            "target_upper_secondary_color",
            "target_upper_pattern",
            "target_upper_type",
            "target_lower_color",
            "target_lower_secondary_color",
            "target_lower_pattern",
            "target_lower_type",
            "target_gender",
            "target_item",
            "search_duration_sec",
            "max_total_vlm_calls",
            "decision_accept",
            "lmstudio_base_url",
        ]:
            cmd.extend([f"--{key}", cfg[key]])
        cmd.extend(["--accept_front_views", "front"])
        cmd.extend(["--no-search_until_found", "--max_search_runtime_sec", cfg["search_duration_sec"]])

    def run_direct(self) -> None:
        try:
            if self.start_process(self.direct_command()):
                cfg = self.current_config()
                self.log(f"人物探索を開始しました。直接実行は {cfg['selected_target_ids'][0]} を単体探索します。")
                self.show_capture_notice("人物探索開始", kind="camera", duration_ms=900)
        except Exception as exc:
            messagebox.showerror("実行できません", str(exc))

    def run_ros_node(self) -> None:
        try:
            if self.start_process(self.ros_command()):
                cfg = self.current_config()
                self.log(f"人物探索サーバー起動中です。{cfg['person_arrival_topic']} の true を待ちます。")
                self.log("探索対象: " + ", ".join(cfg["selected_target_ids"]))
                self.log("受信したら、選択中の対象人物指定とLM Studio URLでMS1/MS2判定を実行します。")
                self.log("v7通常版: trueを受けたら、設定した探索時間内でMS2判定します。")
                self.log("VLM最大回数に達したら、それ以上はVLMへ送りません。")
                self.log("軽量設定: camera 640x480/15fps, process_every=5, VLM crop long_side=320")
                self.log("2エリア運用: 1エリア目で false→true、成功後に2エリア目でもう一度 false→true を受けると2回探索します。")
                self.log("停止: UIの停止ボタン、UI/カメラ画面のF10。ターミナルから起動した場合はCtrl+C。")
                self.show_capture_notice("人物探索待機中", kind="camera", duration_ms=900)
        except Exception as exc:
            messagebox.showerror("実行できません", str(exc))

    def run_topic_capture_node(self) -> None:
        try:
            if not self.start_process(self.topic_capture_command()):
                return
            self.log("写真だけサーバー起動中です。外部からの/camera_capture_requestを待ちます。")
            self.log("これは1枚撮影だけです。対象人物指定とLM Studioは使いません。")
            self.log("先輩の走行側が自動で飛ばしても、このUIの「写真だけ手動撮影」でも撮影できます。")
            self.log("カメラプレビューを前面表示します。撮影はUIの写真だけ手動撮影、F9、またはプレビュー上でC/Space。")
            self.log("軽量設定: camera 640x480/15fps, preview 1440x810")
            self.log("停止: UIの停止ボタン、UI/カメラ画面のF10。ターミナルから起動した場合はCtrl+C。")
            self.log(self.topic_capture_test_command_text())
            self.show_capture_notice("カメラ待機中", kind="camera", duration_ms=800)
        except Exception as exc:
            messagebox.showerror("実行できません", str(exc))

    def start_process(self, cmd: List[str]) -> bool:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("実行中", "すでに処理が動いています。停止してから再実行してください。")
            return False
        self.save_config(silent=True)
        self.stop_requested = False
        self.stop_notice_until = 0.0
        self.stop_notice_shown = False
        self.log("\n$ " + " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        threading.Thread(target=self._reader_thread, daemon=True).start()
        return True

    def _reader_thread(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line.rstrip())
        code = self.process.wait()
        self.log_queue.put(f"[終了] returncode={code}")
        if self.stop_requested or time.monotonic() < self.stop_notice_until:
            self.log_queue.put("__UI_PROCESS_STOPPED__")
        elif code != 0:
            self.log_queue.put("__UI_PROCESS_ERROR__")

    def stop_process(self) -> None:
        if self.process and self.process.poll() is None:
            proc = self.process
            self.stop_requested = True
            self.stop_notice_until = time.monotonic() + 8.0
            self.stop_notice_shown = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                self.log("[停止] F10/停止: プロセスグループへSIGTERMを送りました")
            except Exception as exc:
                self.log(f"[停止] SIGTERM失敗: {exc} / terminateへ切替")
                try:
                    proc.terminate()
                except Exception as term_exc:
                    self.log(f"[停止] terminate失敗: {term_exc}")
            threading.Thread(target=self._kill_process_if_needed, args=(proc,), daemon=True).start()
            self.show_capture_notice("停止", kind="camera", duration_ms=900)
        else:
            self.log("[停止] 実行中の処理はありません")
            self.show_capture_notice("停止", kind="camera", duration_ms=800)

    def _kill_process_if_needed(self, proc: subprocess.Popen[str]) -> None:
        try:
            proc.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            self.log_queue.put("[停止] 終了しないためSIGKILLで強制終了しました")
        except ProcessLookupError:
            pass
        except Exception as exc:
            self.log_queue.put(f"[停止] 強制終了失敗: {exc}")

    def release_camera_processes(self) -> None:
        self.log("[カメラ解放] 残っている撮影/探索サーバーを確認して停止します")
        self.show_capture_notice("カメラ解放中", kind="camera", duration_ms=900)
        threading.Thread(target=self._release_camera_processes_worker, daemon=True).start()

    def _release_camera_processes_worker(self) -> None:
        patterns = [
            "daikoudai_v7.py --capture-server",
            "daikoudai_v7.py --person-search-server",
            "daikoudai_v6.py --capture-server",
            "daikoudai_v6.py --person-search-server",
            "daikoudai_v6_until_found.py --capture-server",
            "daikoudai_v6_until_found.py --person-search-server",
            "daikoudai_v5_until_found.py --capture-server",
            "daikoudai_v5_until_found.py --person-search-server",
            "daikoudai_v5.py --capture-server",
            "daikoudai_v5.py --person-search-server",
            "daikoudai_v4.py --capture-server",
            "daikoudai_v4.py --person-search-server",
            "daikoudai_v3.py --capture-server",
            "daikoudai_v3.py --person-search-server",
            "camera_all_in_one.py --capture-server",
            "camera_all_in_one.py --person-search-server",
            "topic_camera_capture_once.py",
            "person_search_vlm_milestone2_v7.py",
            "person_search_vlm_milestone2_v6.py",
            "person_search_vlm_milestone2_v6_until_found.py",
            "person_search_vlm_milestone2_v5_until_found.py",
            "person_search_vlm_milestone2_v5.py",
            "person_search_vlm_milestone2_v4.py",
            "person_search_vlm_milestone2_v3.py",
            "person_search_vlm_milestone2_sim.py",
        ]
        killed_any = False
        for pattern in patterns:
            try:
                completed = subprocess.run(
                    ["pkill", "-TERM", "-f", pattern],
                    cwd=str(APP_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=2.0,
                )
                if completed.returncode == 0:
                    killed_any = True
                    self.log_queue.put(f"[カメラ解放] 停止: {pattern}")
                elif completed.stdout.strip():
                    self.log_queue.put(f"[カメラ解放] {pattern}: {completed.stdout.strip()}")
            except Exception as exc:
                self.log_queue.put(f"[カメラ解放] {pattern}: {exc}")
        time.sleep(0.8)
        if killed_any:
            self.log_queue.put("[カメラ解放] 完了。LEDが消えない場合はZoom/ブラウザ等も確認してください。")
            self.log_queue.put("__UI_CAMERA_RELEASE_DONE__")
        else:
            self.log_queue.put("[カメラ解放] このUI関連の残りプロセスは見つかりませんでした。")
            self.log_queue.put("__UI_CAMERA_RELEASE_NONE__")

    def on_close(self) -> None:
        self.save_config(silent=True)
        if self.process and self.process.poll() is None:
            proc = self.process
            self.stop_requested = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                self.log("[終了] UI終了前に子プロセスへSIGTERMを送りました")
                proc.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    self.log("[終了] 子プロセスをSIGKILLで終了しました")
                except Exception:
                    pass
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self.destroy()

    def show_command(self) -> None:
        try:
            cmd = self.ros_command() if messagebox.askyesno("表示するコマンド", "人物探索サーバーのコマンドを表示しますか？\nいいえを選ぶと人物探索の直接実行です。") else self.direct_command()
            self.log("\n$ " + " ".join(cmd))
        except Exception as exc:
            messagebox.showerror("コマンド作成エラー", str(exc))

    def topic_capture_test_command_text(self) -> str:
        cfg = self.current_config()
        request = {
            "area_id": "area1",
            "camera_index": int(cfg["camera_index"]),
            "camera_device": cfg["camera_device"],
            "prefix": "area1",
            "out_dir": cfg["capture_out_dir"],
        }
        data = json.dumps(request, ensure_ascii=False)
        return (
            f"ros2 topic pub --once {cfg['capture_request_topic']} std_msgs/msg/String "
            f"\"{{data: '{data}'}}\""
        )

    def topic_capture_pub_command(self) -> List[str]:
        cfg = self.current_config()
        request = {
            "area_id": "area1",
            "camera_index": int(cfg["camera_index"]),
            "camera_device": cfg["camera_device"],
            "prefix": "area1",
            "out_dir": cfg["capture_out_dir"],
        }
        data = json.dumps(request, ensure_ascii=False)
        cmd = [
            "ros2",
            "topic",
            "pub",
            "--once",
            cfg["capture_request_topic"],
            "std_msgs/msg/String",
            f"{{data: {shlex.quote(data)}}}",
        ]
        return ros_wrapped_command(cmd)

    def person_arrival_pub_command(self, value: bool = True) -> List[str]:
        cfg = self.current_config()
        cmd = [
            "ros2",
            "topic",
            "pub",
            "--once",
            "--wait-matching-subscriptions",
            "1",
            "--keep-alive",
            "1.0",
            "--qos-reliability",
            "reliable",
            "--qos-durability",
            "transient_local",
            "--qos-depth",
            "1",
            cfg["person_arrival_topic"],
            "std_msgs/msg/Bool",
            "{data: true}" if value else "{data: false}",
        ]
        return ros_wrapped_command(cmd)

    def person_arrival_pub_sequence(self) -> List[List[str]]:
        return [
            self.person_arrival_pub_command(False),
            self.person_arrival_pub_command(True),
        ]

    def publish_person_arrival_from_ui(self) -> None:
        cfg = self.current_config()
        server_running = bool(self.process and self.process.poll() is None)
        if not server_running:
            self.log("[探索開始中止] V8サーバー起動後に発火してください")
            return
        commands = self.person_arrival_pub_sequence()
        for cmd in commands:
            self.log("\n$ " + " ".join(cmd))
        self.show_capture_notice("探索開始送信", kind="camera", duration_ms=900)
        threading.Thread(target=self._publish_person_arrival_worker, args=(commands,), daemon=True).start()

    def _publish_person_arrival_worker(self, commands: List[List[str]]) -> None:
        try:
            for idx, cmd in enumerate(commands):
                completed = subprocess.run(
                    cmd,
                    cwd=str(APP_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10,
                )
                if completed.stdout:
                    for line in completed.stdout.splitlines():
                        self.log_queue.put(line)
                label = "false" if idx == 0 else "true"
                self.log_queue.put(f"[探索開始{label}送信] returncode={completed.returncode}")
                if completed.returncode != 0:
                    break
                if idx == 0:
                    time.sleep(0.3)
        except Exception as exc:
            self.log_queue.put(f"[探索開始true送信エラー] {exc}")

    def publish_capture_request_from_ui(self) -> None:
        cmd = self.topic_capture_pub_command()
        self.log("\n$ " + " ".join(cmd))
        threading.Thread(target=self._publish_capture_request_worker, args=(cmd,), daemon=True).start()

    def _publish_capture_request_worker(self, cmd: List[str]) -> None:
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            if completed.stdout:
                for line in completed.stdout.splitlines():
                    self.log_queue.put(line)
            self.log_queue.put(f"[撮影要求送信] returncode={completed.returncode}")
        except Exception as exc:
            self.log_queue.put(f"[撮影要求送信エラー] {exc}")

    def show_topic_capture_test_command(self) -> None:
        self.log("\n# 撮影要求テスト")
        self.log(self.topic_capture_test_command_text())

    def save_config(self, silent: bool = False) -> None:
        data = self.current_config()
        data["quick_text"] = self.quick_text.get()
        data["saved_at"] = dt.datetime.now().isoformat(timespec="seconds")
        LAST_CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if not silent:
            self.log(f"設定を保存しました: {LAST_CONFIG_PATH}")

    def _load_last_config(self, silent: bool = False) -> None:
        if not LAST_CONFIG_PATH.exists():
            return
        try:
            data = json.loads(LAST_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            if not silent:
                messagebox.showerror("読込エラー", str(exc))
            return
        self.input_mode.set(str(data.get("input_mode", "camera")))
        self.quick_text.set(str(data.get("quick_text", self.quick_text.get())))
        if "search_target_mode" in data and "search_target_mode" in self.vars:
            self.set_combo_value("search_target_mode", SEARCH_TARGET_MODE_OPTIONS, str(data["search_target_mode"]))
        targets = data.get("all_targets") or data.get("targets")
        if isinstance(targets, list):
            for profile in targets:
                if not isinstance(profile, dict):
                    continue
                target_id = str(profile.get("target_id", ""))
                if target_id in DEFAULT_TARGET_PROFILES:
                    self.apply_target_profile_to_ui(target_id, profile)
        for key in [
            "video",
            "camera_index",
            "camera_device",
            "search_duration_sec",
            "max_total_vlm_calls",
            "lmstudio_base_url",
            "capture_request_topic",
            "capture_done_topic",
            "capture_out_dir",
            "person_arrival_topic",
            "person_done_topic",
            "person_detail_topic",
        ]:
            if key in data and key in self.vars:
                self.vars[key].set(str(data[key]))
        for key, options in [
            ("subject_type", SUBJECT_TYPE_OPTIONS),
            ("target_upper_color", COLOR_OPTIONS),
            ("target_upper_secondary_color", COLOR_OPTIONS),
            ("target_upper_pattern", PATTERN_OPTIONS),
            ("target_upper_type", UPPER_TYPE_OPTIONS),
            ("target_lower_color", COLOR_OPTIONS),
            ("target_lower_secondary_color", COLOR_OPTIONS),
            ("target_lower_pattern", PATTERN_OPTIONS),
            ("target_lower_type", LOWER_TYPE_OPTIONS),
            ("target_gender", GENDER_OPTIONS),
            ("decision_accept", DECISION_OPTIONS),
        ]:
            if key in data and key in self.vars:
                self.set_combo_value(key, options, str(data[key]))
        if not silent:
            self.log(f"前回設定を読み込みました: {LAST_CONFIG_PATH}")

    def reset_mission_state_ui(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showwarning(
                "探索状態リセット",
                "人物探索サーバーが動いています。停止してからリセットしてください。",
            )
            return
        self.area_status_var.set("現在の人物エリア: 0 / 2")
        self.target_a_status_var.set("Target A: NOT_FOUND / found area: -")
        self.target_b_status_var.set("Target B: NOT_FOUND / found area: -")
        self.log("探索状態をリセットしました。次に人物探索サーバーを起動するとTarget A/Bを最初から扱います。")

    def update_mission_status_from_detail(self, detail: Dict[str, Any]) -> None:
        area_index = detail.get("area_index", 0)
        status = str(detail.get("status", "UNKNOWN"))
        mission_status = str(detail.get("mission_status", "MISSION_INCOMPLETE"))
        self.area_status_var.set(f"現在の人物エリア: {area_index} / 2  status: {status}  mission: {mission_status}")
        for target_id, var in [
            ("target_A", self.target_a_status_var),
            ("target_B", self.target_b_status_var),
        ]:
            state = detail.get(target_id, {})
            if not isinstance(state, dict):
                state = {}
            found = bool(state.get("ms2_success", False))
            found_area = state.get("found_area_index")
            found_area_text = "-" if found_area is None else str(found_area)
            label = DEFAULT_TARGET_PROFILES.get(target_id, {}).get("label", target_id)
            state_text = "FOUND" if found else "NOT_FOUND"
            var.set(f"{label}: {state_text} / found area: {found_area_text}")

    def log(self, text: str) -> None:
        line = str(text)
        self.log_history.append(line)
        if (
            self.log_window is not None
            and self.log_window.winfo_exists()
            and self.log_text is not None
        ):
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")

    def show_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
            return

        window = tk.Toplevel(self)
        self.log_window = window
        window.title("人物探索サーバー ログ")
        window.geometry("1000x600")
        window.minsize(700, 400)
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        text_widget = tk.Text(window, wrap="word")
        self.log_text = text_widget
        text_widget.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(window, command=text_widget.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scroll.set)
        if self.log_history:
            text_widget.insert("end", "\n".join(self.log_history) + "\n")
            text_widget.see("end")
        window.protocol("WM_DELETE_WINDOW", self.hide_log_window)

    def hide_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.withdraw()

    def show_capture_notice(self, message: str = "撮影完了", kind: str = "done", duration_ms: int = 2500) -> None:
        if self.notice_window is not None and self.notice_window.winfo_exists():
            self.notice_window.destroy()

        colors = {
            "done": ("#ffcc00", "#111111"),
            "error": ("#cc0022", "white"),
            "camera": ("#0057d8", "white"),
        }
        bg, fg = colors.get(kind, colors["done"])

        win = tk.Toplevel(self)
        self.notice_window = win
        win.title(message)
        win.configure(bg=bg)
        win.attributes("-topmost", True)
        screen_w = max(1200, int(self.winfo_screenwidth()))
        screen_h = max(700, int(self.winfo_screenheight()))
        notice_w = min(1500, max(1100, int(screen_w * 0.82)))
        notice_h = min(460, max(360, int(screen_h * 0.38)))
        notice_x = max(0, int((screen_w - notice_w) / 2))
        notice_y = max(0, int((screen_h - notice_h) / 2))
        win.geometry(f"{notice_w}x{notice_h}+{notice_x}+{notice_y}")
        font_size = 108 if len(message) <= 7 else 90

        label = tk.Label(
            win,
            text=f"⚠ {message} ⚠",
            font=("", font_size, "bold"),
            bg=bg,
            fg=fg,
            padx=40,
            pady=36,
            wraplength=notice_w - 80,
        )
        label.pack(fill="both", expand=True)

        try:
            win.lift()
            win.focus_force()
        except tk.TclError:
            pass
        self.bell()
        if self.capture_notice_after_id is not None:
            self.after_cancel(self.capture_notice_after_id)
        min_duration_ms = 700 if kind == "camera" else 1000
        duration_ms = max(int(duration_ms), min_duration_ms)
        self.capture_notice_after_id = self.after(duration_ms, self.hide_capture_notice)

    def hide_capture_notice(self) -> None:
        if self.notice_window is not None and self.notice_window.winfo_exists():
            self.notice_window.destroy()
        self.notice_window = None
        self.capture_notice_after_id = None

    def handle_log_line(self, line: str) -> None:
        stopping = self.stop_requested or time.monotonic() < self.stop_notice_until
        if line == "__UI_PROCESS_ERROR__":
            if stopping:
                if not self.stop_notice_shown:
                    self.stop_notice_shown = True
                    self.show_capture_notice("停止", kind="camera", duration_ms=900)
                return
            self.show_capture_notice("エラー", kind="error", duration_ms=3000)
            return
        if line == "__UI_PROCESS_STOPPED__":
            self.stop_notice_until = time.monotonic() + 2.0
            self.stop_requested = False
            if not self.stop_notice_shown:
                self.stop_notice_shown = True
                self.show_capture_notice("停止", kind="camera", duration_ms=900)
            return
        if line == "__UI_CAMERA_RELEASE_DONE__":
            self.show_capture_notice("カメラ解放", kind="done", duration_ms=1200)
            return
        if line == "__UI_CAMERA_RELEASE_NONE__":
            self.show_capture_notice("残りなし", kind="camera", duration_ms=1200)
            return
        self.log(line)
        if "CAMERA_OPEN_ERROR" in line:
            self.show_capture_notice("カメラ開けない", kind="error", duration_ms=1800)
            return
        if "preview camera opened:" in line:
            self.show_capture_notice("カメラ待機中", kind="camera", duration_ms=800)
        if stopping and (
            "FOUND_FRONT_PHOTO" in line
            or "FOUND_BUT_NOT_FRONT" in line
            or "NOT_FOUND" in line
            or "found:" in line
            or "ms2=" in line
            or '"ms2_success"' in line
            or "ERROR" in line
            or "Error" in line
            or "failed" in line
            or "Traceback" in line
        ):
            return
        if "mission_detail:" in line:
            try:
                detail_text = line.split("mission_detail:", 1)[1].strip()
                detail = json.loads(detail_text)
                if isinstance(detail, dict):
                    self.update_mission_status_from_detail(detail)
                    if str(detail.get("status", "")).upper() == "ERROR":
                        self.show_capture_notice("エラー", kind="error", duration_ms=1800)
            except Exception:
                pass
        if "FOUND_FRONT_PHOTO" in line or "found: True" in line or "ms2=True" in line or '"ms2_success": true' in line:
            self.show_capture_notice("MS2成功", kind="done", duration_ms=1800)
        elif "FOUND_BUT_NOT_FRONT" in line:
            self.show_capture_notice("人物発見・正面NG", kind="error", duration_ms=1800)
        elif "NOT_FOUND" in line or "found: False" in line or "ms2=False" in line or '"ms2_success": false' in line:
            self.show_capture_notice("未発見", kind="error", duration_ms=1800)
        if "saved:" in line or '"status":"SAVED"' in line or '"status": "SAVED"' in line:
            self.show_capture_notice("撮影完了", kind="done", duration_ms=1500)
        if "Traceback" in line:
            self.show_capture_notice("エラー", kind="error", duration_ms=1800)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self.handle_log_line(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--person-search-server":
        run_person_search_server_mode(sys.argv[2:])
        return
    app = Milestone2TargetUI()

    def handle_ui_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_ui_sigterm)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.on_close()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    main()
