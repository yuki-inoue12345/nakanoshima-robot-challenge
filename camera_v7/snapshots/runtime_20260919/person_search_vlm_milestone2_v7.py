#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 2 person-search simulation.

This script treats the input video as a robot camera after arriving at a
person-detection area. It searches only a short time window, stops as soon as
an accepted candidate is found, and saves evidence images/JSON for the task.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import queue
import signal
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_video_recording import (
    CameraVideoManifest,
    FailSoftVideoSegment,
    HighGuiPreviewManager,
    timestamped_video_path,
)
import person_search_vlm_v3_9_7_final_wait_reduce_v3 as base

base.ALLOWED_COLORS.update({"grey"})
base.ALLOWED_UPPER_TYPES.update({"vest", "sweatshirt", "cardigan", "polo_shirt"})
base.ALLOWED_PATTERNS.update({"solid", "striped", "plaid", "graphic", "floral", "color_block", "unknown"})

SEARCH_VERSION = "person_search_vlm_milestone2_v7"
MILESTONE = 2
CV_F10_KEYS = {65479, 0xFFC7, 16777273, 0x01000039}
STOP_REQUESTED = threading.Event()


def request_stop() -> None:
    STOP_REQUESTED.set()


def reset_stop_request() -> None:
    STOP_REQUESTED.clear()


def stop_requested() -> bool:
    return STOP_REQUESTED.is_set()


def log_search_stage(stage: str, **fields: Any) -> None:
    """Emit one concise timestamped marker around blocking startup boundaries."""
    wall_time = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(
        f"[SEARCH_STAGE] {stage} wall={wall_time} "
        f"monotonic={time.perf_counter():.3f}{suffix}",
        flush=True,
    )


def is_cv_f10_key(key: int) -> bool:
    try:
        key_int = int(key)
    except (TypeError, ValueError):
        return False
    return key_int in CV_F10_KEYS or (key_int & 0xFFFF) in {65479, 0xFFC7}

MS2_PERSON_PROMPT = """
Milestone-2 PERSON attribute JSON extractor.
Input: cropped image with one main person or mannequin-like target.
Output: compact valid JSON only. No markdown. No text.

Extract ONLY target visual attributes. Ignore bags and carried objects.
unknown=unclear. none=clearly absent. confidence=high/medium/low.
All attributes under person/head/par_attributes must be {"value":X,"confidence":C}.
visibility.body_visible is a raw string.

Schema:
{
 "person":{
  "apparent_age":{"value":"child/teen/young_adult/adult/middle_aged/senior/unknown","confidence":"high/medium/low"},
  "apparent_gender":{"value":"male/female/unknown","confidence":"high/medium/low"},
  "hair":{"value":"short/medium/long/unknown","confidence":"high/medium/low"}
 },
 "head":{
  "headwear_type":{"value":"none/cap/hat/hood/helmet/unknown","confidence":"high/medium/low"}
 },
 "par_attributes":{
  "upper_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "upper_secondary_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "upper_pattern":{"value":"solid/striped/plaid/graphic/floral/color_block/unknown","confidence":"high/medium/low"},
  "upper_type":{"value":"hoodie/jacket/coat/sweater/sweatshirt/shirt/t-shirt/t_shirt/vest/cardigan/polo_shirt/unknown","confidence":"high/medium/low"},
  "upper_type_features":{
    "has_collar":{"value":true/false/unknown,"confidence":"high/medium/low"},
    "has_front_buttons":{"value":true/false/unknown,"confidence":"high/medium/low"},
    "has_round_neck":{"value":true/false/unknown,"confidence":"high/medium/low"},
    "is_sleeveless_outer":{"value":true/false/unknown,"confidence":"high/medium/low"},
    "visible_inner_sleeves":{"value":true/false/unknown,"confidence":"high/medium/low"}
  },
  "lower_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "lower_secondary_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "lower_pattern":{"value":"solid/striped/plaid/graphic/floral/color_block/unknown","confidence":"high/medium/low"},
  "lower_type":{"value":"pants/trousers/jeans/shorts/skirt/dress/unknown","confidence":"high/medium/low"}
 },
 "visibility":{
  "body_visible":"full_body/upper_body/partial_body/face_only/unknown"
 }
}

Upper type rules:
- t_shirt = round neck or crew neck, no collar, no front opening/buttons.
- shirt = collar and/or front placket/buttons.
- sweater = knit sweater or pullover knit top.
- sweatshirt = non-knit sweat top without hood.
- vest = sleeveless outer layer worn over another top; inner sleeves may be visible.
- cardigan = front-opening knit top.
- polo_shirt = collar with short placket, casual polo.
- If a beige/tan/brown sleeveless outer layer is worn over a white T-shirt, use upper_type=vest and set is_sleeveless_outer=true, visible_inner_sleeves=true.
- Use upper_pattern/lower_pattern: solid, striped, plaid, graphic, floral, color_block, unknown.
- striped includes border/stripes/horizontal stripes. plaid includes check/grid/plaid. Do not confuse striped and plaid.
- Use secondary color only when a second clothing color is clear; otherwise unknown.
- If unsure between shirt and t_shirt, use the visible features honestly; do not guess.
""".strip()

LOWER_RESCUE_CROP_START_FRACTION = 0.20
LOWER_RESCUE_CROP_X_START_FRACTION = 0.00
LOWER_RESCUE_CROP_X_END_FRACTION = 1.00
LOWER_RESCUE_LONG_SIDE = 384
LOWER_RESCUE_FIELDS = (
    "lower_visible",
    "lower_color",
    "lower_pattern",
    "lower_type",
)
LOWER_RESCUE_ALLOWED_VALUES = {
    "lower_visible": {"full", "partial", "not_visible", "unknown"},
    "lower_color": {
        "black", "white", "gray", "dark", "light_blue", "blue", "navy",
        "beige", "tan", "khaki", "brown", "cream", "green", "olive",
        "pink", "red", "yellow", "unknown",
    },
    "lower_pattern": {
        "solid", "striped", "plaid", "graphic", "floral", "color_block",
        "unknown",
    },
    "lower_type": {"pants", "jeans", "shorts", "skirt", "dress", "unknown"},
}
LOWER_RESCUE_ALLOWED_CONFIDENCE = {"high", "medium", "low"}

LOWER_RESCUE_PROMPT = """
You are inspecting a lower-body-focused crop of a person or mannequin.
The image retains some torso context but is intended only for lower-garment
observation. Describe only the main centered subject's visible lower garment.
Do not use or guess any target identity or desired answer.

Return exactly one compact JSON object and no markdown or extra text.
Copy this object nesting exactly, replacing only each value and confidence:
{"lower_visible":{"value":"unknown","confidence":"low"},"lower_color":{"value":"unknown","confidence":"low"},"lower_pattern":{"value":"unknown","confidence":"low"},"lower_type":{"value":"unknown","confidence":"low"}}

Never simplify a field to a string. Every top-level field must contain an
object with exactly two keys: value and confidence.

Rules:
- lower_visible=full only when the lower garment is clearly visible from waist
  through most of its length; partial when useful evidence is visible but
  cropped or occluded; not_visible when it cannot be inspected.
- Report skirt only when one continuous skirt garment is visually clear.
- Report pants or jeans only from direct garment evidence, never from color.
- If skirt versus pants is ambiguous, return lower_type=unknown with low confidence.
- If multiple people make the main subject or garment ambiguous, prefer unknown.
- Do not reinterpret the visible lower garment as upper-body clothing.
- Every value must use one of the listed categories. Do not add fields.

Allowed values:
- lower_visible: full, partial, not_visible, unknown
- lower_color: black, white, gray, dark, light_blue, blue, navy, beige, tan,
  khaki, brown, cream, green, olive, pink, red, yellow, unknown
- lower_pattern: solid, striped, plaid, graphic, floral, color_block, unknown
- lower_type: pants, jeans, shorts, skirt, dress, unknown
- confidence for every field: high, medium, low
""".strip()

FRONT_PHOTO_PROMPT = """
Milestone-2 front photo checker.
Input: cropped image of one main subject. The subject may be a real person,
a mannequin, or a mannequin-like display body.
Return compact valid JSON only. No markdown. No extra text.

Goal:
Decide whether this crop can be used as a FRONT PHOTO evidence image for a
person-finding milestone.

Schema:
{
  "front_view": {"value": "front/three_quarter/side/back/unknown", "confidence": "high/medium/low"},
  "face_visible": {"value": true/false, "confidence": "high/medium/low"},
  "torso_front_visible": {"value": true/false, "confidence": "high/medium/low"},
  "photo_usable_for_ms2": {"value": true/false, "confidence": "high/medium/low"},
  "reason": "short reason"
}

Rules:
- front = the face/head front or mannequin face/front head is facing the camera,
  and the chest/front torso is mostly facing the camera.
- three_quarter = face and torso are partly turned, but this is NOT a strict front photo.
- side = person is mostly side-on.
- back = back of head/body faces the camera; face is not visible.
- If the back of the shirt/body is visible, or only a cheek/ear/profile is visible, use side/back and photo_usable_for_ms2=false.
- For mannequins/display bodies, a blank face is OK. If the mannequin head/front
  face surface and front torso are visible, set front_view=front,
  face_visible=true, torso_front_visible=true, and photo_usable_for_ms2=true.
- Use photo_usable_for_ms2=true only for a clear front-facing evidence photo.
- Use false for back, side, very small, heavily cropped, or unclear images.
""".strip()

MANNEQUIN_FULL_FRAME_PROMPT = """
Milestone-2 mannequin fallback detector.
Input: one full camera frame.
Return compact valid JSON only. No markdown. No extra text.

Goal:
Find a mannequin or mannequin-like display body that matches the requested
clothing well enough to crop it for evidence. Do not detect ordinary background
objects. If no suitable mannequin-like clothing target is visible, return found=false.

Schema:
{
  "found": true/false,
  "bbox_norm": [x1, y1, x2, y2],
  "person": {
    "apparent_age": {"value": "unknown", "confidence": "low"},
    "apparent_gender": {"value": "unknown", "confidence": "low"},
    "hair": {"value": "unknown", "confidence": "low"}
  },
  "head": {
    "headwear_type": {"value": "none/unknown", "confidence": "high/medium/low"}
  },
  "par_attributes": {
    "upper_color": {"value": "black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown", "confidence": "high/medium/low"},
    "upper_secondary_color": {"value": "black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown", "confidence": "high/medium/low"},
    "upper_pattern": {"value": "solid/striped/plaid/graphic/floral/color_block/unknown", "confidence": "high/medium/low"},
    "upper_type": {"value": "hoodie/jacket/coat/sweater/sweatshirt/shirt/t-shirt/t_shirt/vest/cardigan/polo_shirt/unknown", "confidence": "high/medium/low"},
    "lower_color": {"value": "black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown", "confidence": "high/medium/low"},
    "lower_secondary_color": {"value": "black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/cream/green/olive/pink/red/yellow/unknown", "confidence": "high/medium/low"},
    "lower_pattern": {"value": "solid/striped/plaid/graphic/floral/color_block/unknown", "confidence": "high/medium/low"},
    "lower_type": {"value": "pants/trousers/jeans/shorts/skirt/dress/unknown", "confidence": "high/medium/low"}
  },
  "visibility": {
    "body_visible": "full_body/upper_body/partial_body/unknown"
  },
  "reason": "short reason"
}

bbox_norm must be normalized image coordinates in [0,1]. Use a tight box around
the mannequin/body/clothing target. If the box is uncertain or invalid, return
found=false.
""".strip()

DEFAULT_FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
_FACE_CASCADE_CACHE: Dict[str, cv2.CascadeClassifier] = {}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def call_budget_available(count: int, limit: Any) -> bool:
    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        limit_int = 0
    return limit_int <= 0 or int(count) < limit_int


def decision_rank(decision: str) -> int:
    return {
        "strong_candidate": 3,
        "candidate": 2,
        "weak_candidate": 1,
        "no_match": 0,
        "unknown": -1,
        "not_found": -2,
    }.get(base.normalize_token(decision), -2)


def front_view_rank(view: str) -> int:
    return {
        "front": 4,
        "three_quarter": 3,
        "side": 2,
        "back": 1,
        "unknown": 0,
    }.get(base.normalize_token(view), 0)


def accepted_decision(decision: str, decision_accept: str) -> bool:
    return decision_rank(decision) >= decision_rank(decision_accept)


VLM_EVAL_OK = "OK"
VLM_EVAL_NOT_NEEDED = "NOT_NEEDED"
VLM_EVAL_OPTIONAL_FAILURE = "OPTIONAL_FAILURE"
VLM_EVAL_REQUIRED_FAILURE = "REQUIRED_FAILURE"


class SearchDeadlineReached(RuntimeError):
    """Normal camera-search deadline reached before starting a new VLM call."""


class SearchStopRequested(RuntimeError):
    """Explicit process shutdown interrupted an in-flight VLM wait."""


def remaining_search_time(deadline_mono: Optional[float]) -> Optional[float]:
    if deadline_mono is None:
        return None
    return float(deadline_mono) - time.monotonic()


def vlm_timeout_with_deadline(requested_timeout: Any, deadline_mono: Optional[float]) -> float:
    try:
        requested = float(requested_timeout)
    except (TypeError, ValueError):
        requested = 0.0
    remaining = remaining_search_time(deadline_mono)
    if remaining is not None:
        if remaining <= 0.0:
            raise SearchDeadlineReached("camera search deadline reached before VLM call")
        if requested <= 0.0:
            return remaining
        return min(requested, remaining)
    return requested


def call_lmstudio_json_with_deadline(
    *,
    deadline_mono: Optional[float],
    image_path: Path,
    prompt: str,
    base_url: str,
    model: str,
    timeout_sec: Any,
    max_tokens: int,
    kind: str,
) -> Dict[str, Any]:
    clipped_timeout = vlm_timeout_with_deadline(timeout_sec, deadline_mono)
    if stop_requested():
        raise SearchStopRequested("external stop requested before VLM call")
    completed: "queue.Queue[Tuple[bool, Any]]" = queue.Queue(maxsize=1)

    def request_worker() -> None:
        try:
            result = base.call_lmstudio_json(
                image_path=image_path,
                prompt=prompt,
                base_url=base_url,
                model=model,
                timeout_sec=clipped_timeout,
                max_tokens=max_tokens,
                kind=kind,
            )
            completed.put((True, result))
        except BaseException as exc:
            completed.put((False, exc))

    # urllib's blocking response wait cannot observe STOP_REQUESTED. Keep that
    # transport wait off the non-daemon search worker so SIGTERM can unwind the
    # search finally block and release Camera/Writer without waiting the full
    # LM request timeout. Normal calls still return exactly the same result.
    threading.Thread(
        target=request_worker,
        name=f"ms2-vlm-request-{kind}",
        daemon=True,
    ).start()
    while True:
        if stop_requested():
            raise SearchStopRequested("external stop requested during VLM call")
        try:
            succeeded, value = completed.get(timeout=0.05)
        except queue.Empty:
            continue
        if succeeded:
            return value
        raise value


def classify_vlm_evaluation(raw: Any, *, required: bool) -> str:
    if isinstance(raw, dict) and raw.get("error"):
        return VLM_EVAL_REQUIRED_FAILURE if required else VLM_EVAL_OPTIONAL_FAILURE
    return VLM_EVAL_OK


def record_vlm_evaluation(health: Dict[str, Dict[str, int]], stage: str, state: str) -> None:
    counts = health.setdefault(str(stage), {})
    counts[state] = int(counts.get(state, 0)) + 1


def unresolved_required_vlm_stages(health: Dict[str, Dict[str, int]]) -> List[str]:
    return sorted(
        stage
        for stage, counts in health.items()
        if int(counts.get(VLM_EVAL_REQUIRED_FAILURE, 0)) > 0
        and int(counts.get(VLM_EVAL_OK, 0)) == 0
    )


def candidate_required_vlm_failures(candidate: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(candidate, dict):
        return []
    evaluations = candidate.get("vlm_evaluations")
    failures = []
    if isinstance(evaluations, dict):
        failures.extend(stage for stage, state in evaluations.items() if state == VLM_EVAL_REQUIRED_FAILURE)
    front_result = candidate.get("front_result")
    if isinstance(front_result, dict) and front_result.get("vlm_evaluation") == VLM_EVAL_REQUIRED_FAILURE:
        failures.append("front")
    return sorted(set(failures))


def terminal_search_error(
    *,
    stop_reason: str,
    vlm_health: Dict[str, Dict[str, int]],
    selected_candidate: Optional[Dict[str, Any]],
    ms2_success: bool,
) -> Optional[Dict[str, Any]]:
    if stop_reason == "CAMERA_READ_FAILED":
        return {"error_code": "CAMERA_READ_FAILED", "error_message": "camera read failure tolerance exceeded"}
    selected_failures = candidate_required_vlm_failures(selected_candidate)
    if selected_failures:
        return {
            "error_code": "REQUIRED_VLM_EVALUATION_FAILED",
            "error_message": "selected candidate has failed required VLM evaluation",
            "unresolved_vlm_stages": selected_failures,
        }
    # A safely selected local candidate is a valid fallback even if an earlier
    # candidate's required VLM call failed.
    if ms2_success and isinstance(selected_candidate, dict):
        return None
    unresolved = unresolved_required_vlm_stages(vlm_health)
    if unresolved:
        error_code = (
            "MANNEQUIN_FULL_FRAME_VLM_FAILED"
            if unresolved == ["mannequin_full_frame"]
            else "REQUIRED_VLM_EVALUATION_FAILED"
        )
        return {
            "error_code": error_code,
            "error_message": "required VLM evaluation did not complete successfully",
            "unresolved_vlm_stages": unresolved,
        }
    return None


def memory_decision_after_acceptance_gates(
    decision: str,
    decision_accept: str,
    quality_ok: bool,
    reliability_ok: bool,
    operational_ok: bool,
) -> str:
    if accepted_decision(decision, decision_accept) and not (quality_ok and reliability_ok and operational_ok):
        return "unknown"
    return decision


def bool_value(obj: Any, default: bool = False) -> bool:
    if isinstance(obj, dict):
        obj = obj.get("value", default)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return base.normalize_token(obj) in {"true", "yes", "1", "ok", "usable"}
    return default


def confidence_rank(confidence: Any) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(base.normalize_token(str(confidence)), 0)


def confidence_ok(obj: Any, min_confidence: str) -> bool:
    if not min_confidence or base.normalize_token(min_confidence) in {"none", "any"}:
        return True
    conf = obj.get("confidence", "low") if isinstance(obj, dict) else "low"
    return confidence_rank(conf) >= confidence_rank(min_confidence)


def resize_exact_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    """Resize a rescue input both up and down to the tested image scale."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("lower rescue crop is empty")
    height, width = image.shape[:2]
    scale = float(long_side) / max(height, width)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )


def build_lower_rescue_crop(person_crop: np.ndarray) -> np.ndarray:
    """Build the validated y=0.20, full-width lower crop at 384px."""
    if not isinstance(person_crop, np.ndarray) or person_crop.size == 0:
        raise ValueError("person crop is empty")
    start_y = int(person_crop.shape[0] * LOWER_RESCUE_CROP_START_FRACTION)
    start_x = int(person_crop.shape[1] * LOWER_RESCUE_CROP_X_START_FRACTION)
    end_x = int(person_crop.shape[1] * LOWER_RESCUE_CROP_X_END_FRACTION)
    lower_context = person_crop[start_y:, start_x:end_x].copy()
    return resize_exact_long_side(lower_context, LOWER_RESCUE_LONG_SIDE)


def parse_compact_lower_rescue(
    raw_result: Any,
    *,
    max_tokens: int,
) -> Dict[str, Any]:
    """Strictly parse compact evidence without the full-person normalizer."""
    reasons: List[str] = []
    if not isinstance(raw_result, dict):
        return {"valid": False, "reasons": ["response_not_object"], "fields": {}}
    if raw_result.get("error"):
        return {
            "valid": False,
            "reasons": [f"request_error:{raw_result.get('error')}"],
            "fields": {},
            "raw_text": raw_result.get("raw_text", ""),
            "runtime": raw_result.get("_runtime", {}),
        }

    raw_text = raw_result.get("raw_text", "")
    usage = raw_result.get("_lmstudio_usage")
    usage = usage if isinstance(usage, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens >= int(max_tokens):
        reasons.append("completion_token_limit_reached")
    if not isinstance(raw_text, str) or not raw_text.strip():
        reasons.append("raw_text_missing")
        parsed = None
    else:
        try:
            parsed = json.loads(raw_text.strip())
        except (TypeError, ValueError) as exc:
            parsed = None
            reasons.append(f"strict_json_parse_failed:{type(exc).__name__}")

    fields: Dict[str, Dict[str, str]] = {}
    if parsed is not None:
        if not isinstance(parsed, dict):
            reasons.append("schema_root_not_object")
        else:
            missing = sorted(set(LOWER_RESCUE_FIELDS) - set(parsed))
            extra = sorted(set(parsed) - set(LOWER_RESCUE_FIELDS))
            if missing:
                reasons.append("schema_missing_fields:" + ",".join(missing))
            if extra:
                reasons.append("schema_extra_fields:" + ",".join(extra))
            for field in LOWER_RESCUE_FIELDS:
                value_confidence = parsed.get(field)
                if not isinstance(value_confidence, dict):
                    reasons.append(f"schema_field_not_object:{field}")
                    continue
                if set(value_confidence) != {"value", "confidence"}:
                    reasons.append(f"schema_field_keys_invalid:{field}")
                    continue
                value = base.normalize_token(value_confidence.get("value", "unknown"))
                confidence = base.normalize_token(value_confidence.get("confidence", "low"))
                if value not in LOWER_RESCUE_ALLOWED_VALUES[field]:
                    reasons.append(f"invalid_enum:{field}:{value}")
                    continue
                if confidence not in LOWER_RESCUE_ALLOWED_CONFIDENCE:
                    reasons.append(f"invalid_confidence:{field}:{confidence}")
                    continue
                fields[field] = {"value": value, "confidence": confidence}

    if len(fields) != len(LOWER_RESCUE_FIELDS) and not any(
        reason.startswith(("schema_", "invalid_")) for reason in reasons
    ):
        reasons.append("schema_incomplete")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "fields": fields,
        "raw_text": raw_text if isinstance(raw_text, str) else "",
        "usage": usage,
        "runtime": raw_result.get("_runtime", {}),
        "lower_type_audit_only": fields.get("lower_type"),
    }


def vc_value(obj: Any, default: Any = "unknown") -> Any:
    if isinstance(obj, dict):
        return obj.get("value", default)
    return default if obj is None else obj


def observed_value(obj: Dict[str, Any], path: List[str]) -> str:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return "unknown"
        cur = cur.get(key, "unknown")
    return base.normalize_token(vc_value(cur, "unknown"))


def color_similarity(target_color: str, observed_color: str) -> Tuple[float, str]:
    t = base.normalize_token(target_color)
    o = base.normalize_token(observed_color)
    t = "navy" if t == "dark_blue" else t
    o = "navy" if o == "dark_blue" else o
    if t == o:
        return 1.0, "exact"
    if o in {"unknown", ""}:
        return 0.20, "unknown_partial"

    groups = [
        ({"beige", "tan", "cream", "khaki", "light_brown"}, 0.95, "warm_neutral_same_family"),
        ({"beige", "tan", "cream", "khaki", "brown", "light_brown"}, 0.72, "warm_neutral_brown_family"),
        ({"gray", "grey", "light", "white", "cream"}, 0.55, "light_neutral_family"),
        ({"black", "dark", "gray", "grey"}, 0.62, "dark_gray_family"),
        ({"black", "dark", "navy"}, 0.78, "dark_family"),
        ({"blue", "navy", "light_blue"}, 0.65, "blue_family"),
        ({"green", "olive", "khaki"}, 0.70, "green_khaki_family"),
    ]
    for group, score, reason in groups:
        if t in group and o in group:
            return score, reason
    return 0.0, "mismatch"


def pattern_similarity(target_pattern: str, observed_pattern: str) -> Tuple[float, str]:
    t = base.normalize_token(target_pattern)
    o = base.normalize_token(observed_pattern)
    if t == o:
        return 1.0, "exact"
    if o in {"unknown", ""}:
        return 0.35, "pattern_unknown_partial"
    if t in {"none", "any", "unspecified"}:
        return 1.0, "target_pattern_unspecified"
    if t == "solid" and o == "unknown":
        return 0.35, "solid_unknown_partial"
    if {t, o} == {"striped", "plaid"}:
        return 0.0, "striped_plaid_mismatch"
    near_groups = [
        {"graphic", "color_block"},
        {"solid", "color_block"},
    ]
    for group in near_groups:
        if t in group and o in group:
            return 0.35, "pattern_near"
    return 0.0, "pattern_mismatch"


def feature_bool(features: Dict[str, Any], key: str) -> Optional[bool]:
    raw = features.get(key)
    if isinstance(raw, bool):
        return raw
    if not isinstance(raw, dict):
        return None
    if not confidence_ok(raw, "medium"):
        return None
    value = raw.get("value", "unknown")
    if isinstance(value, bool):
        return value
    token = base.normalize_token(value)
    if token in {"true", "yes", "1"}:
        return True
    if token in {"false", "no", "0"}:
        return False
    return None


def infer_upper_type_from_features(person_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    par = person_result.get("par_attributes") if isinstance(person_result.get("par_attributes"), dict) else {}
    features = par.get("upper_type_features") if isinstance(par.get("upper_type_features"), dict) else {}
    if not features:
        return "unknown", {"status": "NO_FEATURES"}

    has_collar = feature_bool(features, "has_collar")
    has_buttons = feature_bool(features, "has_front_buttons")
    has_round_neck = feature_bool(features, "has_round_neck")
    sleeveless_outer = feature_bool(features, "is_sleeveless_outer")
    inner_sleeves = feature_bool(features, "visible_inner_sleeves")

    if sleeveless_outer is True and inner_sleeves is True:
        inferred = "vest"
        reason = "sleeveless_outer_with_inner_sleeves"
    elif has_collar is True or has_buttons is True:
        inferred = "shirt"
        reason = "collar_or_front_buttons"
    elif has_round_neck is True and has_collar is False and has_buttons is False:
        inferred = "t_shirt"
        reason = "round_neck_no_collar_no_buttons"
    else:
        inferred = "unknown"
        reason = "features_inconclusive"

    return inferred, {
        "status": "OK",
        "reason": reason,
        "has_collar": has_collar,
        "has_front_buttons": has_buttons,
        "has_round_neck": has_round_neck,
        "is_sleeveless_outer": sleeveless_outer,
        "visible_inner_sleeves": inner_sleeves,
    }


def upper_type_similarity(target_type: str, observed_type: str, person_result: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    t = base.normalize_token(target_type)
    o = base.normalize_token(observed_type)
    inferred, detail = infer_upper_type_from_features(person_result)
    detail["observed_upper_type"] = o
    detail["inferred_upper_type"] = inferred

    if t == o:
        return 1.0, "exact", detail
    if t == inferred:
        return 0.90, "upper_type_feature_match", detail
    if o in {"unknown", ""} and inferred == "unknown":
        return 0.20, "unknown_partial", detail

    if (
        {t, o} <= {"sweater", "sweatshirt", "t_shirt"}
        and detail.get("has_round_neck") is True
        and detail.get("has_collar") is False
        and detail.get("has_front_buttons") is False
        and detail.get("is_sleeveless_outer") is False
    ):
        # Compatible pullover structure is not evidence of knit material.
        return 0.72, "pullover_structure_compatible_material_unconfirmed", detail

    if {t, o} <= {"t_shirt", "shirt"}:
        if inferred == "t_shirt" and t == "shirt":
            return 0.30, "shirt_tshirt_feature_contradiction", detail
        if inferred == "shirt" and t == "t_shirt":
            return 0.30, "shirt_tshirt_feature_contradiction", detail
        return 0.55, "shirt_tshirt_near_without_features", detail

    if t == "vest":
        if o in {"shirt", "t_shirt", "sweater", "unknown"}:
            return 0.45, "vest_possible_layer_but_not_confirmed", detail
    if o == "vest" and t in {"shirt", "t_shirt"}:
        return 0.45, "observed_vest_target_inner_top", detail

    if {t, o} <= {"sweater", "sweatshirt", "cardigan"}:
        return 0.58, "knit_sweat_near", detail
    if t == "polo_shirt" and o in {"shirt", "t_shirt"}:
        return 0.55, "polo_near", detail
    if o == "polo_shirt" and t in {"shirt", "t_shirt"}:
        return 0.55, "polo_near", detail

    if t == "hoodie" and o in {"shirt", "t_shirt", "sweater", "sweatshirt", "jacket"}:
        return 0.45, "upper_type_near", detail
    if t in {"shirt", "t_shirt"} and o == "hoodie":
        return 0.45, "upper_type_near", detail

    return 0.0, "mismatch", detail


def ms2_field_score(target_value: str, observed: Dict[str, Any], field: str, person_result: Dict[str, Any]) -> Tuple[float, str, str, Dict[str, Any]]:
    field_paths = {
        "upper_color": ["par_attributes", "upper_color"],
        "upper_secondary_color": ["par_attributes", "upper_secondary_color"],
        "upper_pattern": ["par_attributes", "upper_pattern"],
        "upper_type": ["par_attributes", "upper_type"],
        "lower_color": ["par_attributes", "lower_color"],
        "lower_secondary_color": ["par_attributes", "lower_secondary_color"],
        "lower_pattern": ["par_attributes", "lower_pattern"],
        "lower_type": ["par_attributes", "lower_type"],
        "apparent_gender": ["person", "apparent_gender"],
        "hair": ["person", "hair"],
        "headwear_type": ["head", "headwear_type"],
    }
    observed_token = observed_value(observed, field_paths[field])

    if field.endswith("color"):
        score, reason = color_similarity(target_value, observed_token)
        return score, reason, observed_token, {}
    if field.endswith("pattern"):
        score, reason = pattern_similarity(target_value, observed_token)
        return score, reason, observed_token, {}
    if field == "upper_type":
        score, reason, detail = upper_type_similarity(target_value, observed_token, person_result)
        return score, reason, observed_token, detail
    if field == "apparent_gender" and observed_token in {"unknown", ""}:
        return 0.35, "gender_unknown_ms2_soft", observed_token, {}

    score, reason = base.partial_match_score(target_value, observed_token, field)
    return score, reason, observed_token, {}


def score_person_ms2(target: Dict[str, Any], person_result: Dict[str, Any]) -> Dict[str, Any]:
    subject_type = base.normalize_token(target.get("subject_type", "auto") or "auto")
    if subject_type == "mannequin":
        field_weights = {
            "upper_color": 0.20,
            "upper_secondary_color": 0.10,
            "upper_pattern": 0.30,
            "upper_type": 0.18,
            "lower_color": 0.11,
            "lower_secondary_color": 0.02,
            "lower_pattern": 0.02,
            "lower_type": 0.07,
        }
    else:
        field_weights = {
            "upper_color": 0.26,
            "upper_secondary_color": 0.10,
            "upper_pattern": 0.16,
            "upper_type": 0.20,
            "apparent_gender": 0.10,
            "lower_color": 0.07,
            "lower_secondary_color": 0.02,
            "lower_pattern": 0.02,
            "lower_type": 0.04,
            "hair": 0.02,
            "headwear_type": 0.01,
        }
    matched: List[str] = []
    mismatch: List[Dict[str, Any]] = []
    unknown: List[str] = []
    partial: List[Dict[str, Any]] = []
    feature_notes: Dict[str, Any] = {}
    breakdown: List[Dict[str, Any]] = []
    weight_total = 0.0
    score_sum = 0.0
    specified_count = 0

    for field, weight in field_weights.items():
        tv = target.get(field)
        if tv is None or base.normalize_token(tv) in {"", "none", "unknown", "unspecified", "any"}:
            continue
        if subject_type == "mannequin" and field in {"apparent_gender", "hair", "headwear_type"}:
            continue
        specified_count += 1
        weight_total += float(weight)
        score, reason, ov, detail = ms2_field_score(tv, person_result, field, person_result)
        section = "par_attributes" if field.startswith(("upper_", "lower_")) else ("head" if field == "headwear_type" else "person")
        attribute = (person_result.get(section) or {}).get(field, {})
        confidence = attribute.get("confidence", "low") if isinstance(attribute, dict) else "low"
        if ov not in {"", "unknown"} and not confidence_ok(attribute, "medium"):
            score *= 0.5
        score_sum += float(weight) * score
        if detail:
            feature_notes[field] = detail
        row = {
            "field": field,
            "target": base.normalize_token(tv),
            "observed": ov,
            "score": round(score, 3),
            "reason": reason,
            "confidence": confidence,
            "weight": weight,
            "weighted_score": round(weight * score, 5),
            "evidence_state": "unknown" if ov in {"", "unknown"} else ("contradiction" if score == 0 else "observed"),
        }
        breakdown.append(row)
        if ov in {"unknown", ""}:
            unknown.append(field)
            partial.append(row)
        elif score >= 0.999:
            matched.append(field)
        elif score > 0:
            partial.append(row)
        else:
            mismatch.append({"field": field, "target": base.normalize_token(tv), "observed": ov, "reason": reason})

    if specified_count == 0 or weight_total <= 0:
        return {
            "score": 0.0,
            "specified_count": specified_count,
            "matched": matched,
            "mismatch": mismatch,
            "unknown": unknown,
            "partial": partial,
            "field_weights": {},
            "status": "NO_PERSON_TARGET_FIELDS",
        }

    return {
        "score": round(score_sum / weight_total, 3),
        "score_breakdown": breakdown,
        "weight_total": weight_total,
        "evidence_provenance": person_evidence_provenance(person_result),
        "specified_count": specified_count,
        "matched": matched,
        "mismatch": mismatch,
        "unknown": unknown,
        "partial": partial,
        "field_weights": {k: v for k, v in field_weights.items() if target.get(k) is not None},
        "upper_type_feature_notes": feature_notes.get("upper_type", {}),
        "subject_type": subject_type,
        "status": "OK_MS2_RELAXED_COLOR_FEATURE_TYPE",
    }


def lower_rescue_filtered_rejections(
    target: Dict[str, Any],
    person_result: Dict[str, Any],
    lower_rescue_gate: Optional[Dict[str, Any]],
) -> List[str]:
    reasons = distinct_evidence_rejections(target, person_result)
    try:
        rescue_score = float((lower_rescue_gate or {}).get("provisional_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        rescue_score = 0.0
    gate_allows_missing_type = bool(
        isinstance(lower_rescue_gate, dict)
        and lower_rescue_gate.get("accepted") is True
        and lower_rescue_gate.get("source") == "rescue_vlm_lower_focus"
        and lower_rescue_gate.get("lower_type_policy")
        == "audit_only_never_merge_never_accept"
        and lower_rescue_gate.get("reasons") == []
        and (lower_rescue_gate.get("semantic_policy") or {}).get("eligible") is True
        and lower_rescue_gate.get("remaining_evidence_gate_reasons_before_override")
        == ["lower_evidence_missing:lower_type"]
        and rescue_score >= 0.85
    )
    if gate_allows_missing_type:
        reasons = [
            reason
            for reason in reasons
            if reason != "lower_evidence_missing:lower_type"
        ]
    return reasons


def build_match_json_ms2(
    target: Dict[str, Any],
    person_result: Dict[str, Any],
    items_result: Optional[Dict[str, Any]],
    *,
    lower_rescue_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_match = base.build_match_json(target, person_result, items_result)
    person = score_person_ms2(target, person_result)
    item = base_match.get("item_match", base.score_item(target, items_result))
    person_score = float(person["score"])
    item_score = item.get("score") if isinstance(item, dict) else None
    final_score = person_score if item_score is None else person_score * 0.70 + float(item_score) * 0.30
    final_score = round(final_score, 3)
    if final_score >= 0.85:
        decision = "strong_candidate"
    elif final_score >= 0.75:
        decision = "candidate"
    elif final_score >= 0.50:
        decision = "weak_candidate"
    else:
        decision = "no_match"

    evidence_gate_reasons = lower_rescue_filtered_rejections(
        target, person_result, lower_rescue_gate
    )
    if evidence_gate_reasons and decision in {"strong_candidate", "candidate"}:
        decision = "weak_candidate"

    match = {
        "person_score": person_score,
        "item_score": item_score,
        "final_score": final_score,
        "decision": decision,
        "evidence_gate_reasons": evidence_gate_reasons,
        "person_match": person,
        "item_match": item,
        "matched": person.get("matched", []) + item.get("matched", []),
        "mismatch": person.get("mismatch", []) + item.get("mismatch", []),
        "unknown": person.get("unknown", []) + item.get("unknown", []),
        "partial": person.get("partial", []) + item.get("partial", []),
        "weights": {
            "person": 1.0 if item_score is None else 0.70,
            "item": 0.0 if item_score is None else 0.30,
        },
        "base_match_before_ms2_adjust": base_match,
        "ms2_match_version": "robust_color_pattern_feature_v7",
    }
    if lower_rescue_gate is not None:
        match["lower_rescue_gate"] = copy.deepcopy(lower_rescue_gate)
    return match


def upper_body_color_hint_ms2(frame: np.ndarray, bbox: Tuple[int, int, int, int], target_color: str) -> Tuple[float, Dict[str, Any]]:
    target_color = base.normalize_token(target_color)
    if target_color in {"", "none", "unknown", "unspecified"}:
        return 0.0, {"status": "NO_TARGET_COLOR", "target_color": target_color}

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = base.clamp_bbox(bbox, w, h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 8 or bh < 16:
        return 0.0, {"status": "CROP_TOO_SMALL", "target_color": target_color}

    uy1 = y1 + int(bh * 0.18)
    uy2 = y1 + int(bh * 0.66)
    ux1 = x1 + int(bw * 0.08)
    ux2 = x2 - int(bw * 0.08)
    ux1, uy1, ux2, uy2 = base.clamp_bbox((ux1, uy1, ux2, uy2), w, h)
    crop = frame[uy1:uy2, ux1:ux2]
    if crop.size == 0:
        return 0.0, {"status": "EMPTY_CROP", "target_color": target_color}

    score, detail = target_color_presence_score(crop, target_color)
    detail = dict(detail)
    detail["status"] = "OK_MS2_GENERAL_COLOR_HINT"
    return round(float(score), 3), detail


def clamp_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_face_cascade(path: str) -> Optional[cv2.CascadeClassifier]:
    if not path:
        return None
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    if path not in _FACE_CASCADE_CACHE:
        try:
            cascade = cv2.CascadeClassifier(path)
        except Exception:
            return None
        if cascade.empty():
            return None
        _FACE_CASCADE_CACHE[path] = cascade
    return _FACE_CASCADE_CACHE.get(path)


def detect_frontal_faces(image: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    cascade = load_face_cascade(str(getattr(args, "front_face_cascade_path", DEFAULT_FACE_CASCADE_PATH)))
    if cascade is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "enabled": bool(getattr(args, "require_cv_face_for_front", True)),
            "available": False,
            "count": 0,
            "boxes": [],
            "max_area_ratio": 0.0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    min_size = max(8, int(getattr(args, "front_face_min_size_px", 18)))
    try:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=float(getattr(args, "front_face_scale_factor", 1.05)),
            minNeighbors=int(getattr(args, "front_face_min_neighbors", 3)),
            minSize=(min_size, min_size),
        )
    except Exception as exc:
        return {
            "enabled": bool(getattr(args, "require_cv_face_for_front", True)),
            "available": False,
            "count": 0,
            "boxes": [],
            "max_area_ratio": 0.0,
            "error": str(exc),
        }
    h, w = image.shape[:2]
    boxes = [[int(x), int(y), int(fw), int(fh)] for x, y, fw, fh in faces]
    max_area = max((fw * fh for _x, _y, fw, fh in boxes), default=0)
    return {
        "enabled": bool(getattr(args, "require_cv_face_for_front", True)),
        "available": True,
        "count": len(boxes),
        "boxes": boxes[:5],
        "max_area_ratio": round(max_area / float(max(1, w * h)), 5),
    }


def detect_front_like_skin_region(image: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {"enabled": bool(getattr(args, "front_fast_local_gate", True)), "accepted": False, "reason": "empty_image"}

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return {"enabled": bool(getattr(args, "front_fast_local_gate", True)), "accepted": False, "reason": "invalid_image"}

    y2 = max(1, int(h * 0.52))
    x1 = max(0, int(w * 0.08))
    x2 = min(w, int(w * 0.92))
    roi = image[:y2, x1:x2]
    if roi.size == 0:
        return {"enabled": bool(getattr(args, "front_fast_local_gate", True)), "accepted": False, "reason": "empty_roi"}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    # Haar顔検出は顔が少し切れると外しやすいので、MS2実機用に軽い肌色ゲートも使う。
    skin_mask = (
        (((hue <= 25) | (hue >= 160)) & (sat >= 30) & (sat <= 190) & (val >= 60))
        & (cr >= 135)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 140)
        & (y >= 55)
    )
    mask = skin_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)

    min_area_ratio = float(getattr(args, "front_fast_min_skin_area_ratio", 0.018) or 0.018)
    min_width_ratio = float(getattr(args, "front_fast_min_skin_width_ratio", 0.12) or 0.12)
    min_height_ratio = float(getattr(args, "front_fast_min_skin_height_ratio", 0.10) or 0.10)
    best: Optional[Dict[str, Any]] = None
    for idx in range(1, count):
        bx, by, bw, bh, area = [int(v) for v in stats[idx]]
        if area <= 0:
            continue
        area_ratio = area / float(max(1, w * h))
        width_ratio = bw / float(max(1, w))
        height_ratio = bh / float(max(1, h))
        cx_roi, cy_roi = centers[idx]
        cx = float(cx_roi + x1)
        cy = float(cy_roi)
        row = {
            "box": [int(bx + x1), int(by), int(bw), int(bh)],
            "area_ratio": round(area_ratio, 4),
            "width_ratio": round(width_ratio, 4),
            "height_ratio": round(height_ratio, 4),
            "center": [round(cx / float(max(1, w)), 3), round(cy / float(max(1, h)), 3)],
        }
        if best is None or area_ratio > float(best.get("area_ratio", 0.0) or 0.0):
            best = row

    if best is None:
        return {
            "enabled": bool(getattr(args, "front_fast_local_gate", True)),
            "accepted": False,
            "reason": "no_skin_component",
            "skin_pixel_ratio": round(float(np.count_nonzero(mask)) / float(max(1, mask.size)), 4),
        }

    center_x, center_y = best["center"]
    accepted = (
        float(best["area_ratio"]) >= min_area_ratio
        and float(best["width_ratio"]) >= min_width_ratio
        and float(best["height_ratio"]) >= min_height_ratio
        and 0.10 <= float(center_x) <= 0.90
        and float(center_y) <= 0.42
    )
    reasons: List[str] = []
    if float(best["area_ratio"]) < min_area_ratio:
        reasons.append("skin_area_small")
    if float(best["width_ratio"]) < min_width_ratio:
        reasons.append("skin_width_small")
    if float(best["height_ratio"]) < min_height_ratio:
        reasons.append("skin_height_small")
    if not (0.10 <= float(center_x) <= 0.90):
        reasons.append("skin_center_x_outside")
    if float(center_y) > 0.42:
        reasons.append("skin_center_too_low")
    return {
        "enabled": bool(getattr(args, "front_fast_local_gate", True)),
        "accepted": accepted,
        "reason": "front_like_skin_region" if accepted else ",".join(reasons),
        "best_component": best,
        "thresholds": {
            "min_area_ratio": min_area_ratio,
            "min_width_ratio": min_width_ratio,
            "min_height_ratio": min_height_ratio,
        },
        "skin_pixel_ratio": round(float(np.count_nonzero(mask)) / float(max(1, mask.size)), 4),
    }


def _bbox_tuple(raw_bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in raw_bbox[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou_value(a: Any, b: Any) -> float:
    box_a = _bbox_tuple(a)
    box_b = _bbox_tuple(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / float(max(1, union))


def analyze_front_crop_people(
    target_bbox: Any,
    crop_box: Any,
    frame_detections: Optional[List[Dict[str, Any]]],
    *,
    target_track_id: int = -1,
    target_track_id_is_real: bool = False,
) -> Dict[str, Any]:
    target = _bbox_tuple(target_bbox)
    crop = _bbox_tuple(crop_box)
    if target is None or crop is None:
        return {"accepted": False, "reasons": ["front_crop_geometry_missing"], "other_people": []}
    cx1, cy1, cx2, cy2 = crop
    other_people = []
    for det in frame_detections or []:
        other = _bbox_tuple(det.get("bbox") if isinstance(det, dict) else None)
        if other is None:
            continue
        other_track_id = int(det.get("track_id", -1) or -1)
        other_track_is_real = bool(det.get("track_id_is_real", False))
        if target_track_id_is_real and other_track_is_real and other_track_id == target_track_id:
            continue
        if bbox_iou_value(target, other) >= 0.65:
            continue
        ox1, oy1, ox2, oy2 = other
        ix = max(0, min(cx2, ox2) - max(cx1, ox1))
        iy = max(0, min(cy2, oy2) - max(cy1, oy1))
        overlap_ratio = (ix * iy) / float(max(1, (ox2 - ox1) * (oy2 - oy1)))
        center_x = (ox1 + ox2) / 2.0
        center_y = (oy1 + oy2) / 2.0
        center_inside = cx1 <= center_x <= cx2 and cy1 <= center_y <= cy2
        if center_inside or overlap_ratio >= 0.50:
            other_people.append({
                "bbox": list(other),
                "track_id": other_track_id,
                "center_inside_front_crop": center_inside,
                "overlap_ratio": round(overlap_ratio, 3),
            })
    return {
        "accepted": not other_people,
        "reasons": [] if not other_people else ["multiple_people_in_front_crop"],
        "other_people": other_people,
        "target_bbox": list(target),
        "front_crop_box": list(crop),
    }


def face_target_correspondence(
    face_check: Dict[str, Any],
    candidate: Dict[str, Any],
    subject_type: str,
) -> Dict[str, Any]:
    if subject_type != "person":
        return {"accepted": True, "reason": "face_target_mapping_not_required_for_mannequin_or_auto"}
    if not bool(face_check.get("available", False)):
        return {"accepted": False, "reason": "cv_face_detector_unavailable"}
    count = int(face_check.get("count", 0) or 0)
    if count != 1:
        return {
            "accepted": False,
            "reason": "multiple_faces_in_front_crop" if count > 1 else "cv_frontal_face_not_detected",
            "face_count": count,
        }
    boxes = face_check.get("boxes") if isinstance(face_check.get("boxes"), list) else []
    target_local = _bbox_tuple(candidate.get("target_bbox_in_front_crop"))
    if not boxes or target_local is None:
        return {"accepted": False, "reason": "target_head_region_unavailable", "face_count": count}
    face = boxes[0]
    if not isinstance(face, (list, tuple)) or len(face) < 4:
        return {"accepted": False, "reason": "invalid_face_box", "face_count": count}
    fx, fy, fw, fh = (float(v) for v in face[:4])
    face_center = (fx + fw / 2.0, fy + fh / 2.0)
    tx1, ty1, tx2, ty2 = target_local
    head_bottom = ty1 + (ty2 - ty1) * 0.45
    accepted = tx1 <= face_center[0] <= tx2 and ty1 <= face_center[1] <= head_bottom
    return {
        "accepted": accepted,
        "reason": "face_matches_target_head_region" if accepted else "face_outside_target_head_region",
        "face_count": count,
        "face_center": [round(face_center[0], 2), round(face_center[1], 2)],
        "target_head_region": [tx1, ty1, tx2, round(head_bottom, 2)],
    }


def build_fast_front_photo_result(candidate: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if not bool(getattr(args, "front_fast_local_gate", True)):
        return None
    crop_image = cv2.imread(str(candidate.get("candidate_crop_path", "")))
    if not isinstance(crop_image, np.ndarray):
        return None
    face_check = detect_frontal_faces(crop_image, args)
    skin_check = detect_front_like_skin_region(crop_image, args)
    has_cv_face = bool(face_check.get("available")) and int(face_check.get("count", 0) or 0) > 0
    has_skin_face = bool(skin_check.get("accepted", False))
    if not has_cv_face and not has_skin_face:
        return None

    source = "cv_frontal_face" if has_cv_face else "skin_front_like_region"
    # A local face/skin cue does not establish torso orientation. Keep it as
    # auxiliary evidence, but never manufacture a strict MS2 front success.
    result = {
        "front_view": {"value": "unknown", "confidence": "medium"},
        "face_visible": {"value": True, "confidence": "medium"},
        "torso_front_visible": {"value": False, "confidence": "low"},
        "photo_usable_for_ms2": {"value": False, "confidence": "high"},
        "reason": f"local face cue found by {source}; torso orientation requires front VLM",
        "requires_front_vlm": True,
        "fast_front_gate": {
            "source": source,
            "cv_frontal_face": face_check,
            "skin_region": skin_check,
        },
        "_runtime": {
            "kind": "front_photo_fast_local",
            "used_vlm": False,
            "elapsed_sec": 0.0,
            "model": "local_cv",
            "image_path": str(candidate.get("candidate_crop_path")),
        },
    }
    result["_runtime"].update({
        "frame": candidate.get("frame"),
        "time_sec": candidate.get("time_sec"),
        "track_id": candidate.get("track_id"),
        "memory_id": candidate.get("memory_id"),
        "crop_path": str(candidate.get("candidate_crop_path")),
        "milestone": MILESTONE,
    })
    return apply_front_photo_gate(result, candidate, args)


def candidate_crop_quality(candidate: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    quality: Optional[float] = None
    if isinstance(candidate.get("crop_quality_detail"), dict):
        detail = dict(candidate.get("crop_quality_detail") or {})
    if candidate.get("crop_quality_score") is not None:
        try:
            quality = float(candidate.get("crop_quality_score"))
        except (TypeError, ValueError):
            quality = None

    person_result = candidate.get("person_result") if isinstance(candidate.get("person_result"), dict) else {}
    runtime = person_result.get("_runtime") if isinstance(person_result.get("_runtime"), dict) else {}
    if not detail and isinstance(runtime.get("crop_quality_detail"), dict):
        detail = dict(runtime.get("crop_quality_detail") or {})
    if quality is None and runtime.get("crop_quality_score") is not None:
        try:
            quality = float(runtime.get("crop_quality_score"))
        except (TypeError, ValueError):
            quality = None
    return quality, detail


def mannequin_front_fallback_detail(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    view: str,
    crop_quality: Optional[float],
    crop_detail: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(getattr(args, "enable_mannequin_front_fallback", True)):
        return {"accepted": False, "reason": "disabled"}
    if base.normalize_token(getattr(args, "subject_type", "auto") or "auto") != "mannequin":
        return {"accepted": False, "reason": "not_mannequin"}
    if view != "front":
        return {"accepted": False, "reason": f"strict_front_not_confirmed:{view}"}

    match = candidate.get("match") if isinstance(candidate.get("match"), dict) else {}
    try:
        final_score = float(match.get("final_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        final_score = 0.0

    person_result = candidate.get("person_result") if isinstance(candidate.get("person_result"), dict) else {}
    visibility = person_result.get("visibility") if isinstance(person_result.get("visibility"), dict) else {}
    body_visible = base.normalize_token(visibility.get("body_visible", "unknown"))
    area_ratio = float(crop_detail.get("area_ratio", 0.0) or 0.0)
    edge_hits = int(crop_detail.get("edge_hits", 0) or 0)
    near_side = bool(crop_detail.get("near_left", False)) or bool(crop_detail.get("near_right", False))

    min_score = float(getattr(args, "mannequin_front_fallback_min_score", 0.85) or 0.85)
    min_quality = float(getattr(args, "mannequin_front_fallback_min_quality", 0.70) or 0.70)
    min_area = float(getattr(args, "mannequin_front_fallback_min_area_ratio", 0.06) or 0.06)
    reasons: List[str] = []
    if final_score < min_score:
        reasons.append(f"score_low:{final_score:.3f}")
    if crop_quality is not None and crop_quality < min_quality:
        reasons.append(f"crop_quality_low:{crop_quality:.3f}")
    if area_ratio < min_area:
        reasons.append(f"area_small:{area_ratio:.4f}")
    if near_side:
        reasons.append("near_side_edge")
    if edge_hits >= 2:
        reasons.append(f"edge_hits:{edge_hits}")
    if body_visible not in {"full_body", "upper_body", "unknown"}:
        reasons.append(f"body_visible_not_enough:{body_visible}")

    return {
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "view_before_fallback": view,
        "final_score": round(final_score, 3),
        "crop_quality_score": round(crop_quality, 3) if crop_quality is not None else None,
        "area_ratio": round(area_ratio, 5),
        "body_visible": body_visible,
        "thresholds": {
            "min_score": min_score,
            "min_quality": min_quality,
            "min_area_ratio": min_area,
        },
    }


def accepted_candidate_quality_ok(candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    quality, detail = candidate_crop_quality(candidate)
    min_quality = float(getattr(args, "accepted_candidate_min_quality", 0.45) or 0.45)
    min_area = float(getattr(args, "accepted_candidate_min_area_ratio", 0.012) or 0.012)
    area = float(detail.get("area_ratio", 0.0) or 0.0)
    reasons: List[str] = []
    if quality is not None and quality < min_quality:
        reasons.append(f"accepted_crop_quality_low:{quality:.3f}")
    if area < min_area:
        reasons.append(f"accepted_crop_area_small:{area:.4f}")
    return len(reasons) == 0, reasons


def target_has_distinct_visual_cue(target: Dict[str, Any]) -> bool:
    unspecified = {"", "none", "unknown", "unspecified", "any"}
    upper_pattern = base.normalize_token(target.get("upper_pattern", "none"))
    lower_pattern = base.normalize_token(target.get("lower_pattern", "none"))
    upper_secondary = base.normalize_token(target.get("upper_secondary_color", "none"))
    lower_secondary = base.normalize_token(target.get("lower_secondary_color", "none"))
    if upper_pattern in {"striped", "plaid", "color_block", "graphic", "floral"}:
        return True
    if lower_pattern in {"striped", "plaid", "color_block", "graphic", "floral"}:
        return True
    if upper_secondary not in unspecified or lower_secondary not in unspecified:
        return True

    visual_fields = (
        "upper_color",
        "upper_type",
        "upper_pattern",
        "lower_color",
        "lower_type",
        "lower_pattern",
        "apparent_gender",
        "hair",
        "headwear_type",
    )
    specified = [
        field
        for field in visual_fields
        if base.normalize_token(target.get(field, "none")) not in unspecified
    ]
    upper_specified = any(field.startswith("upper_") for field in specified)
    lower_specified = any(field.startswith("lower_") for field in specified)
    return (upper_specified and lower_specified) or len(specified) >= 2


def candidate_uses_crop_color_heuristic(candidate: Dict[str, Any]) -> bool:
    provenance = candidate.get("person_evidence_provenance")
    if isinstance(provenance, dict) and bool(provenance.get("heuristic_only", False)):
        return True
    if str(candidate.get("person_match_source", "")) in {"fast_heuristic", "crop_heuristic"}:
        return True
    person_result = candidate.get("person_result")
    if not isinstance(person_result, dict):
        return False
    return person_evidence_provenance(person_result)["heuristic_only"]


def person_evidence_provenance(person_result: Dict[str, Any]) -> Dict[str, Any]:
    fallback = person_result.get("_fallback")
    fallback = fallback if isinstance(fallback, dict) else {}
    declared = person_result.get("_evidence_provenance")
    declared = declared if isinstance(declared, dict) else {}
    heuristic = (
        fallback.get("type") == "ms2_crop_color_heuristic"
        or bool(declared.get("heuristic_only"))
    )
    return {
        "source": "local_image_heuristic" if heuristic else declared.get("source", "person_vlm"),
        "heuristic_only": heuristic,
    }


def distinct_evidence_rejections(target: Dict[str, Any], person_result: Dict[str, Any]) -> List[str]:
    """An aggregate score cannot override missing identity evidence."""
    if not target_has_distinct_visual_cue(target):
        return []
    reasons = []
    if person_evidence_provenance(person_result)["heuristic_only"]:
        reasons.append("heuristic_only_distinct_target")
    par = person_result.get("par_attributes") or {}
    for field in ("upper_color", "upper_secondary_color", "upper_pattern", "upper_type", "lower_color", "lower_pattern", "lower_type"):
        tv = target.get(field)
        if tv is None or base.normalize_token(tv) in {"", "none", "unknown", "unspecified", "any"}:
            continue
        attr = par.get(field, {})
        value = base.normalize_token(vc_value(attr))
        score, _, _, _ = ms2_field_score(tv, person_result, field, person_result)
        if value not in {"", "unknown"} and score == 0 and confidence_ok(attr, "high"):
            reasons.append("high_confidence_contradiction:" + field)
        if field.endswith("pattern") and base.normalize_token(tv) in {"striped", "plaid", "graphic", "floral", "color_block"}:
            if score < 1.0 or not confidence_ok(attr, "high"):
                reasons.append("distinct_pattern_unconfirmed:" + field)
        if field in {"lower_color", "lower_type"} and (
            value in {"", "unknown"} or not confidence_ok(attr, "medium")
        ):
            reasons.append("lower_evidence_missing:" + field)
    return reasons


def distinct_target_rescue_policy(candidate: Dict[str, Any], target: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Eligibility for one normal-budget retry, never acceptance itself."""
    person = candidate.get("person_result") or {}
    par = person.get("par_attributes") or {}
    reasons = distinct_evidence_rejections(target, person)
    quality, geometry = candidate_crop_quality(candidate)
    pattern = par.get("upper_pattern", {})
    pattern_match = (
        base.normalize_token(target.get("upper_pattern", "")) in {"striped", "plaid"}
        and pattern_similarity(target["upper_pattern"], vc_value(pattern))[0] == 1.0
        and confidence_ok(pattern, "high")
    )
    upper_score, _, _, _ = ms2_field_score(target.get("upper_type", "unknown"), person, "upper_type", person)
    color_score, _ = color_similarity(target.get("upper_color", "unknown"), vc_value(par.get("upper_color")))
    unspecified = {"", "none", "unknown", "unspecified", "any"}
    solid_upper_match = bool(
        base.normalize_token(target.get("upper_pattern", "")) == "solid"
        and all(
            base.normalize_token(target.get(field, "none")) not in unspecified
            for field in ("upper_color", "upper_type", "lower_color", "lower_pattern")
        )
        and base.normalize_token(target.get("upper_secondary_color", "none")) in unspecified
        and pattern_similarity("solid", vc_value(pattern))[0] == 1.0
        and confidence_ok(pattern, "high")
        and upper_score >= 0.70
        and confidence_ok(par.get("upper_type", {}), "high")
        and color_score >= 0.95
        and confidence_ok(par.get("upper_color", {}), "medium")
    )
    unknown_lower = any(
        target.get(field) not in (None, "", "none", "unknown")
        and observed_value(person, ["par_attributes", field]) in {"", "unknown"}
        for field in ("lower_color", "lower_type")
    )
    eligible = bool(
        getattr(args, "enable_distinct_target_rescue", True)
        and target_has_distinct_visual_cue(target)
        and not candidate_uses_crop_color_heuristic(candidate)
        and not candidate_required_vlm_failures(candidate)
        and (pattern_match or solid_upper_match) and upper_score >= 0.70 and color_score >= 0.65
        and confidence_ok(par.get("upper_color", {}), "medium")
        and unknown_lower
        and not any(r.startswith("high_confidence_contradiction:") for r in reasons)
        and quality is not None and quality >= max(0.50, float(getattr(args, "accepted_candidate_min_quality", 0.45)))
        and float(geometry.get("area_ratio", 0)) >= 0.06
        and float(geometry.get("height_ratio", 0)) >= 0.35
        and observed_value(person, ["visibility", "body_visible"]) not in {"face_only", "upper_body"}
    )
    return {
        "eligible": eligible,
        "reason": (
            "lower_unknown_with_strong_direct_solid_upper"
            if eligible and solid_upper_match
            else "lower_unknown_with_direct_pattern_and_compatible_structure"
            if eligible else "insufficient_direct_evidence_or_geometry"
        ),
        "rescue_branch": "solid_direct_upper" if solid_upper_match else "distinct_pattern",
        "evidence_gate_reasons": reasons,
        "next_action": "retry_compact_lower_context_vlm_within_existing_budget" if eligible else "normal_search",
    }


def build_distinct_target_rescue_context(
    candidate: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Carry only the trusted primary evidence needed by the one-shot retry."""
    return {
        "eligible": bool(policy.get("eligible", False)),
        "policy": copy.deepcopy(policy),
        "primary_candidate": {
            "person_result": copy.deepcopy(candidate.get("person_result") or {}),
            "items_result": copy.deepcopy(candidate.get("items_result")),
            "match": copy.deepcopy(candidate.get("match") or {}),
            "crop_quality_score": candidate.get("crop_quality_score"),
            "crop_quality_detail": copy.deepcopy(candidate.get("crop_quality_detail") or {}),
            "person_evidence_provenance": copy.deepcopy(
                candidate.get("person_evidence_provenance") or {}
            ),
            "vlm_evaluations": copy.deepcopy(candidate.get("vlm_evaluations") or {}),
        },
    }


def integrate_compact_lower_rescue(
    *,
    primary_candidate: Dict[str, Any],
    raw_rescue_result: Any,
    target: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Merge validated color/pattern evidence while keeping direct VLM primary."""
    primary_person = copy.deepcopy(primary_candidate.get("person_result") or {})
    audit = parse_compact_lower_rescue(
        raw_rescue_result,
        max_tokens=int(getattr(args, "person_max_tokens", 260) or 260),
    )
    recomputed_policy = distinct_target_rescue_policy(primary_candidate, target, args)
    support_reasons: List[str] = []
    if not recomputed_policy.get("eligible"):
        support_reasons.append("semantic_rescue_policy_not_eligible")
    if candidate_uses_crop_color_heuristic(primary_candidate):
        support_reasons.append("primary_evidence_is_heuristic_only")
    if not audit.get("valid"):
        support_reasons.extend(
            f"compact_response_invalid:{reason}" for reason in audit.get("reasons", [])
        )

    fields = audit.get("fields") if isinstance(audit.get("fields"), dict) else {}
    visible = fields.get("lower_visible", {})
    lower_color = fields.get("lower_color", {})
    lower_pattern = fields.get("lower_pattern", {})
    color_support_candidate = "none"
    if audit.get("valid"):
        if (
            base.normalize_token(vc_value(visible)) not in {"full", "partial"}
            or not confidence_ok(visible, "medium")
        ):
            support_reasons.append("compact_lower_not_visible_enough")
        color_score, color_reason = color_similarity(
            target.get("lower_color", "unknown"), vc_value(lower_color)
        )
        pattern_score, pattern_reason = pattern_similarity(
            target.get("lower_pattern", "unknown"), vc_value(lower_pattern)
        )
        strong_color = color_score >= 0.95 and confidence_ok(lower_color, "high")
        soft_near_color = bool(
            0.60 <= color_score < 0.95
            and confidence_ok(lower_color, "medium")
            and pattern_score == 1.0
            and confidence_ok(lower_pattern, "high")
            and recomputed_policy.get("eligible")
            and not any(
                reason.startswith("high_confidence_contradiction:")
                for reason in recomputed_policy.get("evidence_gate_reasons", [])
            )
            and base.normalize_token(vc_value(
                (primary_person.get("par_attributes") or {}).get("lower_color")
            )) in {"", "unknown"}
        )
        if strong_color:
            color_support_candidate = "strong_color"
        elif soft_near_color:
            color_support_candidate = "soft_near_color"
        else:
            support_reasons.append(
                f"compact_lower_color_not_strong:{color_reason}:{color_score:.3f}"
            )
        if pattern_score < 1.0 or not confidence_ok(lower_pattern, "medium"):
            support_reasons.append(
                f"compact_lower_pattern_not_confirmed:{pattern_reason}:{pattern_score:.3f}"
            )

    direct_par = primary_person.get("par_attributes")
    direct_par = direct_par if isinstance(direct_par, dict) else {}
    direct_lower_type = direct_par.get("lower_type", {})
    direct_lower_type_value = base.normalize_token(vc_value(direct_lower_type))
    direct_lower_type_score, _, _, _ = ms2_field_score(
        target.get("lower_type", "unknown"),
        primary_person,
        "lower_type",
        primary_person,
    )
    if (
        direct_lower_type_value not in {"", "unknown"}
        and direct_lower_type_score == 0
        and confidence_ok(direct_lower_type, "high")
    ):
        support_reasons.append("direct_lower_type_high_confidence_contradiction")

    if color_support_candidate == "soft_near_color" and not support_reasons:
        trial_person = copy.deepcopy(primary_person)
        trial_par = trial_person.setdefault("par_attributes", {})
        for field in ("lower_color", "lower_pattern"):
            if base.normalize_token(vc_value(trial_par.get(field))) in {"", "unknown"}:
                trial_par[field] = copy.deepcopy(fields[field])
        trial_match = build_match_json_ms2(
            target, trial_person, primary_candidate.get("items_result")
        )
        trial_reasons = list(trial_match.get("evidence_gate_reasons", []))
        if trial_reasons != ["lower_evidence_missing:lower_type"]:
            support_reasons.append(
                "soft_near_color_remaining_evidence_failures:"
                + ",".join(trial_reasons or ["none"])
            )
        trial_score = float(trial_match.get("final_score", 0.0) or 0.0)
        if trial_score < 0.85:
            support_reasons.append(
                f"soft_near_color_provisional_score_below_threshold:{trial_score:.3f}"
            )

    merged_fields: List[str] = []
    corroborated_fields: List[str] = []
    support_ok = not support_reasons
    if support_ok:
        merged_par = primary_person.setdefault("par_attributes", {})
        for field in ("lower_color", "lower_pattern"):
            rescue_value = copy.deepcopy(fields[field])
            direct_value = base.normalize_token(vc_value(merged_par.get(field)))
            if direct_value in {"", "unknown"}:
                merged_par[field] = rescue_value
                merged_fields.append(field)
            else:
                corroborated_fields.append(field)

    provisional_match = build_match_json_ms2(target, primary_person, primary_candidate.get("items_result"))
    remaining_reasons = list(provisional_match.get("evidence_gate_reasons", []))
    gate_reasons = list(support_reasons)
    if support_ok and remaining_reasons != ["lower_evidence_missing:lower_type"]:
        gate_reasons.append(
            "remaining_evidence_failures_not_lower_type_only:"
            + ",".join(remaining_reasons or ["none"])
        )
    provisional_score = float(provisional_match.get("final_score", 0.0) or 0.0)
    if support_ok and provisional_score < 0.85:
        gate_reasons.append(f"merged_score_below_strong_threshold:{provisional_score:.3f}")
    if support_ok and direct_lower_type_value not in {"", "unknown"}:
        gate_reasons.append("direct_lower_type_is_not_missing")

    gate = {
        "accepted": len(gate_reasons) == 0,
        "source": "rescue_vlm_lower_focus",
        "reasons": gate_reasons,
        "semantic_policy": recomputed_policy,
        "merged_fields": merged_fields,
        "corroborated_fields": corroborated_fields,
        "remaining_evidence_gate_reasons_before_override": remaining_reasons,
        "provisional_score": round(provisional_score, 3),
        "lower_type_policy": "audit_only_never_merge_never_accept",
    }
    audit.update(
        {
            "source": "rescue_vlm_lower_focus",
            "crop_start_fraction": LOWER_RESCUE_CROP_START_FRACTION,
            "x_start_fraction": LOWER_RESCUE_CROP_X_START_FRACTION,
            "x_end_fraction": LOWER_RESCUE_CROP_X_END_FRACTION,
            "long_side": LOWER_RESCUE_LONG_SIDE,
            "color_support_path": color_support_candidate if support_ok else "none",
            "support_ok": support_ok,
            "support_reasons": support_reasons,
            "merged_fields": merged_fields,
            "corroborated_fields": corroborated_fields,
            "lower_type_used_for_acceptance": False,
            "acceptance_gate": copy.deepcopy(gate),
        }
    )

    if support_ok:
        provenance = primary_person.get("_evidence_provenance")
        provenance = copy.deepcopy(provenance) if isinstance(provenance, dict) else {}
        provenance.setdefault("source", "person_vlm")
        provenance["heuristic_only"] = False
        field_sources = provenance.get("field_sources")
        field_sources = copy.deepcopy(field_sources) if isinstance(field_sources, dict) else {}
        for field in merged_fields:
            field_sources[field] = "rescue_vlm_lower_focus"
        provenance["field_sources"] = field_sources
        provenance["lower_rescue"] = {
            "source": "rescue_vlm_lower_focus",
            "valid": bool(audit.get("valid")),
            "support_ok": support_ok,
            "merged_fields": merged_fields,
            "corroborated_fields": corroborated_fields,
            "lower_type_used_for_acceptance": False,
        }
        primary_person["_evidence_provenance"] = provenance
        primary_person["_lower_rescue_audit"] = audit
    return primary_person, gate if gate["accepted"] else None, audit


def accepted_candidate_reliability_ok(
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if (
        target_has_distinct_visual_cue(target)
        and candidate_uses_crop_color_heuristic(candidate)
    ):
        reasons.append("crop_color_heuristic_not_enough_for_distinct_target")
    lower_rescue_gate = candidate.get("lower_rescue_gate")
    reasons.extend(candidate.get("match", {}).get("evidence_gate_reasons", []))
    person = candidate.get("person_result")
    if isinstance(person, dict) and person:
        # Keep the identity gate even when the legacy matcher is selected.
        reasons.extend(
            reason
            for reason in lower_rescue_filtered_rejections(
                target, person, lower_rescue_gate
            )
            if reason not in reasons
        )
    return len(reasons) == 0, reasons


def ms2_photo_quality_score(candidate: Dict[str, Any]) -> float:
    front = candidate.get("front_result") if isinstance(candidate.get("front_result"), dict) else {}
    gate = front.get("ms2_front_gate") if isinstance(front.get("ms2_front_gate"), dict) else {}
    crop_quality, crop_detail = candidate_crop_quality(candidate)
    if crop_quality is None and gate.get("crop_quality_score") is not None:
        try:
            crop_quality = float(gate.get("crop_quality_score"))
        except (TypeError, ValueError):
            crop_quality = None
    if not crop_detail and isinstance(gate.get("crop_quality_detail"), dict):
        crop_detail = dict(gate.get("crop_quality_detail") or {})

    face = gate.get("cv_frontal_face") if isinstance(gate.get("cv_frontal_face"), dict) else {}
    face_count = min(1.0, float(face.get("count", 0) or 0))
    face_area = min(1.0, float(face.get("max_area_ratio", 0.0) or 0.0) / 0.08)
    center_score = float(crop_detail.get("center_score", 0.0) or 0.0)
    width_score = float(crop_detail.get("width_score", 0.0) or 0.0)
    height_score = float(crop_detail.get("height_score", 0.0) or 0.0)
    area_score = float(crop_detail.get("area_score", 0.0) or 0.0)
    edge_hits = min(3.0, float(crop_detail.get("edge_hits", 0) or 0))
    near_side = bool(crop_detail.get("near_left", False)) or bool(crop_detail.get("near_right", False))

    quality = float(crop_quality or 0.0)
    score = (
        quality * 0.38
        + face_count * 0.18
        + face_area * 0.16
        + center_score * 0.12
        + width_score * 0.06
        + height_score * 0.05
        + area_score * 0.05
        - edge_hits * 0.08
        - (0.18 if near_side else 0.0)
    )
    return round(max(0.0, min(1.0, score)), 3)


def build_front_photo_unknown_result(candidate: Dict[str, Any], args: argparse.Namespace, reason: str) -> Dict[str, Any]:
    result = {
        "front_view": {"value": "unknown", "confidence": "medium"},
        "face_visible": {"value": False, "confidence": "medium"},
        "torso_front_visible": {"value": False, "confidence": "medium"},
        "photo_usable_for_ms2": {"value": False, "confidence": "high"},
        "reason": reason,
        "_runtime": {
            "kind": "front_photo_fast_local",
            "used_vlm": False,
            "elapsed_sec": 0.0,
            "model": "local_cv",
            "image_path": str(candidate.get("candidate_crop_path")),
            "frame": candidate.get("frame"),
            "time_sec": candidate.get("time_sec"),
            "track_id": candidate.get("track_id"),
            "memory_id": candidate.get("memory_id"),
            "crop_path": str(candidate.get("candidate_crop_path")),
            "milestone": MILESTONE,
        },
    }
    return apply_front_photo_gate(result, candidate, args)


def apply_front_photo_gate(
    front_result: Dict[str, Any],
    candidate: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Apply deterministic MS2 evidence rules after the VLM front-view answer."""
    if not getattr(args, "check_front_view", True):
        front_result["ms2_front_gate"] = {
            "accepted": True,
            "reasons": ["front_view_check_disabled"],
            "strict": False,
        }
        return front_result

    accepted_views = {
        base.normalize_token(v)
        for v in str(getattr(args, "accept_front_views", "front")).split(",")
        if str(v).strip()
    }
    view_obj = front_result.get("front_view") if isinstance(front_result.get("front_view"), dict) else {}
    face_obj = front_result.get("face_visible") if isinstance(front_result.get("face_visible"), dict) else {}
    torso_obj = front_result.get("torso_front_visible") if isinstance(front_result.get("torso_front_visible"), dict) else {}
    usable_obj = front_result.get("photo_usable_for_ms2") if isinstance(front_result.get("photo_usable_for_ms2"), dict) else {}

    view = base.normalize_token(view_obj.get("value", "unknown"))
    raw_usable = bool_value(usable_obj, False)
    reasons: List[str] = []
    subject_type = base.normalize_token(getattr(args, "subject_type", "auto") or "auto")
    is_mannequin = subject_type == "mannequin"

    if front_result.get("vlm_evaluation") == VLM_EVAL_REQUIRED_FAILURE:
        reasons.append("front_vlm_required_failure")
    if view not in accepted_views:
        reasons.append(f"view_not_accepted:{view}")
    if not raw_usable:
        reasons.append("vlm_photo_usable_false")
    if not confidence_ok(view_obj, str(getattr(args, "front_min_confidence", "medium"))):
        reasons.append("front_view_confidence_low")

    if getattr(args, "strict_ms2_front", True):
        if view != "front":
            reasons.append("strict_requires_front")
        if not is_mannequin and not bool_value(face_obj, False):
            reasons.append("face_not_visible")
        if not bool_value(torso_obj, False):
            reasons.append("torso_front_not_visible")
        if not is_mannequin and not confidence_ok(face_obj, str(getattr(args, "front_min_confidence", "medium"))):
            reasons.append("face_confidence_low")
        if not confidence_ok(torso_obj, str(getattr(args, "front_min_confidence", "medium"))):
            reasons.append("torso_confidence_low")

    raw_bbox = candidate.get("bbox", (0, 0, 0, 0))
    bbox = tuple(raw_bbox) if isinstance(raw_bbox, (list, tuple)) else (0, 0, 0, 0)
    bw = max(1, clamp_int(bbox[2]) - clamp_int(bbox[0])) if len(bbox) >= 4 else 1
    bh = max(1, clamp_int(bbox[3]) - clamp_int(bbox[1])) if len(bbox) >= 4 else 1
    bbox_aspect = bw / float(bh)
    min_aspect = float(getattr(args, "front_min_bbox_aspect", 0.0) or 0.0)
    if min_aspect > 0.0 and bbox_aspect < min_aspect:
        reasons.append(f"bbox_too_narrow:{bbox_aspect:.3f}")

    crop_quality, crop_quality_detail = candidate_crop_quality(candidate)
    min_crop_quality = float(getattr(args, "front_min_crop_quality", 0.0) or 0.0)
    if crop_quality is not None and min_crop_quality > 0.0 and crop_quality < min_crop_quality:
        reasons.append(f"front_crop_quality_low:{crop_quality:.3f}")
    crop_area_ratio = float(crop_quality_detail.get("area_ratio", 0.0) or 0.0)
    min_area_ratio = float(getattr(args, "front_min_area_ratio", 0.0) or 0.0)
    if min_area_ratio > 0.0 and crop_area_ratio < min_area_ratio:
        reasons.append(f"front_area_too_small:{crop_area_ratio:.4f}")
    crop_height_ratio = float(crop_quality_detail.get("height_ratio", 0.0) or 0.0)
    min_height_ratio = float(getattr(args, "front_min_height_ratio", 0.0) or 0.0)
    if min_height_ratio > 0.0 and crop_height_ratio < min_height_ratio:
        reasons.append(f"front_height_too_small:{crop_height_ratio:.3f}")
    near_left = bool(crop_quality_detail.get("near_left", False))
    near_right = bool(crop_quality_detail.get("near_right", False))
    near_top = bool(crop_quality_detail.get("near_top", False))
    near_bottom = bool(crop_quality_detail.get("near_bottom", False))
    edge_hits = int(crop_quality_detail.get("edge_hits", 0) or 0)
    edge_names = [
        name
        for name, flag in (
            ("left", near_left),
            ("right", near_right),
            ("top", near_top),
            ("bottom", near_bottom),
        )
        if flag
    ]
    if bool(getattr(args, "front_reject_edge_crops", True)):
        if near_left or near_right:
            reasons.append("bbox_near_side_edge")
        if edge_hits >= 2:
            reasons.append("bbox_multiple_edges:" + ",".join(edge_names))

    multi_person_check = candidate.get("front_multi_person_check")
    if isinstance(multi_person_check, dict) and not bool(multi_person_check.get("accepted", False)):
        reasons.extend(str(reason) for reason in multi_person_check.get("reasons", []) if str(reason))

    crop_image = cv2.imread(str(candidate.get("candidate_crop_path", "")))
    face_check = detect_frontal_faces(crop_image, args) if isinstance(crop_image, np.ndarray) else {
        "enabled": bool(getattr(args, "require_cv_face_for_front", True)),
        "available": False,
        "count": 0,
        "boxes": [],
        "max_area_ratio": 0.0,
    }
    # MS2 target may be a real person, mannequin, or still unknown at UI time.
    # Require OpenCV face only when the target is explicitly a person; for auto
    # targets, a front-like local gate can be enough to keep field tests moving.
    require_cv_face = bool(getattr(args, "require_cv_face_for_front", True)) and subject_type == "person"
    fast_gate = front_result.get("fast_front_gate") if isinstance(front_result.get("fast_front_gate"), dict) else {}
    skin_gate = fast_gate.get("skin_region") if isinstance(fast_gate.get("skin_region"), dict) else {}
    skin_gate_accepted = bool(skin_gate.get("accepted", False))
    cv_face_unavailable_fallback_used = False
    cv_face_missing_fallback_used = False
    if require_cv_face:
        if not face_check.get("available"):
            if skin_gate_accepted and bool_value(face_obj, False):
                cv_face_unavailable_fallback_used = True
            else:
                reasons.append("cv_face_detector_unavailable")
        elif int(face_check.get("count", 0) or 0) <= 0:
            allow_missing_face_fallback = (
                bool(getattr(args, "allow_skin_front_when_cv_face_missing", True))
                and skin_gate_accepted
                and bool_value(face_obj, False)
                and bool_value(torso_obj, False)
                and crop_quality is not None
                and crop_quality >= float(getattr(args, "skin_front_cv_face_missing_min_quality", 0.55) or 0.55)
            )
            if allow_missing_face_fallback:
                cv_face_missing_fallback_used = True
            else:
                reasons.append("cv_frontal_face_not_detected")

    face_correspondence = face_target_correspondence(face_check, candidate, subject_type)
    if subject_type == "person" and not bool(face_correspondence.get("accepted", False)):
        reason = str(face_correspondence.get("reason", "face_target_correspondence_failed"))
        if reason not in reasons:
            reasons.append(reason)

    vertical_edge_with_face_fallback_used = False
    if bool(getattr(args, "front_allow_vertical_edge_with_face", True)):
        has_cv_face = bool(face_check.get("available")) and int(face_check.get("count", 0) or 0) > 0
        vertical_edges_only = bool(edge_names) and all(name in {"top", "bottom"} for name in edge_names)
        min_vertical_edge_quality = float(getattr(args, "front_vertical_edge_face_min_quality", 0.60) or 0.60)
        if (
            has_cv_face
            and vertical_edges_only
            and not near_left
            and not near_right
            and view == "front"
            and raw_usable
            and crop_quality is not None
            and crop_quality >= min_vertical_edge_quality
        ):
            filtered_reasons: List[str] = []
            for reason in reasons:
                if reason.startswith("bbox_multiple_edges:"):
                    reason_edges = {
                        name.strip()
                        for name in reason.split(":", 1)[1].split(",")
                        if name.strip()
                    }
                    if reason_edges and reason_edges.issubset({"top", "bottom"}):
                        vertical_edge_with_face_fallback_used = True
                        continue
                filtered_reasons.append(reason)
            reasons = filtered_reasons

    mannequin_front_fallback = mannequin_front_fallback_detail(candidate, args, view, crop_quality, crop_quality_detail)
    mannequin_front_fallback["applied"] = False

    usable_confidence = base.normalize_token(usable_obj.get("confidence", "medium"))
    multi_person_accepted = (
        isinstance(multi_person_check, dict)
        and bool(multi_person_check.get("accepted", False))
    )
    no_crop_edge_issue = (
        edge_hits == 0
        and not near_left
        and not near_right
        and not near_top
        and not near_bottom
    )
    strong_mannequin_rescue_conditions = (
        is_mannequin
        and bool(getattr(args, "strict_ms2_front", True))
        and bool(mannequin_front_fallback.get("accepted", False))
        and front_result.get("vlm_evaluation") == VLM_EVAL_OK
        and view == "front"
        and bool_value(torso_obj, False)
        and crop_quality is not None
        and no_crop_edge_issue
        and multi_person_accepted
    )
    mannequin_front_fallback["rescue_conditions"] = {
        "strong_conditions_met": strong_mannequin_rescue_conditions,
        "front_vlm_ok": front_result.get("vlm_evaluation") == VLM_EVAL_OK,
        "torso_front_visible": bool_value(torso_obj, False),
        "raw_photo_usable_confidence": usable_confidence,
        "crop_quality_available": crop_quality is not None,
        "no_crop_edge_issue": no_crop_edge_issue,
        "multi_person_accepted": multi_person_accepted,
    }
    if strong_mannequin_rescue_conditions:
        rescued_reasons: List[str] = []
        remaining_reasons: List[str] = []
        for reason in reasons:
            if reason.startswith("bbox_too_narrow:"):
                rescued_reasons.append(reason)
            elif reason == "vlm_photo_usable_false" and not raw_usable and usable_confidence == "low":
                rescued_reasons.append(reason)
            else:
                remaining_reasons.append(reason)
        # Do not partially erase diagnostics. The fallback is applied only when
        # these are the complete set of rejects and every safety gate above held.
        if rescued_reasons and not remaining_reasons:
            reasons = []
            mannequin_front_fallback["applied"] = True
            mannequin_front_fallback["rescued_reject_reasons"] = rescued_reasons
            if "vlm_photo_usable_false" in rescued_reasons:
                front_result["photo_usable_for_ms2"] = {
                    "value": True,
                    "confidence": "medium",
                    "raw_value": False,
                    "raw_confidence": usable_confidence,
                    "mannequin_fallback_override": True,
                }

    accepted = len(reasons) == 0
    gate = {
        "accepted": accepted,
        "reasons": reasons,
        "accepted_views": sorted(accepted_views),
        "strict": bool(getattr(args, "strict_ms2_front", True)),
        "raw_photo_usable_for_ms2": raw_usable,
        "bbox_aspect": round(bbox_aspect, 3),
        "min_bbox_aspect": min_aspect,
        "crop_quality_score": round(crop_quality, 3) if crop_quality is not None else None,
        "front_min_crop_quality": min_crop_quality,
        "front_min_area_ratio": min_area_ratio,
        "front_min_height_ratio": min_height_ratio,
        "crop_quality_detail": crop_quality_detail,
        "front_reject_edge_crops": bool(getattr(args, "front_reject_edge_crops", True)),
        "cv_frontal_face": face_check,
        "subject_type": subject_type,
        "mannequin_face_requirement_disabled": bool(is_mannequin),
        "mannequin_front_fallback": mannequin_front_fallback,
        "front_multi_person_check": multi_person_check,
        "face_target_correspondence": face_correspondence,
        "cv_face_requirement_disabled_for_auto": bool(subject_type == "auto"),
        "cv_face_unavailable_fallback_used": cv_face_unavailable_fallback_used,
        "cv_face_missing_fallback_used": cv_face_missing_fallback_used,
        "vertical_edge_with_face_fallback_used": vertical_edge_with_face_fallback_used,
    }
    front_result["ms2_front_gate"] = gate
    if not accepted:
        front_result["photo_usable_for_ms2"] = {
            "value": False,
            "confidence": "high",
            "gate_rejected_raw_value": raw_usable,
        }
        existing_reason = str(front_result.get("reason", "")).strip()
        front_result["reason"] = (existing_reason + " | " if existing_reason else "") + "MS2 gate rejected: " + ", ".join(reasons)
    return front_result


def normalize_front_photo_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    def value_conf(key: str, default: Any) -> Dict[str, Any]:
        raw = obj.get(key, default)
        if isinstance(raw, dict):
            value = raw.get("value", default)
            conf = raw.get("confidence", "medium")
        else:
            value = raw
            conf = "medium"
        if conf not in {"low", "medium", "high"}:
            conf = "medium"
        return {"value": value, "confidence": conf}

    front_view = value_conf("front_view", "unknown")
    front_value = base.normalize_token(front_view.get("value"))
    if front_value in {"near_front", "three_quarters", "three_quarter_view"}:
        front_value = "three_quarter"
    if front_value not in {"front", "three_quarter", "side", "back", "unknown"}:
        front_value = "unknown"
    front_view["value"] = front_value

    face_visible = value_conf("face_visible", False)
    face_visible["value"] = bool_value(face_visible)
    torso_front_visible = value_conf("torso_front_visible", False)
    torso_front_visible["value"] = bool_value(torso_front_visible)
    usable = value_conf("photo_usable_for_ms2", False)
    usable["value"] = bool_value(usable)

    return {
        "front_view": front_view,
        "face_visible": face_visible,
        "torso_front_visible": torso_front_visible,
        "photo_usable_for_ms2": usable,
        "reason": str(obj.get("reason", "")),
        "raw_front_photo_result": obj,
    }


def front_photo_ok(front_result: Optional[Dict[str, Any]], args: argparse.Namespace) -> bool:
    if not getattr(args, "check_front_view", True):
        return True
    if not isinstance(front_result, dict):
        return False
    gate = front_result.get("ms2_front_gate")
    if isinstance(gate, dict):
        return bool(gate.get("accepted", False))
    accepted = {
        base.normalize_token(v)
        for v in str(getattr(args, "accept_front_views", "front")).split(",")
        if str(v).strip()
    }
    view = base.normalize_token((front_result.get("front_view") or {}).get("value", "unknown"))
    usable = bool_value(front_result.get("photo_usable_for_ms2"), False)
    return usable and view in accepted


def milestone_dirs(out_dir: str = "") -> Dict[str, Path]:
    root = Path(out_dir) if out_dir else Path("outputs") / f"milestone2_{now_stamp()}"
    return {
        "root": ensure_dir(root),
        "crops": ensure_dir(root / "crops"),
        "json": ensure_dir(root / "json"),
    }


def available_camera_sources(primary: Any) -> List[Any]:
    sources: List[Any] = []

    def add(value: Any) -> None:
        if value in sources:
            return
        sources.append(value)

    add(primary)
    if isinstance(primary, int):
        for idx in range(0, 6):
            add(idx)
    for dev in sorted(Path("/dev").glob("video*")):
        add(str(dev))
    for dev in sorted(Path("/dev/v4l/by-id").glob("*")):
        add(str(dev))
    return sources


def open_camera_capture_with_fallback(primary: Any) -> Tuple[cv2.VideoCapture, str, List[str]]:
    tried: List[str] = []
    for source in available_camera_sources(primary):
        for api_name, api in (("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)):
            label = f"{source}({api_name})"
            tried.append(label)
            cap = cv2.VideoCapture(source, api)
            if cap.isOpened():
                read_ok = False
                for _ in range(5):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        read_ok = True
                        break
                    time.sleep(0.03)
                if read_ok:
                    return cap, str(source), tried
                tried.append(f"{label}:read_failed")
            cap.release()
    raise RuntimeError(
        "failed to open camera. "
        f"primary={primary} tried={tried}. "
        "カメラ番号が違う、USB未接続、または別プロセスがカメラを使用中の可能性があります。"
    )


def draw_found_annotated(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    memory_id: int,
    track_id: int,
    decision: str,
    score: float,
    extra_lines: Optional[List[str]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    color = base.decision_color("REUSE_MEMORY", decision)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    base.draw_label(
        annotated,
        (x1, y1, x2, y2),
        [f"M:{memory_id} T:{track_id}", f"{decision} {score:.3f}"] + list(extra_lines or []),
        color,
    )
    return annotated


def build_observed(person_result: Dict[str, Any], items_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "person": person_result.get("person", {}),
        "head": person_result.get("head", {}),
        "par_attributes": person_result.get("par_attributes", {}),
        "visibility": person_result.get("visibility", {}),
        "items": items_result.get("items", {}) if isinstance(items_result, dict) else None,
    }


def vc(value: Any, confidence: str = "medium") -> Dict[str, Any]:
    return {"value": value, "confidence": confidence}


def person_result_has_useful_attributes(person_result: Dict[str, Any]) -> bool:
    if not isinstance(person_result, dict) or person_result.get("error"):
        return False
    par = person_result.get("par_attributes") if isinstance(person_result.get("par_attributes"), dict) else {}
    person = person_result.get("person") if isinstance(person_result.get("person"), dict) else {}
    useful_paths = [
        par.get("upper_color"),
        par.get("upper_type"),
        par.get("lower_color"),
        par.get("lower_type"),
        person.get("apparent_gender"),
    ]
    for raw in useful_paths:
        if isinstance(raw, dict):
            value = base.normalize_token(raw.get("value", "unknown"))
        else:
            value = base.normalize_token(raw)
        if value not in {"", "unknown", "none"}:
            return True
    return False


def infer_basic_color_from_region(region: np.ndarray) -> Tuple[str, str, Dict[str, Any]]:
    if not isinstance(region, np.ndarray) or region.size == 0:
        return "unknown", "low", {"reason": "empty_region"}
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    dark_ratio = float(np.mean(v < 85))
    bright_ratio = float(np.mean(v > 185))
    low_sat_ratio = float(np.mean(s < 55))
    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    detail = {
        "dark_ratio": round(dark_ratio, 3),
        "bright_ratio": round(bright_ratio, 3),
        "low_sat_ratio": round(low_sat_ratio, 3),
        "mean_h": round(mean_h, 1),
        "mean_s": round(mean_s, 1),
        "mean_v": round(mean_v, 1),
    }
    presence = color_presence_scores(region)
    detail["color_presence"] = {k: round(v, 3) for k, v in sorted(presence.items()) if v >= 0.02}
    raw_presence = raw_color_presence_ratios(region)
    detail["raw_color_presence"] = {k: round(v, 3) for k, v in sorted(raw_presence.items()) if v >= 0.02}
    ranked = sorted(presence.items(), key=lambda item: item[1], reverse=True)
    chromatic_ranked = [
        item for item in ranked
        if item[0] not in {"black", "white", "gray", "grey"} and item[1] >= 0.28
    ]
    if ranked and chromatic_ranked and ranked[0][0] in {"gray", "grey", "white"}:
        # Low-saturation patterned clothes often look gray overall. If a real
        # chromatic cue is close, prefer it over a generic neutral label.
        if ranked[0][1] - chromatic_ranked[0][1] <= 0.22:
            color, score = chromatic_ranked[0]
            return color, "high" if score >= 0.62 else "medium", detail
    if ranked and ranked[0][1] >= 0.16:
        color, score = ranked[0]
        if color not in {"gray"} or score >= 0.35:
            return color, "high" if score >= 0.32 else "medium", detail
    if dark_ratio >= 0.35 or mean_v < 95:
        return "black", "high" if dark_ratio >= 0.55 else "medium", detail
    if bright_ratio >= 0.45 and low_sat_ratio >= 0.45:
        return "white", "medium", detail
    if low_sat_ratio >= 0.45:
        return "gray", "medium", detail
    if 90 <= mean_h <= 135:
        return "blue", "medium", detail
    if 15 <= mean_h <= 35 and mean_s >= 45:
        return "brown", "low", detail
    return "unknown", "low", detail


def raw_color_presence_ratios(region: np.ndarray) -> Dict[str, float]:
    if not isinstance(region, np.ndarray) or region.size == 0:
        return {}
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    red_hue = (h <= 8) | (h >= 168)
    return {
        "black": float(np.mean(v < 85)),
        "white": float(np.mean((v > 165) & (s < 85))),
        "gray": float(np.mean((v >= 70) & (v <= 220) & (s < 60))),
        "blue": float(np.mean((h >= 88) & (h <= 135) & (s > 28) & (v > 38))),
        "navy": float(np.mean((h >= 88) & (h <= 135) & (s > 28) & (v <= 135))),
        "light_blue": float(np.mean((h >= 88) & (h <= 135) & (s > 18) & (v > 115))),
        "brown": float(np.mean((h >= 5) & (h <= 32) & (s > 38) & (v > 35) & (v < 190))),
        "beige": float(np.mean((h >= 5) & (h <= 42) & (s >= 12) & (s <= 125) & (v >= 105))),
        "tan": float(np.mean((h >= 7) & (h <= 36) & (s >= 25) & (s <= 155) & (v >= 85) & (v <= 215))),
        "cream": float(np.mean((h >= 5) & (h <= 45) & (s < 90) & (v >= 145))),
        "khaki": float(np.mean((h >= 20) & (h <= 55) & (s >= 20) & (s <= 130) & (v >= 65) & (v <= 205))),
        "green": float(np.mean((h >= 36) & (h <= 85) & (s > 30) & (v > 38))),
        "olive": float(np.mean((h >= 35) & (h <= 70) & (s >= 25) & (v >= 35) & (v <= 145))),
        "red": float(np.mean(red_hue & (s > 45) & (v > 45))),
        "pink": float(np.mean((h >= 145) & (h <= 178) & (s > 25) & (v > 90))),
        "yellow": float(np.mean((h >= 20) & (h <= 38) & (s > 35) & (v > 90))),
    }


def color_presence_scores(region: np.ndarray) -> Dict[str, float]:
    if not isinstance(region, np.ndarray) or region.size == 0:
        return {}
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    raw = raw_color_presence_ratios(region)
    chromatic_total = float(np.mean((s >= 24) & (v >= 38)))
    neutral_total = float(np.mean((s < 90) & (v >= 35)))

    def conf(color: str, denom: float, ratio_weight: float = 2.0, share_weight: float = 0.55) -> float:
        ratio = float(raw.get(color, 0.0) or 0.0)
        share = ratio / max(0.05, denom)
        return min(1.0, ratio * ratio_weight + share * share_weight)

    scores = {
        "black": min(1.0, float(raw.get("black", 0.0)) * 1.55),
        "white": min(1.0, float(raw.get("white", 0.0)) * 1.75),
        "gray": min(1.0, float(raw.get("gray", 0.0)) * 1.10),
        "blue": conf("blue", chromatic_total),
        "navy": conf("navy", chromatic_total),
        "light_blue": conf("light_blue", chromatic_total),
        "brown": conf("brown", chromatic_total),
        "beige": conf("beige", neutral_total, 1.65, 0.48),
        "tan": conf("tan", neutral_total, 1.65, 0.48),
        "cream": conf("cream", neutral_total, 1.65, 0.48),
        "khaki": conf("khaki", chromatic_total),
        "green": conf("green", chromatic_total),
        "olive": conf("olive", chromatic_total),
        "red": conf("red", chromatic_total),
        "pink": conf("pink", chromatic_total),
        "yellow": conf("yellow", chromatic_total),
    }
    return scores


def target_color_presence_score(region: np.ndarray, target_color: str) -> Tuple[float, Dict[str, Any]]:
    color = base.normalize_token(target_color)
    if color in {"", "none", "unknown", "unspecified"}:
        return 0.0, {"target_color": color, "status": "NO_TARGET_COLOR"}
    presence = color_presence_scores(region)
    raw_presence = raw_color_presence_ratios(region)
    families = {
        "white": [("white", 1.0), ("cream", 0.78), ("gray", 0.28)],
        "black": [("black", 1.0), ("navy", 0.38), ("gray", 0.22)],
        "gray": [("gray", 1.0), ("white", 0.38), ("black", 0.30)],
        "grey": [("gray", 1.0), ("white", 0.38), ("black", 0.30)],
        "blue": [("blue", 1.0), ("navy", 0.78), ("light_blue", 0.68), ("black", 0.22)],
        "navy": [("navy", 1.0), ("blue", 0.68), ("black", 0.38)],
        "light_blue": [("light_blue", 1.0), ("blue", 0.55), ("white", 0.20), ("gray", 0.20)],
        "brown": [("brown", 1.0), ("tan", 0.72), ("beige", 0.50), ("cream", 0.28)],
        "beige": [("beige", 1.0), ("tan", 0.88), ("cream", 0.76), ("brown", 0.42), ("khaki", 0.40)],
        "tan": [("tan", 1.0), ("beige", 0.88), ("brown", 0.58), ("cream", 0.58), ("khaki", 0.42)],
        "cream": [("cream", 1.0), ("beige", 0.82), ("white", 0.58), ("tan", 0.52)],
        "khaki": [("khaki", 1.0), ("olive", 0.72), ("beige", 0.58), ("green", 0.48), ("brown", 0.42)],
        "green": [("green", 1.0), ("olive", 0.74), ("khaki", 0.45)],
        "olive": [("olive", 1.0), ("green", 0.72), ("khaki", 0.68), ("brown", 0.35)],
        "red": [("red", 1.0), ("pink", 0.38), ("brown", 0.24)],
        "pink": [("pink", 1.0), ("red", 0.42), ("cream", 0.24)],
        "yellow": [("yellow", 1.0), ("cream", 0.46), ("tan", 0.34), ("khaki", 0.30)],
    }
    keys = families.get(color, [(color, 1.0)])
    score = 0.0
    for key, weight in keys:
        part = float(presence.get(key, 0.0)) * weight
        if color in {"blue", "light_blue"} and key == "navy":
            raw_black = float(raw_presence.get("black", 0.0) or 0.0)
            raw_blue = float(raw_presence.get("blue", 0.0) or 0.0)
            raw_light_blue = float(raw_presence.get("light_blue", 0.0) or 0.0)
            # Very dark clothing often produces a false navy hue. Do not let
            # navy alone become blue unless a real blue/light-blue component is
            # also visible or the region is not dominated by black.
            if raw_black >= 0.50 and max(raw_blue, raw_light_blue) < 0.12:
                part = min(part, 0.24)
        score = max(score, part)
    return score, {
        "target_color": color,
        "score": round(score, 3),
        "presence": {k: round(v, 3) for k, v in sorted(presence.items()) if v >= 0.02},
        "raw_presence": {k: round(v, 3) for k, v in sorted(raw_presence.items()) if v >= 0.02},
    }


def pattern_presence_score(region: np.ndarray, target_pattern: str, secondary_color_score: float = 0.0) -> Tuple[float, Dict[str, Any]]:
    pattern = base.normalize_token(target_pattern)
    if pattern in {"", "none", "unknown", "solid", "unspecified"}:
        return 0.0, {"target_pattern": pattern, "status": "NO_PATTERN_TARGET"}
    if not isinstance(region, np.ndarray) or region.size == 0:
        return 0.0, {"target_pattern": pattern, "status": "EMPTY_REGION"}
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = float(np.mean(edges > 0))
    row_var = float(np.std(np.mean(gray, axis=1)) / 255.0)
    col_var = float(np.std(np.mean(gray, axis=0)) / 255.0)
    texture_score = min(1.0, edge_ratio / 0.10)
    if edge_ratio < 0.05:
        score = min(0.35, secondary_color_score * 0.50)
        return float(score), {
            "target_pattern": pattern,
            "edge_ratio": round(edge_ratio, 3),
            "row_var": round(row_var, 3),
            "col_var": round(col_var, 3),
            "secondary_color_score": round(float(secondary_color_score), 3),
            "score": round(float(score), 3),
            "reason": "edge_too_weak_for_pattern",
        }
    stronger_axis = max(row_var, col_var)
    weaker_axis = min(row_var, col_var)
    orientation_ratio = stronger_axis / max(1e-6, weaker_axis)
    orientation_balance = weaker_axis / max(1e-6, stronger_axis)
    if pattern == "striped":
        # Stripes should have one dominant direction. A plaid shirt has texture,
        # but row/column variation are usually balanced, so do not let texture
        # alone promote plaid to striped.
        axis_strength = min(1.0, stronger_axis / 0.10)
        direction_score = min(1.0, max(0.0, orientation_ratio - 1.25) / 0.75)
        horizontal_band_score = min(1.0, row_var / 0.12) * min(1.0, secondary_color_score / 0.55)
        score = max(axis_strength * direction_score, horizontal_band_score, min(0.42, secondary_color_score * 0.65))
    elif pattern == "plaid":
        # Plaid needs both vertical and horizontal structure. This keeps a
        # single-direction striped shirt from being treated as plaid.
        grid_strength = min(1.0, weaker_axis / 0.075)
        balance_score = min(1.0, orientation_balance / 0.65)
        score = max(texture_score * 0.25, grid_strength * balance_score, min(0.42, secondary_color_score * 0.65))
    else:
        score = max(texture_score * 0.7, secondary_color_score * 2.0)
    return float(min(1.0, score)), {
        "target_pattern": pattern,
        "edge_ratio": round(edge_ratio, 3),
        "row_var": round(row_var, 3),
        "col_var": round(col_var, 3),
        "orientation_ratio": round(orientation_ratio, 3),
        "orientation_balance": round(orientation_balance, 3),
        "secondary_color_score": round(float(secondary_color_score), 3),
        "score": round(float(min(1.0, score)), 3),
    }


def infer_person_result_from_crop_for_ms2(crop: np.ndarray, target: Dict[str, Any]) -> Dict[str, Any]:
    h, w = crop.shape[:2]
    upper = crop[int(h * 0.20):int(h * 0.58), int(w * 0.12):int(w * 0.88)]
    lower = crop[int(h * 0.55):int(h * 0.95), int(w * 0.18):int(w * 0.82)]
    upper_color, upper_conf, upper_detail = infer_basic_color_from_region(upper)
    lower_color, lower_conf, lower_detail = infer_basic_color_from_region(lower)
    target_upper = base.normalize_token(target.get("upper_color", ""))
    target_lower = base.normalize_token(target.get("lower_color", ""))
    target_upper_pattern = base.normalize_token(target.get("upper_pattern", "unknown"))
    target_lower_pattern = base.normalize_token(target.get("lower_pattern", "unknown"))
    target_upper_secondary = base.normalize_token(target.get("upper_secondary_color", "unknown"))
    target_lower_secondary = base.normalize_token(target.get("lower_secondary_color", "unknown"))

    upper_color_score, upper_color_reason = color_similarity(target_upper, upper_color)
    lower_color_score, lower_color_reason = color_similarity(target_lower, lower_color)
    upper_target_score, upper_target_detail = target_color_presence_score(upper, target_upper)
    lower_target_score, lower_target_detail = target_color_presence_score(lower, target_lower)
    upper_secondary_score, upper_secondary_detail = target_color_presence_score(upper, target_upper_secondary)
    lower_secondary_score, lower_secondary_detail = target_color_presence_score(lower, target_lower_secondary)
    upper_pattern_score, upper_pattern_detail = pattern_presence_score(upper, target_upper_pattern, upper_secondary_score)
    lower_pattern_score, lower_pattern_detail = pattern_presence_score(lower, target_lower_pattern, lower_secondary_score)
    upper_has_distinct_pattern = target_upper_pattern in {"striped", "plaid", "color_block"} or target_upper_secondary not in {"", "none", "unknown"}
    if upper_has_distinct_pattern:
        if target_upper_pattern in {"plaid", "striped"}:
            upper_visual_match = (
                (upper_target_score >= 0.42 and upper_pattern_score >= 0.70)
                or (
                    upper_target_score >= 0.34
                    and upper_secondary_score >= 0.30
                    and upper_pattern_score >= 0.70
                )
                or (
                    target_upper_pattern == "striped"
                    and target_upper in {"white", "cream"}
                    and target_upper_secondary in {"black", "navy", "blue", "gray", "grey"}
                    and upper_target_score >= 0.28
                    and upper_secondary_score >= 0.60
                    and upper_pattern_score >= 0.70
                )
            )
        elif target_upper_pattern == "color_block":
            upper_visual_match = upper_target_score >= 0.34 and upper_secondary_score >= 0.22 and upper_pattern_score >= 0.55
        elif target_upper in {"white", "cream"}:
            upper_visual_match = upper_target_score >= 0.36 and upper_secondary_score >= 0.22
        else:
            upper_visual_match = upper_target_score >= 0.42 and upper_secondary_score >= 0.18
    else:
        upper_visual_match = upper_target_score >= 0.42
    lower_visual_match = lower_target_score >= 0.36
    upper_color_match = target_upper in {"", "none", upper_color} or upper_color_score >= 0.55 or upper_visual_match
    lower_color_match = target_lower in {"", "none", lower_color} or lower_color_score >= 0.55 or lower_visual_match
    upper_pattern_match = target_upper_pattern not in {"", "none", "unknown"} and upper_visual_match
    lower_pattern_match = target_lower_pattern not in {"", "none", "unknown"} and (lower_pattern_score >= 0.28 or lower_secondary_score >= 0.05)
    inferred = {
        "person": {
            "apparent_age": vc("unknown", "low"),
            "apparent_gender": vc("unknown", "low"),
            "hair": vc("unknown", "low"),
        },
        "head": {
            "headwear_type": vc("unknown", "low"),
        },
        "par_attributes": {
            "upper_color": vc(upper_color, upper_conf),
            "upper_secondary_color": vc("unknown", "low"),
            "upper_pattern": vc("unknown", "low"),
            "upper_type": vc("unknown", "low"),
            "lower_color": vc(lower_color, lower_conf),
            "lower_secondary_color": vc("unknown", "low"),
            "lower_pattern": vc("unknown", "low"),
            "lower_type": vc("unknown", "low"),
        },
        "visibility": {
            "body_visible": "full_body" if h > w * 1.5 else "upper_body",
        },
        "_fallback": {
            "type": "ms2_crop_color_heuristic",
            "reason": "person_vlm_empty_or_error",
            "upper_region": upper_detail,
            "lower_region": lower_detail,
            "upper_color_match": upper_color_match,
            "lower_color_match": lower_color_match,
            "upper_color_score": round(float(upper_color_score), 3),
            "lower_color_score": round(float(lower_color_score), 3),
            "upper_color_reason": upper_color_reason,
            "lower_color_reason": lower_color_reason,
            "upper_target_color_presence": upper_target_detail,
            "lower_target_color_presence": lower_target_detail,
            "upper_secondary_color_presence": upper_secondary_detail,
            "lower_secondary_color_presence": lower_secondary_detail,
            "upper_pattern_presence": upper_pattern_detail,
            "lower_pattern_presence": lower_pattern_detail,
            "upper_visual_match": upper_visual_match,
            "lower_visual_match": lower_visual_match,
            "upper_pattern_match": upper_pattern_match,
            "lower_pattern_match": lower_pattern_match,
        },
        "_evidence_provenance": {
            "source": "local_image_heuristic",
            "heuristic_only": True,
            "directly_observed": ["upper_color", "lower_color"],
            "not_directly_observed": [
                "upper_secondary_color",
                "upper_pattern",
                "upper_type",
                "lower_secondary_color",
                "lower_pattern",
                "lower_type",
            ],
        },
    }
    return base.normalize_person_json(inferred)


def upper_primary_color_conflicts_with_target(person_result: Dict[str, Any], target: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    target_upper = base.normalize_token(target.get("upper_color", ""))
    if target_upper in {"", "none", "unknown", "unspecified"}:
        return False, {"reason": "no_target_upper_color"}
    par = person_result.get("par_attributes") if isinstance(person_result.get("par_attributes"), dict) else {}
    upper_obj = par.get("upper_color") if isinstance(par.get("upper_color"), dict) else {}
    observed_upper = base.normalize_token(upper_obj.get("value", "unknown"))
    if observed_upper in {"", "none", "unknown", target_upper}:
        return False, {"target_upper": target_upper, "observed_upper": observed_upper, "reason": "not_conflicting"}
    color_score, color_reason = color_similarity(target_upper, observed_upper)
    fallback = person_result.get("_fallback") if isinstance(person_result.get("_fallback"), dict) else {}
    target_presence = fallback.get("upper_target_color_presence") if isinstance(fallback.get("upper_target_color_presence"), dict) else {}
    try:
        target_presence_score = float(target_presence.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_presence_score = 0.0
    raw_presence = target_presence.get("raw_presence") if isinstance(target_presence.get("raw_presence"), dict) else {}
    raw_target_score = float(raw_presence.get(target_upper, 0.0) or 0.0)
    raw_observed_score = float(raw_presence.get(observed_upper, 0.0) or 0.0)
    raw_conflict = raw_target_score < 0.16 and raw_observed_score >= 0.35
    conflict = color_score < 0.45 and (target_presence_score < 0.12 or raw_conflict)
    return conflict, {
        "target_upper": target_upper,
        "observed_upper": observed_upper,
        "color_score": round(float(color_score), 3),
        "color_reason": color_reason,
        "target_presence_score": round(target_presence_score, 3),
        "raw_target_score": round(raw_target_score, 3),
        "raw_observed_score": round(raw_observed_score, 3),
        "raw_conflict": bool(raw_conflict),
        "reason": "upper_primary_color_conflict" if conflict else "weak_or_no_conflict",
    }


def infer_person_result_from_detector_crops(
    *,
    attribute_crop: np.ndarray,
    wide_crop: np.ndarray,
    target: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    attr_result = infer_person_result_from_crop_for_ms2(attribute_crop, target)
    wide_result = infer_person_result_from_crop_for_ms2(wide_crop, target)
    attr_match = build_match_json_ms2(target, attr_result, None)
    wide_match = build_match_json_ms2(target, wide_result, None)
    attr_score = float(attr_match.get("final_score", 0.0) or 0.0)
    wide_score = float(wide_match.get("final_score", 0.0) or 0.0)
    conflict, conflict_detail = upper_primary_color_conflicts_with_target(attr_result, target)
    min_score = float(getattr(args, "wide_crop_attribute_rescue_min_score", 0.75) or 0.75)
    min_gain = float(getattr(args, "wide_crop_attribute_rescue_min_gain", 0.08) or 0.08)
    use_wide = (not conflict) and wide_score >= min_score and wide_score >= attr_score + min_gain
    selected = wide_result if use_wide else attr_result
    selected.setdefault("_attribute_crop_selection", {})
    selected["_attribute_crop_selection"] = {
        "selected": "wide_crop" if use_wide else "attribute_crop",
        "attribute_score": round(attr_score, 3),
        "wide_score": round(wide_score, 3),
        "wide_rescue_min_score": min_score,
        "wide_rescue_min_gain": min_gain,
        "attribute_upper_conflict": conflict_detail,
        "attribute_match": attr_match,
        "wide_match": wide_match,
    }
    return selected


def fast_heuristic_allowed_for_target(target: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> bool:
    upper_pattern = base.normalize_token(target.get("upper_pattern", "none"))
    lower_pattern = base.normalize_token(target.get("lower_pattern", "none"))
    upper_secondary = base.normalize_token(target.get("upper_secondary_color", "none"))
    lower_secondary = base.normalize_token(target.get("lower_secondary_color", "none"))
    upper_color = base.normalize_token(target.get("upper_color", "none"))
    lower_color = base.normalize_token(target.get("lower_color", "none"))
    target_item = base.normalize_token(target.get("target_item", "unspecified"))
    has_distinct_visual_cue = target_has_distinct_visual_cue(target)
    if has_distinct_visual_cue and target_item in {"", "none", "unspecified"}:
        if not bool(getattr(args, "fast_heuristic_allow_distinct_targets", False)):
            return False
        return upper_color not in {"", "none", "unknown"} or lower_color not in {"", "none", "unknown"}
    return (
        upper_pattern in {"", "none", "solid"}
        and lower_pattern in {"", "none", "solid"}
        and upper_secondary in {"", "none", "unknown"}
        and lower_secondary in {"", "none", "unknown"}
        and target_item in {"", "none", "unspecified"}
    )


def build_heuristic_person_candidate(
    *,
    frame: np.ndarray,
    det: Dict[str, Any],
    memory_id: int,
    track_id: int,
    frame_count: int,
    time_sec: float,
    target: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    reason: str,
) -> Dict[str, Any]:
    bbox = tuple(int(v) for v in det["bbox"])
    person_crop, crop_box = base.crop_person(frame, bbox, margin_ratio=args.person_crop_margin)
    person_crop = base.resize_long_side(person_crop, args.person_vlm_long_side)
    attr_crop, attr_crop_box = base.crop_person(frame, bbox, margin_ratio=args.attribute_crop_margin)
    attr_crop = base.resize_long_side(attr_crop, args.person_vlm_long_side)
    candidate_crop_path = dirs["crops"] / f"heuristic_frame{frame_count:06d}_track{track_id}_mem{memory_id}.jpg"
    cv2.imwrite(str(candidate_crop_path), person_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    person_result = infer_person_result_from_detector_crops(
        attribute_crop=attr_crop,
        wide_crop=person_crop,
        target=target,
        args=args,
    )
    person_result.setdefault("_runtime", {})
    person_result["_runtime"].update({
        "kind": "person_heuristic",
        "reason": reason,
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": track_id,
        "track_id_is_real": bool(det.get("track_id_is_real", False)),
        "memory_id": memory_id,
        "bbox": list(bbox),
        "crop_box": list(crop_box),
        "attribute_crop_box": list(attr_crop_box),
        "attribute_crop_margin": float(getattr(args, "attribute_crop_margin", 0.0) or 0.0),
        "crop_path": str(candidate_crop_path),
        "crop_quality_score": det.get("crop_quality_score"),
        "crop_quality_detail": det.get("crop_quality_detail", {}),
        "milestone": MILESTONE,
    })
    items_result = None
    match = build_match_json_ms2(target, person_result, items_result) if getattr(args, "ms2_relaxed_color_feature_type_match", True) else base.build_match_json(target, person_result, items_result)
    match["item_match"]["status"] = "SKIPPED_TARGET_ITEM_UNSPECIFIED"
    match.setdefault("notes", []).append("person_match_from_fast_heuristic")
    return {
        "person_result": person_result,
        "items_result": items_result,
        "match": match,
        "observed": build_observed(person_result, items_result),
        "candidate_crop_path": candidate_crop_path,
        "bbox": bbox,
        "crop_quality_score": det.get("crop_quality_score"),
        "crop_quality_detail": det.get("crop_quality_detail", {}),
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": track_id,
        "track_id_is_real": bool(det.get("track_id_is_real", False)),
        "memory_id": memory_id,
        "source_frame": frame.copy(),
        "person_match_source": "fast_heuristic",
        "person_evidence_provenance": {
            "source": "local_image_heuristic",
            "heuristic_only": True,
        },
    }


def maybe_rescue_person_result_with_crop(
    *,
    person_result: Dict[str, Any],
    person_crop: np.ndarray,
    wide_crop: Optional[np.ndarray] = None,
    target: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not bool(getattr(args, "ms2_enable_crop_attribute_rescue", True)):
        return person_result
    rescue_result = infer_person_result_from_detector_crops(
        attribute_crop=person_crop,
        wide_crop=wide_crop if isinstance(wide_crop, np.ndarray) else person_crop,
        target=target,
        args=args,
    )
    original_match = build_match_json_ms2(target, person_result, None)
    rescue_match = build_match_json_ms2(target, rescue_result, None)
    original_score = float(original_match.get("final_score", 0.0) or 0.0)
    rescue_score = float(rescue_match.get("final_score", 0.0) or 0.0)
    min_score = float(getattr(args, "ms2_crop_rescue_min_score", 0.50) or 0.50)
    min_gain = float(getattr(args, "ms2_crop_rescue_min_gain", 0.10) or 0.10)
    if rescue_score >= min_score and rescue_score >= original_score + min_gain:
        if target_has_distinct_visual_cue(target):
            person_result["_crop_rescue"] = {
                "accepted": False, "original_score": original_score,
                "rescue_score": rescue_score,
                "reason": "keep_direct_vlm_evidence_for_distinct_target",
            }
            return person_result
        rescue_result["_vlm_original_result"] = person_result
        rescue_result["_crop_rescue"] = {
            "accepted": True,
            "original_score": round(original_score, 3),
            "rescue_score": round(rescue_score, 3),
            "min_score": min_score,
            "min_gain": min_gain,
            "reason": "crop_attribute_rescue_better_than_vlm",
            "original_match": original_match,
            "rescue_match": rescue_match,
        }
        return rescue_result
    person_result["_crop_rescue"] = {
        "accepted": False,
        "original_score": round(original_score, 3),
        "rescue_score": round(rescue_score, 3),
        "min_score": min_score,
        "min_gain": min_gain,
        "reason": "crop_attribute_rescue_not_better_enough",
        "rescue_match": rescue_match,
    }
    return person_result


def call_person_vlm_for_candidate(
    *,
    frame: np.ndarray,
    det: Dict[str, Any],
    memory_id: int,
    track_id: int,
    frame_count: int,
    time_sec: float,
    target: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    deadline_mono: Optional[float] = None,
    rescue_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bbox = tuple(int(v) for v in det["bbox"])
    raw_person_crop, crop_box = base.crop_person(
        frame, bbox, margin_ratio=args.person_crop_margin
    )
    person_crop = base.resize_long_side(raw_person_crop, args.person_vlm_long_side)
    attr_crop, attr_crop_box = base.crop_person(frame, bbox, margin_ratio=args.attribute_crop_margin)
    attr_crop = base.resize_long_side(attr_crop, args.person_vlm_long_side)
    candidate_crop_path = dirs["crops"] / f"candidate_frame{frame_count:06d}_track{track_id}_mem{memory_id}.jpg"
    cv2.imwrite(str(candidate_crop_path), person_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    person_prompt = MS2_PERSON_PROMPT if getattr(args, "ms2_use_feature_person_prompt", True) else base.PERSON_PROMPT
    primary_candidate = (
        rescue_context.get("primary_candidate")
        if isinstance(rescue_context, dict)
        and isinstance(rescue_context.get("primary_candidate"), dict)
        else None
    )
    rescue_used = bool(
        rescue_context
        and rescue_context.get("eligible")
        and primary_candidate
        and distinct_target_rescue_policy(primary_candidate, target, args).get("eligible")
    )
    vlm_image_path = candidate_crop_path
    if rescue_used:
        person_prompt = LOWER_RESCUE_PROMPT
        vlm_image_path = dirs["crops"] / f"lower_rescue_frame{frame_count:06d}_track{track_id}_mem{memory_id}.jpg"
        lower_context_crop = build_lower_rescue_crop(raw_person_crop)
        if not cv2.imwrite(str(vlm_image_path), lower_context_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
            raise OSError(f"Could not save rescue input: {vlm_image_path}")
    raw_person_result = call_lmstudio_json_with_deadline(
        deadline_mono=deadline_mono,
        image_path=vlm_image_path,
        prompt=person_prompt,
        base_url=args.lmstudio_base_url,
        model=args.vlm_model,
        timeout_sec=args.lmstudio_timeout,
        max_tokens=args.person_max_tokens,
        kind="lower_rescue" if rescue_used else "person",
    )

    lower_rescue_gate: Optional[Dict[str, Any]] = None
    lower_rescue_audit: Optional[Dict[str, Any]] = None
    if rescue_used:
        person_result, lower_rescue_gate, lower_rescue_audit = integrate_compact_lower_rescue(
            primary_candidate=primary_candidate,
            raw_rescue_result=raw_person_result,
            target=target,
            args=args,
        )
        person_evaluation = (
            VLM_EVAL_OK
            if lower_rescue_audit.get("valid")
            else VLM_EVAL_OPTIONAL_FAILURE
        )
        # The primary person VLM already succeeded. A malformed supplemental
        # observation must fail closed to that primary result, not replace it.
        items_result = copy.deepcopy(primary_candidate.get("items_result"))
        vlm_evaluations = copy.deepcopy(primary_candidate.get("vlm_evaluations") or {})
        vlm_evaluations["person_lower_rescue"] = person_evaluation
    else:
        person_evaluation = classify_vlm_evaluation(raw_person_result, required=True)
        person_result = raw_person_result
        if not person_result_has_useful_attributes(person_result):
            original_person_result = person_result
            person_result = infer_person_result_from_detector_crops(
                attribute_crop=attr_crop,
                wide_crop=person_crop,
                target=target,
                args=args,
            )
            person_result["_vlm_original_result"] = original_person_result
        else:
            person_result = maybe_rescue_person_result_with_crop(
                person_result=person_result,
                person_crop=attr_crop,
                wide_crop=person_crop,
                target=target,
                args=args,
            )
        items_result = None
        vlm_evaluations = {"person": person_evaluation}
        if base.should_run_items(target):
            # Milestone 2 usually skips items, but keep the path open for future targets.
            items_result = call_lmstudio_json_with_deadline(
                deadline_mono=deadline_mono,
                image_path=candidate_crop_path,
                prompt=base.build_items_prompt(target.get("target_item", "none")),
                base_url=args.lmstudio_base_url,
                model=args.vlm_model,
                timeout_sec=args.lmstudio_timeout,
                max_tokens=args.items_max_tokens,
                kind="items",
            )
            vlm_evaluations["items"] = classify_vlm_evaluation(items_result, required=True)

    person_result.setdefault("_runtime", {})
    person_result["_runtime"].update({
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": track_id,
        "memory_id": memory_id,
        "bbox": list(bbox),
        "crop_box": list(crop_box),
        "attribute_crop_box": list(attr_crop_box),
        "attribute_crop_margin": float(getattr(args, "attribute_crop_margin", 0.0) or 0.0),
        "crop_path": str(candidate_crop_path),
        "crop_quality_score": det.get("crop_quality_score"),
        "crop_quality_detail": det.get("crop_quality_detail", {}),
        "milestone": MILESTONE,
    })

    if rescue_used:
        match = build_match_json_ms2(
            target,
            person_result,
            items_result,
            lower_rescue_gate=lower_rescue_gate,
        )
    else:
        match = build_match_json_ms2(target, person_result, items_result) if getattr(args, "ms2_relaxed_color_feature_type_match", True) else base.build_match_json(target, person_result, items_result)
    if items_result is None:
        match["item_match"]["status"] = "SKIPPED_TARGET_ITEM_UNSPECIFIED"

    if rescue_used:
        provenance = copy.deepcopy(
            primary_candidate.get("person_evidence_provenance") or {}
        )
        provenance.setdefault("source", "person_vlm")
        provenance["heuristic_only"] = False
        provenance["lower_rescue"] = {
            "source": "rescue_vlm_lower_focus",
            "evaluation": person_evaluation,
            "valid": bool(lower_rescue_audit.get("valid")),
            "gate_accepted": bool(lower_rescue_gate),
            "merged_fields": list(lower_rescue_audit.get("merged_fields", [])),
            "lower_type_used_for_acceptance": False,
        }
    else:
        fallback = person_result.get("_fallback") if isinstance(person_result.get("_fallback"), dict) else {}
        heuristic_only = fallback.get("type") == "ms2_crop_color_heuristic"
        provenance = {
            "source": "local_image_heuristic" if heuristic_only else "person_vlm",
            "heuristic_only": bool(heuristic_only),
            "person_vlm_evaluation": person_evaluation,
        }

    return {
        "person_result": person_result,
        "items_result": items_result,
        "match": match,
        "observed": build_observed(person_result, items_result),
        "candidate_crop_path": candidate_crop_path,
        "bbox": bbox,
        "crop_quality_score": det.get("crop_quality_score"),
        "crop_quality_detail": det.get("crop_quality_detail", {}),
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": track_id,
        "track_id_is_real": bool(det.get("track_id_is_real", False)),
        "memory_id": memory_id,
        "source_frame": frame.copy(),
        "vlm_evaluations": vlm_evaluations,
        "person_evidence_provenance": provenance,
        "lower_rescue_gate": lower_rescue_gate,
        "distinct_target_rescue": {
            "attempted": rescue_used, "input_path": str(vlm_image_path),
            "prior_policy": (
                rescue_context.get("policy")
                if isinstance(rescue_context, dict)
                else None
            ),
            "compact_evidence": lower_rescue_audit,
        },
    }


def normalized_bbox_to_pixels(raw_bbox: Any, frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(raw_bbox, dict):
        raw_bbox = [raw_bbox.get(k) for k in ("x1", "y1", "x2", "y2")]
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        vals = [float(v) for v in raw_bbox[:4]]
    except Exception:
        return None
    if max(vals) > 1.5:
        return None
    x1, y1, x2, y2 = vals
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    h, w = frame_shape[:2]
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))
    px1, py1, px2, py2 = base.clamp_bbox((px1, py1, px2, py2), w, h)
    if px2 - px1 < 20 or py2 - py1 < 40:
        return None
    return int(px1), int(py1), int(px2), int(py2)


def call_mannequin_full_frame_fallback(
    *,
    frame: np.ndarray,
    frame_count: int,
    time_sec: float,
    target: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    deadline_mono: Optional[float] = None,
) -> Dict[str, Any]:
    frame_path = dirs["crops"] / f"mannequin_fallback_full_frame{frame_count:06d}.jpg"
    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    raw = call_lmstudio_json_with_deadline(
        deadline_mono=deadline_mono,
        image_path=frame_path,
        prompt=MANNEQUIN_FULL_FRAME_PROMPT,
        base_url=args.lmstudio_base_url,
        model=args.vlm_model,
        timeout_sec=args.lmstudio_timeout,
        max_tokens=max(int(args.person_max_tokens), 420),
        kind="mannequin_full_frame",
    )
    evaluation_state = classify_vlm_evaluation(raw, required=True)
    evaluation = {
        "stage": "mannequin_full_frame",
        "state": evaluation_state,
        "found": None if evaluation_state == VLM_EVAL_REQUIRED_FAILURE else bool_value(raw.get("found"), False),
        "error_type": str(raw.get("error", "")),
        "error_message": str(raw.get("message", "")),
    }
    if evaluation_state == VLM_EVAL_REQUIRED_FAILURE:
        return {"candidate": None, "evaluation": evaluation}
    if not bool_value(raw.get("found"), False):
        return {"candidate": None, "evaluation": evaluation}

    bbox = normalized_bbox_to_pixels(raw.get("bbox_norm", raw.get("bbox")), frame.shape)
    if bbox is None:
        raw.setdefault("_postprocess_notes", []).append("mannequin_fallback_invalid_bbox")
        return {"candidate": None, "evaluation": evaluation}

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    fallback_det = {
        "bbox": bbox,
        "conf": 0.0,
        "bbox_w": x2 - x1,
        "bbox_h": y2 - y1,
        "area": float((x2 - x1) * (y2 - y1)),
        "area_ratio": ((x2 - x1) * (y2 - y1)) / float(max(1, w * h)),
        "height_ratio": (y2 - y1) / float(max(1, h)),
        "track_id": -1,
        "track_id_is_real": False,
    }
    crop_quality_score, crop_quality_detail = base.compute_crop_quality(
        fallback_det,
        frame.shape,
        args,
        track_age=0,
    )

    person_result = base.normalize_person_json(raw)
    crop, crop_box = base.crop_person(frame, bbox, margin_ratio=args.person_crop_margin)
    crop = base.resize_long_side(crop, args.person_vlm_long_side)
    candidate_crop_path = dirs["crops"] / f"mannequin_fallback_frame{frame_count:06d}.jpg"
    cv2.imwrite(str(candidate_crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    person_result.setdefault("_runtime", {})
    person_result["_runtime"].update({
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": -1,
        "memory_id": -1,
        "bbox": list(bbox),
        "crop_box": list(crop_box),
        "crop_path": str(candidate_crop_path),
        "crop_quality_score": round(float(crop_quality_score), 3),
        "crop_quality_detail": crop_quality_detail,
        "fallback": "mannequin_full_frame",
        "full_frame_path": str(frame_path),
        "milestone": MILESTONE,
    })

    items_result = None
    vlm_evaluations = {"mannequin_full_frame": VLM_EVAL_OK}
    if base.should_run_items(target):
        items_result = call_lmstudio_json_with_deadline(
            deadline_mono=deadline_mono,
            image_path=candidate_crop_path,
            prompt=base.build_items_prompt(target.get("target_item", "none")),
            base_url=args.lmstudio_base_url,
            model=args.vlm_model,
            timeout_sec=args.lmstudio_timeout,
            max_tokens=args.items_max_tokens,
            kind="items",
        )
        vlm_evaluations["items"] = classify_vlm_evaluation(items_result, required=True)

    match = build_match_json_ms2(target, person_result, items_result) if getattr(args, "ms2_relaxed_color_feature_type_match", True) else base.build_match_json(target, person_result, items_result)
    if items_result is None:
        match["item_match"]["status"] = "SKIPPED_TARGET_ITEM_UNSPECIFIED"

    candidate = {
        "person_result": person_result,
        "items_result": items_result,
        "match": match,
        "observed": build_observed(person_result, items_result),
        "candidate_crop_path": candidate_crop_path,
        "bbox": bbox,
        "crop_quality_score": round(float(crop_quality_score), 3),
        "crop_quality_detail": crop_quality_detail,
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": -1,
        "track_id_is_real": False,
        "memory_id": -1,
        "vlm_evaluations": vlm_evaluations,
        "person_evidence_provenance": {
            "source": "mannequin_full_frame_vlm",
            "heuristic_only": False,
        },
        "source_frame": frame.copy(),
        "fallback_source": "mannequin_full_frame",
    }
    return {"candidate": candidate, "evaluation": evaluation}


def prepare_front_evidence_candidate(
    candidate: Dict[str, Any],
    *,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    frame_detections: Optional[List[Dict[str, Any]]],
    crop_label: str,
) -> Dict[str, Any]:
    frame = candidate.get("source_frame")
    bbox = _bbox_tuple(candidate.get("bbox"))
    if not isinstance(frame, np.ndarray) or bbox is None:
        result = dict(candidate)
        result["front_multi_person_check"] = {
            "accepted": False,
            "reasons": ["front_crop_geometry_missing"],
            "other_people": [],
        }
        return result
    crop, crop_box = base.crop_person(
        frame,
        bbox,
        margin_ratio=float(getattr(args, "front_crop_margin", args.person_crop_margin) or args.person_crop_margin),
    )
    raw_crop_h, raw_crop_w = crop.shape[:2]
    crop = base.resize_long_side(crop, args.person_vlm_long_side)
    crop_h, crop_w = crop.shape[:2]
    scale_x = crop_w / float(max(1, raw_crop_w))
    scale_y = crop_h / float(max(1, raw_crop_h))
    crop_path = dirs["crops"] / (
        f"{crop_label}_frame{int(candidate.get('frame', 0) or 0):06d}_"
        f"track{int(candidate.get('track_id', -1) or -1)}_mem{int(candidate.get('memory_id', -1) or -1)}.jpg"
    )
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    cx1, cy1, _cx2, _cy2 = crop_box
    tx1, ty1, tx2, ty2 = bbox
    result = dict(candidate)
    result.update({
        "candidate_crop_path": crop_path,
        "front_crop_box": list(crop_box),
        "target_bbox_in_front_crop": [
            int(round((tx1 - cx1) * scale_x)),
            int(round((ty1 - cy1) * scale_y)),
            int(round((tx2 - cx1) * scale_x)),
            int(round((ty2 - cy1) * scale_y)),
        ],
        "front_evidence_source": "dedicated_front_crop",
        "front_multi_person_check": analyze_front_crop_people(
            bbox,
            crop_box,
            frame_detections,
            target_track_id=int(candidate.get("track_id", -1) or -1),
            target_track_id_is_real=bool(candidate.get("track_id_is_real", False)),
        ),
    })
    return result


def front_retry_identity_continuity(
    base_candidate: Dict[str, Any],
    current_det: Dict[str, Any],
    current_track_id: int,
    *,
    previous_observation: Optional[Dict[str, Any]] = None,
    expected_memory_id: Optional[int] = None,
) -> Dict[str, Any]:
    base_track_id = int(base_candidate.get("track_id", -1) or -1)
    base_track_real = bool(base_candidate.get("track_id_is_real", False))
    current_track_real = bool(current_det.get("track_id_is_real", False))
    base_bbox = _bbox_tuple(base_candidate.get("bbox"))
    current_bbox = _bbox_tuple(current_det.get("bbox"))

    def geometry_detail(anchor_bbox: Any) -> Tuple[float, bool, bool]:
        anchor = _bbox_tuple(anchor_bbox)
        iou = bbox_iou_value(anchor, current_bbox)
        center_continuity = False
        if anchor is not None and current_bbox is not None:
            bx1, by1, bx2, by2 = anchor
            cx1, cy1, cx2, cy2 = current_bbox
            anchor_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            current_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
            center_continuity = (
                (bx1 <= current_center[0] <= bx2 and by1 <= current_center[1] <= by2)
                or (cx1 <= anchor_center[0] <= cx2 and cy1 <= anchor_center[1] <= cy2)
            )
        return iou, center_continuity, iou >= 0.25 or center_continuity

    base_iou, base_center_continuity, base_geometry_continuity = geometry_detail(base_bbox)
    selected_iou = base_iou
    selected_center_continuity = base_center_continuity
    selected_geometry_continuity = base_geometry_continuity
    anchor_source = "initial_accepted_candidate"
    previous_iou: Optional[float] = None
    previous_center_continuity: Optional[bool] = None

    if expected_memory_id is not None:
        base_memory_id = int(base_candidate.get("memory_id", -1) or -1)
        if base_memory_id != int(expected_memory_id):
            return {
                "accepted": False,
                "reason": "memory_id_mismatch",
                "anchor_source": anchor_source,
                "bbox_iou": round(selected_iou, 3),
                "center_continuity": selected_center_continuity,
                "base_bbox_iou": round(base_iou, 3),
                "base_track_id": base_track_id,
                "current_track_id": int(current_track_id),
                "base_memory_id": base_memory_id,
                "expected_memory_id": int(expected_memory_id),
            }

    if base_track_real and current_track_real:
        same_track = base_track_id == int(current_track_id)
        if not same_track:
            return {
                "accepted": False,
                "reason": "real_track_id_switch",
                "anchor_source": anchor_source,
                "bbox_iou": round(selected_iou, 3),
                "center_continuity": selected_center_continuity,
                "base_bbox_iou": round(base_iou, 3),
                "base_track_id": base_track_id,
                "current_track_id": int(current_track_id),
            }

        if isinstance(previous_observation, dict):
            previous_memory_id = int(previous_observation.get("memory_id", -1) or -1)
            if expected_memory_id is not None and previous_memory_id != int(expected_memory_id):
                return {
                    "accepted": False,
                    "reason": "memory_id_mismatch",
                    "anchor_source": "previous_observation",
                    "bbox_iou": round(selected_iou, 3),
                    "center_continuity": selected_center_continuity,
                    "base_bbox_iou": round(base_iou, 3),
                    "base_track_id": base_track_id,
                    "current_track_id": int(current_track_id),
                    "previous_memory_id": previous_memory_id,
                    "expected_memory_id": int(expected_memory_id),
                }
            previous_track_real = bool(previous_observation.get("track_id_is_real", False))
            previous_track_id = int(previous_observation.get("track_id", -1) or -1)
            previous_bbox = _bbox_tuple(previous_observation.get("bbox"))
            if previous_track_real and previous_track_id != int(current_track_id):
                return {
                    "accepted": False,
                    "reason": "real_track_id_switch",
                    "anchor_source": "previous_observation",
                    "bbox_iou": round(selected_iou, 3),
                    "center_continuity": selected_center_continuity,
                    "base_bbox_iou": round(base_iou, 3),
                    "base_track_id": base_track_id,
                    "previous_track_id": previous_track_id,
                    "current_track_id": int(current_track_id),
                }
            if previous_track_real and previous_bbox is not None:
                previous_iou, previous_center_continuity, selected_geometry_continuity = geometry_detail(previous_bbox)
                selected_iou = previous_iou
                selected_center_continuity = previous_center_continuity
                anchor_source = "previous_same_track_observation"

        accepted = selected_geometry_continuity
        if accepted and anchor_source == "previous_same_track_observation":
            reason = "same_real_track_id_with_previous_observation_continuity"
        elif accepted:
            reason = "same_real_track_id_with_spatial_continuity"
        elif anchor_source == "previous_same_track_observation":
            reason = "same_track_without_previous_observation_continuity"
        else:
            reason = "same_track_without_spatial_continuity"
        return {
            "accepted": accepted,
            "reason": reason,
            "anchor_source": anchor_source,
            "bbox_iou": round(selected_iou, 3),
            "center_continuity": selected_center_continuity,
            "base_bbox_iou": round(base_iou, 3),
            "base_center_continuity": base_center_continuity,
            "previous_bbox_iou": round(previous_iou, 3) if previous_iou is not None else None,
            "previous_center_continuity": previous_center_continuity,
            "base_track_id": base_track_id,
            "current_track_id": int(current_track_id),
        }

    iou = base_iou
    center_continuity = base_center_continuity
    geometry_continuity = base_geometry_continuity
    accepted = geometry_continuity
    return {
        "accepted": accepted,
        "reason": "bbox_continuity_confirmed" if accepted else "identity_continuity_not_confirmed",
        "anchor_source": anchor_source,
        "bbox_iou": round(iou, 3),
        "center_continuity": center_continuity,
        "base_bbox_iou": round(base_iou, 3),
        "base_track_id": base_track_id,
        "current_track_id": int(current_track_id),
    }


def make_front_retry_candidate(
    *,
    memory: Dict[str, Any],
    base_candidate: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    retry_index: int,
    frame: Optional[np.ndarray] = None,
    det: Optional[Dict[str, Any]] = None,
    frame_count: Optional[int] = None,
    time_sec: Optional[float] = None,
    track_id: Optional[int] = None,
    frame_detections: Optional[List[Dict[str, Any]]] = None,
    previous_observation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    src_frame = frame if isinstance(frame, np.ndarray) else memory.get("best_vlm_frame")
    src_det = det if isinstance(det, dict) else memory.get("best_vlm_det")
    if not isinstance(src_frame, np.ndarray) or not isinstance(src_det, dict):
        return None
    bbox = src_det.get("bbox")
    if isinstance(bbox, list):
        bbox = tuple(bbox)
    if not isinstance(bbox, tuple):
        return None

    memory_id = int(memory.get("memory_id", base_candidate.get("memory_id", -1)) or -1)
    use_track_id = int(track_id if track_id is not None else memory.get("best_crop_track_id", memory.get("last_track_id", base_candidate.get("track_id", -1))) or -1)
    use_frame_count = int(frame_count if frame_count is not None else memory.get("best_crop_frame", memory.get("last_seen_frame", base_candidate.get("frame", 0))) or 0)
    use_time_sec = float(time_sec if time_sec is not None else memory.get("best_crop_time", memory.get("last_seen_time", base_candidate.get("time_sec", 0.0))) or 0.0)

    identity_continuity = front_retry_identity_continuity(
        base_candidate,
        src_det,
        use_track_id,
        previous_observation=previous_observation,
        expected_memory_id=memory_id,
    )
    if not bool(identity_continuity.get("accepted", False)):
        memory["front_retry_identity_rejection"] = identity_continuity
        return None

    candidate = dict(base_candidate)
    candidate.update({
        "bbox": tuple(int(v) for v in bbox),
        "crop_quality_score": src_det.get("crop_quality_score"),
        "crop_quality_detail": src_det.get("crop_quality_detail", {}),
        "frame": use_frame_count,
        "time_sec": round(use_time_sec, 3),
        "track_id": use_track_id,
        "track_id_is_real": bool(src_det.get("track_id_is_real", False)),
        "memory_id": memory_id,
        "source_frame": src_frame.copy(),
        "front_retry_index": retry_index,
        "front_retry_identity_continuity": identity_continuity,
    })
    return prepare_front_evidence_candidate(
        candidate,
        dirs=dirs,
        args=args,
        frame_detections=frame_detections,
        crop_label=f"front_retry{retry_index:02d}",
    )


def call_front_photo_vlm(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    *,
    deadline_mono: Optional[float] = None,
) -> Dict[str, Any]:
    fast_result = build_fast_front_photo_result(candidate, args)
    if fast_result is not None and not bool(fast_result.get("requires_front_vlm", False)):
        fast_result["vlm_evaluation"] = VLM_EVAL_NOT_NEEDED
        return fast_result
    subject_type = base.normalize_token(getattr(args, "subject_type", "auto") or "auto")
    allow_vlm_after_fast_miss = bool(getattr(args, "front_vlm_on_fast_gate_miss", False)) or subject_type == "mannequin"
    if not allow_vlm_after_fast_miss:
        result = fast_result or build_front_photo_unknown_result(candidate, args, "fast local front gate missed; front VLM skipped")
        result["vlm_evaluation"] = VLM_EVAL_NOT_NEEDED
        return result

    raw = call_lmstudio_json_with_deadline(
        deadline_mono=deadline_mono,
        image_path=Path(candidate["candidate_crop_path"]),
        prompt=FRONT_PHOTO_PROMPT,
        base_url=args.lmstudio_base_url,
        model=args.vlm_model,
        timeout_sec=args.front_vlm_timeout or args.lmstudio_timeout,
        max_tokens=args.front_max_tokens,
        kind="front_photo",
    )
    front_result = normalize_front_photo_json(raw)
    front_result["vlm_evaluation"] = classify_vlm_evaluation(raw, required=True)
    front_result.setdefault("_runtime", {})
    if isinstance(raw, dict) and isinstance(raw.get("_runtime"), dict):
        front_result["_runtime"] = raw.get("_runtime", {})
    if isinstance(fast_result, dict):
        front_result["local_front_evidence"] = fast_result.get("fast_front_gate", {})
    front_result["_runtime"].update({
        "used_vlm": True,
        "frame": candidate.get("frame"),
        "time_sec": candidate.get("time_sec"),
        "track_id": candidate.get("track_id"),
        "memory_id": candidate.get("memory_id"),
        "crop_path": str(candidate.get("candidate_crop_path")),
        "milestone": MILESTONE,
    })
    return apply_front_photo_gate(front_result, candidate, args)


def save_found_artifacts(
    *,
    dirs: Dict[str, Path],
    frame: np.ndarray,
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    args: argparse.Namespace,
    status: str,
) -> Dict[str, Any]:
    match = candidate["match"]
    decision = match.get("decision", "unknown")
    final_score = float(match.get("final_score", 0.0) or 0.0)
    bbox = tuple(candidate["bbox"])
    front_result = candidate.get("front_result") if isinstance(candidate.get("front_result"), dict) else {}
    view = (front_result.get("front_view") or {}).get("value", "unknown")
    photo_usable = bool_value(front_result.get("photo_usable_for_ms2"), False)
    ms2_front_gate = front_result.get("ms2_front_gate") if isinstance(front_result.get("ms2_front_gate"), dict) else {}
    ms2_success = status == "FOUND_FRONT_PHOTO"

    annotated_path = dirs["root"] / "found_annotated.jpg"
    crop_path = dirs["root"] / "found_crop.jpg"
    person_json_path = dirs["root"] / "found_person.json"
    match_json_path = dirs["root"] / "found_match.json"
    front_json_path = dirs["root"] / "found_front.json"

    if args.save_annotated_image:
        extra = [
            "BEST CROP",
            f"photoQ:{ms2_photo_quality_score(candidate):.3f}",
            f"front:{view}",
            f"ms2:{'OK' if ms2_success else 'NG'}",
        ]
        annotated = draw_found_annotated(
            frame,
            bbox,
            int(candidate["memory_id"]),
            int(candidate["track_id"]),
            decision,
            final_score,
            extra,
        )
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    if args.save_crop:
        shutil.copyfile(candidate["candidate_crop_path"], crop_path)

    write_json(person_json_path, candidate["person_result"])
    if front_result:
        write_json(front_json_path, front_result)
    write_json(match_json_path, {
        "target": target,
        "observed": candidate["observed"],
        "match": match,
        "front_photo": front_result,
        "track": {
            "memory_id": candidate["memory_id"],
            "track_id": candidate["track_id"],
            "bbox": list(bbox),
            "frame": candidate["frame"],
            "time_sec": candidate["time_sec"],
            "time_to_front_photo_sec": candidate.get("time_to_front_photo_sec"),
            "front_retry_index": candidate.get("front_retry_index"),
        },
    })

    return {
        "milestone": MILESTONE,
        "found": True,
        "person_found": True,
        "ms1_success": True,
        "ms2_success": ms2_success,
        "front_photo_saved": bool(args.save_crop and ms2_success),
        "front_view": view,
        "photo_usable_for_ms2": photo_usable,
        "ms2_front_gate": ms2_front_gate,
        "decision": decision,
        "final_score": final_score,
        "person_score": match.get("person_score"),
        "item_score": match.get("item_score"),
        "memory_id": candidate["memory_id"],
        "track_id": candidate["track_id"],
        "bbox": list(bbox),
        "frame": candidate["frame"],
        "time_sec": candidate["time_sec"],
        "time_to_front_photo_sec": candidate.get("time_to_front_photo_sec"),
        "target": target,
        "observed": candidate["observed"],
        "image_path": str(annotated_path) if args.save_annotated_image else "",
        "crop_path": str(crop_path) if args.save_crop else "",
        "person_json_path": str(person_json_path),
        "match_json_path": str(match_json_path),
        "front_json_path": str(front_json_path) if front_result else "",
        "selected_candidate_crop_path": str(candidate.get("candidate_crop_path", "")),
        "ms2_photo_quality_score": ms2_photo_quality_score(candidate),
        "status": status,
    }


def save_best_photo_candidate_sheet(
    *,
    dirs: Dict[str, Path],
    candidates: List[Dict[str, Any]],
    selected: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if not candidates:
        return ""
    selected_path = str(selected.get("candidate_crop_path", ""))
    tile_w, tile_h = 210, 360
    tiles: List[np.ndarray] = []
    for idx, candidate in enumerate(candidates, start=1):
        path = str(candidate.get("candidate_crop_path", ""))
        image = cv2.imread(path)
        if not isinstance(image, np.ndarray):
            continue
        h, w = image.shape[:2]
        scale = min((tile_w - 20) / float(max(1, w)), (tile_h - 86) / float(max(1, h)))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        thumb = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        is_selected = path == selected_path
        bg = (245, 245, 245) if is_selected else (35, 35, 35)
        fg = (0, 0, 0) if is_selected else (255, 255, 255)
        accent = (0, 0, 255) if is_selected else (180, 180, 180)
        tile = np.full((tile_h, tile_w, 3), bg, dtype=np.uint8)
        x = (tile_w - new_w) // 2
        y = 58
        tile[y:y + new_h, x:x + new_w] = thumb
        cv2.rectangle(tile, (4, 4), (tile_w - 5, tile_h - 5), accent, 4 if is_selected else 2)
        title = "BEST" if is_selected else f"CAND {idx}"
        cv2.putText(tile, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, accent, 2, cv2.LINE_AA)
        score = float(candidate.get("ms2_photo_quality_score", ms2_photo_quality_score(candidate)) or 0.0)
        crop_q = candidate_crop_quality(candidate)[0]
        frame = candidate.get("frame", "-")
        retry = candidate.get("front_retry_index")
        lines = [
            f"photoQ {score:.3f}",
            f"cropQ {float(crop_q or 0.0):.3f}",
            f"frame {frame}",
            f"retry {retry if retry is not None else '-'}",
        ]
        for line_idx, line in enumerate(lines):
            cv2.putText(
                tile,
                line,
                (12, tile_h - 82 + line_idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                fg,
                1,
                cv2.LINE_AA,
            )
        tiles.append(tile)
    if not tiles:
        return ""
    sheet = np.hstack(tiles)
    out_path = dirs["root"] / "best_photo_candidates.jpg"
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    return str(out_path)


def better_candidate(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> Dict[str, Any]:
    if a is None:
        return b
    am = a.get("match", {})
    bm = b.get("match", {})
    ar = decision_rank(str(am.get("decision", "unknown")))
    br = decision_rank(str(bm.get("decision", "unknown")))
    if br != ar:
        return b if br > ar else a
    return b if float(bm.get("final_score", 0.0) or 0.0) > float(am.get("final_score", 0.0) or 0.0) else a


def better_ms2_candidate(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> Dict[str, Any]:
    if a is None:
        return b
    af = a.get("front_result") if isinstance(a.get("front_result"), dict) else {}
    bf = b.get("front_result") if isinstance(b.get("front_result"), dict) else {}
    a_usable = bool_value(af.get("photo_usable_for_ms2"), False) if af else False
    b_usable = bool_value(bf.get("photo_usable_for_ms2"), False) if bf else False
    if a_usable != b_usable:
        return b if b_usable else a
    av = front_view_rank((af.get("front_view") or {}).get("value", "unknown") if af else "unknown")
    bv = front_view_rank((bf.get("front_view") or {}).get("value", "unknown") if bf else "unknown")
    if av != bv:
        return b if bv > av else a
    am = a.get("match", {})
    bm = b.get("match", {})
    ar = decision_rank(str(am.get("decision", "unknown")))
    br = decision_rank(str(bm.get("decision", "unknown")))
    if br != ar:
        return b if br > ar else a
    afs = float(am.get("final_score", 0.0) or 0.0)
    bfs = float(bm.get("final_score", 0.0) or 0.0)
    if bfs != afs:
        return b if bfs > afs else a
    aps = ms2_photo_quality_score(a)
    bps = ms2_photo_quality_score(b)
    if bps != aps:
        return b if bps > aps else a
    return b if float(b.get("time_sec", 0.0) or 0.0) > float(a.get("time_sec", 0.0) or 0.0) else a


def candidate_summary(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    match = candidate.get("match", {})
    front = candidate.get("front_result") if isinstance(candidate.get("front_result"), dict) else {}
    gate = front.get("ms2_front_gate") if isinstance(front.get("ms2_front_gate"), dict) else {}
    return {
        "decision": match.get("decision"),
        "final_score": match.get("final_score"),
        "person_score": match.get("person_score"),
        "item_score": match.get("item_score"),
        "front_view": (front.get("front_view") or {}).get("value") if front else None,
        "photo_usable_for_ms2": bool_value(front.get("photo_usable_for_ms2"), False) if front else None,
        "ms2_front_gate_accepted": gate.get("accepted") if gate else None,
        "ms2_front_gate_reasons": gate.get("reasons") if gate else None,
        "memory_id": candidate.get("memory_id"),
        "track_id": candidate.get("track_id"),
        "bbox": list(candidate.get("bbox", [])),
        "frame": candidate.get("frame"),
        "time_sec": candidate.get("time_sec"),
        "time_to_front_photo_sec": candidate.get("time_to_front_photo_sec"),
        "front_retry_index": candidate.get("front_retry_index"),
        "ms2_photo_quality_score": ms2_photo_quality_score(candidate),
        "observed": candidate.get("observed"),
        "crop_path": str(candidate.get("candidate_crop_path", "")),
        "crop_quality_score": candidate_crop_quality(candidate)[0],
        "crop_quality_detail": candidate_crop_quality(candidate)[1],
        "person_evidence_provenance": candidate.get("person_evidence_provenance") or person_evidence_provenance(candidate.get("person_result") or {}),
        "match_evidence": match.get("person_match", {}),
        "evidence_gate_reasons": match.get("evidence_gate_reasons", []),
        "accepted_quality_gate": candidate.get("accepted_quality_gate"),
        "distinct_target_rescue": candidate.get("distinct_target_rescue"),
        "distinct_target_rescue_policy": candidate.get("distinct_target_rescue_policy"),
        "lower_rescue_gate": candidate.get("lower_rescue_gate"),
        "vlm_evaluations": candidate.get("vlm_evaluations", {}),
    }


def not_found_explanation(result: Dict[str, Any], candidate: Optional[Dict[str, Any]], detections_total: int) -> Dict[str, Any]:
    if result.get("status") == "ERROR":
        return {"candidate_reject_gate": "operational_error"}
    if result.get("person_found"):
        return {"not_found_reason": "PERSON_MATCHED_BUT_NO_FRONT_PHOTO", "candidate_reject_gate": "front_photo"}
    if candidate is None:
        reason = "NO_PERSON_OBSERVED" if detections_total == 0 else "NO_VLM_CANDIDATE"
        return {"not_found_reason": reason, "candidate_reject_gate": "detection" if detections_total == 0 else "vlm_candidate"}
    reasons = candidate.get("match", {}).get("evidence_gate_reasons", [])
    if any(r.startswith("high_confidence_contradiction:") for r in reasons):
        reason, gate = "CANDIDATE_REJECTED_BY_ATTRIBUTE_MISMATCH", "attribute_evidence"
    elif (candidate.get("accepted_quality_gate") or {}).get("accepted") is False:
        reason, gate = "CANDIDATE_REJECTED_BY_ACCEPTANCE_GATE", "quality_or_reliability"
    elif (candidate.get("distinct_target_rescue_policy") or {}).get("eligible"):
        reason, gate = "LOWER_EVIDENCE_UNCONFIRMED", "distinct_target_rescue"
    else:
        reason, gate = "CANDIDATE_BELOW_ACCEPT_THRESHOLD", "person_acceptance"
    return {"not_found_reason": reason, "candidate_reject_gate": gate}


def save_non_success_evidence(
    result: Dict[str, Any], candidate: Optional[Dict[str, Any]], *,
    dirs: Dict[str, Path], target: Dict[str, Any], args: argparse.Namespace,
    detections_total: int,
) -> None:
    """Add observation-only fields; never modify found/status/safety flags."""
    if result.get("ms2_success"):
        return
    result.update(not_found_explanation(result, candidate, detections_total))
    if candidate is None:
        return
    evidence = candidate_summary(candidate)
    evidence.update({
        "target": target, "reason": result.get("not_found_reason", result.get("error_code")),
        "reject_gate": result["candidate_reject_gate"],
        "accept_threshold": args.decision_accept,
        "person_result": candidate.get("person_result", {}),
        "match": candidate.get("match", {}),
        "distinct_target_rescue_evaluations": result.get("distinct_target_rescue_evaluations", []),
        "annotated_path": result.get("image_path", ""),
    })
    crop_path = Path(evidence["crop_path"]) if evidence["crop_path"] else None
    evidence["crop_available"] = bool(crop_path and crop_path.is_file())
    frame = candidate.get("source_frame")
    try:
        if not evidence["annotated_path"] and isinstance(frame, np.ndarray) and args.save_annotated_image:
            annotated_path = dirs["root"] / "best_candidate_annotated.jpg"
            annotated = draw_found_annotated(
                frame, tuple(candidate["bbox"]), int(candidate["memory_id"]),
                int(candidate["track_id"]), str(evidence["decision"]),
                float(evidence["final_score"] or 0.0), ["UNCONFIRMED CANDIDATE", str(evidence["reject_gate"])],
            )
            if not cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise OSError("candidate annotation write failed")
            evidence["annotated_path"] = str(annotated_path)
        sidecar = dirs["root"] / "best_candidate_evidence.json"
        evidence["json_path"] = str(sidecar)
        write_json(sidecar, evidence)
    except (OSError, cv2.error) as exc:
        # Saving optional evidence must not convert a Camera ERROR to NOT_FOUND.
        evidence["save_error"] = str(exc)
    result["best_candidate_evidence"] = evidence


def vlm_minimum_candidate_ok(memory: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    detail = memory.get("best_crop_detail") if isinstance(memory.get("best_crop_detail"), dict) else {}
    checks = (
        ("area_ratio", float(getattr(args, "min_area_ratio_for_vlm", 0.0) or 0.0)),
        ("height_ratio", float(getattr(args, "min_height_ratio_for_vlm", 0.0) or 0.0)),
        ("conf", float(getattr(args, "min_conf_for_vlm", 0.0) or 0.0)),
        ("width_ratio", float(getattr(args, "min_width_ratio_for_vlm", 0.0) or 0.0)),
    )
    for field, threshold in checks:
        try:
            observed = float(detail.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            observed = 0.0
        if observed < threshold:
            return False, f"{field}_below_{threshold:.3f}"
    return True, "minimum_geometry_and_confidence_ok"


def should_send_to_vlm(memory: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    minimum_ok, minimum_reason = vlm_minimum_candidate_ok(memory, args)
    if not minimum_ok:
        return False, minimum_reason
    quality = float(memory.get("best_crop_score", 0.0) or 0.0)
    target_color = float(memory.get("best_crop_target_color_score", 0.0) or 0.0)
    if quality < args.min_crop_quality_for_vlm:
        return False, f"quality_below_{args.min_crop_quality_for_vlm:.2f}"
    if target_color >= args.target_color_send_threshold:
        return True, f"target_color_{target_color:.2f}"
    if quality >= args.high_quality_send_threshold:
        return True, f"high_quality_{quality:.2f}"
    return False, "low_priority"


def should_allow_person_vlm_attempt(memory: Dict[str, Any], args: argparse.Namespace, time_sec: float) -> Tuple[bool, str]:
    call_count = int(memory.get("person_vlm_call_count", 0) or 0)
    max_per_memory = max(1, int(getattr(args, "max_person_vlm_calls_per_memory", 1)))
    if call_count <= 0:
        return True, "first_attempt"
    if call_count >= max_per_memory:
        return False, f"max_per_memory_{max_per_memory}"

    last_decision = str(memory.get("decision", "unknown") or "unknown")
    decision_accept = str(getattr(args, "decision_accept", "strong_candidate") or "strong_candidate")
    if accepted_decision(last_decision, decision_accept):
        return False, f"already_decided_{last_decision}"

    last_time = float(memory.get("person_vlm_last_time", -1e9) or -1e9)
    retry_interval = float(getattr(args, "person_vlm_retry_interval_sec", 2.0))
    if time_sec - last_time < retry_interval:
        return False, f"retry_wait_{retry_interval:.1f}s"

    current_quality = float(memory.get("best_crop_score", 0.0) or 0.0)
    last_quality = float(memory.get("person_vlm_last_quality", 0.0) or 0.0)
    min_gain = float(getattr(args, "person_vlm_retry_min_quality_gain", 0.08))
    if current_quality < last_quality + min_gain and current_quality < float(getattr(args, "high_quality_send_threshold", 0.55)):
        return False, f"retry_quality_gain_below_{min_gain:.2f}"

    return True, f"retry_after_{last_decision}"


def init_search_window(
    args: argparse.Namespace, preview_manager: HighGuiPreviewManager
) -> bool:
    window_name = str(getattr(args, "search_window_name", "ms2_person_search_camera"))
    return preview_manager.activate(
        "search",
        window_name=window_name,
        size=(
            int(getattr(args, "search_window_width", 1440)),
            int(getattr(args, "search_window_height", 810)),
        ),
        position=(
            int(getattr(args, "search_window_x", 180)),
            int(getattr(args, "search_window_y", 80)),
        ),
        topmost=bool(getattr(args, "search_window_topmost", True)),
    )


def show_search_window(
    frame: np.ndarray,
    dets: List[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    preview_manager: HighGuiPreviewManager,
    frame_count: int,
    time_sec: float,
    vlm_call_count: int,
    front_vlm_call_count: int,
    last_vlm_skip_reason: str = "",
    status: str = "MS2 SEARCHING",
) -> Optional[bool]:
    preview = frame.copy()
    cv2.putText(preview, status, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(preview, status, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 255, 255), 4, cv2.LINE_AA)
    info = f"frame={frame_count} time={time_sec:.1f}s dets={len(dets)} vlm={vlm_call_count} front={front_vlm_call_count}"
    cv2.putText(preview, info, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(preview, info, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    if last_vlm_skip_reason:
        skip_text = f"last skip: {last_vlm_skip_reason[:95]}"
        cv2.putText(preview, skip_text, (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(preview, skip_text, (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "F10/Q = stop", (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(preview, "F10/Q = stop", (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        track_id = int(det.get("track_id", -1))
        color = (0, 255, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        cv2.putText(preview, f"track:{track_id}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return preview_manager.show("search", preview)


def run_milestone2_search(
    args: argparse.Namespace,
    *,
    preview_manager: Optional[HighGuiPreviewManager] = None,
    preview_stop_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    reset_stop_request()
    log_search_stage("search_begin")
    run_started = dt.datetime.now()
    if base.normalize_token(getattr(args, "subject_type", "auto") or "auto") == "mannequin":
        args.require_cv_face_for_front = False
        args.target_gender = "none"
        args.target_hair = "none"
        args.target_headwear_type = "none"
    dirs = milestone_dirs(args.out_dir)
    target = base.build_target_json(args)
    write_json(dirs["root"] / "target.json", {
        "search_version": SEARCH_VERSION,
        "target": target,
        "args": vars(args),
    })

    use_camera = bool(args.camera_device) or int(args.camera_index) >= 0
    if args.video:
        input_mode = "video"
        video_source: Any = str(args.video)
        source_label = str(args.video)
    elif use_camera:
        input_mode = "camera"
        video_source = str(args.camera_device) if args.camera_device else int(args.camera_index)
        source_label = str(video_source)
    else:
        raise RuntimeError("input source is required: use --video or --camera_index/--camera_device")

    log_search_stage(
        "camera_open_begin", input_mode=input_mode, source=source_label
    )
    if input_mode == "camera":
        cap, source_label, camera_open_tried = open_camera_capture_with_fallback(video_source)
        video_source = source_label
    else:
        camera_open_tried = []
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {input_mode}: {source_label}")
    log_search_stage(
        "camera_open_done", input_mode=input_mode, source=source_label
    )
    if input_mode == "camera":
        log_search_stage("camera_config_begin")
        if int(args.camera_width) > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
        if int(args.camera_height) > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
        if float(args.camera_fps) > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
        log_search_stage("camera_config_done")

    model_initialized = False
    log_search_stage("yolo_init_begin", model=args.yolo_model)
    try:
        model = base.YOLO(args.yolo_model)
        model_initialized = True
    finally:
        if not model_initialized:
            cap.release()
    log_search_stage("yolo_init_done", model=args.yolo_model)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0)
    search_video_segment: Optional[FailSoftVideoSegment] = None
    search_recording_summary: Optional[Dict[str, Any]] = None
    search_recording_enabled = bool(
        input_mode == "camera"
        and bool(getattr(args, "record_search_video", False))
    )
    log_search_stage(
        "recording_init_begin", enabled=search_recording_enabled
    )
    if search_recording_enabled:
        search_recording_started_at = dt.datetime.now().astimezone()
        search_video_role_stem = str(
            getattr(args, "search_video_role_stem", "") or ""
        ).strip()
        if search_video_role_stem:
            search_video_dir_text = str(
                getattr(args, "search_video_dir", "") or ""
            ).strip()
            search_video_dir = (
                Path(search_video_dir_text)
                if search_video_dir_text
                else dirs["root"]
            )
            search_video_path = timestamped_video_path(
                search_video_dir,
                search_video_role_stem,
                started_at=search_recording_started_at,
            )
        else:
            search_video_path_text = str(
                getattr(args, "search_video_path", "") or ""
            )
            search_video_path = (
                Path(search_video_path_text)
                if search_video_path_text
                else dirs["root"] / "raw_search.mp4"
            )
        manifest_path_text = str(
            getattr(args, "camera_video_manifest_path", "") or ""
        )
        manifest_path = (
            Path(manifest_path_text)
            if manifest_path_text
            else dirs["root"] / "camera_video_manifest.json"
        )
        recording_manifest = CameraVideoManifest(
            manifest_path,
            logger=lambda message: print(f"[CAMERA_RECORDING] {message}", flush=True),
        )
        requested_record_fps = float(
            getattr(args, "record_video_fps", 0.0)
            or getattr(args, "camera_fps", 0.0)
            or fps
        )
        search_video_segment = FailSoftVideoSegment(
            path=search_video_path,
            kind="search",
            area_index=getattr(args, "recording_area_index", None),
            target_id=str(getattr(args, "recording_target_id", "") or ""),
            requested_fps=requested_record_fps,
            camera_source=source_label,
            cv2_module=cv2,
            manifest=recording_manifest,
            logger=lambda message: print(f"[CAMERA_RECORDING] {message}", flush=True),
            started_at=search_recording_started_at.isoformat(timespec="seconds"),
        )
    log_search_stage(
        "recording_init_done", enabled=search_recording_enabled
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    searched_from_sec = max(0.0, float(args.area_arrival_sec)) if input_mode == "video" else 0.0
    search_until_found = bool(getattr(args, "search_until_found", False)) and input_mode == "camera"
    max_search_runtime_sec = float(getattr(args, "max_search_runtime_sec", 0.0) or 0.0)
    searched_to_sec = searched_from_sec + max(0.1, float(args.search_duration_sec))
    start_frame = int(searched_from_sec * fps)
    end_frame = int(searched_to_sec * fps)
    if input_mode == "video" and total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if input_mode == "video":
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    show_window = bool(getattr(args, "show_search_window", True)) and input_mode == "camera"
    owns_preview_manager = False
    if show_window and preview_manager is None:
        preview_manager = HighGuiPreviewManager(
            cv2_module=cv2,
            logger=lambda message: print(f"[CAMERA_PREVIEW] {message}", flush=True),
        )
        owns_preview_manager = True
    log_search_stage("preview_init_begin", enabled=show_window)
    if show_window:
        try:
            show_window = bool(init_search_window(args, preview_manager))
        except Exception as exc:
            print(
                f"[CAMERA_PREVIEW] search preview disabled: {type(exc).__name__}: {exc}",
                flush=True,
            )
            show_window = False
    log_search_stage("preview_init_done", enabled=show_window)

    person_memories: Dict[int, Dict[str, Any]] = {}
    track_to_memory: Dict[int, int] = {}
    track_seen_counts: Dict[int, int] = {}
    next_memory_id = 1
    used_vlm_memory_ids = set()
    vlm_call_count = 0
    vlm_skip_reason_counts: Dict[str, int] = {}
    vlm_send_reason_counts: Dict[str, int] = {}
    vlm_send_candidate_count = 0
    vlm_retry_count = 0
    distinct_target_rescue_evaluations: List[Dict[str, Any]] = []
    solid_lower_rescue_attempted = False
    vlm_attempt_skip_reason_counts: Dict[str, int] = {}
    heuristic_person_candidate_count = 0
    heuristic_person_accept_count = 0
    last_vlm_skip_reason = ""
    frames_processed = 0
    detections_total = 0
    best_candidate: Optional[Dict[str, Any]] = None
    best_person_candidate: Optional[Dict[str, Any]] = None
    best_front_photo_candidate: Optional[Dict[str, Any]] = None
    front_photo_candidates: List[Dict[str, Any]] = []
    first_front_photo_time_sec: Optional[float] = None
    best_photo_candidate_count = 0
    best_photo_selected_reason = ""
    found_result: Optional[Dict[str, Any]] = None
    accepted_candidates_by_memory: Dict[int, Dict[str, Any]] = {}
    front_last_checked_time: Dict[int, float] = {}
    front_last_checked_frame: Dict[int, int] = {}
    front_check_counts_by_memory: Dict[int, int] = {}
    front_vlm_call_count = 0
    front_fast_check_count = 0
    front_fast_accept_count = 0
    front_vlm_skipped_by_fast_gate_count = 0
    front_retry_count = 0
    front_retry_success_count = 0
    front_retry_last_view = "unknown"
    time_to_front_photo_sec: Optional[float] = None
    subject_type = base.normalize_token(getattr(args, "subject_type", "auto") or "auto")
    mannequin_fallback_count = 0
    mannequin_fallback_success_count = 0
    last_mannequin_fallback_time = -1e9
    stop_reason = "SEARCH_WINDOW_ENDED"
    camera_started_mono = time.monotonic()
    search_deadline_mono: Optional[float] = None
    if input_mode == "camera":
        if search_until_found:
            if max_search_runtime_sec > 0.0:
                search_deadline_mono = camera_started_mono + max_search_runtime_sec
        else:
            search_deadline_mono = camera_started_mono + max(0.1, float(args.search_duration_sec))
    live_frame_count = 0
    camera_read_fail_count = 0
    vlm_health: Dict[str, Dict[str, int]] = {}
    vlm_error_diagnostics: List[Dict[str, Any]] = []

    def record_candidate_evaluations(candidate: Dict[str, Any]) -> None:
        evaluations = candidate.get("vlm_evaluations") if isinstance(candidate, dict) else None
        if not isinstance(evaluations, dict):
            return
        for stage, state in evaluations.items():
            record_vlm_evaluation(vlm_health, str(stage), str(state))

    def record_front_check(front_result: Dict[str, Any]) -> None:
        nonlocal front_vlm_call_count, front_fast_check_count, front_fast_accept_count, front_vlm_skipped_by_fast_gate_count
        state = front_result.get("vlm_evaluation") if isinstance(front_result, dict) else None
        if state:
            record_vlm_evaluation(vlm_health, "front", str(state))
        runtime = front_result.get("_runtime") if isinstance(front_result, dict) else {}
        used_vlm = bool(runtime.get("used_vlm", True)) if isinstance(runtime, dict) else True
        if used_vlm:
            front_vlm_call_count += 1
            return
        front_fast_check_count += 1
        if front_photo_ok(front_result, args):
            front_fast_accept_count += 1
        allow_vlm_after_fast_miss = bool(getattr(args, "front_vlm_on_fast_gate_miss", False)) or subject_type == "mannequin"
        if not front_photo_ok(front_result, args) and not allow_vlm_after_fast_miss:
            front_vlm_skipped_by_fast_gate_count += 1

    def best_photo_hold_sec() -> float:
        return max(0.0, float(getattr(args, "best_photo_hold_sec", 0.0) or 0.0))

    def remember_front_photo_candidate(candidate: Dict[str, Any]) -> bool:
        nonlocal best_front_photo_candidate, first_front_photo_time_sec
        nonlocal best_photo_candidate_count, time_to_front_photo_sec
        reliability_ok, reliability_reasons = accepted_candidate_reliability_ok(candidate, target, args)
        candidate["final_reliability_gate"] = {
            "accepted": reliability_ok,
            "reasons": reliability_reasons,
        }
        if not reliability_ok:
            return False
        cand_time = float(candidate.get("time_sec", 0.0) or 0.0)
        candidate["time_to_front_photo_sec"] = round(cand_time - searched_from_sec, 3)
        candidate["ms2_photo_quality_score"] = ms2_photo_quality_score(candidate)
        if first_front_photo_time_sec is None:
            first_front_photo_time_sec = cand_time
            time_to_front_photo_sec = round(cand_time - searched_from_sec, 3)
        best_photo_candidate_count += 1
        front_photo_candidates.append(candidate)
        best_front_photo_candidate = better_ms2_candidate(best_front_photo_candidate, candidate)
        return True

    def save_best_front_photo(reason: str) -> None:
        nonlocal found_result, stop_reason, best_photo_selected_reason
        if best_front_photo_candidate is None:
            return
        evidence_frame = best_front_photo_candidate.get("source_frame")
        if not isinstance(evidence_frame, np.ndarray):
            evidence_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        found_result = save_found_artifacts(
            dirs=dirs,
            frame=evidence_frame,
            candidate=best_front_photo_candidate,
            target=target,
            args=args,
            status="FOUND_FRONT_PHOTO",
        )
        best_crop_selected_path = dirs["root"] / "best_crop_selected.jpg"
        try:
            shutil.copyfile(best_front_photo_candidate["candidate_crop_path"], best_crop_selected_path)
        except Exception:
            best_crop_selected_path = Path("")
        candidates_sheet_path = save_best_photo_candidate_sheet(
            dirs=dirs,
            candidates=front_photo_candidates,
            selected=best_front_photo_candidate,
            args=args,
        )
        best_photo_selected_reason = reason
        found_result.update({
            "best_photo_hold_sec": best_photo_hold_sec(),
            "best_photo_candidate_count": best_photo_candidate_count,
            "best_photo_selected_reason": reason,
            "selected_photo_quality_score": ms2_photo_quality_score(best_front_photo_candidate),
            "first_front_photo_time_sec": round(first_front_photo_time_sec, 3) if first_front_photo_time_sec is not None else None,
            "best_crop_selected_path": str(best_crop_selected_path) if str(best_crop_selected_path) else "",
            "best_photo_candidates_path": candidates_sheet_path,
        })
        stop_reason = reason

    try:
        first_read_pending = True
        while True:
            if stop_requested():
                stop_reason = "EXTERNAL_STOP_REQUESTED"
                break
            remaining = remaining_search_time(search_deadline_mono)
            if remaining is not None and remaining <= 0.0:
                stop_reason = "SEARCH_DEADLINE_REACHED"
                break
            if first_read_pending:
                log_search_stage("first_read_begin")
            ok, frame = cap.read()
            if first_read_pending:
                log_search_stage(
                    "first_read_done", ok=bool(ok and frame is not None)
                )
                first_read_pending = False
            if not ok:
                if input_mode == "camera":
                    camera_read_fail_count += 1
                    if camera_read_fail_count <= int(getattr(args, "camera_read_fail_tolerance", 30) or 30):
                        time.sleep(0.05)
                        continue
                    stop_reason = "CAMERA_READ_FAILED"
                    break
                stop_reason = "VIDEO_ENDED"
                break
            camera_read_fail_count = 0
            if search_video_segment is not None:
                # Record the untouched Camera frame before detection/UI overlays.
                search_video_segment.write(frame)
            if input_mode == "camera":
                live_frame_count += 1
                frame_count = live_frame_count
                time_sec = time.monotonic() - camera_started_mono
                if search_until_found and max_search_runtime_sec > 0.0 and time_sec > max_search_runtime_sec:
                    stop_reason = "MAX_SEARCH_RUNTIME_REACHED"
                    break
                if not search_until_found and time_sec > max(0.1, float(args.search_duration_sec)):
                    stop_reason = "SEARCH_WINDOW_ENDED"
                    break
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                time_sec = frame_count / fps if fps > 0 else 0.0
                if time_sec > searched_to_sec or frame_count > end_frame:
                    break
            if frame_count % max(1, int(args.process_every)) != 0:
                continue

            frames_processed += 1
            dets = base.detect_persons_with_tracking(model, frame, args)
            detections_total += len(dets)
            if show_window:
                try:
                    keep_window = show_search_window(
                        frame,
                        dets,
                        args=args,
                        preview_manager=preview_manager,
                        frame_count=frame_count,
                        time_sec=time_sec,
                        vlm_call_count=vlm_call_count,
                        front_vlm_call_count=front_vlm_call_count,
                        last_vlm_skip_reason=last_vlm_skip_reason,
                    )
                except Exception as exc:
                    print(
                        f"[CAMERA_PREVIEW] search preview disabled: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    show_window = False
                    keep_window = None
                if keep_window is None:
                    show_window = False
                    if preview_manager is not None:
                        preview_manager.deactivate("search")
                elif not keep_window:
                    if preview_stop_callback is not None:
                        try:
                            preview_stop_callback()
                        except Exception as exc:
                            print(
                                f"[CAMERA_PREVIEW] shutdown callback failed: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    stop_reason = "USER_CLOSED_SEARCH_WINDOW"
                    break
            used_memory_ids = set()

            for det in dets:
                if stop_requested():
                    stop_reason = "EXTERNAL_STOP_REQUESTED"
                    break
                track_id = int(det["track_id"])
                bbox = tuple(int(v) for v in det["bbox"])
                track_id_is_real = bool(det.get("track_id_is_real", False))
                if track_id_is_real:
                    track_seen_counts[track_id] = track_seen_counts.get(track_id, 0) + 1
                track_age = track_seen_counts.get(track_id, 0) if track_id_is_real else 0
                if track_age < int(args.vlm_min_track_frames):
                    reason = f"track_age_below_{int(args.vlm_min_track_frames)}"
                    vlm_skip_reason_counts[reason] = vlm_skip_reason_counts.get(reason, 0) + 1
                    last_vlm_skip_reason = reason
                    continue

                hist = base.compute_upper_body_hist(frame, bbox) if args.enable_person_memory else None
                memory_id = track_to_memory.get(track_id) if track_id_is_real else None
                if memory_id is None:
                    memory_id = next_memory_id
                    next_memory_id += 1
                    person_memories[memory_id] = base.make_new_memory(memory_id, track_id, det, hist, frame_count, time_sec)
                    if track_id_is_real:
                        track_to_memory[track_id] = memory_id
                if memory_id in used_memory_ids:
                    continue
                used_memory_ids.add(memory_id)

                memory = person_memories[memory_id]
                previous_observation = {
                    "memory_id": memory_id,
                    "track_id": memory.get("last_track_id"),
                    "track_id_is_real": memory.get("last_track_id") is not None,
                    "bbox": memory.get("last_bbox"),
                    "frame": memory.get("last_seen_frame"),
                    "time_sec": memory.get("last_seen_time"),
                }
                base.update_memory_observation(memory, track_id, det, hist, frame_count, time_sec)
                q_score, q_detail = base.compute_crop_quality(det, frame.shape, args, track_age=track_age)
                det["crop_quality_score"] = round(q_score, 3)
                det["crop_quality_detail"] = q_detail
                base.update_best_crop_candidate(
                    memory,
                    frame,
                    det,
                    hist,
                    frame_count,
                    time_sec,
                    q_score,
                    q_detail,
                    track_id,
                    target,
                    None,
                )
                if int(memory.get("best_crop_frame", -1) or -1) == frame_count:
                    memory["best_crop_frame_detections"] = [dict(frame_det) for frame_det in dets]
                target_color = base.normalize_token((target or {}).get("upper_color", ""))
                if int(memory.get("best_crop_frame", -1) or -1) == frame_count and target_color not in {"", "none", "unknown", "unspecified"}:
                    ms2_color_score, ms2_color_detail = upper_body_color_hint_ms2(frame, bbox, target_color)
                    if ms2_color_score > float(memory.get("best_crop_target_color_score", 0.0) or 0.0):
                        memory["best_crop_target_color_score"] = ms2_color_score
                        memory["best_crop_target_color_detail"] = ms2_color_detail

                if (
                    args.continue_after_not_front
                    and memory_id in accepted_candidates_by_memory
                    and args.check_front_view
                    and call_budget_available(front_vlm_call_count, args.max_front_vlm_calls)
                ):
                    last_t = float(front_last_checked_time.get(memory_id, -1e9))
                    last_f = int(front_last_checked_frame.get(memory_id, -1))
                    checked_count = int(front_check_counts_by_memory.get(memory_id, 0))
                    if (
                        call_budget_available(checked_count, args.max_front_vlm_calls)
                        and time_sec - last_t >= float(args.front_retry_interval_sec)
                        and frame_count != last_f
                    ):
                        retry_candidate = make_front_retry_candidate(
                            memory=memory,
                            base_candidate=accepted_candidates_by_memory[memory_id],
                            dirs=dirs,
                            args=args,
                            retry_index=checked_count + 1,
                            frame=frame,
                            det=det,
                            frame_count=frame_count,
                            time_sec=time_sec,
                            track_id=track_id,
                            frame_detections=dets,
                            previous_observation=previous_observation,
                        )
                        if retry_candidate is not None:
                            retry_candidate["front_result"] = call_front_photo_vlm(
                                retry_candidate,
                                args,
                                deadline_mono=search_deadline_mono,
                            )
                            record_front_check(retry_candidate["front_result"])
                            front_retry_count += 1
                            front_check_counts_by_memory[memory_id] = checked_count + 1
                            front_last_checked_time[memory_id] = float(retry_candidate.get("time_sec", time_sec) or time_sec)
                            front_last_checked_frame[memory_id] = int(retry_candidate.get("frame", frame_count) or frame_count)
                            front_retry_last_view = (retry_candidate.get("front_result", {}).get("front_view") or {}).get("value", "unknown")
                            accepted_candidates_by_memory[memory_id] = retry_candidate
                            best_person_candidate = better_ms2_candidate(best_person_candidate, retry_candidate)
                            if front_photo_ok(retry_candidate.get("front_result"), args):
                                if remember_front_photo_candidate(retry_candidate):
                                    front_retry_success_count += 1
                                    if best_photo_hold_sec() <= 0.0:
                                        save_best_front_photo("FOUND_FRONT_PHOTO")
                                        break
                                    stop_reason = "FOUND_FRONT_PHOTO_HOLDING_BEST"

                if memory_id in accepted_candidates_by_memory:
                    continue

                attempt_allowed, attempt_reason = should_allow_person_vlm_attempt(memory, args, time_sec)
                if not attempt_allowed:
                    vlm_attempt_skip_reason_counts[attempt_reason] = vlm_attempt_skip_reason_counts.get(attempt_reason, 0) + 1
                    last_vlm_skip_reason = attempt_reason
                    continue
                if not call_budget_available(vlm_call_count, args.max_total_vlm_calls):
                    reason = "max_total_vlm_calls_reached"
                    vlm_skip_reason_counts[reason] = vlm_skip_reason_counts.get(reason, 0) + 1
                    last_vlm_skip_reason = reason
                    continue
                send, _reason = should_send_to_vlm(memory, args)
                if not send:
                    vlm_skip_reason_counts[_reason] = vlm_skip_reason_counts.get(_reason, 0) + 1
                    last_vlm_skip_reason = _reason
                    continue
                vlm_send_candidate_count += 1
                if int(memory.get("person_vlm_call_count", 0) or 0) > 0:
                    vlm_retry_count += 1
                    _reason = f"{attempt_reason}:{_reason}"
                vlm_send_reason_counts[_reason] = vlm_send_reason_counts.get(_reason, 0) + 1

                best_det = dict(memory.get("best_vlm_det") or det)
                best_frame = memory.get("best_vlm_frame") if isinstance(memory.get("best_vlm_frame"), np.ndarray) else frame
                best_track_id = int(memory.get("best_crop_track_id", track_id) or track_id)
                best_frame_count = int(memory.get("best_crop_frame", frame_count) or frame_count)
                best_time_sec = float(memory.get("best_crop_time", time_sec) or time_sec)
                best_frame_dets = memory.get("best_crop_frame_detections")
                if not isinstance(best_frame_dets, list):
                    best_frame_dets = dets
                candidate = None
                if (
                    bool(getattr(args, "use_fast_heuristic_person_match", False))
                    and int(memory.get("person_vlm_call_count", 0) or 0) == 0
                    and fast_heuristic_allowed_for_target(target, args)
                ):
                    heuristic_person_candidate_count += 1
                    heuristic_candidate = build_heuristic_person_candidate(
                        frame=best_frame,
                        det=best_det,
                        memory_id=memory_id,
                        track_id=best_track_id,
                        frame_count=best_frame_count,
                        time_sec=best_time_sec,
                        target=target,
                        dirs=dirs,
                        args=args,
                        reason=_reason,
                    )
                    heuristic_score = float(heuristic_candidate.get("match", {}).get("final_score", 0.0) or 0.0)
                    heuristic_quality = float(heuristic_candidate.get("crop_quality_score", 0.0) or 0.0)
                    heuristic_detail = heuristic_candidate.get("crop_quality_detail") if isinstance(heuristic_candidate.get("crop_quality_detail"), dict) else {}
                    heuristic_area = float(heuristic_detail.get("area_ratio", 0.0) or 0.0)
                    heuristic_quality_ok = (
                        heuristic_quality >= float(getattr(args, "fast_heuristic_min_quality", 0.45) or 0.45)
                        and heuristic_area >= float(getattr(args, "fast_heuristic_min_area_ratio", 0.012) or 0.012)
                    )
                    if heuristic_score >= float(getattr(args, "fast_heuristic_accept_score", 0.85)) and heuristic_quality_ok:
                        heuristic_person_accept_count += 1
                        candidate = heuristic_candidate
                        print(
                            f"[MS2_HEURISTIC_ACCEPT] memory={memory_id} track={track_id} "
                            f"score={heuristic_score:.3f} reason={_reason}",
                            flush=True,
                        )
                    elif heuristic_score >= float(getattr(args, "fast_heuristic_accept_score", 0.85)):
                        last_vlm_skip_reason = (
                            f"heuristic_quality_gate_failed:"
                            f"q={heuristic_quality:.3f},area={heuristic_area:.4f}"
                        )

                if candidate is None:
                    print(
                        f"[MS2_VLM_SEND] memory={memory_id} track={track_id} reason={_reason} "
                        f"quality={float(memory.get('best_crop_score', 0.0) or 0.0):.3f} "
                        f"target_color={float(memory.get('best_crop_target_color_score', 0.0) or 0.0):.3f}",
                        flush=True,
                    )
                    rescue_context = memory.pop("pending_distinct_rescue", None)
                    if (
                        rescue_context
                        and rescue_context.get("policy", {}).get("rescue_branch") == "solid_direct_upper"
                    ):
                        if solid_lower_rescue_attempted:
                            rescue_context = None
                        else:
                            # Reserve the single supplemental call across all memories.
                            solid_lower_rescue_attempted = True
                    candidate = call_person_vlm_for_candidate(
                        frame=best_frame,
                        det=best_det,
                        memory_id=memory_id,
                        track_id=best_track_id,
                        frame_count=best_frame_count,
                        time_sec=best_time_sec,
                        target=target,
                        dirs=dirs,
                        args=args,
                        deadline_mono=search_deadline_mono,
                        rescue_context=rescue_context,
                    )
                    record_candidate_evaluations(candidate)
                    vlm_call_count += 1
                    used_vlm_memory_ids.add(memory_id)
                    memory["person_vlm_call_count"] = int(memory.get("person_vlm_call_count", 0) or 0) + 1
                    memory["person_vlm_last_time"] = float(time_sec)
                    memory["person_vlm_last_quality"] = float(memory.get("best_crop_score", 0.0) or 0.0)
                    if not memory.get("distinct_rescue_attempted"):
                        policy = distinct_target_rescue_policy(candidate, target, args)
                        candidate["distinct_target_rescue_policy"] = policy
                        if candidate.get("distinct_target_rescue", {}).get("attempted"):
                            memory["distinct_rescue_attempted"] = True
                        elif policy["eligible"] and not (
                            policy.get("rescue_branch") == "solid_direct_upper"
                            and solid_lower_rescue_attempted
                        ):
                            memory["pending_distinct_rescue"] = build_distinct_target_rescue_context(
                                candidate, policy
                            )
                else:
                    memory["person_vlm_call_count"] = int(memory.get("person_vlm_call_count", 0) or 0)
                best_candidate = better_candidate(best_candidate, candidate)

                accept_quality_ok, accept_quality_reasons = accepted_candidate_quality_ok(candidate, args)
                reliability_ok, reliability_reasons = accepted_candidate_reliability_ok(candidate, target, args)
                candidate_decision = str(candidate["match"].get("decision", "unknown"))
                operational_ok = not candidate_required_vlm_failures(candidate)
                if candidate.get("distinct_target_rescue", {}).get("attempted"):
                    # Retain the retry even if its score is lower than the old best.
                    distinct_target_rescue_evaluations.append({
                        **candidate_summary(candidate),
                        "quality_ok": accept_quality_ok, "reliability_ok": reliability_ok,
                        "operational_ok": operational_ok,
                        "gate_reasons": accept_quality_reasons + reliability_reasons,
                    })
                memory["vlm_done"] = True
                memory["decision"] = memory_decision_after_acceptance_gates(
                    candidate_decision,
                    args.decision_accept,
                    accept_quality_ok,
                    reliability_ok,
                    operational_ok,
                )
                memory["final_score"] = candidate["match"].get("final_score")
                memory["person_result"] = candidate["person_result"]
                memory["result_json"] = {
                    "target": target,
                    "observed": candidate["observed"],
                    "match": candidate["match"],
                }
                if accepted_decision(candidate_decision, args.decision_accept) and accept_quality_ok and reliability_ok and operational_ok:
                    accepted_candidates_by_memory[memory_id] = candidate
                    if args.check_front_view and call_budget_available(front_vlm_call_count, args.max_front_vlm_calls):
                        candidate = prepare_front_evidence_candidate(
                            candidate,
                            dirs=dirs,
                            args=args,
                            frame_detections=best_frame_dets,
                            crop_label="front_initial",
                        )
                        candidate["front_result"] = call_front_photo_vlm(
                            candidate,
                            args,
                            deadline_mono=search_deadline_mono,
                        )
                        record_front_check(candidate["front_result"])
                        front_check_counts_by_memory[memory_id] = int(front_check_counts_by_memory.get(memory_id, 0)) + 1
                        front_last_checked_time[memory_id] = float(candidate.get("time_sec", time_sec) or time_sec)
                        front_last_checked_frame[memory_id] = int(candidate.get("frame", frame_count) or frame_count)
                        front_retry_last_view = (candidate.get("front_result", {}).get("front_view") or {}).get("value", "unknown")
                    else:
                        candidate["front_result"] = {
                            "front_view": {"value": "front" if not args.check_front_view else "unknown", "confidence": "low"},
                            "photo_usable_for_ms2": {"value": not args.check_front_view, "confidence": "low"},
                            "reason": "front_view_check_disabled" if not args.check_front_view else "front_vlm_budget_exhausted",
                        }
                    accepted_candidates_by_memory[memory_id] = candidate
                    best_person_candidate = better_ms2_candidate(best_person_candidate, candidate)
                    if front_photo_ok(candidate.get("front_result"), args):
                        if remember_front_photo_candidate(candidate):
                            if best_photo_hold_sec() <= 0.0:
                                save_best_front_photo("FOUND_FRONT_PHOTO")
                                break
                            stop_reason = "FOUND_FRONT_PHOTO_HOLDING_BEST"
                    if not args.continue_after_not_front:
                        found_result = save_found_artifacts(
                            dirs=dirs,
                            frame=memory.get("best_vlm_frame") if isinstance(memory.get("best_vlm_frame"), np.ndarray) else frame,
                            candidate=candidate,
                            target=target,
                            args=args,
                            status="FOUND_BUT_NOT_FRONT",
                        )
                        stop_reason = "FOUND_BUT_NOT_FRONT"
                        break
                elif accepted_decision(candidate_decision, args.decision_accept):
                    candidate["accepted_quality_gate"] = {
                        "accepted": False,
                        "reasons": accept_quality_reasons + reliability_reasons + (
                            ["required_vlm_evaluation_failed"] if not operational_ok else []
                        ),
                    }
            if stop_requested():
                stop_reason = "EXTERNAL_STOP_REQUESTED"
                break
            if (
                found_result is None
                and not stop_requested()
                and subject_type == "mannequin"
                and bool(getattr(args, "enable_mannequin_full_frame_fallback", True))
                and len(dets) == 0
                and call_budget_available(vlm_call_count, args.max_total_vlm_calls)
                and time_sec - last_mannequin_fallback_time >= float(getattr(args, "mannequin_fallback_interval_sec", 3.0))
            ):
                last_mannequin_fallback_time = time_sec
                mannequin_fallback_count += 1
                fallback_result = call_mannequin_full_frame_fallback(
                    frame=frame,
                    frame_count=frame_count,
                    time_sec=time_sec,
                    target=target,
                    dirs=dirs,
                    args=args,
                    deadline_mono=search_deadline_mono,
                )
                evaluation = fallback_result["evaluation"]
                record_vlm_evaluation(
                    vlm_health,
                    str(evaluation["stage"]),
                    str(evaluation["state"]),
                )
                if evaluation["state"] == VLM_EVAL_REQUIRED_FAILURE:
                    vlm_error_diagnostics.append(dict(evaluation))
                candidate = fallback_result["candidate"]
                vlm_call_count += 1
                if candidate is not None:
                    for stage, state in candidate.get("vlm_evaluations", {}).items():
                        if stage != "mannequin_full_frame":
                            record_vlm_evaluation(vlm_health, str(stage), str(state))
                    mannequin_fallback_success_count += 1
                    best_candidate = better_candidate(best_candidate, candidate)
                    accept_quality_ok, accept_quality_reasons = accepted_candidate_quality_ok(candidate, args)
                    reliability_ok, reliability_reasons = accepted_candidate_reliability_ok(candidate, target, args)
                    if accepted_decision(str(candidate["match"].get("decision", "unknown")), args.decision_accept) and accept_quality_ok and reliability_ok:
                        if args.check_front_view and call_budget_available(front_vlm_call_count, args.max_front_vlm_calls):
                            candidate = prepare_front_evidence_candidate(
                                candidate,
                                dirs=dirs,
                                args=args,
                                frame_detections=dets,
                                crop_label="front_mannequin",
                            )
                            candidate["front_result"] = call_front_photo_vlm(
                                candidate,
                                args,
                                deadline_mono=search_deadline_mono,
                            )
                            record_front_check(candidate["front_result"])
                            front_retry_last_view = (candidate.get("front_result", {}).get("front_view") or {}).get("value", "unknown")
                        else:
                            candidate["front_result"] = {
                                "front_view": {"value": "front" if not args.check_front_view else "unknown", "confidence": "low"},
                                "photo_usable_for_ms2": {"value": not args.check_front_view, "confidence": "low"},
                                "reason": "front_view_check_disabled" if not args.check_front_view else "front_vlm_budget_exhausted",
                            }
                        best_person_candidate = better_ms2_candidate(best_person_candidate, candidate)
                        if front_photo_ok(candidate.get("front_result"), args):
                            if remember_front_photo_candidate(candidate):
                                if best_photo_hold_sec() <= 0.0:
                                    save_best_front_photo("FOUND_FRONT_PHOTO")
                                else:
                                    stop_reason = "FOUND_FRONT_PHOTO_HOLDING_BEST"
                    elif accepted_decision(str(candidate["match"].get("decision", "unknown")), args.decision_accept):
                        candidate["accepted_quality_gate"] = {
                            "accepted": False,
                            "reasons": accept_quality_reasons + reliability_reasons,
                        }
            if (
                found_result is None
                and best_front_photo_candidate is not None
                and first_front_photo_time_sec is not None
                and best_photo_hold_sec() > 0.0
                and time_sec - first_front_photo_time_sec >= best_photo_hold_sec()
            ):
                save_best_front_photo("FOUND_FRONT_PHOTO_BEST_SELECTED")
            if found_result is not None:
                break
    except SearchDeadlineReached:
        stop_reason = "SEARCH_DEADLINE_REACHED"
    except SearchStopRequested:
        stop_reason = "EXTERNAL_STOP_REQUESTED"
    finally:
        if search_video_segment is not None:
            search_recording_summary = search_video_segment.close()
        cap.release()
        if preview_manager is not None:
            preview_manager.deactivate("search")
            if owns_preview_manager:
                preview_manager.shutdown()

    if found_result is None and best_front_photo_candidate is not None:
        save_best_front_photo("FOUND_FRONT_PHOTO_BEST_AT_END")

    if found_result is None and best_person_candidate is not None:
        evidence_frame = best_person_candidate.get("source_frame")
        if not isinstance(evidence_frame, np.ndarray):
            evidence_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        found_result = save_found_artifacts(
            dirs=dirs,
            frame=evidence_frame,
            candidate=best_person_candidate,
            target=target,
            args=args,
            status="FOUND_BUT_NOT_FRONT",
        )
        stop_reason = "FOUND_BUT_NOT_FRONT"

    if found_result is None:
        result = {
            "milestone": MILESTONE,
            "found": False,
            "person_found": False,
            "ms1_success": False,
            "ms2_success": False,
            "front_photo_saved": False,
            "decision": "not_found",
            "best_candidate": candidate_summary(best_candidate),
            "searched_from_sec": round(searched_from_sec, 3),
            "searched_to_sec": round(searched_to_sec, 3),
            "search_until_found": bool(search_until_found),
            "max_search_runtime_sec": max_search_runtime_sec,
            "vlm_call_count": vlm_call_count,
            "front_vlm_call_count": front_vlm_call_count,
            "front_fast_check_count": front_fast_check_count,
            "front_fast_accept_count": front_fast_accept_count,
            "front_vlm_skipped_by_fast_gate_count": front_vlm_skipped_by_fast_gate_count,
            "camera_open_tried": camera_open_tried,
            "camera_read_fail_count": camera_read_fail_count,
            "vlm_send_candidate_count": vlm_send_candidate_count,
            "vlm_retry_count": vlm_retry_count,
            "heuristic_person_candidate_count": heuristic_person_candidate_count,
            "heuristic_person_accept_count": heuristic_person_accept_count,
            "vlm_skip_reason_counts": vlm_skip_reason_counts,
            "vlm_attempt_skip_reason_counts": vlm_attempt_skip_reason_counts,
            "vlm_send_reason_counts": vlm_send_reason_counts,
            "last_vlm_skip_reason": last_vlm_skip_reason,
            "mannequin_fallback_count": mannequin_fallback_count,
            "mannequin_fallback_success_count": mannequin_fallback_success_count,
            "status": "NOT_FOUND",
        }
    else:
        result = found_result

    if bool(result.get("ms2_success")):
        selected_candidate = best_front_photo_candidate
    elif bool(result.get("found")):
        selected_candidate = best_person_candidate
    else:
        selected_candidate = None
    terminal_error = terminal_search_error(
        stop_reason=stop_reason,
        vlm_health=vlm_health,
        selected_candidate=selected_candidate,
        ms2_success=bool(result.get("ms2_success")),
    )
    if terminal_error is not None:
        result = {
            "milestone": MILESTONE,
            "status": "ERROR",
            "found": False,
            "person_found": False,
            "ms1_success": False,
            "ms2_success": False,
            "front_photo_saved": False,
            "decision": "error",
            "best_candidate": candidate_summary(best_candidate),
            "stop_reason": stop_reason,
            "vlm_health": vlm_health,
            "vlm_error_diagnostics": vlm_error_diagnostics,
            "terminal_error_reason": terminal_error["error_message"],
            **terminal_error,
        }

    result["distinct_target_rescue_evaluations"] = distinct_target_rescue_evaluations
    save_non_success_evidence(
        result, best_person_candidate or best_candidate,
        dirs=dirs, target=target, args=args, detections_total=detections_total,
    )
    run_finished = dt.datetime.now()
    result_json_path = dirs["root"] / "milestone2_result.json"
    summary_json_path = dirs["root"] / "summary.json"
    result.update({
        "out_dir": str(dirs["root"]),
        "result_json_path": str(result_json_path),
        "summary_json_path": str(summary_json_path),
        "stop_reason": stop_reason,
        "vlm_health": vlm_health,
        "vlm_error_diagnostics": vlm_error_diagnostics,
        "camera_recording": search_recording_summary,
    })
    summary = {
        "search_version": SEARCH_VERSION,
        "milestone": MILESTONE,
        "out_dir": str(dirs["root"]),
        "input_mode": input_mode,
        "input_source": source_label,
        "video": str(args.video),
        "camera_index": int(args.camera_index),
        "camera_device": str(args.camera_device),
        "camera_open_tried": camera_open_tried,
        "camera_read_fail_count": camera_read_fail_count,
        "vlm_health": vlm_health,
        "vlm_error_diagnostics": vlm_error_diagnostics,
        "target": target,
        "subject_type": subject_type,
        "found": bool(result.get("found")),
        "decision": result.get("decision"),
        "final_score": result.get("final_score"),
        "searched_from_sec": round(searched_from_sec, 3),
        "searched_to_sec": round(searched_to_sec, 3),
        "search_duration_sec": round(searched_to_sec - searched_from_sec, 3),
        "search_until_found": bool(search_until_found),
        "max_search_runtime_sec": max_search_runtime_sec,
        "run_started_at": run_started.isoformat(timespec="seconds"),
        "run_finished_at": run_finished.isoformat(timespec="seconds"),
        "total_runtime_sec": round((run_finished - run_started).total_seconds(), 3),
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "memory_count": len(person_memories),
        "vlm_call_count": vlm_call_count,
        "front_vlm_call_count": front_vlm_call_count,
        "front_fast_check_count": front_fast_check_count,
        "front_fast_accept_count": front_fast_accept_count,
        "front_vlm_skipped_by_fast_gate_count": front_vlm_skipped_by_fast_gate_count,
        "vlm_send_candidate_count": vlm_send_candidate_count,
        "vlm_retry_count": vlm_retry_count,
        "heuristic_person_candidate_count": heuristic_person_candidate_count,
        "heuristic_person_accept_count": heuristic_person_accept_count,
        "fast_heuristic_allow_distinct_targets": bool(getattr(args, "fast_heuristic_allow_distinct_targets", False)),
        "vlm_skip_reason_counts": vlm_skip_reason_counts,
        "vlm_attempt_skip_reason_counts": vlm_attempt_skip_reason_counts,
        "vlm_send_reason_counts": vlm_send_reason_counts,
        "last_vlm_skip_reason": last_vlm_skip_reason,
        "mannequin_fallback_count": mannequin_fallback_count,
        "mannequin_fallback_success_count": mannequin_fallback_success_count,
        "mannequin_fallback_interval_sec": float(getattr(args, "mannequin_fallback_interval_sec", 3.0)),
        "front_retry_count": front_retry_count,
        "front_retry_success_count": front_retry_success_count,
        "front_retry_last_view": front_retry_last_view,
        "time_to_front_photo_sec": time_to_front_photo_sec,
        "best_photo_hold_sec": best_photo_hold_sec(),
        "best_photo_candidate_count": best_photo_candidate_count,
        "best_photo_selected_reason": best_photo_selected_reason,
        "first_front_photo_time_sec": round(first_front_photo_time_sec, 3) if first_front_photo_time_sec is not None else None,
        "best_front_photo_candidate": candidate_summary(best_front_photo_candidate),
        "front_photo_candidates": [candidate_summary(candidate) for candidate in front_photo_candidates],
        "front_check_counts_by_memory": {str(k): int(v) for k, v in sorted(front_check_counts_by_memory.items())},
        "max_total_vlm_calls": int(args.max_total_vlm_calls),
        "max_front_vlm_calls": int(args.max_front_vlm_calls),
        "front_retry_interval_sec": float(args.front_retry_interval_sec),
        "front_fast_local_gate": bool(getattr(args, "front_fast_local_gate", True)),
        "front_vlm_on_fast_gate_miss": bool(getattr(args, "front_vlm_on_fast_gate_miss", False)),
        "continue_after_not_front": bool(args.continue_after_not_front),
        "accept_front_views": args.accept_front_views,
        "strict_ms2_front": bool(args.strict_ms2_front),
        "front_min_confidence": args.front_min_confidence,
        "require_cv_face_for_front": bool(args.require_cv_face_for_front),
        "front_face_cascade_path": args.front_face_cascade_path,
        "front_min_bbox_aspect": float(args.front_min_bbox_aspect),
        "front_min_crop_quality": float(args.front_min_crop_quality),
        "front_min_area_ratio": float(getattr(args, "front_min_area_ratio", 0.0)),
        "front_min_height_ratio": float(getattr(args, "front_min_height_ratio", 0.0)),
        "front_crop_margin": float(getattr(args, "front_crop_margin", getattr(args, "person_crop_margin", 0.45))),
        "front_reject_edge_crops": bool(args.front_reject_edge_crops),
        "front_allow_vertical_edge_with_face": bool(getattr(args, "front_allow_vertical_edge_with_face", True)),
        "front_vertical_edge_face_min_quality": float(getattr(args, "front_vertical_edge_face_min_quality", 0.60)),
        "camera_read_fail_tolerance": int(getattr(args, "camera_read_fail_tolerance", 30) or 30),
        "accept_crop_heuristic_for_distinct_targets": bool(getattr(args, "accept_crop_heuristic_for_distinct_targets", False)),
        "ms2_use_feature_person_prompt": bool(args.ms2_use_feature_person_prompt),
        "ms2_relaxed_color_feature_type_match": bool(args.ms2_relaxed_color_feature_type_match),
        "ms2_enable_crop_attribute_rescue": bool(getattr(args, "ms2_enable_crop_attribute_rescue", True)),
        "ms2_crop_rescue_min_score": float(getattr(args, "ms2_crop_rescue_min_score", 0.50)),
        "ms2_crop_rescue_min_gain": float(getattr(args, "ms2_crop_rescue_min_gain", 0.10)),
        "accepted_memory_count": len(accepted_candidates_by_memory),
        "decision_accept": args.decision_accept,
        "stop_reason": stop_reason,
        "ms1_success": bool(result.get("ms1_success")),
        "ms2_success": bool(result.get("ms2_success")),
        "person_found": bool(result.get("person_found", result.get("found"))),
        "front_photo_saved": bool(result.get("front_photo_saved")),
        "front_view": result.get("front_view"),
        "photo_usable_for_ms2": result.get("photo_usable_for_ms2"),
        "best_candidate": candidate_summary(best_candidate),
        "best_person_candidate": candidate_summary(best_person_candidate),
        "not_found_reason": result.get("not_found_reason"),
        "candidate_reject_gate": result.get("candidate_reject_gate"),
        "best_candidate_evidence": result.get("best_candidate_evidence"),
        "enable_distinct_target_rescue": bool(getattr(args, "enable_distinct_target_rescue", True)),
        "distinct_target_rescue_evaluations": distinct_target_rescue_evaluations,
        "camera_recording": search_recording_summary,
    }
    write_json(result_json_path, result)
    write_json(summary_json_path, summary)
    print(f"out_dir: {dirs['root']}")
    print(f"milestone2_result: {dirs['root'] / 'milestone2_result.json'}")
    print(f"found: {result.get('found')} / decision: {result.get('decision')} / score: {result.get('final_score')}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Milestone 2 simulated person search")
    parser.add_argument("--video", default="")
    parser.add_argument("--camera_index", type=int, default=-1)
    parser.add_argument("--camera_device", default="")
    parser.add_argument("--camera_width", type=int, default=0)
    parser.add_argument("--camera_height", type=int, default=0)
    parser.add_argument("--camera_fps", type=float, default=0.0)
    parser.add_argument("--camera_read_fail_tolerance", type=int, default=30)
    parser.add_argument(
        "--record_search_video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--record_video_fps", type=float, default=0.0)
    parser.add_argument("--search_video_path", default="")
    parser.add_argument("--search_video_dir", default="")
    parser.add_argument("--search_video_role_stem", default="")
    parser.add_argument("--camera_video_manifest_path", default="")
    parser.add_argument("--recording_area_index", type=int, default=None)
    parser.add_argument("--recording_target_id", default="")
    parser.add_argument("--subject_type", choices=["auto", "person", "mannequin"], default="auto")
    parser.add_argument("--target_upper_color", default="none")
    parser.add_argument("--target_upper_secondary_color", default="none")
    parser.add_argument("--target_upper_pattern", default="none")
    parser.add_argument("--target_upper_type", default="none")
    parser.add_argument("--target_lower_color", default="none")
    parser.add_argument("--target_lower_secondary_color", default="none")
    parser.add_argument("--target_lower_pattern", default="none")
    parser.add_argument("--target_lower_type", default="none")
    parser.add_argument("--target_gender", default="none")
    parser.add_argument("--target_hair", default="none")
    parser.add_argument("--target_headwear_type", default="none")
    parser.add_argument("--target_item", default="unspecified")
    parser.add_argument("--area_arrival_sec", type=float, default=0.0)
    parser.add_argument("--search_duration_sec", type=float, default=12.0)
    parser.add_argument("--search_until_found", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_search_runtime_sec", type=float, default=0.0)
    parser.add_argument("--max_total_vlm_calls", type=int, default=6)
    parser.add_argument("--decision_accept", choices=["strong_candidate", "candidate", "weak_candidate"], default="strong_candidate")
    parser.add_argument("--save_annotated_image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmstudio_base_url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--ms2_use_feature_person_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ms2_relaxed_color_feature_type_match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ms2_enable_crop_attribute_rescue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_distinct_target_rescue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ms2_crop_rescue_min_score", type=float, default=0.50)
    parser.add_argument("--ms2_crop_rescue_min_gain", type=float, default=0.10)
    parser.add_argument("--wide_crop_attribute_rescue_min_score", type=float, default=0.75)
    parser.add_argument("--wide_crop_attribute_rescue_min_gain", type=float, default=0.08)
    parser.add_argument("--check_front_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accept_front_views", default="front")
    parser.add_argument("--strict_ms2_front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_min_confidence", choices=["low", "medium", "high", "none"], default="medium")
    parser.add_argument("--require_cv_face_for_front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_face_cascade_path", default=DEFAULT_FACE_CASCADE_PATH)
    parser.add_argument("--front_face_min_size_px", type=int, default=18)
    parser.add_argument("--front_face_scale_factor", type=float, default=1.05)
    parser.add_argument("--front_face_min_neighbors", type=int, default=3)
    parser.add_argument("--front_min_bbox_aspect", type=float, default=0.0)
    parser.add_argument("--front_min_crop_quality", type=float, default=0.60)
    parser.add_argument("--front_min_area_ratio", type=float, default=0.06)
    parser.add_argument("--front_min_height_ratio", type=float, default=0.35)
    parser.add_argument("--front_crop_margin", type=float, default=0.12)
    parser.add_argument("--front_reject_edge_crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_allow_vertical_edge_with_face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_vertical_edge_face_min_quality", type=float, default=0.60)
    parser.add_argument("--best_photo_hold_sec", type=float, default=2.0)
    parser.add_argument("--max_front_vlm_calls", type=int, default=4)
    parser.add_argument("--front_retry_interval_sec", type=float, default=1.0)
    parser.add_argument("--continue_after_not_front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_fast_local_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_vlm_on_fast_gate_miss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--front_fast_min_skin_area_ratio", type=float, default=0.018)
    parser.add_argument("--front_fast_min_skin_width_ratio", type=float, default=0.12)
    parser.add_argument("--front_fast_min_skin_height_ratio", type=float, default=0.10)
    parser.add_argument("--allow_skin_front_when_cv_face_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skin_front_cv_face_missing_min_quality", type=float, default=0.55)
    parser.add_argument("--use_fast_heuristic_person_match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_heuristic_allow_distinct_targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast_heuristic_accept_score", type=float, default=0.75)
    parser.add_argument("--fast_heuristic_min_quality", type=float, default=0.45)
    parser.add_argument("--fast_heuristic_min_area_ratio", type=float, default=0.012)
    parser.add_argument("--accepted_candidate_min_quality", type=float, default=0.45)
    parser.add_argument("--accepted_candidate_min_area_ratio", type=float, default=0.012)
    parser.add_argument("--accept_crop_heuristic_for_distinct_targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_person_vlm_calls_per_memory", type=int, default=2)
    parser.add_argument("--person_vlm_retry_interval_sec", type=float, default=2.0)
    parser.add_argument("--person_vlm_retry_min_quality_gain", type=float, default=0.08)
    parser.add_argument("--enable_mannequin_front_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mannequin_front_fallback_min_score", type=float, default=0.85)
    parser.add_argument("--mannequin_front_fallback_min_quality", type=float, default=0.70)
    parser.add_argument("--mannequin_front_fallback_min_area_ratio", type=float, default=0.06)
    parser.add_argument("--enable_mannequin_full_frame_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mannequin_fallback_interval_sec", type=float, default=3.0)
    parser.add_argument("--show_search_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search_window_name", default="ms2_person_search_camera")
    parser.add_argument("--search_window_width", type=int, default=1440)
    parser.add_argument("--search_window_height", type=int, default=810)
    parser.add_argument("--search_window_x", type=int, default=180)
    parser.add_argument("--search_window_y", type=int, default=80)
    parser.add_argument("--search_window_topmost", action=argparse.BooleanOptionalAction, default=True)

    # Practical defaults inherited from v3.9.7.
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--yolo_model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--process_every", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min_area_ratio_for_vlm", type=float, default=0.07)
    parser.add_argument("--min_height_ratio_for_vlm", type=float, default=0.25)
    parser.add_argument("--min_conf_for_vlm", type=float, default=0.45)
    parser.add_argument("--min_width_ratio_for_vlm", type=float, default=0.10)
    parser.add_argument("--vlm_min_track_frames", type=int, default=1)
    parser.add_argument("--enable_person_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--person_crop_margin", type=float, default=0.45)
    parser.add_argument("--attribute_crop_margin", type=float, default=0.0)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--person_vlm_long_side", type=int, default=384)
    parser.add_argument("--items_vlm_long_side", type=int, default=384)
    parser.add_argument("--person_max_tokens", type=int, default=260)
    parser.add_argument("--items_max_tokens", type=int, default=110)
    parser.add_argument("--lmstudio_timeout", type=int, default=8)
    parser.add_argument("--front_vlm_timeout", type=int, default=2)
    parser.add_argument("--front_max_tokens", type=int, default=140)
    parser.add_argument("--vlm_model", default="gemma-4-e2b-it-qat")
    parser.add_argument("--min_crop_quality_for_vlm", type=float, default=0.35)
    parser.add_argument("--target_color_send_threshold", type=float, default=0.05)
    parser.add_argument("--high_quality_send_threshold", type=float, default=0.55)

    # Values required by reused helpers.
    parser.add_argument("--best_crop_edge_margin_ratio", type=float, default=0.03)
    parser.add_argument("--best_crop_preferred_width_ratio", type=float, default=0.22)
    parser.add_argument("--best_crop_preferred_height_ratio", type=float, default=0.62)
    parser.add_argument("--best_crop_preferred_area_ratio", type=float, default=0.18)
    parser.add_argument("--item_margin_x", type=float, default=-1.0)
    parser.add_argument("--item_margin_top", type=float, default=-1.0)
    parser.add_argument("--item_margin_bottom", type=float, default=-1.0)
    parser.add_argument("--no_draw_target_box", action="store_true")
    return parser


def main() -> None:
    def _handle_stop(_signum, _frame) -> None:
        request_stop()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    args = build_parser().parse_args()
    run_milestone2_search(args)


if __name__ == "__main__":
    main()
