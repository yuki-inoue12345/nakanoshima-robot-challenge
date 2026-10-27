#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
person_search_vlm.py

YOLO + LM Studio / Gemma QAT を用いた人物探索プロトタイプ。

できること:
- 画像 / 動画 / カメラ入力に対応
- YOLOで人物検出・track_id追跡
- 1段目: person-only VLMで服装・人物属性を抽出
- 2段目: target_item が none 以外のときだけ item-context VLMで持ち物を抽出
- 入力条件を target.json として保存
- person JSON / items JSON / result JSON を分けて保存
- target と observed を比較し、person_score / item_score / final_score / decision を出す
- track_idとは別にperson_memoryを持ち、同一人物ならVLMへ再問い合わせしない

注意:
- 探索対象の文字情報はLLMに投げない。argparseで受け取った値をPython側でJSON化するだけ。
- 顔認証はまだ未統合。今版は服装 + 持ち物 + track_idキャッシュまで。
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime as dt
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# Version
# =========================================================
PERSON_PROMPT_VERSION = "person_only_from_fixed9_v5_no_view_angle_body_visible_only"
ITEMS_PROMPT_VERSION = "items_context_attr_v3_fast_negative_guard"
SEARCH_VERSION = "person_search_vlm_v3_9_7_final_wait_reduce"

CONF_VALUES = {"low", "medium", "high"}

SUPPORTED_ITEM_TARGETS = {
    "suitcase", "umbrella", "phone", "shopping_bag", "paper_bag", "bottle",
    "backpack", "shoulder_bag", "handbag", "tote_bag", "bag",
    # absence is also a real search condition
    "none", "no_item", "no_belongings",
}
NO_ITEM_TARGETS = {"none", "no_item", "no_belongings", "nothing"}
SKIP_ITEM_TARGETS = {"unspecified", "any", "unknown", ""}
BAG_LIKE_TYPES = {"backpack", "shoulder_bag", "handbag", "tote_bag", "paper_bag", "bag"}
CONCRETE_ITEM_TYPES = {"suitcase", "umbrella", "phone", "shopping_bag", "paper_bag", "bottle"}
CONCRETE_BAG_TYPES = {"backpack", "shoulder_bag", "handbag", "tote_bag", "paper_bag"}

ALLOWED_COLORS = {
    "none", "black", "blue", "brown", "gray", "white", "pink", "red", "green",
    "yellow", "navy", "tan", "beige", "orange", "purple", "dark", "light", "grey",
    "light_blue", "khaki", "olive", "cream", "unknown",
}
ALLOWED_UPPER_TYPES = {
    "hoodie", "jacket", "coat", "sweater", "sweatshirt", "shirt", "t_shirt",
    "t-shirt", "vest", "cardigan", "polo_shirt", "unknown",
}
ALLOWED_LOWER_TYPES = {"pants", "trousers", "jeans", "shorts", "skirt", "dress", "unknown"}
ALLOWED_PATTERNS = {"solid", "striped", "plaid", "graphic", "floral", "color_block", "unknown"}
ALLOWED_HEADWEAR = {"none", "cap", "hat", "hood", "helmet", "unknown"}
ALLOWED_GENDER = {"male", "female", "unknown"}
ALLOWED_HAIR = {"short", "medium", "long", "unknown"}


# =========================================================
# Prompts
# =========================================================
PERSON_PROMPT = """
Fast PERSON-ONLY attribute JSON extractor.
Input: cropped image with one main person.
Output: compact valid JSON only. No markdown. No text.

Extract ONLY person visual attributes.
Do NOT extract bags, carried items, suitcases, umbrellas, phones, bottles, or other objects.
Do NOT add an items field.
Do NOT add view_angle.

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
  "upper_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/green/olive/cream/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "upper_secondary_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/green/olive/cream/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "upper_pattern":{"value":"solid/striped/plaid/graphic/floral/color_block/unknown","confidence":"high/medium/low"},
  "upper_type":{"value":"hoodie/jacket/coat/sweater/sweatshirt/shirt/t-shirt/t_shirt/vest/cardigan/polo_shirt/unknown","confidence":"high/medium/low"},
  "lower_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/green/olive/cream/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "lower_secondary_color":{"value":"black/white/gray/grey/dark/light-blue/blue/navy/beige/tan/khaki/brown/green/olive/cream/pink/red/yellow/unknown","confidence":"high/medium/low"},
  "lower_pattern":{"value":"solid/striped/plaid/graphic/floral/color_block/unknown","confidence":"high/medium/low"},
  "lower_type":{"value":"pants/trousers/jeans/shorts/skirt/dress/unknown","confidence":"high/medium/low"}
 },
 "visibility":{
  "body_visible":"full_body/upper_body/partial_body/face_only/unknown"
 }
}

body_visible rules:
- full_body = head, torso, legs, and feet are inside the crop.
- full_body is still correct even if a suitcase/bag/object overlaps small parts.
- partial_body = important body parts are cut off by the crop boundary.
- upper_body = legs/feet are not visible.
- face_only = only face/head is visible.

Other rules:
- Use only the main person in the crop. Ignore other people.
- upper_type/lower_type are clothing only.
- Use upper_secondary_color/lower_secondary_color only for clear two-color clothing such as stripes, plaid, or color blocking; otherwise use unknown.
- Use upper_pattern/lower_pattern: solid=no visible pattern, striped=border/stripes, plaid=check/grid/plaid, graphic=printed graphic/text, floral=flower pattern, color_block=large blocks of different colors.
- striped and plaid are different. Do not mix them.
- Never put bag/backpack/suitcase/umbrella/phone/bottle/cap/hat/helmet/object into upper_type or lower_type.
- If clothing is hidden by an object, another person, or crop boundary, use unknown.
- age/gender are rough visual categories only. Use unknown if unclear.
""".strip()


def build_items_prompt(target_item: str) -> str:
    target_item = normalize_token(target_item or "unspecified")

    if target_item in NO_ITEM_TARGETS:
        target_hint = (
            "Target item: none / no belongings. "
            "Decide whether the MAIN PERSON has NO visible bag and NO visible carried item. "
            "Set has_bag=false and bag_type=none only when no backpack/shoulder_bag/handbag/tote/paper_bag is visible on or near the main person. "
            "Set has_item=false and item_type=none only when no suitcase/umbrella/phone/shopping_bag/paper_bag/bottle is clearly held, pulled, worn, or near the main person. "
            "Do NOT mark has_item=true for empty hands, arms, clothing folds, shadows, background edges, or ambiguous small regions. "
            "If an object is not clearly identifiable as a carried item, prefer has_item=false and item_type=none."
        )
    elif target_item and target_item not in SKIP_ITEM_TARGETS:
        target_hint = (
            f"Target item: {target_item}. "
            "Decide if this item is an attribute of the MAIN PERSON: worn, held, pulled, or clearly near them. "
            "For suitcase/luggage, check handle, wheels, lower-left/lower-right, and feet/side area. "
            "For umbrella, check hand area, upper area, and long dark/bright canopy/stick. "
        )
    else:
        target_hint = "No specific target item. Check only belongings associated with the MAIN PERSON."

    return f"""Analyze PERSON-ITEM context for person search.

MAIN PERSON = person inside the green rectangle.
Ignore belongings of other people.
Count only objects attached to, held by, pulled by, or clearly near the main person.

{target_hint}

Return compact JSON only, using scalar values:
{{
  "items": {{
    "bag": {{
      "has_bag": true/false,
      "bag_type": "none/backpack/shoulder_bag/handbag/tote_bag/paper_bag/unknown",
      "bag_color": "none/black/blue/brown/gray/white/pink/red/green/yellow/navy/tan/beige/dark/light/unknown",
      "bag_relation": "none/on_back/on_shoulder/held/near_person/worn/unknown"
    }},
    "carried_item": {{
      "has_item": true/false,
      "item_type": "none/suitcase/umbrella/phone/shopping_bag/paper_bag/bottle/unknown",
      "item_relation": "none/held/near_person/pulled/worn/unknown"
    }}
  }}
}}

Rules:
- backpack -> bag_type=backpack, bag_relation=on_back.
- shoulder/crossbody bag -> bag_type=shoulder_bag, bag_relation=on_shoulder.
- handbag/tote/paper bag held by main person -> bag relation held.
- suitcase/luggage pulled or near feet/side -> carried_item suitcase, relation pulled or near_person.
- umbrella in/near hand -> carried_item umbrella, relation held.
- phone clearly in hand -> carried_item phone, relation held.
- If no associated item, use false and none.
JSON only."""


# =========================================================
# General utilities
# =========================================================
def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def safe_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def normalize_token(value: Any) -> str:
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    if not v:
        return "unknown"
    v = v.replace(" ", "_").replace("-", "_")
    aliases = {
        "grey": "gray",
        "light_blue": "light_blue",
        "lightblue": "light_blue",
        "tshirt": "t_shirt",
        "tee_shirt": "t_shirt",
        "t_shirt": "t_shirt",
        "t-shirt": "t_shirt",
        "trouser": "pants",
        "trousers": "pants",
        "suit_case": "suitcase",
        "luggage": "suitcase",
        "rolling_bag": "suitcase",
        "carry_on": "suitcase",
        "cell_phone": "phone",
        "smartphone": "phone",
        "mobile_phone": "phone",
        "shoppingbag": "shopping_bag",
        "paperbag": "paper_bag",
        "rucksack": "backpack",
        "back_pack": "backpack",
        "shoulderbag": "shoulder_bag",
        "crossbody_bag": "shoulder_bag",
        "none_": "none",
        "no": "none",
        "false": "none",
        "not_present": "none",
    }
    return aliases.get(v, v)


def make_vc(value: Any, confidence: str = "medium") -> Dict[str, Any]:
    if confidence not in CONF_VALUES:
        confidence = "medium"
    if isinstance(value, str):
        value = normalize_token(value)
    return {"value": value, "confidence": confidence}


def vc_get(obj: Any, default_value: Any, default_conf: str = "medium") -> Dict[str, Any]:
    if isinstance(obj, dict):
        val = obj.get("value", default_value)
        conf = obj.get("confidence", default_conf)
        if conf not in CONF_VALUES:
            conf = default_conf
        return make_vc(val, conf)
    if obj is None:
        return make_vc(default_value, default_conf)
    return make_vc(obj, default_conf)


def normalize_bool_vc(v: Any, default: bool = False, default_conf: str = "medium") -> Dict[str, Any]:
    vc = vc_get(v, default, default_conf)
    val = vc["value"]
    if isinstance(val, str):
        low = normalize_token(val)
        if low in {"true", "yes", "1"}:
            val = True
        elif low in {"false", "no", "0", "none"}:
            val = False
        else:
            val = default
    elif not isinstance(val, bool):
        val = default
    return {"value": val, "confidence": vc["confidence"]}


def normalize_enum_vc(v: Any, allowed: set, default: str = "unknown", default_conf: str = "medium") -> Dict[str, Any]:
    vc = vc_get(v, default, default_conf)
    val = normalize_token(vc["value"])
    conf = vc["confidence"]
    if val not in allowed:
        val = default
        conf = "low"
    return make_vc(val, conf)


def best_conf(*confs: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    best = "low"
    for c in confs:
        if c in rank and rank[c] > rank[best]:
            best = c
    return best


def image_file_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    if long_side <= 0:
        return image
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= long_side:
        return image
    scale = long_side / current
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def crop_person(frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.08) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    cx1 = clamp(x1 - mx, 0, w - 1)
    cy1 = clamp(y1 - my, 0, h - 1)
    cx2 = clamp(x2 + mx, 0, w - 1)
    cy2 = clamp(y2 + my, 0, h - 1)
    if cx2 <= cx1:
        cx2 = min(w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(h, cy1 + 1)
    return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


def get_target_item_margins(target_item: str) -> Tuple[float, float, float]:
    t = normalize_token(target_item or "none")
    if t == "suitcase":
        return 1.25, 0.15, 0.90
    if t == "umbrella":
        return 0.55, 0.45, 0.45
    if t in {"shopping_bag", "paper_bag", "handbag", "tote_bag"}:
        return 0.55, 0.18, 0.65
    if t in {"phone", "bottle"}:
        return 0.35, 0.15, 0.30
    if t in {"backpack", "shoulder_bag", "bag"}:
        return 0.45, 0.25, 0.40
    return 0.60, 0.25, 0.65


def make_item_context_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_x: float,
    margin_top: float,
    margin_bottom: float,
    draw_target_box: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx1 = clamp(int(x1 - bw * margin_x), 0, w - 1)
    cx2 = clamp(int(x2 + bw * margin_x), 0, w - 1)
    cy1 = clamp(int(y1 - bh * margin_top), 0, h - 1)
    cy2 = clamp(int(y2 + bh * margin_bottom), 0, h - 1)
    if cx2 <= cx1:
        cx2 = min(w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(h, cy1 + 1)

    crop = image[cy1:cy2, cx1:cx2].copy()
    if draw_target_box:
        rx1 = clamp(x1 - cx1, 0, crop.shape[1] - 1)
        ry1 = clamp(y1 - cy1, 0, crop.shape[0] - 1)
        rx2 = clamp(x2 - cx1, 0, crop.shape[1] - 1)
        ry2 = clamp(y2 - cy1, 0, crop.shape[0] - 1)
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
        cv2.putText(crop, "target person", (rx1, max(20, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return crop, (cx1, cy1, cx2, cy2)


# =========================================================
# JSON parsing and normalization
# =========================================================
def _repair_common_json_text(candidate: str) -> Tuple[str, List[str]]:
    """
    VLMがよく出す軽いJSON崩れだけを補正する。
    内容の推測はしない。形式だけを直す。
    """
    s = (candidate or "").strip()
    notes: List[str] = []

    # code fence / BOM / 全角記号 / スマートクォート
    before = s
    s = s.lstrip("\ufeff")
    s = re.sub(r"^\s*```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s).strip()
    replacements = {
        "，": ",", "：": ":", "｛": "{", "｝": "}", "［": "[", "］": "]",
        "“": '"', "”": '"', "＂": '"',
    }
    for old_ch, new_ch in replacements.items():
        s = s.replace(old_ch, new_ch)
    if s != before:
        notes.append("json_repaired_common_symbols")

    # Python風のリテラルが混ざるケースへの保険
    before = s
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if s != before:
        notes.append("json_repaired_python_literals")

    # 末尾カンマ:  ,} / ,] を削除
    before = s
    s = re.sub(r",\s*([}\]])", r"\1", s)
    if s != before:
        notes.append("json_repaired_trailing_commas")

    # 行末カンマ抜け: } の次の行に "key": が来る典型例だけ補正
    # かなり壊れたJSONを無理に直すのではなく、隣接するキー間だけに限定する。
    before = s
    s = re.sub(r'([}\]"0-9a-zA-Z_])\s*\n\s*("[A-Za-z0-9_]+"\s*:)', r'\1,\n\2', s)
    if s != before:
        notes.append("json_repaired_missing_commas_between_lines")

    return s, notes


def _loads_json_with_optional_repair(candidate: str, note_prefix: str = "") -> Optional[Dict[str, Any]]:
    candidate = (candidate or "").strip()
    if not candidate:
        return None

    def add_note(obj: Dict[str, Any], note: str) -> Dict[str, Any]:
        if note:
            obj.setdefault("_postprocess_notes", []).append(note)
        return obj

    def try_load(s: str, note: str = "", allow_ast: bool = True) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
            return add_note(obj, note) if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        # single quote / Python dict風の出力への軽い保険。コード実行はせずliteralだけ読む。
        if allow_ast:
            try:
                obj = ast.literal_eval(s)
                return add_note(obj, note or f"{note_prefix}json_repaired_python_literal_eval") if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    obj = try_load(candidate)
    if obj is not None:
        return obj

    common, common_notes = _repair_common_json_text(candidate)
    if common != candidate:
        obj = try_load(common, f"{note_prefix}{'+'.join(common_notes)}")
        if obj is not None:
            return obj

    # QATが object の閉じ括弧を ] にする軽い崩れへの保険
    repaired = re.sub(r'\]\s*,\s*("visibility"\s*:)', r'},\1', common)
    repaired = re.sub(r'\]\s*,\s*("items"\s*:)', r'},\1', repaired)
    if repaired != common:
        obj = try_load(repaired, f"{note_prefix}json_repaired_wrong_square_bracket")
        if obj is not None:
            return obj

    for base_text, prefix in [
        (repaired, f"{note_prefix}json_repaired_wrong_bracket_and_missing_closing_brace"),
        (common, f"{note_prefix}json_repaired_missing_closing_brace"),
        (candidate, f"{note_prefix}json_repaired_missing_closing_brace_raw"),
    ]:
        open_n = base_text.count("{")
        close_n = base_text.count("}")
        missing = open_n - close_n
        if 0 < missing <= 8:
            obj = try_load(base_text + ("}" * missing), f"{prefix}:{missing}")
            if obj is not None:
                return obj

    return None


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    obj = _loads_json_with_optional_repair(text)
    if obj is not None:
        return obj

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _loads_json_with_optional_repair(text[start:end + 1])
        if obj is not None:
            return obj
        obj = _loads_json_with_optional_repair(text[start:])
        if obj is not None:
            return obj

    return {"error": "JSON_DECODE_ERROR", "raw_text": text}


def normalize_person_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict) or obj.get("error"):
        return obj

    # Gemmaがvisibilityをitems内に入れた場合だけ救出
    if "visibility" not in obj and isinstance(obj.get("items"), dict) and "visibility" in obj["items"]:
        obj["visibility"] = obj["items"].get("visibility")
        obj.setdefault("_postprocess_notes", []).append("visibility_moved_from_items")

    person = obj.setdefault("person", {}) if isinstance(obj.get("person", {}), dict) else {}
    head = obj.setdefault("head", {}) if isinstance(obj.get("head", {}), dict) else {}
    par = obj.setdefault("par_attributes", {}) if isinstance(obj.get("par_attributes", {}), dict) else {}
    vis = obj.setdefault("visibility", {}) if isinstance(obj.get("visibility", {}), dict) else {}

    person["apparent_age"] = vc_get(person.get("apparent_age"), "unknown", "low")
    person["apparent_gender"] = normalize_enum_vc(person.get("apparent_gender"), ALLOWED_GENDER, "unknown", "low")
    person["hair"] = normalize_enum_vc(person.get("hair"), ALLOWED_HAIR, "unknown", "low")
    head["headwear_type"] = normalize_enum_vc(head.get("headwear_type"), ALLOWED_HEADWEAR, "unknown", "low")

    par["upper_color"] = normalize_enum_vc(par.get("upper_color"), ALLOWED_COLORS, "unknown", "low")
    par["upper_secondary_color"] = normalize_enum_vc(par.get("upper_secondary_color"), ALLOWED_COLORS, "unknown", "low")
    par["upper_pattern"] = normalize_enum_vc(par.get("upper_pattern"), ALLOWED_PATTERNS, "unknown", "low")
    par["upper_type"] = normalize_enum_vc(par.get("upper_type"), ALLOWED_UPPER_TYPES, "unknown", "low")
    par["lower_color"] = normalize_enum_vc(par.get("lower_color"), ALLOWED_COLORS, "unknown", "low")
    par["lower_secondary_color"] = normalize_enum_vc(par.get("lower_secondary_color"), ALLOWED_COLORS, "unknown", "low")
    par["lower_pattern"] = normalize_enum_vc(par.get("lower_pattern"), ALLOWED_PATTERNS, "unknown", "low")
    par["lower_type"] = normalize_enum_vc(par.get("lower_type"), ALLOWED_LOWER_TYPES, "unknown", "low")

    invalid_clothing_types = {
        "bag", "backpack", "tote_bag", "shoulder_bag", "handbag", "paper_bag",
        "suitcase", "luggage", "umbrella", "shopping_bag", "phone", "bottle",
        "cap", "hat", "hood", "helmet", "object", "item",
    }
    for key in ["upper_type", "lower_type"]:
        v = normalize_token(par[key]["value"])
        if v in invalid_clothing_types:
            par[key] = make_vc("unknown", "low")
            obj.setdefault("_postprocess_notes", []).append(f"{key}_invalid_object_fixed:{v}->unknown")

    body_visible = vis.get("body_visible", "unknown")
    if isinstance(body_visible, dict):
        body_visible = body_visible.get("value", "unknown")
    body_visible = normalize_token(body_visible)
    if body_visible not in {"full_body", "upper_body", "partial_body", "face_only", "unknown"}:
        body_visible = "unknown"

    obj["person"] = person
    obj["head"] = head
    obj["par_attributes"] = par
    obj["visibility"] = {"body_visible": body_visible}

    if "items" in obj:
        obj.pop("items", None)
        obj.setdefault("_postprocess_notes", []).append("items_removed_by_person_only")

    if isinstance(vis, dict) and "view_angle" in vis:
        obj.setdefault("_postprocess_notes", []).append("view_angle_removed_by_no_view_angle_version")

    return obj


def normalize_items_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    notes: List[str] = []
    if not isinstance(obj, dict):
        obj = {}

    if "items" not in obj and ("bag" in obj or "carried_item" in obj):
        root_notes = obj.get("_postprocess_notes", []) if isinstance(obj.get("_postprocess_notes"), list) else []
        obj = {"items": obj, "_postprocess_notes": root_notes}

    raw_notes = obj.get("_postprocess_notes", [])
    if isinstance(raw_notes, list):
        notes.extend([str(x) for x in raw_notes])

    items = obj.get("items", {}) if isinstance(obj.get("items", {}), dict) else {}
    bag = items.get("bag", {}) if isinstance(items.get("bag", {}), dict) else {}
    carried = items.get("carried_item", {}) if isinstance(items.get("carried_item", {}), dict) else {}

    bag_norm = {
        "has_bag": normalize_bool_vc(bag.get("has_bag"), False, "medium"),
        "bag_type": normalize_enum_vc(bag.get("bag_type"), {"none", "backpack", "shoulder_bag", "handbag", "tote_bag", "paper_bag", "unknown"}, "unknown", "medium"),
        "bag_color": normalize_enum_vc(bag.get("bag_color"), ALLOWED_COLORS, "unknown", "low"),
        "bag_relation": normalize_enum_vc(bag.get("bag_relation"), {"none", "on_back", "on_shoulder", "held", "near_person", "worn", "unknown"}, "unknown", "medium"),
    }
    carried_norm = {
        "has_item": normalize_bool_vc(carried.get("has_item"), False, "medium"),
        "item_type": normalize_enum_vc(carried.get("item_type"), {"none", "suitcase", "umbrella", "phone", "shopping_bag", "paper_bag", "bottle", "unknown"} | BAG_LIKE_TYPES, "unknown", "medium"),
        "item_relation": normalize_enum_vc(carried.get("item_relation"), {"none", "held", "near_person", "worn", "pulled", "on_back", "on_shoulder", "unknown"}, "unknown", "medium"),
    }

    # backpack等がcarried_item側に出た場合はbag側へ移動
    item_type_v = carried_norm["item_type"]["value"]
    if item_type_v in BAG_LIKE_TYPES:
        move_conf = best_conf(carried_norm["item_type"]["confidence"], carried_norm["has_item"]["confidence"], bag_norm["has_bag"]["confidence"])
        bag_type = "unknown" if item_type_v == "bag" else item_type_v
        bag_norm["has_bag"] = make_vc(True, move_conf)
        bag_norm["bag_type"] = make_vc(bag_type, move_conf)
        if item_type_v == "backpack":
            bag_norm["bag_relation"] = make_vc("on_back", move_conf)
        elif item_type_v == "shoulder_bag":
            bag_norm["bag_relation"] = make_vc("on_shoulder", move_conf)
        elif item_type_v in {"handbag", "tote_bag", "paper_bag"}:
            bag_norm["bag_relation"] = make_vc("held", move_conf)
        if bag_norm["bag_color"]["value"] in {"none", "unknown", None}:
            bag_norm["bag_color"] = make_vc("unknown", "low")
        carried_norm["has_item"] = make_vc(False, "high")
        carried_norm["item_type"] = make_vc("none", "high")
        carried_norm["item_relation"] = make_vc("none", "high")
        notes.append(f"misplaced_bag_item_moved_to_bag:{item_type_v}")

    bag_type_v = bag_norm["bag_type"]["value"]
    if bag_type_v in CONCRETE_BAG_TYPES:
        bag_norm["has_bag"] = make_vc(True, best_conf(bag_norm["has_bag"]["confidence"], bag_norm["bag_type"]["confidence"]))
        if bag_norm["bag_relation"]["value"] in {"none", "unknown"}:
            if bag_type_v == "backpack":
                bag_norm["bag_relation"] = make_vc("on_back", bag_norm["bag_type"]["confidence"])
            elif bag_type_v == "shoulder_bag":
                bag_norm["bag_relation"] = make_vc("on_shoulder", bag_norm["bag_type"]["confidence"])
            elif bag_type_v in {"handbag", "tote_bag", "paper_bag"}:
                bag_norm["bag_relation"] = make_vc("held", bag_norm["bag_type"]["confidence"])
        notes.append(f"has_bag_fixed_by_bag_type:{bag_type_v}")

    if bag_norm["has_bag"]["value"] is True and bag_norm["bag_type"]["value"] == "none":
        bag_norm["bag_type"] = make_vc("unknown", "low")
        notes.append("bag_type_none_fixed_to_unknown_when_has_bag_true")

    if bag_norm["has_bag"]["value"] is False:
        if bag_norm["bag_type"]["value"] not in {"none", "unknown"}:
            notes.append(f"bag_type_cleared_when_has_bag_false:{bag_norm['bag_type']['value']}")
        bag_norm["bag_type"] = make_vc("none", "high")
        bag_norm["bag_color"] = make_vc("none", "high")
        bag_norm["bag_relation"] = make_vc("none", "high")

    item_type_v = carried_norm["item_type"]["value"]
    if item_type_v in CONCRETE_ITEM_TYPES:
        carried_norm["has_item"] = make_vc(True, best_conf(carried_norm["has_item"]["confidence"], carried_norm["item_type"]["confidence"]))
        notes.append(f"has_item_fixed_by_item_type:{item_type_v}")

    rel_v = carried_norm["item_relation"]["value"]
    if rel_v in {"held", "near_person", "worn", "pulled"}:
        carried_norm["has_item"] = make_vc(True, best_conf(carried_norm["has_item"]["confidence"], carried_norm["item_relation"]["confidence"]))
        if carried_norm["item_type"]["value"] == "none":
            carried_norm["item_type"] = make_vc("unknown", "medium")
            notes.append(f"item_type_unknown_by_relation:{rel_v}")
        notes.append(f"has_item_fixed_by_relation:{rel_v}")

    # negative guard: has_item=true / item_type=none / relation=none は false に戻す
    if carried_norm["has_item"]["value"] is True and carried_norm["item_type"]["value"] == "none":
        if carried_norm["item_relation"]["value"] in {"none", "unknown", None}:
            carried_norm["has_item"] = make_vc(False, "medium")
            carried_norm["item_type"] = make_vc("none", "high")
            carried_norm["item_relation"] = make_vc("none", "high")
            notes.append("has_item_false_by_none_type_and_none_relation")
        else:
            carried_norm["item_type"] = make_vc("unknown", "low")
            notes.append("item_type_none_fixed_to_unknown_when_has_item_true")

    if carried_norm["has_item"]["value"] is False:
        carried_norm["item_type"] = make_vc("none", "high")
        carried_norm["item_relation"] = make_vc("none", "high")

    out = {"items": {"bag": bag_norm, "carried_item": carried_norm}}
    uniq_notes: List[str] = []
    for n in notes:
        if n not in uniq_notes:
            uniq_notes.append(n)
    if uniq_notes:
        out["_postprocess_notes"] = uniq_notes
    return out


# =========================================================
# LM Studio calls
# =========================================================
def extract_lmstudio_usage(obj: Dict[str, Any]) -> Dict[str, Any]:
    usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    completion_details = usage.get("completion_tokens_details", {})
    if not isinstance(completion_details, dict):
        completion_details = {}
    stats = obj.get("stats", {}) if isinstance(obj, dict) else {}
    timings = obj.get("timings", {}) if isinstance(obj, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    if not isinstance(timings, dict):
        timings = {}

    return {
        "prompt_tokens": safe_int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": safe_int_or_none(usage.get("completion_tokens")),
        "total_tokens": safe_int_or_none(usage.get("total_tokens")),
        "reasoning_tokens": safe_int_or_none(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens")),
        "server_tokens_per_second": (
            safe_float_or_none(stats.get("tokens_per_second"))
            or safe_float_or_none(stats.get("tokens_per_sec"))
            or safe_float_or_none(timings.get("tokens_per_second"))
            or safe_float_or_none(timings.get("tokens_per_sec"))
        ),
        "raw_usage": usage,
    }


def call_lmstudio_json(
    image_path: Path,
    prompt: str,
    base_url: str,
    model: str,
    timeout_sec: int,
    max_tokens: int,
    kind: str,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    data_url = image_file_to_data_url(image_path)
    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        **sampling,
        "max_tokens": max_tokens,
    }

    started = time.time()
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as res:
            raw = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {
            "error": "HTTPError",
            "status_code": e.code,
            "message": body,
            "_runtime": {
                "kind": kind,
                "elapsed_sec": round(time.time() - started, 3),
                "started_at": started_at,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "image_path": str(image_path),
                "sampling": dict(sampling),
            },
        }
    except Exception as e:
        return {
            "error": type(e).__name__,
            "message": str(e),
            "_runtime": {
                "kind": kind,
                "elapsed_sec": round(time.time() - started, 3),
                "started_at": started_at,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "image_path": str(image_path),
                "sampling": dict(sampling),
            },
        }

    finished_at = dt.datetime.now().isoformat(timespec="seconds")
    elapsed = round(time.time() - started, 3)

    try:
        response_obj = json.loads(raw)
        text = response_obj["choices"][0]["message"]["content"]
        usage = extract_lmstudio_usage(response_obj)
    except Exception:
        return {
            "error": "LMSTUDIO_RESPONSE_PARSE_ERROR",
            "raw_response": raw,
            "_runtime": {
                "kind": kind,
                "elapsed_sec": elapsed,
                "started_at": started_at,
                "finished_at": finished_at,
                "model": model,
                "image_path": str(image_path),
                "sampling": dict(sampling),
            },
        }

    obj = extract_json_from_text(text)
    if kind == "person":
        obj = normalize_person_json(obj)
        prompt_version = PERSON_PROMPT_VERSION
    elif kind == "items":
        obj = normalize_items_json(obj)
        prompt_version = ITEMS_PROMPT_VERSION
    else:
        prompt_version = "unknown"

    obj["raw_text"] = text
    obj["converted_from"] = f"{kind}_gemma_direct_json"
    obj["_lmstudio_usage"] = usage
    obj.setdefault("_runtime", {})
    obj["_runtime"].update({
        "kind": kind,
        "elapsed_sec": elapsed,
        "started_at": started_at,
        "finished_at": finished_at,
        "prompt_version": prompt_version,
        "model": model,
        "image_path": str(image_path),
        "max_tokens": max_tokens,
        "sampling": dict(sampling),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })
    return obj


def check_lmstudio_models(base_url: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as res:
            raw = res.read().decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


# =========================================================
# Target and scoring
# =========================================================
def maybe_target_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = normalize_token(v)
    if s in {"", "none", "unknown", "any"}:
        return None
    return s


def build_target_json(args: argparse.Namespace) -> Dict[str, Any]:
    target = {
        "subject_type": maybe_target_value(getattr(args, "subject_type", "auto")),
        "upper_color": maybe_target_value(args.target_upper_color),
        "upper_secondary_color": maybe_target_value(getattr(args, "target_upper_secondary_color", "none")),
        "upper_pattern": maybe_target_value(getattr(args, "target_upper_pattern", "none")),
        "upper_type": maybe_target_value(args.target_upper_type),
        "lower_color": maybe_target_value(args.target_lower_color),
        "lower_secondary_color": maybe_target_value(getattr(args, "target_lower_secondary_color", "none")),
        "lower_pattern": maybe_target_value(getattr(args, "target_lower_pattern", "none")),
        "lower_type": maybe_target_value(args.target_lower_type),
        "apparent_gender": maybe_target_value(args.target_gender),
        "hair": maybe_target_value(args.target_hair),
        "headwear_type": maybe_target_value(args.target_headwear_type),
        "target_item": normalize_token(args.target_item or "unspecified"),
    }
    # Noneの項目は比較対象外。ただしtarget_itemは常に残す。
    return {k: v for k, v in target.items() if v is not None or k == "target_item"}


def should_run_items(target: Dict[str, Any]) -> bool:
    item = normalize_token(target.get("target_item", "unspecified"))
    # "none" means "search for a person with no belongings", so items VLM must run.
    # "unspecified/any/unknown" means "do not use belongings as a condition", so skip.
    return item not in SKIP_ITEM_TARGETS


def get_nested_value(obj: Dict[str, Any], path: List[str]) -> str:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return "unknown"
        cur = cur.get(p, "unknown")
    if isinstance(cur, dict):
        cur = cur.get("value", "unknown")
    return normalize_token(cur)


def values_match(target_value: str, observed_value: str, field: str) -> bool:
    t = normalize_token(target_value)
    o = normalize_token(observed_value)
    if t == o:
        return True

    # 色の近似。研究初期なのでゆるめにする。
    color_groups = {
        "gray": {"gray", "grey"},
        "black": {"black", "dark"},
        "beige": {"beige", "tan", "cream"},
        "green": {"green", "olive", "khaki"},
    }
    if field.endswith("color"):
        for group in color_groups.values():
            if t in group and o in group:
                return True

    # 種類の近似
    type_groups = [
        {"pants", "trousers"},
        {"t_shirt", "t-shirt"},
    ]
    for group in type_groups:
        if t in group and o in group:
            return True

    return False


def partial_match_score(target_value: str, observed_value: str, field: str) -> Tuple[float, str]:
    """ロボット視点向けの曖昧一致。完全一致以外も弱い候補として残す。"""
    t = normalize_token(target_value)
    o = normalize_token(observed_value)
    if o in {"unknown", ""}:
        if field == "apparent_gender":
            return 0.0, "gender_unknown_soft"
        return 0.20, "unknown_partial"
    if values_match(t, o, field):
        return 1.0, "exact"

    if field.endswith("pattern"):
        if o in {"unknown", ""}:
            return 0.35, "pattern_unknown_partial"
        if {t, o} == {"striped", "plaid"}:
            return 0.0, "striped_plaid_mismatch"
        if {t, o} <= {"graphic", "color_block", "solid"}:
            return 0.35, "pattern_near"

    if field == "upper_type":
        if t == "hoodie" and o in {"shirt", "t_shirt", "t-shirt", "sweater", "jacket"}:
            return 0.45, "upper_type_near"
        if t in {"shirt", "t_shirt", "t-shirt"} and o == "hoodie":
            return 0.45, "upper_type_near"
        if {t, o} <= {"shirt", "t_shirt", "t-shirt", "sweater"}:
            return 0.50, "upper_type_near"

    if field == "apparent_gender":
        # 見た目genderはVLMが揺れやすいので、外れても完全には落とさない。
        if t in {"male", "female"} and o in {"male", "female"}:
            return 0.0, "gender_mismatch_soft"

    if field.endswith("color"):
        if {t, o} <= {"gray", "grey"}:
            return 1.0, "exact"
        if {t, o} == {"blue", "light_blue"}:
            return 0.60, "color_near_light_blue_blue"
        color_near = [
            {"blue", "navy"},
            {"light_blue", "light", "white", "gray", "grey"},
            {"beige", "tan", "cream", "khaki", "brown"},
            {"gray", "white", "light"},
            {"black", "dark", "navy"},
        ]
        for group in color_near:
            if t in group and o in group:
                return 0.30 if "light_blue" in group else 0.55, "color_near"

    return 0.0, "mismatch"


def score_person(target: Dict[str, Any], observed: Dict[str, Any]) -> Dict[str, Any]:
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
    mismatch: List[Dict[str, str]] = []
    unknown: List[str] = []
    partial: List[Dict[str, Any]] = []

    total = 0
    weight_total = 0.0
    score_sum = 0.0
    for field, path in field_paths.items():
        tv = target.get(field)
        if tv is None:
            continue
        total += 1
        weight = float(field_weights.get(field, 0.10))
        weight_total += weight
        ov = get_nested_value(observed, path)
        part_score, reason = partial_match_score(tv, ov, field)
        score_sum += weight * part_score
        if ov in {"unknown", ""}:
            unknown.append(field)
            partial.append({"field": field, "target": normalize_token(tv), "observed": ov, "score": part_score, "reason": reason})
        elif part_score >= 0.999:
            matched.append(field)
        elif part_score > 0:
            partial.append({"field": field, "target": normalize_token(tv), "observed": ov, "score": part_score, "reason": reason})
        else:
            mismatch.append({"field": field, "target": normalize_token(tv), "observed": ov})

    if total == 0:
        return {
            "score": 0.0,
            "specified_count": 0,
            "matched": matched,
            "mismatch": mismatch,
            "unknown": unknown,
            "partial": partial,
            "field_weights": field_weights,
            "status": "NO_PERSON_TARGET_FIELDS",
        }

    score = score_sum / weight_total if weight_total > 0 else 0.0
    return {
        "score": round(score, 3),
        "specified_count": total,
        "matched": matched,
        "mismatch": mismatch,
        "unknown": unknown,
        "partial": partial,
        "field_weights": {k: v for k, v in field_weights.items() if target.get(k) is not None},
        "status": "OK",
    }


def score_item(target: Dict[str, Any], items_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    item = normalize_token(target.get("target_item", "unspecified"))

    # Not a belongings condition: do not run / do not score items.
    if item in SKIP_ITEM_TARGETS:
        return {
            "score": None,
            "target_item": item,
            "status": "SKIPPED_TARGET_ITEM_UNSPECIFIED",
            "matched": [],
            "mismatch": [],
            "unknown": [],
        }

    if not items_result or items_result.get("error"):
        return {
            "score": 0.0,
            "target_item": item,
            "status": "ITEMS_ERROR_OR_NOT_RUN",
            "matched": [],
            "mismatch": [{"field": "target_item", "target": item, "observed": "error_or_missing"}],
            "unknown": [],
        }

    bag_type = get_nested_value(items_result, ["items", "bag", "bag_type"])
    carried_type = get_nested_value(items_result, ["items", "carried_item", "item_type"])
    has_bag = get_nested_value(items_result, ["items", "bag", "has_bag"])
    has_item = get_nested_value(items_result, ["items", "carried_item", "has_item"])

    # Real absence condition:
    # target_item=none means "person with no bag and no carried item".
    if item in NO_ITEM_TARGETS:
        bag_absent = (bag_type == "none" or has_bag in {"false", "none", "0"})
        item_absent = (carried_type == "none" or has_item in {"false", "none", "0"})

        if bag_type == "unknown" or carried_type == "unknown" or has_bag == "unknown" or has_item == "unknown":
            return {
                "score": 0.5,
                "target_item": item,
                "observed_item": f"bag:{bag_type},carried:{carried_type}",
                "status": "UNKNOWN_NO_ITEM",
                "matched": [],
                "mismatch": [],
                "unknown": ["target_item:none"],
            }

        if bag_absent and item_absent:
            return {
                "score": 1.0,
                "target_item": item,
                "observed_item": "none",
                "status": "OK_NO_ITEM",
                "matched": ["target_item:none"],
                "mismatch": [],
                "unknown": [],
            }

        return {
            "score": 0.0,
            "target_item": item,
            "observed_item": f"bag:{bag_type},carried:{carried_type}",
            "status": "MISMATCH_HAS_ITEM",
            "matched": [],
            "mismatch": [{"field": "target_item", "target": "none", "observed": f"bag:{bag_type},carried:{carried_type}"}],
            "unknown": [],
        }

    if item in BAG_LIKE_TYPES:
        observed = bag_type
        field = "bag_type"
        # target_item=bag は具体バッグなら合格扱い
        if item == "bag" and observed in CONCRETE_BAG_TYPES:
            return {
                "score": 1.0,
                "target_item": item,
                "observed_item": observed,
                "status": "OK",
                "matched": [f"target_item:{item}"],
                "mismatch": [],
                "unknown": [],
            }
    else:
        observed = carried_type
        field = "item_type"

    if observed in {"unknown", ""}:
        return {
            "score": 0.5,
            "target_item": item,
            "observed_item": observed,
            "status": "UNKNOWN_ITEM",
            "matched": [],
            "mismatch": [],
            "unknown": ["target_item"],
        }

    if observed == item:
        return {
            "score": 1.0,
            "target_item": item,
            "observed_item": observed,
            "status": "OK",
            "matched": [f"target_item:{item}"],
            "mismatch": [],
            "unknown": [],
        }

    return {
        "score": 0.0,
        "target_item": item,
        "observed_item": observed,
        "status": "MISMATCH",
        "matched": [],
        "mismatch": [{"field": field, "target": item, "observed": observed}],
        "unknown": [],
    }

def build_match_json(target: Dict[str, Any], person_result: Dict[str, Any], items_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    person = score_person(target, person_result)
    item = score_item(target, items_result)

    person_score = float(person["score"])
    item_score = item["score"]

    if item_score is None:
        final_score = person_score
    else:
        final_score = person_score * 0.70 + float(item_score) * 0.30

    final_score = round(final_score, 3)
    if final_score >= 0.85:
        decision = "strong_candidate"
    elif final_score >= 0.75:
        decision = "candidate"
    elif final_score >= 0.50:
        decision = "weak_candidate"
    else:
        decision = "no_match"

    return {
        "person_score": person_score,
        "item_score": item_score,
        "final_score": final_score,
        "decision": decision,
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
    }


# =========================================================
# YOLO detection and tracking
# =========================================================
def xyxy_from_box(box: Any) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
    return int(x1), int(y1), int(x2), int(y2)


def detect_persons_with_tracking(model: YOLO, frame: np.ndarray, args: argparse.Namespace) -> List[Dict[str, Any]]:
    results = model.track(
        frame,
        persist=True,
        conf=args.yolo_conf,
        classes=[0],
        verbose=False,
        tracker=args.tracker if args.tracker else None,
    )
    if not results:
        return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    h, w = frame.shape[:2]
    dets: List[Dict[str, Any]] = []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = xyxy_from_box(box)
        conf = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
        track_id = None
        if getattr(box, "id", None) is not None and box.id is not None:
            try:
                track_id = int(box.id[0].detach().cpu().item())
            except Exception:
                track_id = None
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = float(bw * bh)
        dets.append({
            "raw_index": i,
            "track_id": track_id if track_id is not None else -(i + 1),
            "track_id_is_real": track_id is not None,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "bbox_w": bw,
            "bbox_h": bh,
            "area": area,
            "area_ratio": area / float(max(1, w * h)),
            "height_ratio": bh / float(max(1, h)),
        })

    dets.sort(key=lambda d: d["area"], reverse=True)
    for rank, d in enumerate(dets, start=1):
        d["rank_by_area"] = rank
    return dets


def should_reanalyze_cached_feature(
    track_id: int,
    det: Dict[str, Any],
    now_time: float,
    feature_cache: Dict[int, Dict[str, Any]],
    cooldown_sec: float,
    area_change_ratio: float,
    allow_area_reanalyze_before_interval: bool = False,
) -> Tuple[bool, str]:
    entry = feature_cache.get(track_id)
    if entry is None:
        return True, "NO_CACHE"

    age_sec = now_time - float(entry.get("created_time", now_time))
    prev_area = float(entry.get("area", 0.0) or 0.0)
    current_area = float(det.get("area", 0.0) or 0.0)
    rel_change: Optional[float] = None
    if prev_area > 0 and current_area > 0:
        rel_change = abs(current_area - prev_area) / prev_area

    # v3.2: 動画ではVLM再送信が増えすぎないよう、intervalを基本的に最短再送信間隔として扱う。
    # 旧挙動に戻したい場合だけ --allow_area_reanalyze_before_interval を付ける。
    if cooldown_sec > 0 and age_sec < cooldown_sec:
        if (
            allow_area_reanalyze_before_interval
            and rel_change is not None
            and area_change_ratio > 0
            and rel_change >= area_change_ratio
        ):
            return True, f"AREA_CHANGED_{rel_change:.2f}_BEFORE_INTERVAL"
        if rel_change is not None and area_change_ratio > 0 and rel_change >= area_change_ratio:
            return False, f"CACHE_VALID_{age_sec:.1f}s_AREA_HELD_{rel_change:.2f}"
        return False, f"CACHE_VALID_{age_sec:.1f}s"

    if cooldown_sec > 0 and age_sec >= cooldown_sec:
        if rel_change is not None and area_change_ratio > 0 and rel_change >= area_change_ratio:
            return True, f"CACHE_EXPIRED_{age_sec:.1f}s_AREA_CHANGED_{rel_change:.2f}"
        return True, f"CACHE_EXPIRED_{age_sec:.1f}s"

    # cooldown_sec <= 0 のときだけ、面積変化で即再解析する。
    if rel_change is not None and area_change_ratio > 0 and rel_change >= area_change_ratio:
        return True, f"AREA_CHANGED_{rel_change:.2f}"

    return False, f"CACHE_VALID_{age_sec:.1f}s"


# =========================================================
# Person memory / lightweight ReID helpers
# =========================================================
def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def compute_upper_body_hist(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """VLMを使わない軽量な見た目特徴。上半身寄りのHSV 3Dヒストグラムを返す。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 8 or bh < 16:
        return None

    # 顔や背景より服の影響が出やすいよう、bboxの上半分〜胴体付近を使う。
    uy1 = y1 + int(bh * 0.15)
    uy2 = y1 + int(bh * 0.68)
    ux1 = x1 + int(bw * 0.05)
    ux2 = x2 - int(bw * 0.05)
    ux1, uy1, ux2, uy2 = clamp_bbox((ux1, uy1, ux2, uy2), w, h)
    crop = frame[uy1:uy2, ux1:ux2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def hist_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    if h1 is None or h2 is None:
        return 0.0
    try:
        d = float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
        return max(0.0, min(1.0, 1.0 - d))
    except Exception:
        return 0.0


def upper_body_color_hint(frame: np.ndarray, bbox: Tuple[int, int, int, int], target_color: str) -> Tuple[float, Dict[str, Any]]:
    """VLM前に使う軽量なtarget色らしさ。上半身HSVからざっくり評価する。"""
    target_color = normalize_token(target_color)
    if target_color in {"", "none", "unknown"}:
        return 0.0, {"status": "NO_TARGET_COLOR"}

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 8 or bh < 16:
        return 0.0, {"status": "CROP_TOO_SMALL"}

    uy1 = y1 + int(bh * 0.18)
    uy2 = y1 + int(bh * 0.66)
    ux1 = x1 + int(bw * 0.08)
    ux2 = x2 - int(bw * 0.08)
    ux1, uy1, ux2, uy2 = clamp_bbox((ux1, uy1, ux2, uy2), w, h)
    crop = frame[uy1:uy2, ux1:ux2]
    if crop.size == 0:
        return 0.0, {"status": "EMPTY_CROP"}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    if target_color == "light_blue":
        mask_exact = (hue >= 85) & (hue <= 112) & (sat >= 35) & (val >= 95)
        mask_near = (hue >= 78) & (hue <= 120) & (sat >= 25) & (val >= 70)
    elif target_color in {"blue", "navy"}:
        mask_exact = (hue >= 90) & (hue <= 125) & (sat >= 45) & (val >= 45)
        mask_near = (hue >= 82) & (hue <= 130) & (sat >= 30) & (val >= 35)
    elif target_color in {"black", "dark"}:
        mask_exact = val <= 70
        mask_near = val <= 95
    elif target_color in {"white", "light"}:
        mask_exact = (sat <= 45) & (val >= 165)
        mask_near = (sat <= 70) & (val >= 130)
    elif target_color in {"gray", "grey"}:
        mask_exact = (sat <= 55) & (val >= 65) & (val <= 190)
        mask_near = (sat <= 80) & (val >= 45) & (val <= 210)
    else:
        return 0.0, {"status": "UNSUPPORTED_TARGET_COLOR", "target_color": target_color}

    total = max(1, crop.shape[0] * crop.shape[1])
    exact_ratio = float(np.count_nonzero(mask_exact)) / float(total)
    near_ratio = float(np.count_nonzero(mask_near)) / float(total)
    score = min(1.0, exact_ratio * 2.2 + max(0.0, near_ratio - exact_ratio) * 0.9)
    return round(score, 3), {
        "status": "OK",
        "target_color": target_color,
        "exact_ratio": round(exact_ratio, 3),
        "near_ratio": round(near_ratio, 3),
        "score": round(score, 3),
    }




def compute_crop_quality(
    det: Dict[str, Any],
    frame_shape: Tuple[int, int, int],
    args: argparse.Namespace,
    track_age: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """VLMへ送る人物cropの良さをbbox情報だけで軽量評価する。"""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = det["bbox"]
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    width_ratio = bw / float(max(1, w))
    height_ratio = float(det.get("height_ratio", bh / float(max(1, h))) or 0.0)
    area_ratio = float(det.get("area_ratio", (bw * bh) / float(max(1, w * h))) or 0.0)
    conf = float(det.get("conf", 0.0) or 0.0)

    edge_margin = float(getattr(args, "best_crop_edge_margin_ratio", 0.03))
    edge_px_x = max(1.0, w * edge_margin)
    edge_px_y = max(1.0, h * edge_margin)
    near_left = x1 <= edge_px_x
    near_right = x2 >= (w - edge_px_x)
    near_top = y1 <= edge_px_y
    near_bottom = y2 >= (h - edge_px_y)
    edge_hits = int(near_left) + int(near_right) + int(near_top) + int(near_bottom)

    cx, cy = bbox_center(det["bbox"])
    center_dx = abs(cx - w / 2.0) / max(1.0, w / 2.0)
    center_dy = abs(cy - h / 2.0) / max(1.0, h / 2.0)
    center_score = max(0.0, 1.0 - 0.75 * center_dx - 0.25 * center_dy)

    width_score = min(1.0, width_ratio / max(1e-6, float(getattr(args, "best_crop_preferred_width_ratio", 0.22))))
    height_score = min(1.0, height_ratio / max(1e-6, float(getattr(args, "best_crop_preferred_height_ratio", 0.62))))
    area_score = min(1.0, area_ratio / max(1e-6, float(getattr(args, "best_crop_preferred_area_ratio", 0.18))))
    conf_score = max(0.0, min(1.0, (conf - 0.35) / 0.60))
    stable_score = min(1.0, track_age / max(1, int(getattr(args, "vlm_min_track_frames", 3)) + 5))

    too_narrow_penalty = 0.20 if width_ratio < float(getattr(args, "min_width_ratio_for_vlm", 0.10)) else 0.0
    edge_penalty = min(0.35, 0.11 * edge_hits)
    partial_risk_penalty = 0.15 if (width_ratio < 0.12 and height_ratio > 0.65) else 0.0

    score = (
        0.22 * area_score
        + 0.20 * width_score
        + 0.16 * height_score
        + 0.18 * center_score
        + 0.14 * conf_score
        + 0.10 * stable_score
        - edge_penalty
        - too_narrow_penalty
        - partial_risk_penalty
    )
    score = max(0.0, min(1.0, score))
    detail = {
        "score": round(score, 3),
        "width_ratio": round(width_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "area_ratio": round(area_ratio, 5),
        "conf": round(conf, 4),
        "center_score": round(center_score, 3),
        "width_score": round(width_score, 3),
        "height_score": round(height_score, 3),
        "area_score": round(area_score, 3),
        "conf_score": round(conf_score, 3),
        "stable_score": round(stable_score, 3),
        "edge_hits": edge_hits,
        "edge_penalty": round(edge_penalty, 3),
        "too_narrow_penalty": round(too_narrow_penalty, 3),
        "partial_risk_penalty": round(partial_risk_penalty, 3),
        "near_left": near_left,
        "near_right": near_right,
        "near_top": near_top,
        "near_bottom": near_bottom,
    }
    return score, detail


def update_best_crop_candidate(
    memory: Dict[str, Any],
    frame: np.ndarray,
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    frame_count: int,
    time_sec: float,
    quality_score: float,
    quality_detail: Dict[str, Any],
    track_id: int,
    target: Optional[Dict[str, Any]] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> bool:
    """memory内にVLM用best crop候補を保持する。保存はせず、best更新時だけframeをメモリにコピーする。"""
    memory.setdefault("best_crop_status", "COLLECTING")
    if "best_crop_first_time" not in memory:
        memory["best_crop_first_time"] = round(time_sec, 3)
        memory["best_crop_first_frame"] = frame_count
    current_best = float(memory.get("best_crop_score", -1.0) or -1.0)
    if quality_score > current_best:
        det_copy = dict(det)
        if isinstance(det_copy.get("bbox"), list):
            det_copy["bbox"] = tuple(det_copy["bbox"])
        memory["best_crop_score"] = round(float(quality_score), 3)
        memory["best_crop_detail"] = quality_detail
        memory["best_crop_frame"] = frame_count
        memory["best_crop_time"] = round(time_sec, 3)
        memory["best_crop_track_id"] = track_id
        target_color = normalize_token((target or {}).get("upper_color", ""))
        color_hint_score, color_hint_detail = upper_body_color_hint(frame, det_copy["bbox"], target_color) if target_color else (0.0, {"status": "NO_TARGET_COLOR"})
        memory["best_crop_target_color_score"] = color_hint_score
        memory["best_crop_target_color_detail"] = color_hint_detail
        memory["best_vlm_frame"] = frame.copy()
        memory["best_vlm_det"] = det_copy
        memory["best_vlm_hist"] = hist
        memory["last_quality_improve_frame"] = frame_count
        if memory.get("best_crop_status") not in {"READY", "QUEUED", "DONE"}:
            memory["best_crop_status"] = "COLLECTING"
        inc_perf(perf_stats, "best_crop_updates")
        return True
    return False


def set_best_crop_ready(
    memory: Dict[str, Any],
    reason: str,
    frame_count: int,
    time_sec: float,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> None:
    if memory.get("best_crop_status") != "READY":
        inc_perf(perf_stats, "best_crop_ready_count")
    memory["best_crop_status"] = "READY"
    memory["best_crop_ready_reason"] = reason
    memory["best_crop_ready_frame"] = frame_count
    memory["best_crop_ready_time"] = round(time_sec, 3)


def drop_low_quality_best_crop(
    memory: Dict[str, Any],
    reason: str,
    frame_count: int,
    time_sec: float,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> None:
    if memory.get("best_crop_status") != "DROPPED_LOW_QUALITY":
        inc_perf(perf_stats, "dropped_low_quality_count")
    memory["best_crop_status"] = "DROPPED_LOW_QUALITY"
    memory["best_crop_drop_reason"] = reason
    memory["best_crop_drop_frame"] = frame_count
    memory["best_crop_drop_time"] = round(time_sec, 3)


def best_crop_send_decision(
    memory: Dict[str, Any],
    frame_count: int,
    time_sec: float,
    args: argparse.Namespace,
    force: bool = False,
) -> Tuple[bool, str]:
    """best cropをVLMへ送るべきか判定する。良ければ即送信、悪ければ最大待機までbestを探す。"""
    if memory.get("vlm_pending", False) or memory.get("vlm_done", False):
        return False, "ALREADY_HAS_VLM"
    if memory.get("best_vlm_frame") is None or memory.get("best_vlm_det") is None:
        return False, "NO_BEST_CROP"
    score = float(memory.get("best_crop_score", 0.0) or 0.0)
    first_time = float(memory.get("best_crop_first_time", time_sec) or time_sec)
    elapsed = max(0.0, time_sec - first_time)
    last_imp = int(memory.get("last_quality_improve_frame", frame_count) or frame_count)
    no_improve_frames = max(0, frame_count - last_imp)
    good_th = float(getattr(args, "best_crop_good_threshold", 0.62))
    min_th = float(getattr(args, "best_crop_min_threshold", 0.45))
    max_wait = float(getattr(args, "best_crop_max_wait_sec", 0.8))
    min_wait = float(getattr(args, "best_crop_min_wait_sec", 0.15))
    patience = int(getattr(args, "best_crop_patience_frames", 6))
    detail = memory.get("best_crop_detail") if isinstance(memory.get("best_crop_detail"), dict) else {}
    edge_hits = int(detail.get("edge_hits", 0) or 0)
    width_score = safe_float(detail.get("width_score"), 1.0)
    good_edge_limit = int(getattr(args, "best_crop_good_max_edge_hits", 1))
    good_width_min = float(getattr(args, "best_crop_good_min_width_score", 0.70))
    good_crop_ok = edge_hits <= good_edge_limit and width_score >= good_width_min
    if force and score >= min_th:
        return True, f"END_FLUSH_BEST_CROP_{score:.2f}"
    if score >= good_th and good_crop_ok:
        return True, f"GOOD_QUALITY_{score:.2f}"
    if score >= good_th and not good_crop_ok:
        reasons = []
        if edge_hits > good_edge_limit:
            reasons.append(f"EDGE_{edge_hits}")
        if width_score < good_width_min:
            reasons.append(f"WIDTH_{width_score:.2f}")
        quality_note = "_".join(reasons) if reasons else "CROP_GATE"
        if elapsed < max_wait:
            return False, f"BEST_CROP_WAIT_GOOD_BUT_{quality_note}_{score:.2f}_{elapsed:.2f}s"
    if elapsed >= max_wait and score >= min_th:
        return True, f"MAX_WAIT_BEST_CROP_{score:.2f}_{elapsed:.2f}s"
    if elapsed >= min_wait and no_improve_frames >= patience and score >= min_th:
        return True, f"QUALITY_PEAKED_{score:.2f}_{no_improve_frames}f"
    return False, f"BEST_CROP_WAIT_{score:.2f}_{elapsed:.2f}s"


def vlm_budget_used(perf_stats: Optional[Dict[str, Any]]) -> int:
    if not isinstance(perf_stats, dict):
        return 0
    return int(perf_stats.get("async_jobs_queued", 0) or 0)


def vlm_budget_remaining(
    args: argparse.Namespace,
    perf_stats: Optional[Dict[str, Any]],
    time_sec: Optional[float] = None,
) -> Optional[int]:
    limit = int(getattr(args, "max_total_vlm_calls", 0) or 0)
    per_min = float(getattr(args, "max_vlm_calls_per_min", 0.0) or 0.0)
    if per_min > 0 and time_sec is not None:
        elapsed_min = max(float(time_sec), 1.0) / 60.0
        per_min_limit = max(1, int(per_min * elapsed_min))
        limit = per_min_limit if limit <= 0 else min(limit, per_min_limit)
    if limit <= 0:
        return None
    return max(0, limit - vlm_budget_used(perf_stats))


def is_vlm_budget_available(
    args: argparse.Namespace,
    perf_stats: Optional[Dict[str, Any]],
    time_sec: Optional[float] = None,
) -> bool:
    remaining = vlm_budget_remaining(args, perf_stats, time_sec)
    return remaining is None or remaining > 0


def near_video_end_vlm_block_reason(
    memory: Dict[str, Any],
    args: argparse.Namespace,
    time_sec: Optional[float],
    video_duration_sec: Optional[float],
) -> str:
    guard_sec = float(getattr(args, "near_end_vlm_guard_sec", 0.0) or 0.0)
    if guard_sec <= 0 or not video_duration_sec or time_sec is None:
        return ""
    remaining = float(video_duration_sec) - float(time_sec)
    if remaining > guard_sec:
        return ""
    color_score = safe_float(memory.get("best_crop_target_color_score"), 0.0)
    min_color = float(getattr(args, "near_end_min_target_color_score", 0.25))
    quality = safe_float(memory.get("best_crop_score"), 0.0)
    min_quality = float(getattr(args, "near_end_min_quality", 0.68))
    if color_score >= min_color and quality >= min_quality:
        return ""
    return f"near_end_guard_{remaining:.1f}s_color_{color_score:.2f}_quality_{quality:.2f}"


def best_crop_flush_priority(memory: Dict[str, Any]) -> float:
    score = safe_float(memory.get("best_crop_score"), 0.0)
    target_color_score = safe_float(memory.get("best_crop_target_color_score"), 0.0)
    detail = memory.get("best_crop_detail") if isinstance(memory.get("best_crop_detail"), dict) else {}
    edge_hits = int(detail.get("edge_hits", 0) or 0)
    width_score = safe_float(detail.get("width_score"), 0.0)
    height_score = safe_float(detail.get("height_score"), 0.0)
    area_score = safe_float(detail.get("area_score"), 0.0)
    stable_score = safe_float(detail.get("stable_score"), 0.0)
    edge_penalty = min(0.20, 0.05 * max(0, edge_hits))
    return round(
        score * 1.00
        + target_color_score * 0.45
        + width_score * 0.12
        + height_score * 0.08
        + area_score * 0.06
        + stable_score * 0.04
        - edge_penalty,
        3,
    )


def final_flush_skip_reason(memory: Dict[str, Any], args: argparse.Namespace) -> str:
    if memory.get("vlm_done", False):
        return "already_done"
    if memory.get("vlm_pending", False):
        return "already_pending"
    if memory.get("best_vlm_frame") is None or memory.get("best_vlm_det") is None:
        return "no_best_crop"

    score = safe_float(memory.get("best_crop_score"), 0.0)
    target_color_score = safe_float(memory.get("best_crop_target_color_score"), 0.0)
    detail = memory.get("best_crop_detail") if isinstance(memory.get("best_crop_detail"), dict) else {}
    min_quality = float(getattr(args, "final_flush_min_quality", 0.60))
    target_color_min = float(getattr(args, "final_flush_min_target_color_score", 0.0) or 0.0)
    target_color_rescue = float(getattr(args, "final_flush_target_color_rescue_score", 0.35))
    quality_floor = float(getattr(args, "final_flush_quality_floor_with_target_color", 0.52))
    min_width = float(getattr(args, "final_flush_min_width_score", 0.55))
    max_edge = int(getattr(args, "final_flush_max_edge_hits", 2))
    min_height = float(getattr(args, "final_flush_min_height_score", 0.45))
    if target_color_min > 0 and target_color_score < target_color_min:
        return f"target_color_below_{target_color_min:.2f}"
    if score < min_quality and not (target_color_score >= target_color_rescue and score >= quality_floor):
        return f"quality_below_{min_quality:.2f}"
    if safe_float(detail.get("width_score"), 1.0) < min_width:
        return f"width_below_{min_width:.2f}"
    if safe_float(detail.get("height_score"), 1.0) < min_height:
        return f"height_below_{min_height:.2f}"
    if int(detail.get("edge_hits", 0) or 0) > max_edge:
        return f"edge_hits_over_{max_edge}"
    return ""


def choose_final_flush_memories(
    person_memories: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
    perf_stats: Optional[Dict[str, Any]] = None,
    time_sec: Optional[float] = None,
) -> List[Tuple[int, Dict[str, Any]]]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    skipped_reasons: Dict[str, int] = {}
    for mid, mem in sorted(person_memories.items()):
        if not isinstance(mid, int) or not isinstance(mem, dict):
            continue
        reason = final_flush_skip_reason(mem, args)
        if reason:
            if reason not in {"already_done", "already_pending", "no_best_crop"}:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                mem["best_crop_status"] = mem.get("best_crop_status") or "COLLECTING"
                mem["final_flush_skipped_reason"] = reason
            continue
        candidates.append((mid, mem))

    candidates.sort(
        key=lambda item: (
            -best_crop_flush_priority(item[1]),
            -safe_float(item[1].get("best_crop_score"), 0.0),
            int(item[1].get("best_crop_frame", item[1].get("created_frame", 0)) or 0),
        )
    )

    max_memories = int(getattr(args, "final_flush_max_memories", 4) or 0)
    if max_memories > 0:
        selected = candidates[:max_memories]
        skipped_by_limit = max(0, len(candidates) - len(selected))
    else:
        selected = candidates
        skipped_by_limit = 0

    remaining = vlm_budget_remaining(args, perf_stats, time_sec)
    if remaining is not None:
        selected_by_budget = selected[:remaining]
        skipped_by_budget = max(0, len(selected) - len(selected_by_budget))
        selected = selected_by_budget
    else:
        skipped_by_budget = 0

    if skipped_by_limit:
        skipped_reasons["priority_limit"] = skipped_reasons.get("priority_limit", 0) + skipped_by_limit
    if skipped_by_budget:
        skipped_reasons["vlm_budget"] = skipped_reasons.get("vlm_budget", 0) + skipped_by_budget

    inc_perf(perf_stats, "final_flush_candidate_count", len(candidates))
    inc_perf(perf_stats, "final_flush_sent_count", len(selected))
    inc_perf(perf_stats, "final_flush_skipped_count", sum(skipped_reasons.values()))
    if isinstance(perf_stats, dict):
        perf_stats["final_flush_skipped_reasons"] = skipped_reasons
        perf_stats["final_flush_selected_memory_ids"] = [mid for mid, _ in selected]

    selected_ids = {mid for mid, _ in selected}
    for mid, mem in candidates:
        if mid not in selected_ids and not mem.get("final_flush_skipped_reason"):
            mem["final_flush_skipped_reason"] = "priority_or_budget_limit"

    return selected


def submit_best_crop_vlm_job(
    *,
    memory_id: int,
    memory: Dict[str, Any],
    target: Dict[str, Any],
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    executor: ThreadPoolExecutor,
    async_jobs: Dict[int, Future],
    reason: str,
    perf_stats: Optional[Dict[str, Any]] = None,
    video_duration_sec: Optional[float] = None,
    allow_near_end: bool = False,
) -> Optional[Dict[str, Any]]:
    """memory内のbest crop候補を使ってVLMジョブを投入する。"""
    if memory_id in async_jobs or memory.get("vlm_pending", False) or memory.get("vlm_done", False):
        return None
    frame = memory.get("best_vlm_frame")
    det = memory.get("best_vlm_det")
    if not isinstance(frame, np.ndarray) or not isinstance(det, dict):
        return None
    track_id = int(memory.get("best_crop_track_id", memory.get("last_track_id", -1)) or -1)
    frame_count = int(memory.get("best_crop_frame", memory.get("last_seen_frame", 0)) or 0)
    time_sec = float(memory.get("best_crop_time", memory.get("last_seen_time", 0.0)) or 0.0)
    if not allow_near_end:
        block_reason = near_video_end_vlm_block_reason(memory, args, time_sec, video_duration_sec)
        if block_reason:
            inc_perf(perf_stats, "near_end_vlm_block_count")
            memory["near_end_vlm_block_reason"] = block_reason
            return None
    if not is_vlm_budget_available(args, perf_stats, time_sec):
        inc_perf(perf_stats, "vlm_budget_reached_count")
        memory["final_flush_skipped_reason"] = "vlm_budget"
        return None
    bbox = det.get("bbox")
    if isinstance(bbox, list):
        bbox = tuple(bbox)
    if not isinstance(bbox, tuple):
        return None
    memory.update({
        "vlm_pending": True,
        "vlm_done": False,
        "decision": "pending",
        "best_crop_status": "QUEUED",
        "queued_frame": frame_count,
        "queued_time": round(time_sec, 3),
        "selected_crop_frame": frame_count,
        "selected_crop_time": round(time_sec, 3),
        "selected_crop_quality": memory.get("best_crop_score", ""),
        "selected_crop_reason": reason,
    })
    _t_job = time.perf_counter()
    job = build_async_vlm_job(
        frame=frame,
        det=det,
        bbox=bbox,
        track_id=track_id,
        track_id_is_real=bool(det.get("track_id_is_real", True)),
        memory_id=memory_id,
        memory_status=str(memory.get("memory_status_note", "BEST_CROP_SELECTED") or "BEST_CROP_SELECTED"),
        frame_count=frame_count,
        time_sec=time_sec,
        now_time=time_sec,
        target=target,
        args=args,
        dirs=dirs,
        reason=reason,
    )
    job["crop_quality_score"] = memory.get("best_crop_score", "")
    job["crop_quality_detail"] = memory.get("best_crop_detail", {})
    add_perf(perf_stats, "async_job_build_sec", time.perf_counter() - _t_job)
    _t_submit = time.perf_counter()
    async_jobs[memory_id] = executor.submit(run_async_vlm_job, job)
    add_perf(perf_stats, "async_submit_sec", time.perf_counter() - _t_submit)
    inc_perf(perf_stats, "async_jobs_queued")
    append_perf_record(perf_stats, "best_crop_events", {
        "memory_id": memory_id,
        "action": "queue_vlm",
        "reason": reason,
        "selected_crop_frame": frame_count,
        "selected_crop_time": round(time_sec, 3),
        "quality_score": memory.get("best_crop_score", ""),
        "quality_detail": memory.get("best_crop_detail", {}),
    }, max_records=int(getattr(args, "merge_debug_max_records", 80)))
    return job


def dispatch_ready_best_crop_jobs(
    *,
    person_memories: Dict[int, Dict[str, Any]],
    target: Dict[str, Any],
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    executor: ThreadPoolExecutor,
    async_jobs: Dict[int, Future],
    perf_stats: Optional[Dict[str, Any]] = None,
    max_to_queue: Optional[int] = None,
    video_duration_sec: Optional[float] = None,
    allow_near_end: bool = False,
) -> int:
    """READYになったbest cropを、VLMキューの空きに合わせて順次投入する。"""
    queued = 0
    capacity = max(1, int(args.max_pending_vlm))
    if isinstance(perf_stats, dict):
        perf_stats["vlm_backlog_max"] = max(int(perf_stats.get("vlm_backlog_max", 0) or 0), len(async_jobs))
    ready_items = [
        (mid, mem)
        for mid, mem in person_memories.items()
        if isinstance(mid, int)
        and isinstance(mem, dict)
        and mem.get("best_crop_status") == "READY"
        and not mem.get("vlm_pending", False)
        and not mem.get("vlm_done", False)
        and mid not in async_jobs
    ]
    ready_items.sort(
        key=lambda item: (
            -best_crop_flush_priority(item[1]),
            -safe_float(item[1].get("best_crop_target_color_score"), 0.0),
            -float(item[1].get("best_crop_score", 0.0) or 0.0),
            int(item[1].get("best_crop_frame", item[1].get("created_frame", 0)) or 0),
        )
    )

    for mid, mem in ready_items:
        if not is_vlm_budget_available(args, perf_stats):
            inc_perf(perf_stats, "vlm_budget_reached_count")
            break
        if len(async_jobs) >= capacity:
            break
        if max_to_queue is not None and queued >= max_to_queue:
            break
        reason = str(mem.get("best_crop_ready_reason") or "READY_QUEUE")
        job = submit_best_crop_vlm_job(
            memory_id=mid,
            memory=mem,
            target=target,
            args=args,
            dirs=dirs,
            executor=executor,
            async_jobs=async_jobs,
            reason=reason,
            perf_stats=perf_stats,
            video_duration_sec=video_duration_sec,
            allow_near_end=allow_near_end,
        )
        if job is None:
            continue
        queued += 1
        inc_perf(perf_stats, "queued_from_ready_count")
    return queued

def memory_jsonable(memory: Dict[str, Any]) -> Dict[str, Any]:
    """color_histやbest_crop候補画像などJSON化しにくいものを除いた保存用memory。"""
    out = {}
    for k, v in memory.items():
        if k == "color_hist":
            out["color_hist_shape"] = list(v.shape) if isinstance(v, np.ndarray) else None
        elif k == "best_vlm_frame":
            out["best_vlm_frame_shape"] = list(v.shape) if isinstance(v, np.ndarray) else None
        elif k == "best_vlm_hist":
            out["best_vlm_hist_shape"] = list(v.shape) if isinstance(v, np.ndarray) else None
        elif k == "best_vlm_det":
            if isinstance(v, dict):
                tmp = dict(v)
                if isinstance(tmp.get("bbox"), tuple):
                    tmp["bbox"] = list(tmp["bbox"])
                out[k] = tmp
            else:
                out[k] = v
        elif isinstance(v, set):
            out[k] = sorted(list(v))
        else:
            out[k] = v
    return out


def write_memory_snapshot(dirs: Dict[str, Path], person_memories: Dict[int, Dict[str, Any]]) -> None:
    try:
        path = dirs["memory_json"] / "person_memory_latest.json"
        data = {
            "search_version": SEARCH_VERSION,
            "memory_count": len(person_memories),
            "memories": [memory_jsonable(m) for _, m in sorted(person_memories.items())],
        }
        write_json(path, data)
    except Exception:
        pass




def _as_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def add_perf(perf_stats: Optional[Dict[str, Any]], key: str, sec: float) -> None:
    """軽量な区間別タイマーを加算する。perf_stats=Noneなら何もしない。"""
    if perf_stats is None:
        return
    perf_stats[key] = float(perf_stats.get(key, 0.0)) + float(sec)


def inc_perf(perf_stats: Optional[Dict[str, Any]], key: str, n: int = 1) -> None:
    """回数系の計測値を加算する。"""
    if perf_stats is None:
        return
    perf_stats[key] = int(perf_stats.get(key, 0)) + int(n)


def append_perf_record(
    perf_stats: Optional[Dict[str, Any]],
    key: str,
    record: Dict[str, Any],
    max_records: int = 80,
) -> None:
    """run_summaryへ入れる診断用の小さな記録を追加する。多すぎる場合は先頭から落とす。"""
    if perf_stats is None:
        return
    lst = perf_stats.setdefault(key, [])
    if not isinstance(lst, list):
        perf_stats[key] = lst = []
    lst.append(record)
    if len(lst) > max_records:
        del lst[0:len(lst) - max_records]


def should_write_log_row(args: argparse.Namespace, status: str, memory_status: str, feature_source: str) -> bool:
    """CSV出力を間引く。fullなら従来通り、importantならVLM/新規memory/ReID中心、noneなら書かない。"""
    mode = getattr(args, "csv_log_mode", "full")
    if mode == "none":
        return False
    if mode == "full":
        return True
    if mode == "vlm_only":
        return status in {"SEND_VLM", "QUEUE_VLM"} or feature_source in {"SEND_VLM", "ASYNC_QUEUE", "ASYNC_QUEUE_BEST_CROP"}
    # important: 速度測定向け。小さい誤検出や毎フレームのMEMORY_REUSEは原則書かない。
    if status in {"SEND_VLM", "QUEUE_VLM", "TRACK_ONLY_VLM_QUEUE_FULL", "TRACK_ONLY_LIMIT"}:
        return True
    if memory_status in {"NEW_MEMORY", "REID_MATCH"}:
        return True
    return False


def build_profile_summary(perf_stats: Dict[str, Any], total_runtime_sec: float) -> Dict[str, Any]:
    """総処理時間の内訳をrun_summaryに入れる。秒数系は_pctも付ける。"""
    sec_keys = [
        "video_read_sec",
        "async_collect_sec",
        "frame_process_sec",
        "yolo_track_sec",
        "hist_sec",
        "reid_sec",
        "async_job_build_sec",
        "async_submit_sec",
        "draw_sec",
        "csv_write_sec",
        "memory_snapshot_sec",
        "video_write_sec",
        "display_sec",
        "async_final_wait_sec",
    ]
    count_keys = [
        "frames_read",
        "frames_processed",
        "frames_skipped_by_process_every",
        "detections_total",
        "csv_rows_written",
        "csv_rows_skipped",
        "async_jobs_queued",
        "async_jobs_finished",
        "memory_merge_count",
        "memory_merge_attempt_count",
        "memory_merge_blocked_count",
        "memory_merge_below_threshold_count",
        "memory_merge_soft_overlap_count",
        "best_crop_updates",
        "best_crop_wait_count",
        "best_crop_good_quality_count",
        "best_crop_max_wait_count",
        "best_crop_peak_count",
        "best_crop_end_flush_count",
        "best_crop_ready_count",
        "queued_from_ready_count",
        "dropped_low_quality_count",
        "vlm_budget_reached_count",
        "vlm_backlog_max",
        "final_flush_candidate_count",
        "final_flush_sent_count",
        "final_flush_skipped_count",
        "final_wait_timeout_count",
        "async_jobs_cancelled_at_final_timeout",
        "near_end_vlm_block_count",
    ]
    out: Dict[str, Any] = {}
    for k in sec_keys:
        v = float(perf_stats.get(k, 0.0))
        out[k] = round(v, 3)
        out[f"{k}_pct"] = round((v / total_runtime_sec) * 100.0, 1) if total_runtime_sec > 0 else None
    for k in count_keys:
        out[k] = int(perf_stats.get(k, 0))
    yolo = float(perf_stats.get("yolo_track_sec", 0.0))
    frames_processed = int(perf_stats.get("frames_processed", 0))
    # memory merge diagnostics are intentionally included in the summary because tuning merge without this is guesswork.
    out["memory_merge_attempts"] = perf_stats.get("memory_merge_attempts", []) if isinstance(perf_stats.get("memory_merge_attempts", []), list) else []
    out["memory_merge_events"] = perf_stats.get("memory_merge_events", []) if isinstance(perf_stats.get("memory_merge_events", []), list) else []
    out["best_crop_events"] = perf_stats.get("best_crop_events", []) if isinstance(perf_stats.get("best_crop_events", []), list) else []

    if frames_processed > 0:
        out["yolo_track_ms_per_processed_frame"] = round((yolo / frames_processed) * 1000.0, 2)
        fp = float(perf_stats.get("frame_process_sec", 0.0))
        out["frame_process_ms_per_processed_frame"] = round((fp / frames_processed) * 1000.0, 2)
    else:
        out["yolo_track_ms_per_processed_frame"] = None
        out["frame_process_ms_per_processed_frame"] = None
    return out


def build_run_summary(
    *,
    args: argparse.Namespace,
    input_type: str,
    out_dir: Path,
    csv_path: Path,
    target_path: Path,
    run_started_at: dt.datetime,
    run_finished_at: dt.datetime,
    total_runtime_sec: float,
    frame_count: int,
    fps: Optional[float],
    video_duration_sec: Optional[float],
    person_memories: Dict[int, Dict[str, Any]],
    async_jobs_left: int,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """実行全体の処理時間とVLM回数をまとめる。研究報告で使いやすいように独立JSONとして保存する。"""
    person_elapsed_values: List[float] = []
    item_elapsed_values: List[float] = []
    queue_to_done_values: List[float] = []
    decisions: Dict[str, int] = {}
    vlm_done_count = 0
    vlm_pending_count = 0
    vlm_error_count = 0
    best_crop_status_counts: Dict[str, int] = {}
    crop_quality_values: List[float] = []

    for _, mem in sorted(person_memories.items()):
        if mem.get("vlm_done"):
            vlm_done_count += 1
        if mem.get("vlm_pending"):
            vlm_pending_count += 1
        if mem.get("vlm_error"):
            vlm_error_count += 1

        decision = str(mem.get("decision") or "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
        best_crop_status = str(mem.get("best_crop_status") or "none")
        best_crop_status_counts[best_crop_status] = best_crop_status_counts.get(best_crop_status, 0) + 1
        cq = _as_float_or_none(mem.get("selected_crop_quality") or mem.get("best_crop_score"))
        if cq is not None:
            crop_quality_values.append(cq)

        ps = runtime_summary(mem.get("person_result"))
        pe = _as_float_or_none(ps.get("elapsed_sec"))
        if pe is not None:
            person_elapsed_values.append(pe)

        is_ = runtime_summary(mem.get("items_result"))
        ie = _as_float_or_none(is_.get("elapsed_sec"))
        if ie is not None:
            item_elapsed_values.append(ie)

        rj = mem.get("result_json") if isinstance(mem.get("result_json"), dict) else {}
        async_summary = (rj.get("runtime_summary", {}) or {}).get("async", {}) if isinstance(rj, dict) else {}
        qd = _as_float_or_none(async_summary.get("queue_to_done_sec"))
        if qd is not None:
            queue_to_done_values.append(qd)

    total_person_vlm_sec = round(sum(person_elapsed_values), 3)
    total_items_vlm_sec = round(sum(item_elapsed_values), 3)
    total_queue_to_done_sec = round(sum(queue_to_done_values), 3)

    realtime_ratio = None
    realtime_ratio_without_final_wait = None
    processing_fps = None
    async_final_wait_sec = float((perf_stats or {}).get("async_final_wait_sec", 0.0) or 0.0)
    runtime_without_final_wait = max(0.0, total_runtime_sec - async_final_wait_sec)
    if total_runtime_sec > 0:
        processing_fps = round(frame_count / total_runtime_sec, 3) if frame_count else None
    if video_duration_sec and video_duration_sec > 0:
        realtime_ratio = round(total_runtime_sec / video_duration_sec, 3)
        realtime_ratio_without_final_wait = round(runtime_without_final_wait / video_duration_sec, 3)

    return {
        "search_version": SEARCH_VERSION,
        "input_type": input_type,
        "out_dir": str(out_dir),
        "csv_path": str(csv_path),
        "target_path": str(target_path),
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "run_finished_at": run_finished_at.isoformat(timespec="seconds"),
        "total_runtime_sec": round(total_runtime_sec, 3),
        "runtime_without_final_wait_sec": round(runtime_without_final_wait, 3),
        "video_duration_sec": round(video_duration_sec, 3) if video_duration_sec is not None else None,
        "realtime_ratio_total_over_video": realtime_ratio,
        "realtime_ratio_without_final_wait": realtime_ratio_without_final_wait,
        "realtime_ratio_with_final_wait": realtime_ratio,
        "processing_fps_avg": processing_fps,
        "frame_count": frame_count,
        "source_fps": round(float(fps), 3) if fps else None,
        "memory_count": len(person_memories),
        "decision_counts_by_memory": decisions,
        "best_crop_status_counts": best_crop_status_counts,
        "ready_memory_count": best_crop_status_counts.get("READY", 0),
        "robot_eval": {
            "strong_candidate_count": decisions.get("strong_candidate", 0),
            "candidate_count": decisions.get("candidate", 0),
            "weak_candidate_count": decisions.get("weak_candidate", 0),
            "no_match_count": decisions.get("no_match", 0),
            "unknown_count": decisions.get("unknown", 0),
            "vlm_calls_per_min": round((len(person_elapsed_values) / (video_duration_sec / 60.0)), 3) if video_duration_sec and video_duration_sec > 0 else None,
            "avg_crop_quality": round(sum(crop_quality_values) / len(crop_quality_values), 3) if crop_quality_values else None,
            "min_crop_quality": round(min(crop_quality_values), 3) if crop_quality_values else None,
            "dropped_low_quality_count": best_crop_status_counts.get("DROPPED_LOW_QUALITY", 0),
            "realtime_pass": bool(realtime_ratio is not None and realtime_ratio <= 1.0),
            "realtime_pass_without_final_wait": bool(realtime_ratio_without_final_wait is not None and realtime_ratio_without_final_wait <= 1.0),
        },
        "vlm": {
            "async_vlm": bool(args.async_vlm),
            "vlm_workers": int(args.vlm_workers),
            "max_pending_vlm": int(args.max_pending_vlm),
            "wait_async_at_end": bool(args.wait_async_at_end),
            "vlm_done_count": vlm_done_count,
            "vlm_pending_count": vlm_pending_count,
            "vlm_error_count": vlm_error_count,
            "async_jobs_left": async_jobs_left,
            "person_vlm_call_count": len(person_elapsed_values),
            "items_vlm_call_count": len(item_elapsed_values),
            "person_vlm_total_elapsed_sec": total_person_vlm_sec,
            "items_vlm_total_elapsed_sec": total_items_vlm_sec,
            "vlm_total_elapsed_sec": round(total_person_vlm_sec + total_items_vlm_sec, 3),
            "queue_to_done_count": len(queue_to_done_values),
            "queue_to_done_total_sec": total_queue_to_done_sec,
            "queue_to_done_avg_sec": round(total_queue_to_done_sec / len(queue_to_done_values), 3) if queue_to_done_values else None,
            "max_total_vlm_calls": int(getattr(args, "max_total_vlm_calls", 0) or 0),
            "vlm_budget_reached_count": int((perf_stats or {}).get("vlm_budget_reached_count", 0) or 0),
            "vlm_backlog_max": int((perf_stats or {}).get("vlm_backlog_max", 0) or 0),
            "vlm_wait_after_video_end_sec": round(async_final_wait_sec, 3),
            "final_flush_candidate_count": int((perf_stats or {}).get("final_flush_candidate_count", 0) or 0),
            "final_flush_sent_count": int((perf_stats or {}).get("final_flush_sent_count", 0) or 0),
            "final_flush_skipped_count": int((perf_stats or {}).get("final_flush_skipped_count", 0) or 0),
            "final_flush_skipped_reasons": (perf_stats or {}).get("final_flush_skipped_reasons", {}),
            "final_flush_selected_memory_ids": (perf_stats or {}).get("final_flush_selected_memory_ids", []),
            "final_wait_timeout_sec": float(getattr(args, "final_wait_timeout_sec", 0.0) or 0.0),
            "final_wait_timed_out": bool((perf_stats or {}).get("final_wait_timed_out", False)),
            "final_wait_timeout_count": int((perf_stats or {}).get("final_wait_timeout_count", 0) or 0),
            "near_end_vlm_block_count": int((perf_stats or {}).get("near_end_vlm_block_count", 0) or 0),
        },
        "profiling": build_profile_summary(perf_stats or {}, total_runtime_sec),
        "settings": {
            "process_every": args.process_every,
            "csv_log_mode": getattr(args, "csv_log_mode", "full"),
            "save_annotated_video": bool(args.save_annotated_video),
            "show": bool(args.show),
            "min_area_ratio_for_vlm": args.min_area_ratio_for_vlm,
            "min_height_ratio_for_vlm": args.min_height_ratio_for_vlm,
            "vlm_min_track_frames": args.vlm_min_track_frames,
            "skip_pseudo_track_vlm": bool(args.skip_pseudo_track_vlm),
            "reid_match_threshold": args.reid_match_threshold,
            "reid_memory_ttl_sec": args.reid_memory_ttl_sec,
            "enable_memory_merge": bool(getattr(args, "enable_memory_merge", True)),
            "merge_memory_threshold": getattr(args, "merge_memory_threshold", None),
            "merge_memory_ttl_sec": getattr(args, "merge_memory_ttl_sec", None),
            "merge_min_color_score": getattr(args, "merge_min_color_score", None),
            "merge_soft_overlap": bool(getattr(args, "merge_soft_overlap", True)),
            "merge_soft_overlap_max_sec": getattr(args, "merge_soft_overlap_max_sec", None),
            "merge_soft_overlap_attr_score": getattr(args, "merge_soft_overlap_attr_score", None),
            "merge_soft_overlap_candidate_bonus": getattr(args, "merge_soft_overlap_candidate_bonus", None),
            "merge_soft_overlap_low_quality": getattr(args, "merge_soft_overlap_low_quality", None),
            "merge_soft_overlap_edge_hits": getattr(args, "merge_soft_overlap_edge_hits", None),
            "merge_soft_overlap_width_score": getattr(args, "merge_soft_overlap_width_score", None),
            "merge_soft_overlap_allow_color_hist": bool(getattr(args, "merge_soft_overlap_allow_color_hist", True)),
            "merge_soft_overlap_min_color_score": getattr(args, "merge_soft_overlap_min_color_score", None),
            "merge_debug_max_records": getattr(args, "merge_debug_max_records", None),
            "enable_best_crop_timing": bool(getattr(args, "enable_best_crop_timing", True)),
            "best_crop_max_wait_sec": getattr(args, "best_crop_max_wait_sec", None),
            "best_crop_good_threshold": getattr(args, "best_crop_good_threshold", None),
            "best_crop_min_threshold": getattr(args, "best_crop_min_threshold", None),
            "best_crop_good_max_edge_hits": getattr(args, "best_crop_good_max_edge_hits", None),
            "best_crop_good_min_width_score": getattr(args, "best_crop_good_min_width_score", None),
            "best_crop_patience_frames": getattr(args, "best_crop_patience_frames", None),
            "min_width_ratio_for_vlm": getattr(args, "min_width_ratio_for_vlm", None),
            "max_total_vlm_calls": getattr(args, "max_total_vlm_calls", None),
            "max_vlm_calls_per_min": getattr(args, "max_vlm_calls_per_min", None),
            "final_flush_max_memories": getattr(args, "final_flush_max_memories", None),
            "final_flush_min_quality": getattr(args, "final_flush_min_quality", None),
            "final_flush_min_width_score": getattr(args, "final_flush_min_width_score", None),
            "final_flush_min_height_score": getattr(args, "final_flush_min_height_score", None),
            "final_flush_max_edge_hits": getattr(args, "final_flush_max_edge_hits", None),
            "final_flush_min_target_color_score": getattr(args, "final_flush_min_target_color_score", None),
            "final_flush_target_color_rescue_score": getattr(args, "final_flush_target_color_rescue_score", None),
            "final_flush_quality_floor_with_target_color": getattr(args, "final_flush_quality_floor_with_target_color", None),
            "final_wait_timeout_sec": getattr(args, "final_wait_timeout_sec", None),
            "near_end_vlm_guard_sec": getattr(args, "near_end_vlm_guard_sec", None),
            "near_end_min_target_color_score": getattr(args, "near_end_min_target_color_score", None),
            "near_end_min_quality": getattr(args, "near_end_min_quality", None),
        },
    }

def make_new_memory(
    memory_id: int,
    track_id: int,
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    frame_count: int,
    time_sec: float,
) -> Dict[str, Any]:
    bbox = det["bbox"]
    return {
        "memory_id": memory_id,
        "track_ids": [track_id] if det.get("track_id_is_real", False) else [],
        "last_track_id": track_id if det.get("track_id_is_real", False) else None,
        "last_seen_frame": frame_count,
        "last_seen_time": round(time_sec, 3),
        "last_bbox": list(bbox),
        "last_center": list(bbox_center(bbox)),
        "last_area": float(det.get("area", 0.0) or 0.0),
        "last_area_ratio": float(det.get("area_ratio", 0.0) or 0.0),
        "last_height_ratio": float(det.get("height_ratio", 0.0) or 0.0),
        "last_conf": float(det.get("conf", 0.0) or 0.0),
        "color_hist": hist,
        "vlm_done": False,
        "decision": "",
        "person_score": "",
        "final_score": "",
        "person_result": None,
        "items_result": None,
        "result_json": None,
        "person_json_path": "",
        "items_json_path": "",
        "result_json_path": "",
        "person_crop_path": "",
        "items_crop_path": "",
        "created_frame": frame_count,
        "created_time": round(time_sec, 3),
    }


def update_memory_observation(
    memory: Dict[str, Any],
    track_id: int,
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    frame_count: int,
    time_sec: float,
) -> None:
    bbox = det["bbox"]
    if det.get("track_id_is_real", False):
        memory["last_track_id"] = track_id
        ids = list(memory.get("track_ids", []))
        if track_id not in ids:
            ids.append(track_id)
        memory["track_ids"] = ids
    memory["last_seen_frame"] = frame_count
    memory["last_seen_time"] = round(time_sec, 3)
    memory["last_bbox"] = list(bbox)
    memory["last_center"] = list(bbox_center(bbox))
    memory["last_area"] = float(det.get("area", 0.0) or 0.0)
    memory["last_area_ratio"] = float(det.get("area_ratio", 0.0) or 0.0)
    memory["last_height_ratio"] = float(det.get("height_ratio", 0.0) or 0.0)
    memory["last_conf"] = float(det.get("conf", 0.0) or 0.0)
    if hist is not None:
        # 急な照明変化に少しだけ追従させる。ただし過去特徴を完全には捨てない。
        prev = memory.get("color_hist")
        if isinstance(prev, np.ndarray) and prev.shape == hist.shape:
            mixed = (prev * 0.7 + hist * 0.3).astype(np.float32)
            s = float(np.sum(mixed))
            memory["color_hist"] = mixed / s if s > 0 else hist
        else:
            memory["color_hist"] = hist


def compute_memory_match_score(
    memory: Dict[str, Any],
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    time_sec: float,
    frame_shape: Tuple[int, int, int],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, Any]]:
    h, w = frame_shape[:2]
    diag = float(np.hypot(w, h)) or 1.0
    cur_center = bbox_center(det["bbox"])
    prev_center = memory.get("last_center") or cur_center
    dx = float(cur_center[0]) - float(prev_center[0])
    dy = float(cur_center[1]) - float(prev_center[1])
    center_dist_ratio = float(np.hypot(dx, dy)) / diag
    pos_score = max(0.0, min(1.0, 1.0 - center_dist_ratio / max(1e-6, args.reid_max_center_distance_ratio)))

    prev_area = float(memory.get("last_area", 0.0) or 0.0)
    cur_area = float(det.get("area", 0.0) or 0.0)
    if prev_area > 0 and cur_area > 0:
        size_score = min(prev_area, cur_area) / max(prev_area, cur_area)
    else:
        size_score = 0.0

    color_score = hist_similarity(memory.get("color_hist"), hist)

    last_seen_time = float(memory.get("last_seen_time", time_sec) or time_sec)
    dt_sec = max(0.0, time_sec - last_seen_time)
    if dt_sec > args.reid_memory_ttl_sec:
        age_score = 0.0
    else:
        age_score = max(0.0, 1.0 - dt_sec / max(1e-6, args.reid_memory_ttl_sec))

    total_w = args.reid_position_weight + args.reid_color_weight + args.reid_size_weight + args.reid_age_weight
    if total_w <= 0:
        total_w = 1.0
    score = (
        args.reid_position_weight * pos_score
        + args.reid_color_weight * color_score
        + args.reid_size_weight * size_score
        + args.reid_age_weight * age_score
    ) / total_w

    detail = {
        "score": round(float(score), 3),
        "position_score": round(float(pos_score), 3),
        "color_score": round(float(color_score), 3),
        "size_score": round(float(size_score), 3),
        "age_score": round(float(age_score), 3),
        "center_dist_ratio": round(float(center_dist_ratio), 3),
        "time_gap_sec": round(float(dt_sec), 3),
    }
    return float(score), detail


def find_best_memory_for_det(
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    time_sec: float,
    frame_shape: Tuple[int, int, int],
    args: argparse.Namespace,
    person_memories: Dict[int, Dict[str, Any]],
    used_memory_ids: set,
) -> Tuple[Optional[int], float, Dict[str, Any]]:
    best_id: Optional[int] = None
    best_score = -1.0
    best_detail: Dict[str, Any] = {}
    for memory_id, memory in person_memories.items():
        if memory_id in used_memory_ids:
            continue
        # VLM済みの人物だけ引き継ぐ。未判定memoryは誤紐付けの価値が低い。
        if not memory.get("vlm_done", False):
            continue
        last_seen_time = float(memory.get("last_seen_time", time_sec) or time_sec)
        if time_sec - last_seen_time > args.reid_memory_ttl_sec:
            continue
        score, detail = compute_memory_match_score(memory, det, hist, time_sec, frame_shape, args)
        if detail.get("center_dist_ratio", 999) > args.reid_max_center_distance_ratio:
            continue
        if detail.get("color_score", 0.0) < args.reid_min_color_score:
            continue
        if score > best_score:
            best_id = memory_id
            best_score = score
            best_detail = detail
    if best_id is not None and best_score >= args.reid_match_threshold:
        return best_id, best_score, best_detail
    return None, max(0.0, best_score), best_detail


# =========================================================
# Output helpers
# =========================================================
def runtime_summary(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    rt = obj.get("_runtime", {}) if isinstance(obj.get("_runtime"), dict) else {}
    usage = obj.get("_lmstudio_usage", {}) if isinstance(obj.get("_lmstudio_usage"), dict) else {}
    return {
        "elapsed_sec": rt.get("elapsed_sec"),
        "prompt_tokens": rt.get("prompt_tokens") or usage.get("prompt_tokens"),
        "completion_tokens": rt.get("completion_tokens") or usage.get("completion_tokens"),
        "total_tokens": rt.get("total_tokens") or usage.get("total_tokens"),
        "prompt_version": rt.get("prompt_version"),
    }


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def open_csv_writer(csv_path: Path) -> Tuple[Any, csv.DictWriter]:
    f = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "frame", "time_sec", "track_id", "rank_by_area", "status", "reason",
        "memory_id", "memory_status", "reid_score", "reid_detail", "matched_previous_track_id", "vlm_already_done",
        "decision", "final_score", "person_score", "item_score", "items_status",
        "match_reason", "partial_match_detail",
        "conf", "area_ratio", "height_ratio", "bbox",
        "person_json_path", "items_json_path", "result_json_path",
        "person_crop_path", "items_crop_path",
        "person_elapsed_sec", "items_elapsed_sec", "feature_source", "cached_from_frame", "cache_age_sec",
        "crop_quality_score", "best_crop_status", "selected_crop_frame", "selected_crop_time", "selected_crop_reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def draw_label(annotated: np.ndarray, bbox: Tuple[int, int, int, int], lines: List[str], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    y = max(20, y1 - 8 - 18 * (len(lines) - 1))
    for line in lines:
        cv2.putText(annotated, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        y += 18


def compact_match_reason(match: Dict[str, Any]) -> str:
    if not isinstance(match, dict):
        return ""
    person = match.get("person_match", {}) if isinstance(match.get("person_match"), dict) else {}
    matched = set(person.get("matched", []) or [])
    unknown = set(person.get("unknown", []) or [])
    mismatch_fields = {m.get("field") for m in person.get("mismatch", []) or [] if isinstance(m, dict)}
    partial_reasons = {
        p.get("field"): str(p.get("reason", "near"))
        for p in person.get("partial", []) or []
        if isinstance(p, dict)
    }

    def mark(field: str, label: str) -> str:
        if field in matched:
            return f"{label}:OK"
        if field in partial_reasons:
            reason = partial_reasons[field]
            if reason.startswith("unknown"):
                return f"{label}:?"
            return f"{label}:near"
        if field in unknown:
            return f"{label}:?"
        if field in mismatch_fields:
            return f"{label}:NG"
        return f"{label}:-"

    return " ".join([
        mark("upper_color", "color"),
        mark("upper_type", "type"),
        mark("apparent_gender", "gender"),
    ])


def decision_color(status: str, decision: str) -> Tuple[int, int, int]:
    if status in {"REUSE_CACHE", "REUSE_MEMORY"}:
        return (255, 0, 255)  # purple
    if status.startswith("TRACK_ONLY"):
        return (0, 165, 255)  # orange
    if decision == "strong_candidate":
        return (0, 0, 255)  # red
    if decision == "candidate":
        return (0, 255, 0)  # green
    if decision == "weak_candidate":
        return (0, 220, 255)  # yellow
    return (255, 255, 0)  # cyan


# =========================================================
# Main frame processing
# =========================================================
def process_frame(
    frame: np.ndarray,
    frame_count: int,
    time_sec: float,
    model: YOLO,
    args: argparse.Namespace,
    target: Dict[str, Any],
    dirs: Dict[str, Path],
    csv_writer: csv.DictWriter,
    person_memories: Dict[int, Dict[str, Any]],
    track_to_memory: Dict[int, int],
    next_memory_id_ref: Dict[str, int],
    perf_stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    _t0 = time.perf_counter()
    dets = detect_persons_with_tracking(model, frame, args)
    add_perf(perf_stats, "yolo_track_sec", time.perf_counter() - _t0)
    inc_perf(perf_stats, "detections_total", len(dets))
    sent_count = 0
    now_time = time_sec
    used_memory_ids: set = set()

    for det in dets:
        track_id = int(det["track_id"])
        bbox = det["bbox"]
        track_id_is_real = bool(det.get("track_id_is_real", False))
        _t_hist = time.perf_counter()
        hist = compute_upper_body_hist(frame, bbox) if args.enable_person_memory else None
        add_perf(perf_stats, "hist_sec", time.perf_counter() - _t_hist)

        reason_parts: List[str] = []
        status: Optional[str] = None
        memory_id: Optional[int] = None
        memory_status = ""
        reid_score: Any = ""
        reid_detail: Dict[str, Any] = {}
        matched_previous_track_id: Any = ""
        vlm_already_done: Any = ""

        # 1) 既存track_idがmemoryに紐づいている場合は、同一人物として即再利用。
        if args.enable_person_memory and track_id_is_real and track_id in track_to_memory:
            mid = track_to_memory.get(track_id)
            if mid in person_memories and mid not in used_memory_ids:
                entry0 = person_memories[mid]
                memory_id = mid
                # best crop待機中のmemoryは、同じtrack_idで引き続き候補を更新する。
                # VLM済み/処理中になってから初めてREUSE_MEMORY扱いにする。
                if entry0.get("vlm_done", False) or entry0.get("vlm_pending", False):
                    memory_status = "TRACK_ID_MATCH"
                    status = "REUSE_MEMORY"
                    reason_parts.append("MEMORY_BY_TRACK_ID")
                else:
                    memory_status = "TRACK_ID_CANDIDATE"
                    reason_parts.append("MEMORY_CANDIDATE_BY_TRACK_ID")

        # 2) track_idが新規/途切れた場合だけ、直近memoryと軽量ReID比較。
        if args.enable_person_memory and status is None and memory_id is None:
            best_mid, best_score, best_detail = find_best_memory_for_det(
                det=det,
                hist=hist,
                time_sec=now_time,
                frame_shape=frame.shape,
                args=args,
                person_memories=person_memories,
                used_memory_ids=used_memory_ids,
            )
            add_perf(perf_stats, "reid_sec", time.perf_counter() - _t_reid)
            if best_mid is not None:
                memory_id = best_mid
                memory_status = "REID_MATCH"
                reid_score = round(best_score, 3)
                reid_detail = best_detail
                prev_tid = person_memories[best_mid].get("last_track_id")
                matched_previous_track_id = "" if prev_tid is None else prev_tid
                if track_id_is_real:
                    track_to_memory[track_id] = best_mid
                status = "REUSE_MEMORY"
                reason_parts.append(f"MEMORY_BY_REID_{best_score:.2f}")
            else:
                memory_status = "NEW_OR_UNMATCHED"
                if best_score > 0:
                    reid_score = round(best_score, 3)
                    reid_detail = best_detail
                    reason_parts.append(f"REID_NO_MATCH_{best_score:.2f}")

        # 3) memory再利用できない場合だけ、VLMに送れるか判定。
        if status is None:
            if (not track_id_is_real) and args.skip_pseudo_track_vlm:
                status = "TRACK_ONLY_PSEUDO_ID"
                reason_parts.append("PSEUDO_TRACK_ID")
            elif det["conf"] < args.min_conf_for_vlm:
                status = "TRACK_ONLY_LOW_CONF"
                reason_parts.append("LOW_CONF")
            elif det["area_ratio"] < args.min_area_ratio_for_vlm:
                status = "TRACK_ONLY_SMALL"
                reason_parts.append("SMALL_AREA")
            elif det["height_ratio"] < args.min_height_ratio_for_vlm:
                status = "TRACK_ONLY_SMALL"
                reason_parts.append("SMALL_HEIGHT")

        if status is None:
            if args.max_vlm_per_frame > 0 and sent_count >= args.max_vlm_per_frame:
                status = "TRACK_ONLY_LIMIT"
                reason_parts.append("MAX_VLM_PER_FRAME")
            else:
                status = "SEND_VLM"
                reason_parts.append("OK")

        person_result: Optional[Dict[str, Any]] = None
        items_result: Optional[Dict[str, Any]] = None
        result_json: Optional[Dict[str, Any]] = None
        person_json_path = ""
        items_json_path = ""
        result_json_path = ""
        person_crop_path = ""
        items_crop_path = ""
        feature_source = status
        cached_from_frame = ""
        cache_age_sec = ""

        if status == "SEND_VLM":
            sent_count += 1
            if args.enable_person_memory:
                if memory_id is None:
                    memory_id = int(next_memory_id_ref["next"])
                    next_memory_id_ref["next"] += 1
                    person_memories[memory_id] = make_new_memory(memory_id, track_id, det, hist, frame_count, now_time)
                    memory_status = "NEW_MEMORY"
                if track_id_is_real:
                    track_to_memory[track_id] = memory_id

            mem_suffix = f"_mem{memory_id}" if memory_id is not None else ""
            base_name = f"frame{frame_count:06d}_track{track_id}{mem_suffix}"

            # 1段目: person-only
            person_crop, person_box = crop_person(frame, bbox, margin_ratio=args.person_crop_margin)
            person_crop = resize_long_side(person_crop, args.person_vlm_long_side)
            person_crop_file = dirs["person_crops"] / f"{base_name}_person.jpg"
            cv2.imwrite(str(person_crop_file), person_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            person_crop_path = str(person_crop_file)

            person_result = call_lmstudio_json(
                image_path=person_crop_file,
                prompt=PERSON_PROMPT,
                base_url=args.lmstudio_base_url,
                model=args.vlm_model,
                timeout_sec=args.lmstudio_timeout,
                max_tokens=args.person_max_tokens,
                kind="person",
            )
            person_result.setdefault("_runtime", {})
            person_result["_runtime"].update({
                "frame": frame_count,
                "time_sec": round(time_sec, 3),
                "track_id": track_id,
                "memory_id": memory_id,
                "rank_by_area": det["rank_by_area"],
                "bbox": list(bbox),
                "conf": det["conf"],
                "area_ratio": det["area_ratio"],
                "height_ratio": det["height_ratio"],
                "crop_path": person_crop_path,
            })
            person_json_file = dirs["person_json"] / f"{base_name}_person.json"
            write_json(person_json_file, person_result)
            person_json_path = str(person_json_file)

            # 2段目: items。target_item unspecified/anyなら完全に飛ばす。target_item noneは「持ち物なし」条件なので実行する。
            items_status = "SKIPPED_TARGET_ITEM_UNSPECIFIED"
            if should_run_items(target):
                mx, mt, mb = get_target_item_margins(target.get("target_item", "none"))
                if args.item_margin_x >= 0:
                    mx = args.item_margin_x
                if args.item_margin_top >= 0:
                    mt = args.item_margin_top
                if args.item_margin_bottom >= 0:
                    mb = args.item_margin_bottom

                item_crop, context_box = make_item_context_crop(
                    frame,
                    bbox,
                    margin_x=mx,
                    margin_top=mt,
                    margin_bottom=mb,
                    draw_target_box=not args.no_draw_target_box,
                )
                item_crop = resize_long_side(item_crop, args.items_vlm_long_side)
                item_crop_file = dirs["items_crops"] / f"{base_name}_items_context.jpg"
                cv2.imwrite(str(item_crop_file), item_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                items_crop_path = str(item_crop_file)

                items_result = call_lmstudio_json(
                    image_path=item_crop_file,
                    prompt=build_items_prompt(target.get("target_item", "none")),
                    base_url=args.lmstudio_base_url,
                    model=args.vlm_model,
                    timeout_sec=args.lmstudio_timeout,
                    max_tokens=args.items_max_tokens,
                    kind="items",
                )
                items_result.setdefault("_runtime", {})
                items_result["_runtime"].update({
                    "frame": frame_count,
                    "time_sec": round(time_sec, 3),
                    "track_id": track_id,
                    "memory_id": memory_id,
                    "rank_by_area": det["rank_by_area"],
                    "bbox": list(bbox),
                    "conf": det["conf"],
                    "area_ratio": det["area_ratio"],
                    "height_ratio": det["height_ratio"],
                    "context_box": list(context_box),
                    "item_margin_x": mx,
                    "item_margin_top": mt,
                    "item_margin_bottom": mb,
                    "target_item": target.get("target_item", "none"),
                    "crop_path": items_crop_path,
                })
                items_json_file = dirs["items_json"] / f"{base_name}_items.json"
                write_json(items_json_file, items_result)
                items_json_path = str(items_json_file)
                items_status = "RUN_ITEMS_VLM"

            match = build_match_json(target, person_result, items_result)
            if items_result is None:
                match["item_match"]["status"] = items_status

            result_json = {
                "search_version": SEARCH_VERSION,
                "target": target,
                "observed": {
                    "person": person_result.get("person", {}),
                    "head": person_result.get("head", {}),
                    "par_attributes": person_result.get("par_attributes", {}),
                    "visibility": person_result.get("visibility", {}),
                    "items": items_result.get("items", {}) if isinstance(items_result, dict) else None,
                },
                "match": match,
                "track": {
                    "frame": frame_count,
                    "time_sec": round(time_sec, 3),
                    "track_id": track_id,
                    "track_id_is_real": track_id_is_real,
                    "memory_id": memory_id,
                    "memory_status": memory_status,
                    "rank_by_area": det["rank_by_area"],
                    "bbox": list(bbox),
                    "conf": det["conf"],
                    "area_ratio": det["area_ratio"],
                    "height_ratio": det["height_ratio"],
                    "status": status,
                    "reason": "+".join(reason_parts),
                },
                "sources": {
                    "person_json_path": person_json_path,
                    "items_json_path": items_json_path,
                    "result_json_path": "",
                    "person_crop_path": person_crop_path,
                    "items_crop_path": items_crop_path,
                },
                "runtime_summary": {
                    "person": runtime_summary(person_result),
                    "items": runtime_summary(items_result),
                },
            }

            result_json_file = dirs["result_json"] / f"{base_name}_result.json"
            result_json["sources"]["result_json_path"] = str(result_json_file)
            write_json(result_json_file, result_json)
            result_json_path = str(result_json_file)

            if args.enable_person_memory and memory_id is not None:
                mem = person_memories[memory_id]
                update_memory_observation(mem, track_id, det, hist, frame_count, now_time)
                used_memory_ids.add(memory_id)
                mem.update({
                    "vlm_done": True,
                    "decision": match.get("decision", ""),
                    "person_score": match.get("person_score", ""),
                    "final_score": match.get("final_score", ""),
                    "person_result": person_result,
                    "items_result": items_result,
                    "result_json": result_json,
                    "person_json_path": person_json_path,
                    "items_json_path": items_json_path,
                    "result_json_path": result_json_path,
                    "person_crop_path": person_crop_path,
                    "items_crop_path": items_crop_path,
                    "created_frame": mem.get("created_frame", frame_count),
                    "created_time": mem.get("created_time", round(now_time, 3)),
                })
                _t_snap = time.perf_counter()
                write_memory_snapshot(dirs, person_memories)
                add_perf(perf_stats, "memory_snapshot_sec", time.perf_counter() - _t_snap)

        elif status == "REUSE_MEMORY" and args.enable_person_memory and memory_id in person_memories:
            used_memory_ids.add(memory_id)
            entry = person_memories[memory_id]
            person_result = entry.get("person_result")
            items_result = entry.get("items_result")
            result_json = entry.get("result_json")
            person_json_path = entry.get("person_json_path", "")
            items_json_path = entry.get("items_json_path", "")
            result_json_path = entry.get("result_json_path", "")
            person_crop_path = entry.get("person_crop_path", "")
            items_crop_path = entry.get("items_crop_path", "")
            cached_from_frame = str(entry.get("created_frame", ""))
            cache_age_sec = f"{now_time - float(entry.get('created_time', now_time)):.2f}"
            vlm_already_done = bool(entry.get("vlm_done", False))
            feature_source = "MEMORY_REUSE"
            update_memory_observation(entry, track_id, det, hist, frame_count, now_time)
            write_memory_snapshot(dirs, person_memories)

        # 描画・CSV
        match = result_json.get("match", {}) if isinstance(result_json, dict) else {}
        decision = match.get("decision", "")
        final_score = match.get("final_score", "")
        person_score = match.get("person_score", "")
        item_score = match.get("item_score", "")
        items_status = ""
        if isinstance(match.get("item_match"), dict):
            items_status = match["item_match"].get("status", "")

        color = decision_color(status, decision)
        label_status = {
            "SEND_VLM": "SEND",
            "REUSE_CACHE": "CACHE",
            "REUSE_MEMORY": "MEM",
            "TRACK_ONLY_SMALL": "SMALL",
            "TRACK_ONLY_LIMIT": "LIMIT",
            "TRACK_ONLY_LOW_CONF": "LOWCONF",
            "TRACK_ONLY_PSEUDO_ID": "PSEUDO",
        }.get(status, status)
        mem_label = f" M:{memory_id}" if memory_id is not None else ""
        reason_line = compact_match_reason(match)
        lines = [
            f"ID:{track_id}{mem_label} {label_status}",
            f"score:{final_score} {decision}" if final_score != "" else f"rank:{det['rank_by_area']}",
            f"a:{det['area_ratio']:.3f} h:{det['height_ratio']:.2f} c:{det['conf']:.2f}",
        ]
        if reason_line:
            lines.insert(2, reason_line)
        _t_draw = time.perf_counter()
        draw_label(annotated, bbox, lines, color)
        add_perf(perf_stats, "draw_sec", time.perf_counter() - _t_draw)

        if should_write_log_row(args, status, memory_status, feature_source or status):
            _t_csv = time.perf_counter()
            csv_writer.writerow({
            "frame": frame_count,
            "time_sec": round(time_sec, 3),
            "track_id": track_id,
            "rank_by_area": det["rank_by_area"],
            "status": status,
            "reason": "+".join(reason_parts),
            "memory_id": memory_id if memory_id is not None else "",
            "memory_status": memory_status,
            "reid_score": reid_score,
            "reid_detail": json.dumps(reid_detail, ensure_ascii=False) if reid_detail else "",
            "matched_previous_track_id": matched_previous_track_id,
            "vlm_already_done": vlm_already_done,
            "decision": decision,
            "final_score": final_score,
            "person_score": person_score,
            "item_score": item_score,
            "items_status": items_status,
            "match_reason": reason_line,
            "partial_match_detail": json.dumps(match.get("partial", []), ensure_ascii=False) if isinstance(match, dict) else "",
            "conf": round(det["conf"], 4),
            "area_ratio": round(det["area_ratio"], 5),
            "height_ratio": round(det["height_ratio"], 5),
            "bbox": list(bbox),
            "person_json_path": person_json_path,
            "items_json_path": items_json_path,
            "result_json_path": result_json_path,
            "person_crop_path": person_crop_path,
            "items_crop_path": items_crop_path,
            "person_elapsed_sec": runtime_summary(person_result).get("elapsed_sec", "") if person_result else "",
            "items_elapsed_sec": runtime_summary(items_result).get("elapsed_sec", "") if items_result else "",
            "feature_source": feature_source,
            "cached_from_frame": cached_from_frame,
            "cache_age_sec": cache_age_sec,
            "crop_quality_score": crop_quality_score,
            "selected_crop_frame": selected_crop_frame,
            "selected_crop_time": selected_crop_time,
            "selected_crop_reason": selected_crop_reason,
            })
            add_perf(perf_stats, "csv_write_sec", time.perf_counter() - _t_csv)
            inc_perf(perf_stats, "csv_rows_written")
        else:
            inc_perf(perf_stats, "csv_rows_skipped")

    return annotated




# =========================================================
# Practical live/async VLM helpers
# =========================================================
def find_best_memory_for_det_live(
    det: Dict[str, Any],
    hist: Optional[np.ndarray],
    time_sec: float,
    frame_shape: Tuple[int, ...],
    args: argparse.Namespace,
    person_memories: Dict[int, Dict[str, Any]],
    used_memory_ids: set,
) -> Tuple[Optional[int], float, Dict[str, Any]]:
    """実用想定版: VLM済みだけでなく、VLM処理中(pending)のmemoryにも紐付ける。"""
    best_id: Optional[int] = None
    best_score = -1.0
    best_detail: Dict[str, Any] = {}
    for memory_id, memory in person_memories.items():
        if memory_id in used_memory_ids:
            continue
        if not (memory.get("vlm_done", False) or memory.get("vlm_pending", False)):
            continue
        last_seen_time = float(memory.get("last_seen_time", time_sec) or time_sec)
        if time_sec - last_seen_time > args.reid_memory_ttl_sec:
            continue
        score, detail = compute_memory_match_score(memory, det, hist, time_sec, frame_shape, args)
        if score > best_score:
            best_score = score
            best_id = memory_id
            best_detail = detail
    if best_id is not None and best_score >= args.reid_match_threshold:
        return best_id, best_score, best_detail
    return None, max(0.0, best_score), best_detail


def build_async_vlm_job(
    *,
    frame: np.ndarray,
    det: Dict[str, Any],
    bbox: Tuple[int, int, int, int],
    track_id: int,
    track_id_is_real: bool,
    memory_id: int,
    memory_status: str,
    frame_count: int,
    time_sec: float,
    now_time: float,
    target: Dict[str, Any],
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    reason: str,
) -> Dict[str, Any]:
    mem_suffix = f"_mem{memory_id}"
    base_name = f"frame{frame_count:06d}_track{track_id}{mem_suffix}"

    person_crop, _ = crop_person(frame, bbox, margin_ratio=args.person_crop_margin)
    person_crop = resize_long_side(person_crop, args.person_vlm_long_side)
    person_crop_file = dirs["person_crops"] / f"{base_name}_person.jpg"
    cv2.imwrite(str(person_crop_file), person_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    item_crop_file: Optional[Path] = None
    context_box: Optional[Tuple[int, int, int, int]] = None
    item_margins = None
    if should_run_items(target):
        mx, mt, mb = get_target_item_margins(target.get("target_item", "none"))
        if args.item_margin_x >= 0:
            mx = args.item_margin_x
        if args.item_margin_top >= 0:
            mt = args.item_margin_top
        if args.item_margin_bottom >= 0:
            mb = args.item_margin_bottom
        item_crop, context_box = make_item_context_crop(
            frame,
            bbox,
            margin_x=mx,
            margin_top=mt,
            margin_bottom=mb,
            draw_target_box=not args.no_draw_target_box,
        )
        item_crop = resize_long_side(item_crop, args.items_vlm_long_side)
        item_crop_file = dirs["items_crops"] / f"{base_name}_items_context.jpg"
        cv2.imwrite(str(item_crop_file), item_crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        item_margins = (mx, mt, mb)

    return {
        "base_name": base_name,
        "frame": frame_count,
        "time_sec": round(time_sec, 3),
        "track_id": track_id,
        "track_id_is_real": track_id_is_real,
        "memory_id": memory_id,
        "memory_status": memory_status,
        "rank_by_area": det["rank_by_area"],
        "bbox": list(bbox),
        "conf": det["conf"],
        "area_ratio": det["area_ratio"],
        "height_ratio": det["height_ratio"],
        "crop_quality_score": det.get("crop_quality_score", ""),
        "crop_quality_detail": det.get("crop_quality_detail", {}),
        "target": target,
        "args_snapshot": {
            "lmstudio_base_url": args.lmstudio_base_url,
            "vlm_model": args.vlm_model,
            "lmstudio_timeout": args.lmstudio_timeout,
            "person_max_tokens": args.person_max_tokens,
            "items_max_tokens": args.items_max_tokens,
        },
        "person_crop_file": person_crop_file,
        "item_crop_file": item_crop_file,
        "context_box": list(context_box) if context_box is not None else None,
        "item_margins": item_margins,
        "dirs": dirs,
        "queued_wall_time": time.time(),
        "queued_at": dt.datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
    }


def run_async_vlm_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """別スレッドでVLMを実行する。メインの動画処理は待たない。"""
    snap = job["args_snapshot"]
    target = job["target"]
    dirs = job["dirs"]
    base_name = job["base_name"]
    memory_id = job["memory_id"]
    track_id = job["track_id"]

    person_crop_file: Path = job["person_crop_file"]
    person_result = call_lmstudio_json(
        image_path=person_crop_file,
        prompt=PERSON_PROMPT,
        base_url=snap["lmstudio_base_url"],
        model=snap["vlm_model"],
        timeout_sec=snap["lmstudio_timeout"],
        max_tokens=snap["person_max_tokens"],
        kind="person",
    )
    person_result.setdefault("_runtime", {})
    person_result["_runtime"].update({
        "frame": job["frame"],
        "time_sec": job["time_sec"],
        "track_id": track_id,
        "memory_id": memory_id,
        "rank_by_area": job["rank_by_area"],
        "bbox": job["bbox"],
        "conf": job["conf"],
        "area_ratio": job["area_ratio"],
        "height_ratio": job["height_ratio"],
        "crop_path": str(person_crop_file),
        "crop_quality_score": job.get("crop_quality_score", ""),
        "crop_quality_detail": job.get("crop_quality_detail", {}),
        "async_vlm": True,
        "queued_at": job.get("queued_at"),
    })
    person_json_file = dirs["person_json"] / f"{base_name}_person.json"
    write_json(person_json_file, person_result)

    items_result: Optional[Dict[str, Any]] = None
    items_json_file: Optional[Path] = None
    items_status = "SKIPPED_TARGET_ITEM_UNSPECIFIED"
    item_crop_file = job.get("item_crop_file")
    if item_crop_file is not None:
        items_result = call_lmstudio_json(
            image_path=item_crop_file,
            prompt=build_items_prompt(target.get("target_item", "none")),
            base_url=snap["lmstudio_base_url"],
            model=snap["vlm_model"],
            timeout_sec=snap["lmstudio_timeout"],
            max_tokens=snap["items_max_tokens"],
            kind="items",
        )
        items_result.setdefault("_runtime", {})
        mx, mt, mb = job.get("item_margins") or (None, None, None)
        items_result["_runtime"].update({
            "frame": job["frame"],
            "time_sec": job["time_sec"],
            "track_id": track_id,
            "memory_id": memory_id,
            "rank_by_area": job["rank_by_area"],
            "bbox": job["bbox"],
            "conf": job["conf"],
            "area_ratio": job["area_ratio"],
            "height_ratio": job["height_ratio"],
            "context_box": job.get("context_box"),
            "item_margin_x": mx,
            "item_margin_top": mt,
            "item_margin_bottom": mb,
            "target_item": target.get("target_item", "none"),
            "crop_path": str(item_crop_file),
            "async_vlm": True,
            "queued_at": job.get("queued_at"),
        })
        items_json_file = dirs["items_json"] / f"{base_name}_items.json"
        write_json(items_json_file, items_result)
        items_status = "RUN_ITEMS_VLM"

    match = build_match_json(target, person_result, items_result)
    if items_result is None:
        match["item_match"]["status"] = items_status

    result_json = {
        "search_version": SEARCH_VERSION,
        "target": target,
        "observed": {
            "person": person_result.get("person", {}),
            "head": person_result.get("head", {}),
            "par_attributes": person_result.get("par_attributes", {}),
            "visibility": person_result.get("visibility", {}),
            "items": items_result.get("items", {}) if isinstance(items_result, dict) else None,
        },
        "match": match,
        "track": {
            "frame": job["frame"],
            "time_sec": job["time_sec"],
            "track_id": track_id,
            "track_id_is_real": job["track_id_is_real"],
            "memory_id": memory_id,
            "memory_status": job["memory_status"],
            "rank_by_area": job["rank_by_area"],
            "bbox": job["bbox"],
            "conf": job["conf"],
            "area_ratio": job["area_ratio"],
            "height_ratio": job["height_ratio"],
            "status": "ASYNC_VLM_DONE",
            "reason": job.get("reason", ""),
        },
        "sources": {
            "person_json_path": str(person_json_file),
            "items_json_path": str(items_json_file) if items_json_file else "",
            "result_json_path": "",
            "person_crop_path": str(person_crop_file),
            "items_crop_path": str(item_crop_file) if item_crop_file else "",
        },
        "runtime_summary": {
            "person": runtime_summary(person_result),
            "items": runtime_summary(items_result),
            "async": {
                "queued_at": job.get("queued_at"),
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "queue_to_done_sec": round(time.time() - float(job.get("queued_wall_time", time.time())), 3),
            },
        },
    }
    result_json_file = dirs["result_json"] / f"{base_name}_result.json"
    result_json["sources"]["result_json_path"] = str(result_json_file)
    write_json(result_json_file, result_json)

    return {
        "memory_id": memory_id,
        "track_id": track_id,
        "person_result": person_result,
        "items_result": items_result,
        "result_json": result_json,
        "person_json_path": str(person_json_file),
        "items_json_path": str(items_json_file) if items_json_file else "",
        "result_json_path": str(result_json_file),
        "person_crop_path": str(person_crop_file),
        "items_crop_path": str(item_crop_file) if item_crop_file else "",
        "crop_quality_score": job.get("crop_quality_score", ""),
        "crop_quality_detail": job.get("crop_quality_detail", {}),
        "selected_crop_frame": job.get("frame"),
        "selected_crop_time": job.get("time_sec"),
        "created_frame": job["frame"],
        "created_time": job["time_sec"],
        "match": match,
    }



# =========================================================
# Memory merge helpers (v3.8: merge diagnostics + VLM-attribute assisted merge)
# =========================================================
def _norm_attr_value(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, dict):
        v = v.get("value", "unknown")
    v = str(v).strip().lower().replace("-", "_")
    if v in {"", "none", "null", "n/a"}:
        return "none"
    if v in {"tshirt", "t_shirt", "t-shirt"}:
        return "t_shirt"
    return v


def _nested_value(obj: Any, path: List[str]) -> str:
    cur = obj
    try:
        for k in path:
            if not isinstance(cur, dict):
                return "unknown"
            cur = cur.get(k)
        return _norm_attr_value(cur)
    except Exception:
        return "unknown"


def memory_vlm_attrs(memory: Dict[str, Any]) -> Dict[str, str]:
    """VLMが返した外観属性をmemory統合用に正規化して取り出す。"""
    pr = memory.get("person_result") if isinstance(memory.get("person_result"), dict) else {}
    return {
        "upper_color": _nested_value(pr, ["par_attributes", "upper_color"]),
        "upper_type": _nested_value(pr, ["par_attributes", "upper_type"]),
        "lower_color": _nested_value(pr, ["par_attributes", "lower_color"]),
        "lower_type": _nested_value(pr, ["par_attributes", "lower_type"]),
        "gender": _nested_value(pr, ["person", "apparent_gender"]),
        "hair": _nested_value(pr, ["person", "hair"]),
        "headwear": _nested_value(pr, ["head", "headwear_type"]),
        "body_visible": _nested_value(pr, ["visibility", "body_visible"]),
    }


def _known(v: str) -> bool:
    return v not in {"", "unknown", "none", "null"}


def _compatible_attr(field: str, av: str, bv: str) -> bool:
    """
    VLM表記ゆれを吸収するための互換判定。

    注意:
    - shirt と t_shirt は服種として別物なので、ここでは互換扱いしない。
      memory統合では soft mismatch として少し減点するだけにする。
    - t-shirt / t_shirt のような表記ゆれは _norm_attr_value() 側で t_shirt に正規化済み。
    """
    if av == bv:
        return True
    if field in {"upper_type", "lower_type"}:
        pair = {av, bv}
        if pair == {"pants", "trousers"}:
            return True
    if field == "upper_color":
        # dark は black と近いことが多いが、white等とは混ぜない
        pair = {av, bv}
        if pair == {"black", "dark"}:
            return True
        earth_tones = {"beige", "brown", "tan", "khaki", "cream"}
        if av in earth_tones and bv in earth_tones:
            return True
    return False


def appearance_attr_score(a: Dict[str, str], b: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
    """VLM属性の一致度。unknownは不一致ではなく弱い根拠として扱う。"""
    weights = {
        "upper_color": 0.45,
        "upper_type": 0.10,
        "gender": 0.10,
        "hair": 0.15,
        "headwear": 0.08,
        "lower_color": 0.09,
        "lower_type": 0.03,
    }
    score_sum = 0.0
    weight_sum = 0.0
    matched = []
    compatible = []
    mismatched = []
    unknown = []
    for k, w in weights.items():
        av, bv = a.get(k, "unknown"), b.get(k, "unknown")
        ak, bk = _known(av), _known(bv)
        if not ak or not bk:
            unknown.append(k)
            continue
        weight_sum += w
        if av == bv:
            score_sum += w
            matched.append(k)
        elif _compatible_attr(k, av, bv):
            score_sum += w * 0.85
            compatible.append({"field": k, "a": av, "b": bv})
        else:
            mismatched.append({"field": k, "a": av, "b": bv})
    score = score_sum / weight_sum if weight_sum > 0 else 0.0
    return float(score), {"matched": matched, "compatible": compatible, "mismatched": mismatched, "unknown": unknown}


def same_candidate_decision_bonus(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """両方が同じ探索対象の候補なら統合根拠を少し強める。"""
    da = str(a.get("decision", ""))
    db = str(b.get("decision", ""))
    if da == "strong_candidate" and db == "strong_candidate":
        return 1.0
    if da in {"strong_candidate", "candidate"} and db in {"strong_candidate", "candidate"}:
        return 0.7
    if da == db and da == "no_match":
        return 0.25
    return 0.0

def decision_rank(decision: Any) -> int:
    d = str(decision or "")
    if d == "strong_candidate":
        return 3
    if d == "candidate":
        return 2
    if d == "weak_candidate":
        return 1
    if d in {"uncertain", "pending", "pending_recheck"}:
        return 1
    return 0


def memory_known_attr_count(memory: Dict[str, Any]) -> int:
    attrs = memory_vlm_attrs(memory)
    return sum(1 for k, v in attrs.items() if k != "body_visible" and _known(v))


def memory_intervals_overlap(a: Dict[str, Any], b: Dict[str, Any], tolerance_sec: float = 0.5) -> Tuple[bool, float, float]:
    """2つのmemoryが同時に存在していたかを見る。重なりが大きいなら別人の可能性が高い。"""
    a0 = float(a.get("created_time", a.get("last_seen_time", 0.0)) or 0.0)
    a1 = float(a.get("last_seen_time", a0) or a0)
    b0 = float(b.get("created_time", b.get("last_seen_time", 0.0)) or 0.0)
    b1 = float(b.get("last_seen_time", b0) or b0)
    overlap = min(a1, b1) - max(a0, b0)
    if overlap > tolerance_sec:
        return True, round(overlap, 3), 0.0
    gap = max(0.0, max(a0, b0) - min(a1, b1))
    return False, round(max(0.0, overlap), 3), round(gap, 3)


def best_crop_quality_flags(memory: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """time_overlapをsoft化してよい一時ID分裂らしさを、best crop品質から見る。"""
    detail = memory.get("best_crop_detail") if isinstance(memory.get("best_crop_detail"), dict) else {}
    score = safe_float(memory.get("best_crop_score"), 0.0)
    edge_hits = int(detail.get("edge_hits", 0) or 0)
    width_score = safe_float(detail.get("width_score"), 1.0)
    edge_th = int(getattr(args, "merge_soft_overlap_edge_hits", 2))
    low_q_th = float(getattr(args, "merge_soft_overlap_low_quality", 0.65))
    narrow_th = float(getattr(args, "merge_soft_overlap_width_score", 0.70))
    flags = {
        "score": round(float(score), 3),
        "edge_hits": edge_hits,
        "width_score": round(float(width_score), 3),
        "near_left": bool(detail.get("near_left", False)),
        "near_right": bool(detail.get("near_right", False)),
        "near_top": bool(detail.get("near_top", False)),
        "near_bottom": bool(detail.get("near_bottom", False)),
        "low_quality": score > 0 and score <= low_q_th,
        "edge_heavy": edge_hits >= edge_th,
        "narrow": width_score <= narrow_th,
        "partial_risk": safe_float(detail.get("partial_risk_penalty"), 0.0) > 0,
        "too_narrow": safe_float(detail.get("too_narrow_penalty"), 0.0) > 0,
    }
    flags["suspect_crop"] = any([
        flags["low_quality"],
        flags["edge_heavy"],
        flags["narrow"],
        flags["partial_risk"],
        flags["too_narrow"],
    ])
    return flags


def can_soften_time_overlap_block(
    *,
    a: Dict[str, Any],
    b: Dict[str, Any],
    attrs_a: Dict[str, str],
    attrs_b: Dict[str, str],
    attr_score: float,
    candidate_bonus: float,
    hard_blocks: List[Dict[str, Any]],
    overlap_sec: float,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    """短時間の重複だけが阻害要因なら、一時的なtrack分裂として扱えるかを判定する。"""
    if not bool(getattr(args, "merge_soft_overlap", True)):
        return False, {"reason": "disabled"}
    hard_fields = [hb.get("field") for hb in hard_blocks]
    allowed_fields = {"time_overlap"}
    allow_color_soft = bool(getattr(args, "merge_soft_overlap_allow_color_hist", True))
    if allow_color_soft:
        allowed_fields.add("color_hist")
    if not hard_fields or any(field not in allowed_fields for field in hard_fields) or "time_overlap" not in hard_fields:
        return False, {"reason": "hard_blocks_not_softenable", "hard_fields": hard_fields}

    attr_th = float(getattr(args, "merge_soft_overlap_attr_score", 0.90))
    bonus_th = float(getattr(args, "merge_soft_overlap_candidate_bonus", 1.0))
    max_overlap = float(getattr(args, "merge_soft_overlap_max_sec", 2.5))
    min_soft_color = float(getattr(args, "merge_soft_overlap_min_color_score", 0.10))
    if attr_score < attr_th:
        return False, {"reason": "attr_score_low", "threshold": attr_th}
    if candidate_bonus < bonus_th:
        return False, {"reason": "candidate_bonus_low", "threshold": bonus_th}
    if overlap_sec > max_overlap:
        return False, {"reason": "overlap_too_long", "threshold": max_overlap}
    color_blocks = [hb for hb in hard_blocks if hb.get("field") == "color_hist"]
    for hb in color_blocks:
        color_score = safe_float(hb.get("color_score"), 0.0)
        if color_score < min_soft_color:
            return False, {
                "reason": "color_hist_too_low_for_soft_block",
                "color_score": round(float(color_score), 3),
                "threshold": min_soft_color,
            }
    if attrs_a.get("upper_color") != attrs_b.get("upper_color"):
        return False, {"reason": "upper_color_not_exact"}
    if attrs_a.get("upper_type") != attrs_b.get("upper_type"):
        return False, {"reason": "upper_type_not_exact"}

    qa = best_crop_quality_flags(a, args)
    qb = best_crop_quality_flags(b, args)
    if not (qa.get("suspect_crop") or qb.get("suspect_crop")):
        return False, {"reason": "no_suspect_crop", "quality_a": qa, "quality_b": qb}

    return True, {
        "reason": "softened_short_overlap_with_strong_attrs_and_suspect_crop",
        "overlap_sec": round(float(overlap_sec), 3),
        "attr_score": round(float(attr_score), 3),
        "candidate_bonus": round(float(candidate_bonus), 3),
        "softened_fields": hard_fields,
        "min_soft_color_score": round(float(min_soft_color), 3),
        "quality_a": qa,
        "quality_b": qb,
    }


def score_memory_merge(
    a: Dict[str, Any],
    b: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, Any]]:
    """VLM属性 + 色ヒストグラム + 時間関係で、2つのmemoryを統合してよいか評価する。"""
    attrs_a = memory_vlm_attrs(a)
    attrs_b = memory_vlm_attrs(b)
    attr_score, attr_detail = appearance_attr_score(attrs_a, attrs_b)
    color_score = hist_similarity(a.get("color_hist"), b.get("color_hist"))
    overlap, overlap_sec, gap_sec = memory_intervals_overlap(a, b, getattr(args, "merge_overlap_tolerance_sec", 0.5))
    ttl = float(getattr(args, "merge_memory_ttl_sec", 12.0))
    min_color = float(getattr(args, "merge_min_color_score", 0.20))
    candidate_bonus = same_candidate_decision_bonus(a, b)

    # 強い矛盾は即除外。特に黒服と白服を混ぜない。
    # upper_type の shirt vs t_shirt は別物なので matched/compatible にはしないが、
    # 同一人物統合では単独で即ブロックせず soft_mismatch として減点だけにする。
    hard_blocks = []
    soft_mismatches = []
    for mm in attr_detail.get("mismatched", []):
        field = mm.get("field")
        if field in {"upper_color", "headwear"}:
            hard_blocks.append(mm)
        else:
            soft_mismatches.append(mm)
    if overlap:
        hard_blocks.append({"field": "time_overlap", "overlap_sec": overlap_sec})
    if gap_sec > ttl:
        hard_blocks.append({"field": "time_gap", "gap_sec": round(float(gap_sec), 3), "ttl_sec": ttl})

    # VLM属性がかなり一致している場合だけ、色ヒストグラム下限を少し緩める。
    effective_min_color = min_color
    if attr_score >= 0.80 and candidate_bonus >= 0.7:
        effective_min_color = min(effective_min_color, 0.18)
    if color_score < effective_min_color:
        hard_blocks.append({
            "field": "color_hist",
            "color_score": round(float(color_score), 3),
            "min_color_score": round(float(effective_min_color), 3),
        })

    soft_block_overrides: List[Dict[str, Any]] = []
    softened, soften_detail = can_soften_time_overlap_block(
        a=a,
        b=b,
        attrs_a=attrs_a,
        attrs_b=attrs_b,
        attr_score=attr_score,
        candidate_bonus=candidate_bonus,
        hard_blocks=hard_blocks,
        overlap_sec=overlap_sec,
        args=args,
    )
    if softened:
        softened_fields = set(soften_detail.get("softened_fields", ["time_overlap"]))
        softened_blocks = [hb for hb in hard_blocks if hb.get("field") in softened_fields]
        hard_blocks = [hb for hb in hard_blocks if hb.get("field") not in softened_fields]
        for hb in softened_blocks:
            soft_block_overrides.append({**hb, **soften_detail})

    time_score = max(0.0, 1.0 - gap_sec / max(1e-6, ttl))
    # v3.8ではVLM属性を主役にする。色ヒストグラムは画面端/照明/姿勢でぶれやすいため補助に下げる。
    final = 0.70 * attr_score + 0.15 * color_score + 0.10 * time_score + 0.05 * candidate_bonus

    if hard_blocks:
        action = "blocked"
    elif final >= float(getattr(args, "merge_memory_threshold", 0.70)):
        action = "merge"
    else:
        action = "below_threshold"

    detail = {
        "score": round(float(final), 3),
        "action": action,
        "attr_score": round(float(attr_score), 3),
        "color_score": round(float(color_score), 3),
        "time_score": round(float(time_score), 3),
        "candidate_bonus": round(float(candidate_bonus), 3),
        "gap_sec": round(float(gap_sec), 3),
        "overlap_sec": overlap_sec,
        "merge_threshold": round(float(getattr(args, "merge_memory_threshold", 0.70)), 3),
        "effective_min_color_score": round(float(effective_min_color), 3),
        "attrs_a": attrs_a,
        "attrs_b": attrs_b,
        "attr_detail": attr_detail,
        "soft_mismatches": soft_mismatches,
        "soft_block_overrides": soft_block_overrides,
        "time_overlap_softened": bool(soft_block_overrides),
        "time_overlap_soften_detail": soften_detail,
        "hard_blocks": hard_blocks,
    }
    if hard_blocks:
        return 0.0, detail
    return float(final), detail

def merge_memory_records(
    survivor: Dict[str, Any],
    absorbed: Dict[str, Any],
    merge_score: float,
    merge_detail: Dict[str, Any],
) -> None:
    """absorbedをsurvivorへ統合する。memory_idはsurvivorを維持する。"""
    survivor.setdefault("merged_memory_ids", [])
    absorbed_id = absorbed.get("memory_id")
    if absorbed_id not in survivor["merged_memory_ids"]:
        survivor["merged_memory_ids"].append(absorbed_id)

    ids = list(survivor.get("track_ids", []))
    for tid in absorbed.get("track_ids", []) or []:
        if tid not in ids:
            ids.append(tid)
    survivor["track_ids"] = ids

    # より新しく見えている方の観測情報で追跡状態を更新する。
    if float(absorbed.get("last_seen_time", -1) or -1) >= float(survivor.get("last_seen_time", -1) or -1):
        for k in [
            "last_track_id", "last_seen_frame", "last_seen_time", "last_bbox", "last_center",
            "last_area", "last_area_ratio", "last_height_ratio", "last_conf",
        ]:
            survivor[k] = absorbed.get(k)
        if isinstance(absorbed.get("color_hist"), np.ndarray):
            survivor["color_hist"] = absorbed.get("color_hist")

    # 判定結果はより強い/情報量の多い方を採用する。
    use_absorbed_result = False
    if decision_rank(absorbed.get("decision")) > decision_rank(survivor.get("decision")):
        use_absorbed_result = True
    elif decision_rank(absorbed.get("decision")) == decision_rank(survivor.get("decision")):
        if memory_known_attr_count(absorbed) > memory_known_attr_count(survivor):
            use_absorbed_result = True
    if use_absorbed_result:
        for k in [
            "decision", "person_score", "final_score", "person_result", "items_result", "result_json",
            "person_json_path", "items_json_path", "result_json_path", "person_crop_path", "items_crop_path",
        ]:
            survivor[k] = absorbed.get(k)

    survivor.setdefault("merge_history", [])
    survivor["merge_history"].append({
        "absorbed_memory_id": absorbed_id,
        "merge_score": round(float(merge_score), 3),
        "merge_detail": merge_detail,
        "merged_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    survivor["memory_status_note"] = "MERGED_DUPLICATE_MEMORY"


def try_merge_memory_after_vlm(
    finished_memory_id: int,
    person_memories: Dict[int, Dict[str, Any]],
    track_to_memory: Dict[int, int],
    args: argparse.Namespace,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """VLM結果が返った直後に、同一人物らしい既存memoryと統合する。"""
    if not getattr(args, "enable_memory_merge", True):
        return None
    if finished_memory_id not in person_memories:
        return None
    cur = person_memories[finished_memory_id]
    if not cur.get("vlm_done", False):
        return None

    best_id = None
    best_score = -1.0
    best_detail: Dict[str, Any] = {}
    threshold = float(getattr(args, "merge_memory_threshold", 0.70))
    max_debug = int(getattr(args, "merge_debug_max_records", 80))

    for mid, other in list(person_memories.items()):
        if mid == finished_memory_id:
            continue
        if not other.get("vlm_done", False):
            continue
        score, detail = score_memory_merge(other, cur, args)
        action = str(detail.get("action", ""))
        if detail.get("time_overlap_softened"):
            inc_perf(perf_stats, "memory_merge_soft_overlap_count")
        if action == "blocked":
            inc_perf(perf_stats, "memory_merge_blocked_count")
        elif score < threshold:
            inc_perf(perf_stats, "memory_merge_below_threshold_count")
        inc_perf(perf_stats, "memory_merge_attempt_count")
        append_perf_record(perf_stats, "memory_merge_attempts", {
            "finished_memory_id": int(finished_memory_id),
            "other_memory_id": int(mid),
            "score": round(float(score), 3),
            "action": action,
            "threshold": round(float(threshold), 3),
            "attr_score": detail.get("attr_score"),
            "color_score": detail.get("color_score"),
            "time_score": detail.get("time_score"),
            "candidate_bonus": detail.get("candidate_bonus"),
            "gap_sec": detail.get("gap_sec"),
            "overlap_sec": detail.get("overlap_sec"),
            "time_overlap_softened": detail.get("time_overlap_softened", False),
            "soft_block_overrides": detail.get("soft_block_overrides", []),
            "time_overlap_soften_detail": detail.get("time_overlap_soften_detail", {}),
            "hard_blocks": detail.get("hard_blocks", []),
            "attr_matched": (detail.get("attr_detail", {}) or {}).get("matched", []),
            "attr_compatible": (detail.get("attr_detail", {}) or {}).get("compatible", []),
            "attr_mismatched": (detail.get("attr_detail", {}) or {}).get("mismatched", []),
        }, max_records=max_debug)
        if score > best_score:
            best_score = score
            best_id = mid
            best_detail = detail

    if best_id is None or best_score < threshold:
        return None

    survivor_id = min(best_id, finished_memory_id)
    absorbed_id = max(best_id, finished_memory_id)
    survivor = person_memories[survivor_id]
    absorbed = person_memories[absorbed_id]
    merge_memory_records(survivor, absorbed, best_score, best_detail)

    for tid in absorbed.get("track_ids", []) or []:
        track_to_memory[int(tid)] = survivor_id
    last_tid = absorbed.get("last_track_id")
    if last_tid is not None:
        track_to_memory[int(last_tid)] = survivor_id

    del person_memories[absorbed_id]
    inc_perf(perf_stats, "memory_merge_count")
    append_perf_record(perf_stats, "memory_merge_events", {
        "survivor_memory_id": int(survivor_id),
        "absorbed_memory_id": int(absorbed_id),
        "merge_score": round(float(best_score), 3),
        "attr_score": best_detail.get("attr_score"),
        "color_score": best_detail.get("color_score"),
        "time_score": best_detail.get("time_score"),
        "candidate_bonus": best_detail.get("candidate_bonus"),
        "gap_sec": best_detail.get("gap_sec"),
        "overlap_sec": best_detail.get("overlap_sec"),
        "time_overlap_softened": best_detail.get("time_overlap_softened", False),
        "soft_block_overrides": best_detail.get("soft_block_overrides", []),
        "attr_matched": (best_detail.get("attr_detail", {}) or {}).get("matched", []),
        "attr_compatible": (best_detail.get("attr_detail", {}) or {}).get("compatible", []),
    }, max_records=max_debug)
    return {
        "survivor_memory_id": survivor_id,
        "absorbed_memory_id": absorbed_id,
        "merge_score": round(float(best_score), 3),
        "merge_detail": best_detail,
    }

def process_finished_async_jobs(
    async_jobs: Dict[int, Future],
    person_memories: Dict[int, Dict[str, Any]],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    track_to_memory: Dict[int, int],
    perf_stats: Optional[Dict[str, Any]] = None,
) -> int:
    done_count = 0
    for memory_id, fut in list(async_jobs.items()):
        if not fut.done():
            continue
        done_count += 1
        try:
            res = fut.result()
        except Exception as e:
            mem = person_memories.get(memory_id)
            if mem is not None:
                mem["vlm_pending"] = False
                mem["vlm_error"] = f"{type(e).__name__}: {e}"
                mem["best_crop_status"] = "ERROR"
            del async_jobs[memory_id]
            continue

        mem = person_memories.get(memory_id)
        if mem is not None:
            match = res.get("match", {})
            mem.update({
                "vlm_pending": False,
                "vlm_done": True,
                "best_crop_status": "DONE",
                "decision": match.get("decision", ""),
                "person_score": match.get("person_score", ""),
                "final_score": match.get("final_score", ""),
                "person_result": res.get("person_result"),
                "items_result": res.get("items_result"),
                "result_json": res.get("result_json"),
                "person_json_path": res.get("person_json_path", ""),
                "items_json_path": res.get("items_json_path", ""),
                "result_json_path": res.get("result_json_path", ""),
                "person_crop_path": res.get("person_crop_path", ""),
                "items_crop_path": res.get("items_crop_path", ""),
                "created_frame": mem.get("created_frame", res.get("created_frame")),
                "created_time": mem.get("created_time", res.get("created_time")),
                "vlm_finished_wall_time": time.time(),
                "selected_crop_frame": res.get("selected_crop_frame", mem.get("selected_crop_frame", mem.get("queued_frame"))),
                "selected_crop_time": res.get("selected_crop_time", mem.get("selected_crop_time", mem.get("queued_time"))),
                "selected_crop_quality": res.get("crop_quality_score", mem.get("selected_crop_quality", "")),
            })
            merge_info = try_merge_memory_after_vlm(memory_id, person_memories, track_to_memory, args, perf_stats)
            if merge_info is not None:
                person_memories.setdefault("_last_merge_info", merge_info) if False else None
        del async_jobs[memory_id]
    if done_count:
        write_memory_snapshot(dirs, person_memories)
    return done_count


def process_frame_live_async(
    frame: np.ndarray,
    frame_count: int,
    time_sec: float,
    model: YOLO,
    args: argparse.Namespace,
    target: Dict[str, Any],
    dirs: Dict[str, Path],
    csv_writer: csv.DictWriter,
    person_memories: Dict[int, Dict[str, Any]],
    track_to_memory: Dict[int, int],
    next_memory_id_ref: Dict[str, int],
    executor: ThreadPoolExecutor,
    async_jobs: Dict[int, Future],
    track_seen_counts: Dict[int, int],
    video_duration_sec: Optional[float] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    _t0 = time.perf_counter()
    dets = detect_persons_with_tracking(model, frame, args)
    add_perf(perf_stats, "yolo_track_sec", time.perf_counter() - _t0)
    inc_perf(perf_stats, "detections_total", len(dets))
    queued_count = 0
    now_time = time_sec
    used_memory_ids: set = set()

    for det in dets:
        track_id = int(det["track_id"])
        bbox = det["bbox"]
        track_id_is_real = bool(det.get("track_id_is_real", False))
        if track_id_is_real:
            track_seen_counts[track_id] = track_seen_counts.get(track_id, 0) + 1
        track_age = track_seen_counts.get(track_id, 0) if track_id_is_real else 0
        _t_hist = time.perf_counter()
        hist = compute_upper_body_hist(frame, bbox) if args.enable_person_memory else None
        add_perf(perf_stats, "hist_sec", time.perf_counter() - _t_hist)

        reason_parts: List[str] = []
        status: Optional[str] = None
        memory_id: Optional[int] = None
        memory_status = ""
        reid_score: Any = ""
        reid_detail: Dict[str, Any] = {}
        matched_previous_track_id: Any = ""
        vlm_already_done: Any = ""
        feature_source = ""
        cached_from_frame = ""
        cache_age_sec = ""
        person_json_path = ""
        items_json_path = ""
        result_json_path = ""
        person_crop_path = ""
        items_crop_path = ""
        person_result = None
        items_result = None
        result_json = None

        if args.enable_person_memory and track_id_is_real and track_id in track_to_memory:
            mid = track_to_memory.get(track_id)
            if mid in person_memories and mid not in used_memory_ids:
                memory_id = mid
                memory_status = "TRACK_ID_MATCH"
                entry0 = person_memories[mid]
                if entry0.get("vlm_done", False) or entry0.get("vlm_pending", False):
                    status = "REUSE_MEMORY"
                    reason_parts.append("MEMORY_BY_TRACK_ID")
                else:
                    reason_parts.append("MEMORY_BY_TRACK_ID_COLLECT_BEST_CROP")

        if args.enable_person_memory and status is None and memory_id is None:
            _t_reid = time.perf_counter()
            best_mid, best_score, best_detail = find_best_memory_for_det_live(
                det=det,
                hist=hist,
                time_sec=now_time,
                frame_shape=frame.shape,
                args=args,
                person_memories=person_memories,
                used_memory_ids=used_memory_ids,
            )
            add_perf(perf_stats, "reid_sec", time.perf_counter() - _t_reid)
            if best_mid is not None:
                memory_id = best_mid
                memory_status = "REID_MATCH"
                reid_score = round(best_score, 3)
                reid_detail = best_detail
                prev_tid = person_memories[best_mid].get("last_track_id")
                matched_previous_track_id = "" if prev_tid is None else prev_tid
                if track_id_is_real:
                    track_to_memory[track_id] = best_mid
                status = "REUSE_MEMORY"
                reason_parts.append(f"MEMORY_BY_REID_{best_score:.2f}")
            else:
                memory_status = "NEW_OR_UNMATCHED"
                if best_score > 0:
                    reid_score = round(best_score, 3)
                    reid_detail = best_detail
                    reason_parts.append(f"REID_NO_MATCH_{best_score:.2f}")

        if status is None:
            if (not track_id_is_real) and args.skip_pseudo_track_vlm:
                status = "TRACK_ONLY_PSEUDO_ID"
                reason_parts.append("PSEUDO_TRACK_ID")
            elif track_id_is_real and track_age < args.vlm_min_track_frames:
                status = "TRACK_ONLY_NOT_STABLE"
                reason_parts.append(f"TRACK_AGE_{track_age}_LT_{args.vlm_min_track_frames}")
            elif det["conf"] < args.min_conf_for_vlm:
                status = "TRACK_ONLY_LOW_CONF"
                reason_parts.append("LOW_CONF")
            elif det["area_ratio"] < args.min_area_ratio_for_vlm:
                status = "TRACK_ONLY_SMALL"
                reason_parts.append("SMALL_AREA")
            elif det["height_ratio"] < args.min_height_ratio_for_vlm:
                status = "TRACK_ONLY_SMALL"
                reason_parts.append("SMALL_HEIGHT")

        selected_crop_reason = ""
        crop_quality_score: Any = ""
        selected_crop_frame: Any = ""
        selected_crop_time: Any = ""

        if status is None and getattr(args, "enable_best_crop_timing", True):
            # 条件を満たした最初のcropを即送信せず、人物ごとにbest crop候補を更新する。
            if memory_id is None:
                memory_id = int(next_memory_id_ref["next"])
                next_memory_id_ref["next"] += 1
                person_memories[memory_id] = make_new_memory(memory_id, track_id, det, hist, frame_count, now_time)
                memory_status = "NEW_MEMORY"
            if track_id_is_real:
                # 1 track_id -> 1 memory_id を優先。別memoryに同じtrack_idが残るのを防ぐ。
                old_mid = track_to_memory.get(track_id)
                if old_mid is not None and old_mid != memory_id and old_mid in person_memories:
                    old_ids = [tid for tid in person_memories[old_mid].get("track_ids", []) if tid != track_id]
                    person_memories[old_mid]["track_ids"] = old_ids
                track_to_memory[track_id] = memory_id

            mem = person_memories[memory_id]
            update_memory_observation(mem, track_id, det, hist, frame_count, now_time)
            q_score, q_detail = compute_crop_quality(det, frame.shape, args, track_age=track_age)
            det["crop_quality_score"] = round(q_score, 3)
            det["crop_quality_detail"] = q_detail
            update_best_crop_candidate(mem, frame, det, hist, frame_count, now_time, q_score, q_detail, track_id, target, perf_stats)
            ready, timing_reason = best_crop_send_decision(mem, frame_count, now_time, args)
            crop_quality_score = mem.get("best_crop_score", round(q_score, 3))
            selected_crop_frame = mem.get("best_crop_frame", "")
            selected_crop_time = mem.get("best_crop_time", "")
            selected_crop_reason = timing_reason
            if ready:
                set_best_crop_ready(mem, timing_reason, frame_count, now_time, perf_stats)
                if args.max_vlm_per_frame > 0 and queued_count >= args.max_vlm_per_frame:
                    status = "TRACK_ONLY_LIMIT"
                    reason_parts.append("MAX_VLM_PER_FRAME")
                elif not is_vlm_budget_available(args, perf_stats, now_time):
                    status = "TRACK_ONLY_VLM_BUDGET_REACHED"
                    reason_parts.append("MAX_TOTAL_VLM_CALLS")
                    inc_perf(perf_stats, "vlm_budget_reached_count")
                elif len(async_jobs) >= args.max_pending_vlm:
                    status = "TRACK_ONLY_VLM_QUEUE_FULL"
                    reason_parts.append("MAX_PENDING_VLM")
                else:
                    status = "QUEUE_VLM"
                    reason_parts.append(timing_reason)
            else:
                status = "TRACK_ONLY_BEST_CROP_WAIT"
                reason_parts.append(timing_reason)
                mem["best_crop_status"] = "COLLECTING"
                inc_perf(perf_stats, "best_crop_wait_count")

        if status is None:
            if args.max_vlm_per_frame > 0 and queued_count >= args.max_vlm_per_frame:
                status = "TRACK_ONLY_LIMIT"
                reason_parts.append("MAX_VLM_PER_FRAME")
            elif not is_vlm_budget_available(args, perf_stats, now_time):
                status = "TRACK_ONLY_VLM_BUDGET_REACHED"
                reason_parts.append("MAX_TOTAL_VLM_CALLS")
                inc_perf(perf_stats, "vlm_budget_reached_count")
            elif len(async_jobs) >= args.max_pending_vlm:
                status = "TRACK_ONLY_VLM_QUEUE_FULL"
                reason_parts.append("MAX_PENDING_VLM")
            else:
                status = "QUEUE_VLM"
                reason_parts.append("ASYNC_OK")

        if status == "QUEUE_VLM":
            queued_count += 1
            if memory_id is None:
                memory_id = int(next_memory_id_ref["next"])
                next_memory_id_ref["next"] += 1
                person_memories[memory_id] = make_new_memory(memory_id, track_id, det, hist, frame_count, now_time)
                memory_status = "NEW_MEMORY"
            if track_id_is_real:
                old_mid = track_to_memory.get(track_id)
                if old_mid is not None and old_mid != memory_id and old_mid in person_memories:
                    old_ids = [tid for tid in person_memories[old_mid].get("track_ids", []) if tid != track_id]
                    person_memories[old_mid]["track_ids"] = old_ids
                track_to_memory[track_id] = memory_id

            mem = person_memories[memory_id]
            update_memory_observation(mem, track_id, det, hist, frame_count, now_time)
            if getattr(args, "enable_best_crop_timing", True) and mem.get("best_vlm_frame") is not None:
                job = submit_best_crop_vlm_job(
                    memory_id=memory_id,
                    memory=mem,
                    target=target,
                    args=args,
                    dirs=dirs,
                    executor=executor,
                    async_jobs=async_jobs,
                    reason="+".join(reason_parts),
                    perf_stats=perf_stats,
                    video_duration_sec=video_duration_sec,
                )
                if job is None:
                    status = "TRACK_ONLY_VLM_QUEUE_FULL"
                    reason_parts.append("BEST_CROP_SUBMIT_FAILED")
                    set_best_crop_ready(mem, "+".join(reason_parts), frame_count, now_time, perf_stats)
                else:
                    inc_perf(perf_stats, "queued_from_ready_count")
                    person_crop_path = str(job["person_crop_file"])
                    items_crop_path = str(job["item_crop_file"]) if job.get("item_crop_file") else ""
                    crop_quality_score = job.get("crop_quality_score", mem.get("best_crop_score", ""))
                    selected_crop_frame = job.get("frame", "")
                    selected_crop_time = job.get("time_sec", "")
                    selected_crop_reason = "+".join(reason_parts)
                    if "GOOD_QUALITY" in selected_crop_reason:
                        inc_perf(perf_stats, "best_crop_good_quality_count")
                    elif "MAX_WAIT" in selected_crop_reason:
                        inc_perf(perf_stats, "best_crop_max_wait_count")
                    elif "QUALITY_PEAKED" in selected_crop_reason:
                        inc_perf(perf_stats, "best_crop_peak_count")
                    feature_source = "ASYNC_QUEUE_BEST_CROP"
                    vlm_already_done = False
            else:
                mem.update({
                    "vlm_pending": True,
                    "vlm_done": False,
                    "decision": "pending",
                    "queued_frame": frame_count,
                    "queued_time": round(now_time, 3),
                    "created_frame": mem.get("created_frame", frame_count),
                    "created_time": mem.get("created_time", round(now_time, 3)),
                })
                _t_job = time.perf_counter()
                job = build_async_vlm_job(
                    frame=frame,
                    det=det,
                    bbox=bbox,
                    track_id=track_id,
                    track_id_is_real=track_id_is_real,
                    memory_id=memory_id,
                    memory_status=memory_status,
                    frame_count=frame_count,
                    time_sec=time_sec,
                    now_time=now_time,
                    target=target,
                    args=args,
                    dirs=dirs,
                    reason="+".join(reason_parts),
                )
                add_perf(perf_stats, "async_job_build_sec", time.perf_counter() - _t_job)
                person_crop_path = str(job["person_crop_file"])
                items_crop_path = str(job["item_crop_file"]) if job.get("item_crop_file") else ""
                _t_submit = time.perf_counter()
                async_jobs[memory_id] = executor.submit(run_async_vlm_job, job)
                add_perf(perf_stats, "async_submit_sec", time.perf_counter() - _t_submit)
                inc_perf(perf_stats, "async_jobs_queued")
                feature_source = "ASYNC_QUEUE"
                vlm_already_done = False

        elif status == "REUSE_MEMORY" and args.enable_person_memory and memory_id in person_memories:
            used_memory_ids.add(memory_id)
            entry = person_memories[memory_id]
            person_result = entry.get("person_result")
            items_result = entry.get("items_result")
            result_json = entry.get("result_json")
            person_json_path = entry.get("person_json_path", "")
            items_json_path = entry.get("items_json_path", "")
            result_json_path = entry.get("result_json_path", "")
            person_crop_path = entry.get("person_crop_path", "")
            items_crop_path = entry.get("items_crop_path", "")
            cached_from_frame = str(entry.get("created_frame", ""))
            cache_age_sec = f"{now_time - float(entry.get('created_time', now_time)):.2f}"
            vlm_already_done = bool(entry.get("vlm_done", False))
            feature_source = "MEMORY_REUSE_PENDING" if entry.get("vlm_pending", False) else "MEMORY_REUSE"
            update_memory_observation(entry, track_id, det, hist, frame_count, now_time)

        if args.memory_snapshot_interval_frames > 0 and frame_count % args.memory_snapshot_interval_frames == 0:
            _t_snap = time.perf_counter()
            write_memory_snapshot(dirs, person_memories)
            add_perf(perf_stats, "memory_snapshot_sec", time.perf_counter() - _t_snap)

        match = result_json.get("match", {}) if isinstance(result_json, dict) else {}
        decision = match.get("decision", "pending" if status in {"QUEUE_VLM", "REUSE_MEMORY"} and memory_id in person_memories and person_memories[memory_id].get("vlm_pending") else "")
        final_score = match.get("final_score", "")
        person_score = match.get("person_score", "")
        item_score = match.get("item_score", "")
        items_status = ""
        if isinstance(match.get("item_match"), dict):
            items_status = match["item_match"].get("status", "")

        color = decision_color("REUSE_MEMORY" if status == "QUEUE_VLM" else status, decision)
        label_status = {
            "QUEUE_VLM": "Q-VLM",
            "REUSE_MEMORY": "MEM",
            "TRACK_ONLY_SMALL": "SMALL",
            "TRACK_ONLY_LIMIT": "LIMIT",
            "TRACK_ONLY_VLM_QUEUE_FULL": "QFULL",
            "TRACK_ONLY_LOW_CONF": "LOWCONF",
            "TRACK_ONLY_PSEUDO_ID": "PSEUDO",
            "TRACK_ONLY_NOT_STABLE": "WAIT",
            "TRACK_ONLY_BEST_CROP_WAIT": "BEST",
        }.get(status, status)
        mem_label = f" M:{memory_id}" if memory_id is not None else ""
        reason_line = compact_match_reason(match)
        lines = [
            f"ID:{track_id}{mem_label} {label_status}",
            f"score:{final_score} {decision}" if final_score != "" else f"rank:{det['rank_by_area']} {decision}",
            f"a:{det['area_ratio']:.3f} h:{det['height_ratio']:.2f} c:{det['conf']:.2f}",
        ]
        if reason_line:
            lines.insert(2, reason_line)
        _t_draw = time.perf_counter()
        draw_label(annotated, bbox, lines, color)
        add_perf(perf_stats, "draw_sec", time.perf_counter() - _t_draw)

        if should_write_log_row(args, status, memory_status, feature_source or status):
            _t_csv = time.perf_counter()
            csv_writer.writerow({
            "frame": frame_count,
            "time_sec": round(time_sec, 3),
            "track_id": track_id,
            "rank_by_area": det["rank_by_area"],
            "status": status,
            "reason": "+".join(reason_parts),
            "memory_id": memory_id if memory_id is not None else "",
            "memory_status": memory_status,
            "reid_score": reid_score,
            "reid_detail": json.dumps(reid_detail, ensure_ascii=False) if reid_detail else "",
            "matched_previous_track_id": matched_previous_track_id,
            "vlm_already_done": vlm_already_done,
            "decision": decision,
            "final_score": final_score,
            "person_score": person_score,
            "item_score": item_score,
            "items_status": items_status,
            "match_reason": reason_line,
            "partial_match_detail": json.dumps(match.get("partial", []), ensure_ascii=False) if isinstance(match, dict) else "",
            "conf": round(det["conf"], 4),
            "area_ratio": round(det["area_ratio"], 5),
            "height_ratio": round(det["height_ratio"], 5),
            "bbox": list(bbox),
            "person_json_path": person_json_path,
            "items_json_path": items_json_path,
            "result_json_path": result_json_path,
            "person_crop_path": person_crop_path,
            "items_crop_path": items_crop_path,
            "person_elapsed_sec": runtime_summary(person_result).get("elapsed_sec", "") if person_result else "",
            "items_elapsed_sec": runtime_summary(items_result).get("elapsed_sec", "") if items_result else "",
            "feature_source": feature_source or status,
            "cached_from_frame": cached_from_frame,
            "cache_age_sec": cache_age_sec,
            "crop_quality_score": crop_quality_score,
            "best_crop_status": person_memories.get(memory_id, {}).get("best_crop_status", "") if memory_id is not None else "",
            "selected_crop_frame": selected_crop_frame,
            "selected_crop_time": selected_crop_time,
            "selected_crop_reason": selected_crop_reason,
            })
            add_perf(perf_stats, "csv_write_sec", time.perf_counter() - _t_csv)
            inc_perf(perf_stats, "csv_rows_written")
        else:
            inc_perf(perf_stats, "csv_rows_skipped")

    return annotated


# =========================================================
# CLI / input loop
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + LM Studio VLM person search")

    # input
    parser.add_argument("--image", type=str, default=None, help="静止画パス")
    parser.add_argument("--video", type=str, default=None, help="動画パス")
    parser.add_argument("--camera_id", type=int, default=None, help="カメラID。image/videoがない場合に使用")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    # target structured input -> target_json
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
    parser.add_argument("--target_item", default="unspecified", help="探索対象の持ち物条件。unspecified/anyならitems処理を飛ばす。noneなら「持ち物なし」を条件としてitems処理を実行。例: none/suitcase/umbrella/backpack")

    # yolo / tracking
    parser.add_argument("--yolo_model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker yaml。空文字ならデフォルト")
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--process_every", type=int, default=2, help="動画/カメラで何フレームに1回処理するか。v3.6は速度測定向けにデフォルト2")
    parser.add_argument("--max_vlm_per_frame", type=int, default=1, help="1処理フレームでVLMに投入/キュー投入する最大人数。0なら全員")
    parser.add_argument("--min_area_ratio_for_vlm", type=float, default=0.07)
    parser.add_argument("--min_height_ratio_for_vlm", type=float, default=0.25, help="実用想定では小さい部分検出をVLMに送らない")
    parser.add_argument("--min_conf_for_vlm", type=float, default=0.45)
    parser.add_argument("--vlm_interval_sec", type=float, default=3.0, help="同じtrack_idに再度VLMを投げる最短間隔")
    parser.add_argument("--cache_area_change_ratio", type=float, default=0.80, help="bbox面積変化の再解析しきい値。v3.2ではinterval内の早期再送はデフォルト抑制")
    parser.add_argument("--allow_area_reanalyze_before_interval", action="store_true", help="旧挙動: vlm_interval_sec内でもbbox面積変化が大きければVLM再送する")

    # person memory / lightweight ReID
    parser.add_argument("--enable_person_memory", action=argparse.BooleanOptionalAction, default=True, help="track_idとは別に人物メモリを作り、一度VLM済みの人物を再利用する")
    parser.add_argument("--skip_pseudo_track_vlm", action=argparse.BooleanOptionalAction, default=True, help="track_id=-1などtracker IDなしの検出を新規VLM送信しない")
    parser.add_argument("--reid_match_threshold", type=float, default=0.65, help="新規trackを既存memoryに紐付ける同一人物スコア閾値")
    parser.add_argument("--reid_memory_ttl_sec", type=float, default=6.0, help="見失ったmemoryを同一人物候補として保持する秒数")
    parser.add_argument("--reid_max_center_distance_ratio", type=float, default=0.45, help="画面対角線に対する中心距離の最大比率")
    parser.add_argument("--reid_min_color_score", type=float, default=0.25, help="色ヒストグラム類似度の最低条件")
    parser.add_argument("--reid_position_weight", type=float, default=0.35)
    parser.add_argument("--reid_color_weight", type=float, default=0.45)
    parser.add_argument("--reid_size_weight", type=float, default=0.15)
    parser.add_argument("--reid_age_weight", type=float, default=0.05)

    # memory merge after VLM result
    parser.add_argument("--enable_memory_merge", action=argparse.BooleanOptionalAction, default=True, help="VLM結果が返った後に、同一人物らしいmemoryを統合する")
    parser.add_argument("--merge_memory_threshold", type=float, default=0.70, help="memory統合の最終スコア閾値")
    parser.add_argument("--merge_min_color_score", type=float, default=0.20, help="memory統合時に必要な色ヒストグラム最低類似度。v3.8ではVLM属性一致を主に使うため少し緩める")
    parser.add_argument("--merge_memory_ttl_sec", type=float, default=12.0, help="memory統合で許容する出現間隔の最大秒数")
    parser.add_argument("--merge_overlap_tolerance_sec", type=float, default=0.5, help="同時存在しているmemoryは別人の可能性が高いため、この秒数以上重なれば統合しない")
    parser.add_argument("--merge_soft_overlap", action=argparse.BooleanOptionalAction, default=True, help="短時間のtime_overlapだけが阻害要因で、属性一致とcrop品質条件を満たす場合にsoft block化する")
    parser.add_argument("--merge_soft_overlap_max_sec", type=float, default=2.5, help="time_overlapをsoft block化できる最大重複秒数")
    parser.add_argument("--merge_soft_overlap_attr_score", type=float, default=0.90, help="time_overlap soft block化に必要なVLM属性一致度")
    parser.add_argument("--merge_soft_overlap_candidate_bonus", type=float, default=1.0, help="time_overlap soft block化に必要な探索候補一致ボーナス")
    parser.add_argument("--merge_soft_overlap_low_quality", type=float, default=0.65, help="片方のbest crop品質がこの値以下なら一時ID分裂の根拠にする")
    parser.add_argument("--merge_soft_overlap_edge_hits", type=int, default=2, help="片方のbest cropがこの数以上画面端に触れていれば一時ID分裂の根拠にする")
    parser.add_argument("--merge_soft_overlap_width_score", type=float, default=0.70, help="片方のbest crop横幅スコアがこの値以下なら一時ID分裂の根拠にする")
    parser.add_argument("--merge_soft_overlap_allow_color_hist", action=argparse.BooleanOptionalAction, default=True, help="属性一致が強い短時間overlapでは、低めのcolor_histも条件付きでsoft block化する")
    parser.add_argument("--merge_soft_overlap_min_color_score", type=float, default=0.10, help="color_histをsoft block化できる最低類似度")
    parser.add_argument("--merge_debug_max_records", type=int, default=80, help="run_summaryに残すmemory merge候補診断ログの最大件数")

    # practical live / async VLM
    parser.add_argument("--async_vlm", action=argparse.BooleanOptionalAction, default=True, help="VLMを別スレッドで実行し、動画処理を止めない")
    parser.add_argument("--vlm_workers", type=int, default=1, help="LM Studioへ同時問い合わせするワーカー数。ローカルVLMなら基本1")
    parser.add_argument("--max_pending_vlm", type=int, default=2, help="未完了VLMジョブの最大数")
    parser.add_argument("--max_total_vlm_calls", type=int, default=8, help="1回の動画/カメラ実行でVLMに送る人物memoryの最大数。0なら無制限")
    parser.add_argument("--max_vlm_calls_per_min", type=float, default=0.0, help="動画時刻1分あたりのVLM投入上限。0なら無効")
    parser.add_argument("--vlm_min_track_frames", type=int, default=3, help="新規trackを何処理フレーム安定させてからVLMへ入れるか")
    parser.add_argument("--enable_best_crop_timing", action=argparse.BooleanOptionalAction, default=True, help="条件を満たした最初のcropではなく、一定時間内のbest cropをVLMへ送る")
    parser.add_argument("--best_crop_max_wait_sec", type=float, default=0.8, help="best cropを探す最大待機秒数。映像処理は止めず、VLM投入だけ少し遅らせる")
    parser.add_argument("--best_crop_min_wait_sec", type=float, default=0.15, help="quality peak判定に必要な最短待機秒数")
    parser.add_argument("--best_crop_good_threshold", type=float, default=0.62, help="この品質を超えたら最大待機を待たずにVLM投入")
    parser.add_argument("--best_crop_min_threshold", type=float, default=0.45, help="最大待機/終了時にVLM投入を許可する最低品質")
    parser.add_argument("--best_crop_good_max_edge_hits", type=int, default=1, help="GOOD_QUALITYとして即投入してよい画面端接触数の上限")
    parser.add_argument("--best_crop_good_min_width_score", type=float, default=0.70, help="GOOD_QUALITYとして即投入してよいbest crop横幅スコアの下限")
    parser.add_argument("--best_crop_patience_frames", type=int, default=6, help="品質が伸びない状態が何処理フレーム続けばpeakとみなすか")
    parser.add_argument("--best_crop_edge_margin_ratio", type=float, default=0.03, help="画面端接触ペナルティの幅")
    parser.add_argument("--best_crop_preferred_width_ratio", type=float, default=0.22, help="best crop評価で好ましいbbox横幅比")
    parser.add_argument("--best_crop_preferred_height_ratio", type=float, default=0.62, help="best crop評価で好ましいbbox高さ比")
    parser.add_argument("--best_crop_preferred_area_ratio", type=float, default=0.18, help="best crop評価で好ましいbbox面積比")
    parser.add_argument("--min_width_ratio_for_vlm", type=float, default=0.10, help="横幅が細すぎる人物部分cropをVLMに送りにくくする")
    parser.add_argument("--final_flush_max_memories", type=int, default=2, help="動画終了時に追加でVLMへ送る未判定memoryの最大数。0なら上限なし")
    parser.add_argument("--final_flush_min_quality", type=float, default=0.60, help="動画終了時flushを許可するbest crop品質の下限")
    parser.add_argument("--final_flush_min_width_score", type=float, default=0.55, help="動画終了時flushを許可するbest crop横幅スコアの下限")
    parser.add_argument("--final_flush_min_height_score", type=float, default=0.45, help="動画終了時flushを許可するbest crop高さスコアの下限")
    parser.add_argument("--final_flush_max_edge_hits", type=int, default=2, help="動画終了時flushを許可する画面端接触数の上限")
    parser.add_argument("--final_flush_min_target_color_score", type=float, default=0.15, help="動画終了時flushを許可するtarget色らしさの下限。0なら色だけでは除外しない")
    parser.add_argument("--final_flush_target_color_rescue_score", type=float, default=0.35, help="target色らしさがこの値以上なら、少し低いcrop品質でもflush候補に残す")
    parser.add_argument("--final_flush_quality_floor_with_target_color", type=float, default=0.52, help="target色らしさで救済する場合でも必要な最低crop品質")
    parser.add_argument("--final_wait_timeout_sec", type=float, default=30.0, help="動画終了後のVLM最終待ち上限。0なら全VLM完了まで待つ")
    parser.add_argument("--near_end_vlm_guard_sec", type=float, default=8.0, help="動画終端からこの秒数以内は、targetらしさが低い新規VLM投入を抑制する")
    parser.add_argument("--near_end_min_target_color_score", type=float, default=0.25, help="終端付近でもVLM投入を許可するtarget色らしさの下限")
    parser.add_argument("--near_end_min_quality", type=float, default=0.68, help="終端付近でもVLM投入を許可するcrop品質の下限")
    parser.add_argument("--wait_async_at_end", action=argparse.BooleanOptionalAction, default=True, help="動画終了時に未完了VLMを待ってJSONを完成させる")
    parser.add_argument("--memory_snapshot_interval_frames", type=int, default=0, help="memory snapshotを書き出す間隔。0なら最後/非同期完了時のみ。v3.6はI/O削減のためデフォルト0")

    # LM Studio / VLM
    parser.add_argument("--lmstudio_base_url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--vlm_model", default="gemma-4-e2b-it-qat")
    parser.add_argument("--lmstudio_timeout", type=int, default=600)
    parser.add_argument("--person_vlm_long_side", type=int, default=512)
    parser.add_argument("--items_vlm_long_side", type=int, default=384)
    parser.add_argument("--person_max_tokens", type=int, default=320)
    parser.add_argument("--items_max_tokens", type=int, default=110)
    parser.add_argument("--person_crop_margin", type=float, default=0.08)
    parser.add_argument("--jpeg_quality", type=int, default=90)

    # item crop options
    parser.add_argument("--no_draw_target_box", action="store_true", help="items context cropに緑枠を描かない")
    parser.add_argument("--item_margin_x", type=float, default=-1.0)
    parser.add_argument("--item_margin_top", type=float, default=-1.0)
    parser.add_argument("--item_margin_bottom", type=float, default=-1.0)

    # output / display
    parser.add_argument("--out_dir", default="", help="出力先。空なら outputs/person_search_YYYYMMDD_HHMMSS")
    parser.add_argument("--show", action="store_true", help="処理中にcv2.imshowで表示する")
    parser.add_argument("--csv_log_mode", choices=["full", "important", "vlm_only", "none"], default="important", help="CSVログ量。full=従来通り全検出、important=VLM/新規/ReID中心、vlm_only=VLM投入のみ、none=書かない")
    parser.add_argument("--save_annotated_video", action=argparse.BooleanOptionalAction, default=True, help="動画/カメラ入力でannotated_result.mp4を保存する")
    parser.add_argument("--check_lmstudio", action="store_true", help="LM Studioの/modelsを確認して終了")

    return parser


def open_input(args: argparse.Namespace) -> Tuple[str, Optional[np.ndarray], Optional[cv2.VideoCapture]]:
    if args.image:
        frame = cv2.imread(os.path.expanduser(args.image))
        if frame is None:
            raise RuntimeError(f"画像を読み込めませんでした: {args.image}")
        return "image", frame, None

    if args.video:
        cap = cv2.VideoCapture(os.path.expanduser(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {args.video}")
        return "video", None, cap

    camera_id = 0 if args.camera_id is None else args.camera_id
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: {camera_id}")
    return "camera", None, cap


def make_output_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": ensure_dir(out_dir),
        "person_crops": ensure_dir(out_dir / "crops" / "person"),
        "items_crops": ensure_dir(out_dir / "crops" / "items"),
        "person_json": ensure_dir(out_dir / "json" / "person"),
        "items_json": ensure_dir(out_dir / "json" / "items"),
        "result_json": ensure_dir(out_dir / "json" / "result"),
        "memory_json": ensure_dir(out_dir / "json" / "memory"),
    }
    return dirs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_started_at = dt.datetime.now()
    run_perf_start = time.perf_counter()
    perf_stats: Dict[str, Any] = {}

    if args.check_lmstudio:
        print(json.dumps(check_lmstudio_models(args.lmstudio_base_url), ensure_ascii=False, indent=2))
        return

    target = build_target_json(args)
    out_dir = Path(os.path.expanduser(args.out_dir)) if args.out_dir else Path("outputs") / f"person_search_{now_stamp()}"
    dirs = make_output_dirs(out_dir)
    target_path = dirs["root"] / "target.json"
    write_json(target_path, {
        "search_version": SEARCH_VERSION,
        "target": target,
        "notes": {
            "target_item_unspecified_skips_items_vlm": True,
            "target_json_source": "argparse_structured_input_no_llm",
        },
        "args": vars(args),
    })

    print("person search VLM 起動中")
    print(f"search_version: {SEARCH_VERSION}")
    print(f"person_prompt_version: {PERSON_PROMPT_VERSION}")
    print(f"items_prompt_version: {ITEMS_PROMPT_VERSION}")
    print(f"target: {json.dumps(target, ensure_ascii=False)}")
    print(f"target.json: {target_path}")
    print(f"out_dir: {out_dir}")
    print(f"target_item: {target.get('target_item', 'none')} / run_items: {should_run_items(target)}")
    print(f"max_vlm_per_frame: {args.max_vlm_per_frame}")
    print(f"person_memory: {args.enable_person_memory} / reid_threshold: {args.reid_match_threshold}")

    input_type, image_frame, cap = open_input(args)
    model = YOLO(args.yolo_model)
    person_memories: Dict[int, Dict[str, Any]] = {}
    track_to_memory: Dict[int, int] = {}
    next_memory_id_ref: Dict[str, int] = {"next": 1}
    async_jobs: Dict[int, Future] = {}
    track_seen_counts: Dict[int, int] = {}
    executor: Optional[ThreadPoolExecutor] = None
    if args.async_vlm:
        executor = ThreadPoolExecutor(max_workers=max(1, int(args.vlm_workers)))
        print(f"async_vlm: ON / workers={args.vlm_workers} / max_pending={args.max_pending_vlm}")
    else:
        print("async_vlm: OFF / sync mode")

    csv_path = dirs["root"] / "search_log.csv"
    csv_file, csv_writer = open_csv_writer(csv_path)

    writer = None
    frame_count = 0
    fps: Optional[float] = None
    width = int(args.width)
    height = int(args.height)
    video_duration_sec: Optional[float] = None
    try:
        if input_type == "image":
            annotated = process_frame(
                image_frame,
                frame_count=1,
                time_sec=0.0,
                model=model,
                args=args,
                target=target,
                dirs=dirs,
                csv_writer=csv_writer,
                person_memories=person_memories,
                track_to_memory=track_to_memory,
                next_memory_id_ref=next_memory_id_ref,
                perf_stats=perf_stats,
            )
            out_image = dirs["root"] / "annotated_result.jpg"
            cv2.imwrite(str(out_image), annotated)
            print(f"annotated_result: {out_image}")
            print(f"CSV: {csv_path}")
            return

        assert cap is not None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3 or np.isnan(fps):
            fps = float(args.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and fps and fps > 0:
            video_duration_sec = total_frames / float(fps)

        if args.save_annotated_video:
            out_video = dirs["root"] / "annotated_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
            print(f"annotated_video: {out_video}")

        frame_count = 0
        start_wall = time.time()
        last_annotated = None
        while True:
            _t_read = time.perf_counter()
            ok, frame = cap.read()
            add_perf(perf_stats, "video_read_sec", time.perf_counter() - _t_read)
            if not ok:
                break
            frame_count += 1
            inc_perf(perf_stats, "frames_read")

            if input_type == "video":
                time_sec = frame_count / fps
            else:
                time_sec = time.time() - start_wall

            if args.async_vlm:
                _t_async_collect = time.perf_counter()
                done_n = process_finished_async_jobs(async_jobs, person_memories, dirs, args, track_to_memory, perf_stats)
                if executor is not None:
                    dispatch_ready_best_crop_jobs(
                        person_memories=person_memories,
                        target=target,
                        args=args,
                        dirs=dirs,
                        executor=executor,
                        async_jobs=async_jobs,
                        perf_stats=perf_stats,
                        video_duration_sec=video_duration_sec,
                    )
                add_perf(perf_stats, "async_collect_sec", time.perf_counter() - _t_async_collect)
                inc_perf(perf_stats, "async_jobs_finished", done_n)

            if args.process_every <= 1 or frame_count % args.process_every == 0:
                inc_perf(perf_stats, "frames_processed")
                _t_frame_process = time.perf_counter()
                if args.async_vlm:
                    assert executor is not None
                    annotated = process_frame_live_async(
                        frame,
                        frame_count=frame_count,
                        time_sec=time_sec,
                        model=model,
                        args=args,
                        target=target,
                        dirs=dirs,
                        csv_writer=csv_writer,
                        person_memories=person_memories,
                        track_to_memory=track_to_memory,
                        next_memory_id_ref=next_memory_id_ref,
                        executor=executor,
                        async_jobs=async_jobs,
                        track_seen_counts=track_seen_counts,
                        video_duration_sec=video_duration_sec,
                        perf_stats=perf_stats,
                    )
                else:
                    annotated = process_frame(
                        frame,
                        frame_count=frame_count,
                        time_sec=time_sec,
                        model=model,
                        args=args,
                        target=target,
                        dirs=dirs,
                        csv_writer=csv_writer,
                        person_memories=person_memories,
                        track_to_memory=track_to_memory,
                        next_memory_id_ref=next_memory_id_ref,
                        perf_stats=perf_stats,
                    )
                add_perf(perf_stats, "frame_process_sec", time.perf_counter() - _t_frame_process)
                last_annotated = annotated
            else:
                inc_perf(perf_stats, "frames_skipped_by_process_every")
                annotated = last_annotated if last_annotated is not None else frame

            if writer is not None:
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated_to_write = cv2.resize(annotated, (width, height))
                else:
                    annotated_to_write = annotated
                _t_write = time.perf_counter()
                writer.write(annotated_to_write)
                add_perf(perf_stats, "video_write_sec", time.perf_counter() - _t_write)

            if args.show:
                _t_display = time.perf_counter()
                cv2.imshow("person_search_vlm", annotated)
                key = cv2.waitKey(1) & 0xFF
                add_perf(perf_stats, "display_sec", time.perf_counter() - _t_display)
                if key == ord("q"):
                    break

    finally:
        if args.async_vlm and executor is not None:
            if args.wait_async_at_end:
                # 動画終端で、best crop候補を保持したまま未送信のmemoryを最後に投入する。
                if getattr(args, "enable_best_crop_timing", True):
                    end_time_sec = (float(frame_count) / float(fps)) if fps else 0.0
                    for mid, mem in choose_final_flush_memories(person_memories, args, perf_stats, end_time_sec):
                        ready, flush_reason = best_crop_send_decision(mem, frame_count, end_time_sec, args, force=True)
                        if ready:
                            priority = best_crop_flush_priority(mem)
                            set_best_crop_ready(mem, f"{flush_reason}+FINAL_PRIORITY_{priority:.2f}", frame_count, end_time_sec, perf_stats)
                            inc_perf(perf_stats, "best_crop_end_flush_count")
                        elif mem.get("best_vlm_frame") is not None:
                            drop_low_quality_best_crop(mem, flush_reason, frame_count, end_time_sec, perf_stats)
                _t_final_wait = time.perf_counter()
                final_wait_timeout_sec = float(getattr(args, "final_wait_timeout_sec", 0.0) or 0.0)
                final_wait_timed_out = False
                while async_jobs or any(
                    isinstance(mem, dict)
                    and mem.get("best_crop_status") == "READY"
                    and not mem.get("vlm_done", False)
                    and not mem.get("vlm_pending", False)
                    for mem in person_memories.values()
                ):
                    if getattr(args, "enable_best_crop_timing", True):
                        dispatch_ready_best_crop_jobs(
                            person_memories=person_memories,
                            target=target,
                            args=args,
                            dirs=dirs,
                            executor=executor,
                            async_jobs=async_jobs,
                            perf_stats=perf_stats,
                            video_duration_sec=video_duration_sec,
                            allow_near_end=True,
                        )
                    done_n = process_finished_async_jobs(async_jobs, person_memories, dirs, args, track_to_memory, perf_stats)
                    inc_perf(perf_stats, "async_jobs_finished", done_n)
                    if final_wait_timeout_sec > 0 and (time.perf_counter() - _t_final_wait) >= final_wait_timeout_sec:
                        done_n = process_finished_async_jobs(async_jobs, person_memories, dirs, args, track_to_memory, perf_stats)
                        inc_perf(perf_stats, "async_jobs_finished", done_n)
                        final_wait_timed_out = True
                        inc_perf(perf_stats, "final_wait_timeout_count")
                        break
                    if async_jobs:
                        time.sleep(0.05)
                    elif any(
                        isinstance(mem, dict)
                        and mem.get("best_crop_status") == "READY"
                        and not mem.get("vlm_done", False)
                        and not mem.get("vlm_pending", False)
                        for mem in person_memories.values()
                    ):
                        break
                add_perf(perf_stats, "async_final_wait_sec", time.perf_counter() - _t_final_wait)
                done_n = process_finished_async_jobs(async_jobs, person_memories, dirs, args, track_to_memory, perf_stats)
                inc_perf(perf_stats, "async_jobs_finished", done_n)
                if final_wait_timed_out:
                    if isinstance(perf_stats, dict):
                        perf_stats["final_wait_timed_out"] = True
                        perf_stats["final_wait_timeout_sec"] = final_wait_timeout_sec
                        perf_stats["async_jobs_cancelled_at_final_timeout"] = len(async_jobs)
                    for fut in list(async_jobs.values()):
                        fut.cancel()
                    async_jobs.clear()
                    for mid, mem in person_memories.items():
                        if (
                            isinstance(mem, dict)
                            and (mem.get("vlm_pending", False) or mem.get("best_crop_status") == "READY")
                            and not mem.get("vlm_done", False)
                        ):
                            mem["vlm_pending"] = False
                            if not mem.get("decision") or mem.get("decision") == "pending":
                                mem["decision"] = "unknown"
                            if mem.get("best_crop_status") in {"READY", "QUEUED"}:
                                mem["best_crop_status"] = "TIMEOUT"
                            mem["final_wait_skipped_reason"] = "final_wait_timeout"
            timed_out = bool((perf_stats or {}).get("final_wait_timed_out", False))
            executor.shutdown(wait=(args.wait_async_at_end and not timed_out), cancel_futures=(not args.wait_async_at_end or timed_out))
        csv_file.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    write_memory_snapshot(dirs, person_memories)

    run_finished_at = dt.datetime.now()
    total_runtime_sec = time.perf_counter() - run_perf_start
    if input_type == "video" and fps and fps > 0 and frame_count > 0:
        video_duration_sec = frame_count / float(fps)
    elif input_type == "camera":
        video_duration_sec = total_runtime_sec
    elif input_type == "image":
        video_duration_sec = 0.0

    run_summary = build_run_summary(
        args=args,
        input_type=input_type,
        out_dir=out_dir,
        csv_path=csv_path,
        target_path=target_path,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        total_runtime_sec=total_runtime_sec,
        frame_count=frame_count,
        fps=fps,
        video_duration_sec=video_duration_sec,
        person_memories=person_memories,
        async_jobs_left=len(async_jobs),
        perf_stats=perf_stats,
    )
    run_summary_path = dirs["root"] / "run_summary.json"
    write_json(run_summary_path, run_summary)

    print(f"完了: {out_dir}")
    print(f"CSV: {csv_path}")
    print(f"run_summary: {run_summary_path}")
    print(f"person_memory_count: {len(person_memories)}")
    print(f"async_pending_jobs_left: {len(async_jobs)}")
    print(f"total_runtime_sec: {run_summary['total_runtime_sec']}")
    print(f"video_duration_sec: {run_summary['video_duration_sec']}")
    print(f"realtime_ratio_total_over_video: {run_summary['realtime_ratio_total_over_video']}")
    print(f"vlm_total_elapsed_sec: {run_summary['vlm']['vlm_total_elapsed_sec']}")
    prof = run_summary.get("profiling", {})
    print(f"yolo_track_sec: {prof.get('yolo_track_sec')} / {prof.get('yolo_track_ms_per_processed_frame')} ms per processed frame")
    print(f"frame_process_sec: {prof.get('frame_process_sec')} / {prof.get('frame_process_ms_per_processed_frame')} ms per processed frame")
    print(f"csv_rows_written/skipped: {prof.get('csv_rows_written')}/{prof.get('csv_rows_skipped')}")


if __name__ == "__main__":
    main()
