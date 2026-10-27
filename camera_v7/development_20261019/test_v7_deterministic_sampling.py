"""Regression tests for deterministic LM Studio generation parameters."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "person_search_vlm_v3_9_7_final_wait_reduce_v3.py"
MS2_PATH = ROOT / "person_search_vlm_milestone2_v7.py"
MODEL = "gemma-4-e2b-it-qat"


def read_prompt(name: str) -> str:
    tree = ast.parse(MS2_PATH.read_text(encoding="utf-8"), filename=str(MS2_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
        ):
            value = value.func.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value.strip()
    raise AssertionError(f"string prompt not found: {name}")


def load_base_module():
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    with mock.patch.dict(sys.modules, {"ultralytics": ultralytics}):
        spec = importlib.util.spec_from_file_location("v7_sampling_test_base", BASE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


BASE = load_base_module()


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return json.dumps(response).encode("utf-8")


class DeterministicSamplingTests(unittest.TestCase):
    def call_and_capture(self, *, kind: str, prompt: str, max_tokens: int):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return FakeResponse()

        with (
            mock.patch.object(BASE, "image_file_to_data_url", return_value="data:image/jpeg;base64,AA=="),
            mock.patch.object(BASE.urllib.request, "urlopen", side_effect=fake_urlopen),
            mock.patch.object(BASE, "normalize_person_json", side_effect=lambda value: value),
        ):
            result = BASE.call_lmstudio_json(
                image_path=Path("unused.jpg"),
                prompt=prompt,
                base_url="http://127.0.0.1:1234/v1",
                model=MODEL,
                timeout_sec=1,
                max_tokens=max_tokens,
                kind=kind,
            )

        self.assertEqual(1, len(requests))
        raw_payload = requests[0].data.decode("utf-8")
        return json.loads(raw_payload), raw_payload, result

    def assert_deterministic_payload(self, payload, raw_payload, result, *, prompt, max_tokens):
        self.assertEqual(MODEL, payload["model"])
        self.assertEqual(prompt, payload["messages"][0]["content"][0]["text"])
        self.assertEqual(max_tokens, payload["max_tokens"])
        self.assertEqual(0.0, payload["temperature"])
        self.assertEqual(1.0, payload["top_p"])
        self.assertNotIn("seed", payload)
        self.assertEqual(1, raw_payload.count('"temperature"'))
        self.assertEqual(1, raw_payload.count('"top_p"'))
        self.assertEqual(
            {"temperature": 0.0, "top_p": 1.0},
            result["_runtime"]["sampling"],
        )

    def test_person_request_uses_deterministic_sampling(self):
        prompt = read_prompt("MS2_PERSON_PROMPT")
        payload, raw_payload, result = self.call_and_capture(
            kind="person", prompt=prompt, max_tokens=384
        )
        self.assert_deterministic_payload(
            payload, raw_payload, result, prompt=prompt, max_tokens=384
        )

    def test_front_request_uses_same_deterministic_sampling(self):
        prompt = read_prompt("FRONT_PHOTO_PROMPT")
        payload, raw_payload, result = self.call_and_capture(
            kind="front_photo", prompt=prompt, max_tokens=140
        )
        self.assert_deterministic_payload(
            payload, raw_payload, result, prompt=prompt, max_tokens=140
        )

    def test_person_and_front_prompts_are_unchanged(self):
        expected_hashes = {
            "MS2_PERSON_PROMPT": "94abc619e21424b420f6faf19542e4b149303f36c9428ddca17f8c5bc332d7de",
            "FRONT_PHOTO_PROMPT": "482706fa783943c0b443e81e8b3a508be9490bff5214105ec72bf7f7c3f6ee3e",
        }
        for name, expected in expected_hashes.items():
            actual = hashlib.sha256(read_prompt(name).encode("utf-8")).hexdigest()
            self.assertEqual(expected, actual, name)


if __name__ == "__main__":
    unittest.main()
