"""Regression tests for detection-independent capture_best evidence."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MS2_PATH = ROOT / "person_search_vlm_milestone2_v7.py"


def load_ms2_module():
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    with mock.patch.dict(sys.modules, {"ultralytics": ultralytics}):
        spec = importlib.util.spec_from_file_location("v7_capture_best_test_module", MS2_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


V7 = load_ms2_module()


class CaptureBestTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.args = SimpleNamespace(jpeg_quality=95)
        self.target = {"target_id": "target_B"}

    def make_candidate(self, root: Path, *, score=0.875, quality=0.824, frame=15785):
        crops = root / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        crop_path = crops / f"candidate_frame{frame:06d}_track1_mem2.jpg"
        image = np.full((240, 160, 3), (190, 90, 30), dtype=np.uint8)
        cv2.rectangle(image, (30, 20), (130, 220), (245, 245, 245), -1)
        self.assertTrue(cv2.imwrite(str(crop_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]))
        return {
            "candidate_crop_path": crop_path,
            "bbox": (239, 128, 370, 431),
            "frame": frame,
            "time_sec": 1052.333,
            "memory_id": 2,
            "track_id": 1,
            "crop_quality_score": quality,
            "crop_quality_detail": {"quality": quality},
            "match": {
                "final_score": score,
                "person_score": score,
                "decision": "strong_candidate" if score is not None and score >= 0.9 else "weak_candidate",
                "evidence_gate_reasons": [] if score is not None and score >= 0.9 else ["high_confidence_contradiction:lower_type"],
            },
            "vlm_request_audit": {
                "audit_path": str(root / "json" / "person_vlm_request001.json"),
                "image_path": str(crop_path),
                "image_sha256": V7.sha256_file(crop_path),
                "prompt_version": "MS2_PERSON_PROMPT",
                "prompt_sha256": V7.sha256_text(V7.MS2_PERSON_PROMPT),
                "model": "gemma-4-e2b-it-qat",
                "temperature": 0.0,
                "top_p": 1.0,
                "seed_present": False,
                "max_tokens": 384,
            },
        }

    def dirs(self, root: Path):
        return {
            "root": root,
            "crops": root / "crops",
            "json": root / "json",
        }

    def test_found_not_found_and_front_fail_all_save_capture(self):
        scenarios = {
            "found": {
                "status": "FOUND_FRONT_PHOTO", "found": True, "ms1_success": True,
                "ms2_success": True, "front_photo_saved": True,
            },
            "attribute_mismatch": {
                "status": "NOT_FOUND", "found": False, "ms1_success": False,
                "ms2_success": False, "front_photo_saved": False,
                "not_found_reason": "CANDIDATE_REJECTED_BY_ATTRIBUTE_MISMATCH",
                "candidate_reject_gate": "attribute_evidence",
            },
            "front_fail": {
                "status": "FOUND_BUT_NOT_FRONT", "found": True, "ms1_success": True,
                "ms2_success": False, "front_photo_saved": False,
                "not_found_reason": "PERSON_MATCHED_BUT_NO_FRONT_PHOTO",
                "candidate_reject_gate": "front_photo",
            },
        }
        protected_flags = ("found", "ms1_success", "ms2_success", "front_photo_saved")
        for name, result in scenarios.items():
            with self.subTest(name=name):
                root = self.base / name
                root.mkdir()
                candidate = self.make_candidate(root)
                before = {key: result.get(key) for key in protected_flags}
                metadata = V7.save_capture_best_artifacts(
                    candidate=candidate,
                    result=result,
                    dirs=self.dirs(root),
                    target=self.target,
                    args=self.args,
                    selection_source="person_vlm_request",
                )
                self.assertTrue(metadata["saved"])
                self.assertEqual(before, {key: result.get(key) for key in protected_flags})
                self.assertEqual(
                    candidate["candidate_crop_path"].read_bytes(),
                    (root / "capture_best.jpg").read_bytes(),
                    "capture_best.jpg must be the unannotated original JPEG bytes",
                )
                self.assertTrue((root / "capture_best_annotated.jpg").is_file())
                self.assertNotEqual(
                    (root / "capture_best.jpg").read_bytes(),
                    (root / "capture_best_annotated.jpg").read_bytes(),
                )
                sidecar = json.loads((root / "capture_best.json").read_text(encoding="utf-8"))
                self.assertEqual("target_B", sidecar["target_id"])
                self.assertEqual(result["status"], sidecar["final_result"])
                self.assertTrue(sidecar["detection_result_independent"])
                for field in (
                    "frame", "time_sec", "memory_id", "track_id", "bbox", "crop_path",
                    "final_score", "person_score", "decision", "evidence_gate_reasons",
                    "reject_gate", "reason", "crop_quality_score",
                ):
                    self.assertIn(field, sidecar)

    def test_ranking_prefers_final_score_then_crop_quality(self):
        root = self.base / "ranking"
        root.mkdir()
        low_score_high_quality = self.make_candidate(root, score=0.80, quality=0.95, frame=1)
        high_score_low_quality = self.make_candidate(root, score=0.90, quality=0.50, frame=2)
        tied_score_high_quality = self.make_candidate(root, score=0.90, quality=0.85, frame=3)
        best = V7.better_capture_candidate(None, low_score_high_quality)
        best = V7.better_capture_candidate(best, high_score_low_quality)
        best = V7.better_capture_candidate(best, tied_score_high_quality)
        self.assertIs(tied_score_high_quality, best)

        no_score_low_quality = self.make_candidate(root, score=None, quality=0.40, frame=4)
        no_score_high_quality = self.make_candidate(root, score=None, quality=0.90, frame=5)
        self.assertIs(
            no_score_high_quality,
            V7.better_capture_candidate(no_score_low_quality, no_score_high_quality),
        )
        self.assertIs(
            tied_score_high_quality,
            V7.better_capture_candidate(tied_score_high_quality, no_score_high_quality),
        )

    def test_search_exit_wires_capture_save_without_using_it_as_a_gate(self):
        source = inspect.getsource(V7.run_milestone2_search)
        self.assertIn('capture_selection_source = "person_vlm_request"', source)
        self.assertIn('result["capture_best"] = save_capture_best_artifacts(', source)
        self.assertNotIn('if result["capture_best"]', source)

    def test_vlm_runtime_records_image_and_prompt_hashes(self):
        image_path = self.base / "fixed.jpg"
        image_path.write_bytes(b"fixed-image-bytes")
        prompt = "fixed prompt"
        fake_result = {
            "_runtime": {
                "sampling": {"temperature": 0.0, "top_p": 1.0},
                "model": "gemma-4-e2b-it-qat",
                "max_tokens": 384,
            }
        }
        V7.reset_stop_request()
        with mock.patch.object(V7.base, "call_lmstudio_json", return_value=fake_result):
            result = V7.call_lmstudio_json_with_deadline(
                deadline_mono=None,
                image_path=image_path,
                prompt=prompt,
                base_url="http://127.0.0.1:1234/v1",
                model="gemma-4-e2b-it-qat",
                timeout_sec=1,
                max_tokens=384,
                kind="person",
            )
        runtime = result["_runtime"]
        self.assertEqual(V7.sha256_file(image_path), runtime["image_sha256"])
        self.assertEqual(V7.sha256_text(prompt), runtime["prompt_sha256"])
        self.assertFalse(runtime["seed_present"])


if __name__ == "__main__":
    unittest.main()
