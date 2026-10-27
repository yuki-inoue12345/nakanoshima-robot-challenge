"""Regression tests for person-match plus capture MS2 success policy."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
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
UI_PATH = ROOT / "daikoudai_v7.py.new"


def load_ms2_module():
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    with mock.patch.dict(sys.modules, {"ultralytics": ultralytics}):
        spec = importlib.util.spec_from_file_location("v7_ms2_success_policy_module", MS2_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def load_ui_module():
    loader = importlib.machinery.SourceFileLoader("v7_ms2_success_policy_ui", str(UI_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


V7 = load_ms2_module()
UI = load_ui_module()


class Ms2SuccessPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.args = SimpleNamespace(
            jpeg_quality=95,
            save_annotated_image=True,
            save_crop=True,
            check_front_view=True,
            accept_front_views="front",
            front_min_confidence="medium",
            decision_accept="candidate",
        )

    def candidate(
        self, root: Path, *, target_id: str, view: str,
        multi_person=False, accepted=True, score=None,
    ):
        crops = root / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        (root / "json").mkdir(exist_ok=True)
        crop_path = crops / "candidate_frame000100_track1_mem1.jpg"
        crop = np.full((240, 160, 3), (40, 120, 220), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(crop_path), crop))
        decision = "strong_candidate" if accepted else "weak_candidate"
        reasons = [] if accepted else ["high_confidence_contradiction:lower_type"]
        final_score = float(score if score is not None else (0.911 if accepted else 0.875))
        person_result = {
            "par_attributes": {
                "upper_color": {"value": "white", "confidence": "high"},
                "lower_type": {
                    "value": "skirt" if accepted else "pants",
                    "confidence": "high",
                },
            }
        }
        return {
            "candidate_crop_path": crop_path,
            "bbox": (20, 20, 140, 220),
            "frame": 100,
            "time_sec": 10.0,
            "memory_id": 1,
            "track_id": 1,
            "crop_quality_score": 0.824,
            "crop_quality_detail": {"quality": 0.824},
            "person_result": person_result,
            "items_result": None,
            "observed": {"par_attributes": person_result["par_attributes"]},
            "match": {
                "decision": decision,
                "final_score": final_score,
                "person_score": final_score,
                "item_score": None,
                "evidence_gate_reasons": reasons,
            },
            "front_result": {
                "front_view": {"value": view, "confidence": "low"},
                "photo_usable_for_ms2": {"value": view == "front", "confidence": "low"},
                "ms2_front_gate": {
                    "accepted": view == "front" and not multi_person,
                    "reasons": ["multiple_people_in_front_crop"] if multi_person else [],
                },
                "front_multi_person_check": {
                    "accepted": not multi_person,
                    "other_people": [{"bbox": [1, 1, 5, 5]}] if multi_person else [],
                },
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
        return {"root": root, "crops": root / "crops", "json": root / "json"}

    def finish_success(self, root: Path, candidate, target_id):
        self.assertTrue(
            V7.person_match_ready_for_photo_success(
                candidate,
                self.args,
                quality_ok=True,
                reliability_ok=True,
                operational_ok=True,
            )
        )
        frame = np.full((260, 180, 3), 100, dtype=np.uint8)
        with mock.patch.object(
            V7, "call_front_photo_vlm", side_effect=AssertionError("front VLM must not be called")
        ):
            return V7.save_person_match_success_artifacts(
                dirs=self.dirs(root),
                frame=frame,
                candidate=candidate,
                capture_candidate=candidate,
                target={"target_id": target_id},
                args=self.args,
            )

    def assert_capture_success(self, result):
        self.assertTrue(result["found"])
        self.assertTrue(result["ms1_success"])
        self.assertTrue(result["ms2_success"])
        self.assertTrue(result["photo_saved"])
        self.assertTrue(result["capture_best_saved"])
        self.assertFalse(result["front_photo_saved"])
        self.assertFalse(result["front_validation_performed"])
        self.assertFalse(result["front_validation_required"])
        self.assertEqual("person_match_and_capture", result["photo_acceptance_basis"])
        self.assertEqual("FOUND_FRONT_PHOTO", result["status"])
        self.assertEqual(result["capture_best_path"], result["photo_path"])

    def test_front_unknown_and_side_all_succeed_after_capture(self):
        for view in ("front", "unknown", "side"):
            with self.subTest(view=view):
                root = self.base / view
                root.mkdir()
                candidate = self.candidate(root, target_id="target_B", view=view)
                result = self.finish_success(root, candidate, "target_B")
                self.assert_capture_success(result)

    def test_target_a_unknown_with_multiple_people_succeeds(self):
        root = self.base / "target_a_multi"
        root.mkdir()
        candidate = self.candidate(
            root, target_id="target_A", view="unknown", multi_person=True, score=0.856
        )
        result = self.finish_success(root, candidate, "target_A")
        self.assert_capture_success(result)
        self.assertEqual(0.856, result["final_score"])

    def test_target_b_side_succeeds(self):
        root = self.base / "target_b_side"
        root.mkdir()
        candidate = self.candidate(root, target_id="target_B", view="side", score=0.911)
        result = self.finish_success(root, candidate, "target_B")
        self.assert_capture_success(result)
        self.assertEqual(0.911, result["final_score"])

    def test_pants_contradiction_remains_rejected(self):
        root = self.base / "pants_reject"
        root.mkdir()
        candidate = self.candidate(root, target_id="target_B", view="front", accepted=False)
        self.assertFalse(
            V7.person_match_ready_for_photo_success(
                candidate,
                self.args,
                quality_ok=True,
                reliability_ok=True,
                operational_ok=True,
            )
        )
        explanation = V7.not_found_explanation(
            {"status": "NOT_FOUND", "person_found": False}, candidate, detections_total=1
        )
        self.assertEqual("CANDIDATE_REJECTED_BY_ATTRIBUTE_MISMATCH", explanation["not_found_reason"])
        self.assertEqual("attribute_evidence", explanation["candidate_reject_gate"])

    def test_capture_failure_is_error_and_never_success(self):
        root = self.base / "capture_fail"
        root.mkdir()
        candidate = self.candidate(root, target_id="target_A", view="unknown")
        with mock.patch.object(
            V7,
            "save_capture_best_artifacts",
            return_value={"saved": False, "save_error": "injected I/O failure"},
        ):
            result = V7.save_person_match_success_artifacts(
                dirs=self.dirs(root),
                frame=np.zeros((260, 180, 3), dtype=np.uint8),
                candidate=candidate,
                capture_candidate=candidate,
                target={"target_id": "target_A"},
                args=self.args,
            )
        self.assertEqual("ERROR", result["status"])
        self.assertEqual("CAPTURE_BEST_SAVE_FAILED", result["error_code"])
        self.assertFalse(result["found"])
        self.assertFalse(result["ms2_success"])
        self.assertFalse(result["photo_saved"])

    def test_ab_and_safe_to_advance_contract_is_unchanged(self):
        success = {"status": "FOUND_FRONT_PHOTO", "ms2_success": True, "final_score": 0.9}
        not_found = {"status": "NOT_FOUND", "ms2_success": False}
        outcome_a = UI.classify_area_search_results({"target_A": success, "target_B": not_found})
        outcome_b = UI.classify_area_search_results({"target_A": not_found, "target_B": success})
        self.assertEqual("target_A", outcome_a["found_target_id"])
        self.assertEqual("target_B", outcome_b["found_target_id"])
        self.assertTrue(UI.area_outcome_safe_to_advance(outcome_a))
        self.assertTrue(UI.area_outcome_safe_to_advance(outcome_b))

        error = UI.classify_area_search_results(
            {"target_A": {"status": "ERROR", "ms2_success": False}}
        )
        self.assertFalse(error["success"])
        self.assertFalse(UI.area_outcome_safe_to_advance(error))

    def test_capture_best_is_the_official_evidence_path(self):
        result = {
            "capture_best": {"saved": True, "capture_path": "/tmp/capture_best.jpg"},
            "image_path": "/tmp/found_annotated.jpg",
            "crop_path": "/tmp/found_crop.jpg",
        }
        self.assertEqual("/tmp/capture_best.jpg", UI.person_search_evidence_path(result))
        result["capture_best"]["saved"] = False
        self.assertEqual("/tmp/found_annotated.jpg", UI.person_search_evidence_path(result))

    def test_person_and_front_prompts_are_unchanged(self):
        self.assertEqual(
            "94abc619e21424b420f6faf19542e4b149303f36c9428ddca17f8c5bc332d7de",
            hashlib.sha256(V7.MS2_PERSON_PROMPT.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            "482706fa783943c0b443e81e8b3a508be9490bff5214105ec72bf7f7c3f6ee3e",
            hashlib.sha256(V7.FRONT_PHOTO_PROMPT.encode("utf-8")).hexdigest(),
        )

    def test_normal_person_accept_path_skips_front_vlm_and_multi_person_crop(self):
        source = inspect.getsource(V7.run_milestone2_search)
        accepted_path = source.split("if person_match_ready_for_photo_success(", 1)[1]
        accepted_path = accepted_path.split("elif accepted_decision(candidate_decision", 1)[0]
        self.assertIn("save_person_match_success_artifacts(", accepted_path)
        self.assertNotIn("call_front_photo_vlm(", accepted_path)
        self.assertNotIn("prepare_front_evidence_candidate(", accepted_path)


if __name__ == "__main__":
    unittest.main()
