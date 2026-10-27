"""Regression tests for explicit shirt/t-shirt MS2 matching semantics."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
MS2_PATH = ROOT / "person_search_vlm_milestone2_v7.py"


def load_ms2_module():
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    with mock.patch.dict(sys.modules, {"ultralytics": ultralytics}):
        spec = importlib.util.spec_from_file_location(
            "v7_upper_type_matching_module", MS2_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


V7 = load_ms2_module()


def vc(value, confidence="high"):
    return {"value": value, "confidence": confidence}


def target(upper_type):
    return {
        "subject_type": "person",
        "upper_color": "white",
        "upper_type": upper_type,
        "apparent_gender": "male",
        "lower_color": "black",
        "lower_type": "pants",
    }


def person(upper_type, *, features=None):
    result = {
        "person": {"apparent_gender": vc("male")},
        "head": {},
        "par_attributes": {
            "upper_color": vc("white"),
            "upper_type": vc(upper_type),
            "lower_color": vc("black"),
            "lower_type": vc("pants"),
        },
    }
    if features is not None:
        result["par_attributes"]["upper_type_features"] = {
            key: vc(value) for key, value in features.items()
        }
    return result


TSHIRT_FEATURES = {
    "has_collar": False,
    "has_front_buttons": False,
    "has_round_neck": True,
    "is_sleeveless_outer": False,
    "visible_inner_sleeves": False,
}

SHIRT_FEATURES = {
    "has_collar": True,
    "has_front_buttons": True,
    "has_round_neck": False,
    "is_sleeveless_outer": False,
    "visible_inner_sleeves": False,
}


def upper_type_row(match):
    return next(
        row
        for row in match["person_match"]["score_breakdown"]
        if row["field"] == "upper_type"
    )


class UpperTypeMatchingTests(unittest.TestCase):
    def test_shirt_target_rejects_high_confidence_feature_confirmed_tshirt(self):
        match = V7.build_match_json_ms2(
            target("shirt"), person("t_shirt", features=TSHIRT_FEATURES), None
        )

        row = upper_type_row(match)
        self.assertEqual(0.0, row["score"])
        self.assertEqual("shirt_tshirt_feature_contradiction", row["reason"])
        self.assertEqual("contradiction", row["evidence_state"])
        self.assertIn(
            "high_confidence_contradiction:upper_type",
            match["evidence_gate_reasons"],
        )
        self.assertNotIn(match["decision"], {"strong_candidate", "candidate"})

    def test_shirt_target_and_high_confidence_shirt_match_normally(self):
        match = V7.build_match_json_ms2(target("shirt"), person("shirt"), None)

        self.assertEqual(1.0, upper_type_row(match)["score"])
        self.assertEqual("strong_candidate", match["decision"])
        self.assertEqual([], match["evidence_gate_reasons"])

    def test_unknown_upper_type_is_not_a_hard_contradiction(self):
        match = V7.build_match_json_ms2(target("shirt"), person("unknown"), None)

        row = upper_type_row(match)
        self.assertEqual("unknown_partial", row["reason"])
        self.assertNotIn(
            "high_confidence_contradiction:upper_type",
            match["evidence_gate_reasons"],
        )

    def test_sweater_tshirt_pullover_policy_is_unchanged(self):
        score, reason, _detail = V7.upper_type_similarity(
            "sweater", "t_shirt", person("t_shirt", features=TSHIRT_FEATURES)
        )

        self.assertEqual(0.72, score)
        self.assertEqual(
            "pullover_structure_compatible_material_unconfirmed", reason
        )

    def test_tshirt_target_and_high_confidence_tshirt_match_normally(self):
        match = V7.build_match_json_ms2(
            target("t_shirt"), person("t_shirt", features=TSHIRT_FEATURES), None
        )

        self.assertEqual(1.0, upper_type_row(match)["score"])
        self.assertEqual("strong_candidate", match["decision"])
        self.assertEqual([], match["evidence_gate_reasons"])

    def test_reverse_tshirt_target_rejects_feature_confirmed_shirt(self):
        match = V7.build_match_json_ms2(
            target("t_shirt"), person("shirt", features=SHIRT_FEATURES), None
        )

        row = upper_type_row(match)
        self.assertEqual(0.0, row["score"])
        self.assertEqual("shirt_tshirt_feature_contradiction", row["reason"])
        self.assertIn(
            "high_confidence_contradiction:upper_type",
            match["evidence_gate_reasons"],
        )
        self.assertNotIn(match["decision"], {"strong_candidate", "candidate"})


if __name__ == "__main__":
    unittest.main()
