"""Regression tests for the Camera V7 person-attribute prompt."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MS2_PATH = ROOT / "person_search_vlm_milestone2_v7.py"


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


class PersonPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = read_prompt("MS2_PERSON_PROMPT")

    def test_dress_and_skirt_are_distinguished_by_garment_structure(self):
        self.assertIn("dress = a one-piece garment combining the upper and lower body", self.prompt)
        self.assertIn("skirt = a lower-body garment worn separately from an independent top", self.prompt)
        self.assertIn("do not classify the lower garment as dress", self.prompt)

    def test_all_skirt_lengths_remain_skirt(self):
        self.assertIn("Mini, knee-length, midi, long, and maxi skirts", self.prompt)
        self.assertIn("all lower_type=skirt", self.prompt)
        self.assertIn("Do not classify a lower garment as dress based on length alone", self.prompt)

    def test_skirt_and_pants_require_visible_leg_structure(self):
        self.assertIn("not separated into two sections for the left and right legs", self.prompt)
        self.assertIn("pants have two leg sections covering the legs separately", self.prompt)
        self.assertIn("Classify pants only when a crotch/leg split or two-leg structure is visually confirmed", self.prompt)
        self.assertIn("Do not infer an unseen split when uncertain", self.prompt)

    def test_length_width_and_flare_do_not_imply_pants(self):
        self.assertIn("Do not classify an ankle-length garment as pants based on length alone", self.prompt)
        self.assertIn("when no left/right leg split is visible", self.prompt)
        self.assertIn("wide, flared, or long lower garment as skirt", self.prompt)
        self.assertIn("when it remains one continuous garment", self.prompt)

    def test_lower_type_schema_is_unchanged(self):
        self.assertIn(
            '"lower_type":{"value":"pants/trousers/jeans/shorts/skirt/dress/unknown"',
            self.prompt,
        )
        self.assertNotIn("long_skirt", self.prompt)


class FrontPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = read_prompt("FRONT_PHOTO_PROMPT")

    def test_mannequin_front_prioritizes_torso_and_body_orientation(self):
        self.assertIn("Mannequin-only front-view rules", self.prompt)
        self.assertIn("apply only to mannequins/display bodies", self.prompt)
        self.assertIn("primarily from torso and body orientation", self.prompt)
        self.assertIn("both shoulders or both sides of the upper torso", self.prompt)
        self.assertIn("front clothing surface is visible", self.prompt)
        self.assertIn("A visible frontal face is not required", self.prompt)

    def test_head_hair_or_wig_alone_does_not_make_mannequin_side(self):
        self.assertIn("not from head direction, face visibility, hair direction, or wig direction", self.prompt)
        self.assertIn("Wind-blown hair, a wig covering the face, or a rotated mannequin head alone", self.prompt)
        self.assertIn("must not cause side when the torso clearly faces the camera", self.prompt)

    def test_real_person_rules_are_not_unconditionally_relaxed(self):
        self.assertIn("These supplemental rules do not relax the front-view requirements for real people", self.prompt)
        self.assertIn("front = the face/head front or mannequin face/front head is facing the camera", self.prompt)
        self.assertIn('"front_view": {"value": "front/three_quarter/side/back/unknown"', self.prompt)


if __name__ == "__main__":
    unittest.main()
