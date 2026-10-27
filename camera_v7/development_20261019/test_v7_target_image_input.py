"""Regression tests for Target A/B attribute input from one reference image."""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"
LOADER = importlib.machinery.SourceFileLoader("daikoudai_v7_target_image_test", str(UI_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
V7 = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = V7
LOADER.exec_module(V7)


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeWidget:
    def __init__(self, **options):
        self.options = dict(options)

    def configure(self, **kwargs):
        self.options.update(kwargs)


OPTION_MAP = {
    "subject_type": V7.SUBJECT_TYPE_OPTIONS,
    "target_upper_color": V7.COLOR_OPTIONS,
    "target_upper_secondary_color": V7.COLOR_OPTIONS,
    "target_upper_pattern": V7.PATTERN_OPTIONS,
    "target_upper_type": V7.UPPER_TYPE_OPTIONS,
    "target_lower_color": V7.COLOR_OPTIONS,
    "target_lower_secondary_color": V7.COLOR_OPTIONS,
    "target_lower_pattern": V7.PATTERN_OPTIONS,
    "target_lower_type": V7.LOWER_TYPE_OPTIONS,
    "target_gender": V7.GENDER_OPTIONS,
}


def make_ui():
    ui = object.__new__(V7.Milestone2TargetUI)
    ui.vars = {}
    for target_id in ("target_A", "target_B"):
        profile = V7.DEFAULT_TARGET_PROFILES[target_id]
        for field, options in OPTION_MAP.items():
            ui.vars[f"{target_id}_{field}"] = FakeVar(
                V7.reverse_label(options, profile[field])
            )
        ui.vars[f"{target_id}_target_item"] = FakeVar(profile["target_item"])
    ui.vars["lmstudio_base_url"] = FakeVar("http://127.0.0.1:1234/v1")
    ui.quick_text_target_A = FakeVar("白いシャツ、青いズボンの人")
    ui.quick_text_target_B = FakeVar("黒いスカートのマネキン")
    ui.target_image_paths = {"target_A": "", "target_B": ""}
    ui.target_image_display_vars = {"target_A": FakeVar("未選択"), "target_B": FakeVar("未選択")}
    ui.target_preview_labels = {}
    ui.target_preview_images = {}
    ui.target_image_apply_buttons = {}
    ui._target_image_vlm_running = set()
    ui.messages = []
    ui.log = ui.messages.append
    return ui


def sample_result(**overrides):
    result = {
        "subject_type": "person",
        "target_upper_color": "white",
        "target_upper_secondary_color": "navy",
        "target_upper_pattern": "striped",
        "target_upper_type": "sweater",
        "target_lower_color": "blue",
        "target_lower_secondary_color": "none",
        "target_lower_pattern": "solid",
        "target_lower_type": "skirt",
        "target_gender": "female",
        "target_item": "unspecified",
    }
    result.update(overrides)
    return result


class TargetImageInputTests(unittest.TestCase):
    def setUp(self):
        self.original_dialog = V7.filedialog.askopenfilename
        self.original_showerror = V7.messagebox.showerror
        self.original_showinfo = V7.messagebox.showinfo

    def tearDown(self):
        V7.filedialog.askopenfilename = self.original_dialog
        V7.messagebox.showerror = self.original_showerror
        V7.messagebox.showinfo = self.original_showinfo

    def test_01_target_a_image_selection_updates_only_a(self):
        ui = make_ui()
        V7.filedialog.askopenfilename = lambda **_kwargs: "/tmp/person-a.jpg"
        ui.choose_target_image("target_A")
        self.assertEqual("/tmp/person-a.jpg", ui.target_image_paths["target_A"])
        self.assertEqual("", ui.target_image_paths["target_B"])
        self.assertEqual("person-a.jpg", ui.target_image_display_vars["target_A"].get())

    def test_02_target_b_image_selection_updates_only_b(self):
        ui = make_ui()
        V7.filedialog.askopenfilename = lambda **_kwargs: "/tmp/person-b.webp"
        ui.choose_target_image("target_B")
        self.assertEqual("", ui.target_image_paths["target_A"])
        self.assertEqual("/tmp/person-b.webp", ui.target_image_paths["target_B"])

    def test_03_target_a_result_updates_only_a(self):
        ui = make_ui()
        before_b = ui.target_profile_from_ui("target_B")
        ui.apply_target_attribute_image_profile_to_ui("target_A", sample_result())
        self.assertEqual("skirt", ui.target_profile_from_ui("target_A")["target_lower_type"])
        self.assertEqual(before_b, ui.target_profile_from_ui("target_B"))

    def test_04_target_b_result_updates_only_b(self):
        ui = make_ui()
        before_a = ui.target_profile_from_ui("target_A")
        profile = sample_result(target_upper_color="red", target_lower_type="pants")
        ui.apply_target_attribute_image_profile_to_ui("target_B", profile)
        self.assertEqual("red", ui.target_profile_from_ui("target_B")["target_upper_color"])
        self.assertEqual(before_a, ui.target_profile_from_ui("target_A"))

    def test_05_aliases_normalize_to_existing_ui_enums(self):
        normalized = V7.normalize_target_attribute_image_result(sample_result(
            target_upper_color="navy blue",
            target_upper_secondary_color="light blue",
            target_upper_pattern="horizontal stripes",
            target_upper_type="T-shirt",
            target_lower_pattern="checkered",
            target_lower_type="long skirt",
        ))
        self.assertEqual("navy", normalized["target_upper_color"])
        self.assertEqual("light_blue", normalized["target_upper_secondary_color"])
        self.assertEqual("striped", normalized["target_upper_pattern"])
        self.assertEqual("t_shirt", normalized["target_upper_type"])
        self.assertEqual("plaid", normalized["target_lower_pattern"])
        self.assertEqual("skirt", normalized["target_lower_type"])

    def test_06_unknown_values_are_neutral_and_subject_is_preserved(self):
        normalized = V7.normalize_target_attribute_image_result(
            sample_result(
                subject_type="unknown",
                target_upper_color="purple",
                target_upper_type="unknown",
                target_lower_type="culottes",
                target_gender="unknown",
                target_item="mystery object",
            ),
            current_subject_type="mannequin",
        )
        self.assertEqual("mannequin", normalized["subject_type"])
        self.assertEqual("none", normalized["target_upper_color"])
        self.assertEqual("none", normalized["target_upper_type"])
        self.assertEqual("none", normalized["target_lower_type"])
        self.assertEqual("none", normalized["target_gender"])
        self.assertEqual("unspecified", normalized["target_item"])

    def test_07_mannequin_never_adds_gender(self):
        normalized = V7.normalize_target_attribute_image_result(
            sample_result(subject_type="mannequin", target_gender="female")
        )
        self.assertEqual("none", normalized["target_gender"])

    def test_08_request_uses_exact_model_prompt_and_deterministic_shared_helper(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.jpeg"
            image_path.write_bytes(b"not-decoded-by-fake-request")

            def fake_request(**kwargs):
                captured.update(kwargs)
                return sample_result()

            result = V7.extract_target_attributes_from_image(
                image_path,
                "http://127.0.0.1:1234/v1",
                request_fn=fake_request,
            )
        self.assertEqual("gemma-4-e2b-it-qat", captured["model"])
        self.assertEqual(V7.TARGET_ATTRIBUTE_FROM_IMAGE_PROMPT, captured["prompt"])
        self.assertEqual("target_attribute_image", captured["kind"])
        self.assertEqual(384, captured["max_tokens"])
        self.assertEqual("skirt", result["target_lower_type"])

    def test_09_invalid_json_error_is_fail_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            image_path.write_bytes(b"x")
            with self.assertRaises(V7.TargetAttributeImageError) as caught:
                V7.extract_target_attributes_from_image(
                    image_path,
                    "http://127.0.0.1:1234/v1",
                    request_fn=lambda **_kwargs: {"error": "JSON_DECODE_ERROR"},
                )
        self.assertEqual("TARGET_ATTRIBUTE_IMAGE_PARSE_FAILED", caught.exception.code)

    def test_10_lm_unavailable_error_is_fail_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.jpg"
            image_path.write_bytes(b"x")
            with self.assertRaises(V7.TargetAttributeImageError) as caught:
                V7.extract_target_attributes_from_image(
                    image_path,
                    "http://127.0.0.1:1234/v1",
                    request_fn=lambda **_kwargs: {
                        "error": "ConnectionRefusedError", "message": "refused"
                    },
                )
        self.assertEqual("TARGET_ATTRIBUTE_IMAGE_LM_UNAVAILABLE", caught.exception.code)

    def test_11_apply_without_image_shows_error_and_does_not_start_worker(self):
        ui = make_ui()
        shown = []
        V7.messagebox.showerror = lambda *args: shown.append(args)
        ui.apply_target_attributes_from_image("target_A")
        self.assertTrue(shown)
        self.assertFalse(ui._target_image_vlm_running)
        self.assertIn("TARGET_ATTRIBUTE_IMAGE_NOT_SELECTED", ui.messages[-1])

    def test_12_existing_japanese_text_apply_remains_independent(self):
        ui = make_ui()
        before_b = ui.target_profile_from_ui("target_B")
        ui.apply_quick_text("target_A")
        result_a = ui.target_profile_from_ui("target_A")
        self.assertEqual("white", result_a["target_upper_color"])
        self.assertEqual("blue", result_a["target_lower_color"])
        self.assertEqual(before_b, ui.target_profile_from_ui("target_B"))
        self.assertEqual({"target_A": "", "target_B": ""}, ui.target_image_paths)

    def test_13_manual_fields_remain_editable_and_serializable(self):
        ui = make_ui()
        ui.set_combo_value("target_A_target_upper_color", V7.COLOR_OPTIONS, "red")
        ui.vars["target_A_target_item"].set("umbrella")
        profile = ui.target_profile_from_ui("target_A")
        self.assertEqual("red", profile["target_upper_color"])
        self.assertEqual("umbrella", profile["target_item"])

    def test_14_ab_profiles_still_generate_the_existing_targets_json_shape(self):
        ui = make_ui()
        targets = [ui.target_profile_from_ui("target_A"), ui.target_profile_from_ui("target_B")]
        parsed = json.loads(json.dumps(targets, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(["target_A", "target_B"], [item["target_id"] for item in parsed])
        self.assertEqual(V7.TARGET_FIELD_KEYS, [key for key in V7.TARGET_FIELD_KEYS if key in parsed[0]])

    def test_15_existing_person_and_front_prompt_sha_are_unchanged(self):
        expected = {
            "MS2_PERSON_PROMPT": "94abc619e21424b420f6faf19542e4b149303f36c9428ddca17f8c5bc332d7de",
            "FRONT_PHOTO_PROMPT": "482706fa783943c0b443e81e8b3a508be9490bff5214105ec72bf7f7c3f6ee3e",
        }
        prompt_path = ROOT / "person_search_vlm_milestone2_v7.py"
        tree = ast.parse(prompt_path.read_text(encoding="utf-8"), filename=str(prompt_path))
        prompts = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not any(name in expected for name in names):
                continue
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                value = value.func.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                for name in names:
                    prompts[name] = value.value.strip()
        for name, digest in expected.items():
            self.assertEqual(digest, hashlib.sha256(prompts[name].encode()).hexdigest())

    def test_16_person_search_core_file_matches_reviewed_hash(self):
        digest = hashlib.sha256(
            (ROOT / "person_search_vlm_milestone2_v7.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            "c5061b82ce2a8cfcc11b5b2bc51faeb3b19b1d0e9591349829e50df7ae675d89",
            digest,
        )

    def test_17_ros_interface_and_qos_contract_remain_present(self):
        source = UI_PATH.read_text(encoding="utf-8")
        for token in (
            "DEFAULT_ARRIVAL_TOPIC = \"/person_area_arrived\"",
            "DEFAULT_SEARCH_DONE_TOPIC = \"/person_search_done\"",
            "DEFAULT_SEARCH_DETAIL_TOPIC = \"/person_search_done_detail\"",
            "ReliabilityPolicy.RELIABLE",
            "DurabilityPolicy.TRANSIENT_LOCAL",
        ):
            self.assertIn(token, source)

    def test_18_layout_has_both_independent_photo_controls_in_scroll_content(self):
        build_source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        panel_source = inspect.getsource(V7.Milestone2TargetUI._build_target_photo_panel)
        self.assertIn('self._build_target_photo_panel(target_a, "target_A")', build_source)
        self.assertIn('self._build_target_photo_panel(target_b, "target_B")', build_source)
        self.assertIn('"target_A",', build_source)
        self.assertIn('"target_B",', build_source)
        self.assertIn('self.choose_target_image(selected)', panel_source)
        self.assertIn('self.apply_target_attributes_from_image(selected)', panel_source)
        self.assertIn("self.target_preview_labels[target_id] = preview", panel_source)
        self.assertIn("content_canvas", build_source)

    def test_19_prompt_is_json_only_and_has_exact_requested_schema(self):
        prompt = V7.TARGET_ATTRIBUTE_FROM_IMAGE_PROMPT
        self.assertIn("JSON object only", prompt)
        for field in V7.TARGET_FIELD_KEYS:
            self.assertIn(f'"{field}"', prompt)
        self.assertNotIn("MS2_PERSON_PROMPT", prompt)
        self.assertNotIn("FRONT_PHOTO_PROMPT", prompt)

    def test_20_logging_markers_exist_without_base64_logging(self):
        source = inspect.getsource(V7.Milestone2TargetUI)
        for marker in (
            "TARGET_ATTRIBUTE_IMAGE_SELECTED",
            "TARGET_ATTRIBUTE_IMAGE_VLM_BEGIN",
            "TARGET_ATTRIBUTE_IMAGE_VLM_SUCCESS",
            "TARGET_ATTRIBUTE_IMAGE_APPLIED",
            "TARGET_ATTRIBUTE_IMAGE_FAILED",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("base64", source)

    def test_21_preview_fit_preserves_aspect_ratio_without_upscaling(self):
        self.assertEqual((260, 130), V7.fit_preview_size(1000, 500))
        self.assertEqual((110, 220), V7.fit_preview_size(500, 1000))
        self.assertEqual((120, 80), V7.fit_preview_size(120, 80))
        with self.assertRaises(V7.TargetPreviewError):
            V7.fit_preview_size(0, 100)

    def test_22_preview_loader_returns_contained_ppm(self):
        class FakeImage:
            ndim = 3
            shape = (500, 1000, 3)

            def __getitem__(self, _key):
                return self

            def tobytes(self):
                return b"rgb"

        class FakeCv2:
            IMREAD_COLOR = 1
            INTER_AREA = 2

            def __init__(self):
                self.resize_args = None

            def imread(self, _path, _mode):
                return FakeImage()

            def resize(self, image, size, interpolation):
                self.resize_args = (size, interpolation)
                image.shape = (size[1], size[0], 3)
                return image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.jpg"
            path.write_bytes(b"fixture")
            cv2 = FakeCv2()
            ppm, size = V7.load_target_preview_ppm(path, cv2_module=cv2)
        self.assertEqual((260, 130), size)
        self.assertEqual(((260, 130), FakeCv2.INTER_AREA), cv2.resize_args)
        self.assertTrue(ppm.startswith(b"P6\n260 130\n255\n"))

    def test_23_invalid_preview_is_reported_without_crashing(self):
        class FakeCv2:
            IMREAD_COLOR = 1

            @staticmethod
            def imread(_path, _mode):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(V7.TargetPreviewError):
                V7.load_target_preview_ppm(path, cv2_module=FakeCv2())

    def test_24_invalid_selected_image_updates_only_its_preview(self):
        class FakeLabel:
            def __init__(self):
                self.options = {}

            def configure(self, **kwargs):
                self.options.update(kwargs)

        ui = make_ui()
        label_a = FakeLabel()
        label_b = FakeLabel()
        ui.target_preview_labels = {"target_A": label_a, "target_B": label_b}
        ui.target_image_paths["target_A"] = "/tmp/broken.jpg"
        with mock.patch.object(
            V7, "load_target_preview_ppm", side_effect=V7.TargetPreviewError("bad")
        ):
            self.assertFalse(ui._update_target_image_preview("target_A"))
        self.assertIn("表示できません", label_a.options["text"])
        self.assertEqual({}, label_b.options)
        self.assertIn("TARGET_IMAGE_PREVIEW_FAILED target=A", ui.messages[-1])

    def test_25_a_and_b_preview_images_are_independent(self):
        class FakeLabel:
            def __init__(self):
                self.options = {}

            def configure(self, **kwargs):
                self.options.update(kwargs)

        ui = make_ui()
        ui.target_preview_labels = {
            "target_A": FakeLabel(),
            "target_B": FakeLabel(),
        }
        ui.target_image_paths = {
            "target_A": "/tmp/a.jpg",
            "target_B": "/tmp/b.jpg",
        }
        created = []

        def fake_photo_image(**kwargs):
            token = object()
            created.append((token, kwargs))
            return token

        with (
            mock.patch.object(V7, "load_target_preview_ppm", return_value=(b"ppm", (10, 8))),
            mock.patch.object(V7.tk, "PhotoImage", side_effect=fake_photo_image),
        ):
            self.assertTrue(ui._update_target_image_preview("target_A"))
            self.assertNotIn("target_B", ui.target_preview_images)
            image_a = ui.target_preview_images["target_A"]
            self.assertTrue(ui._update_target_image_preview("target_B"))

        self.assertIs(image_a, ui.target_preview_images["target_A"])
        self.assertIsNot(image_a, ui.target_preview_images["target_B"])
        self.assertEqual(2, len(created))

    def test_26_lm_status_labels_cover_all_required_states(self):
        class FakeLabel:
            def __init__(self):
                self.options = {}

            def configure(self, **kwargs):
                self.options.update(kwargs)

        ui = make_ui()
        ui.lm_status_var = FakeVar()
        ui.lm_status_label = FakeLabel()
        expected = {
            "unprepared": "● 未準備",
            "starting": "● 起動中",
            "ready": "● 接続OK",
            "failed": "● 失敗",
        }
        for state, text in expected.items():
            ui._set_lm_status(state)
            self.assertEqual(text, ui.lm_status_var.get())
            self.assertIn("foreground", ui.lm_status_label.options)

    def test_27_clear_target_a_resets_only_image_state_and_preserves_attributes(self):
        ui = make_ui()
        label_a = FakeWidget(image="a", text="")
        label_b = FakeWidget(image="b", text="")
        button_a = FakeWidget(state="normal")
        button_b = FakeWidget(state="normal")
        image_a = object()
        image_b = object()
        ui.target_preview_labels = {"target_A": label_a, "target_B": label_b}
        ui.target_preview_images = {"target_A": image_a, "target_B": image_b}
        ui.target_image_apply_buttons = {"target_A": button_a, "target_B": button_b}
        ui.target_image_paths = {"target_A": "/tmp/a.jpg", "target_B": "/tmp/b.jpg"}
        ui.target_image_display_vars["target_A"].set("a.jpg")
        ui.target_image_display_vars["target_B"].set("b.jpg")
        before_a = ui.target_profile_from_ui("target_A")
        before_b = ui.target_profile_from_ui("target_B")
        before_text_a = ui.quick_text_target_A.get()
        before_text_b = ui.quick_text_target_B.get()

        ui.clear_target_image("target_A")

        self.assertEqual("", ui.target_image_paths["target_A"])
        self.assertEqual("未選択", ui.target_image_display_vars["target_A"].get())
        self.assertNotIn("target_A", ui.target_preview_images)
        self.assertEqual("", label_a.options["image"])
        self.assertIn("画像未選択", label_a.options["text"])
        self.assertEqual("disabled", button_a.options["state"])
        self.assertEqual("/tmp/b.jpg", ui.target_image_paths["target_B"])
        self.assertIs(image_b, ui.target_preview_images["target_B"])
        self.assertEqual("b", label_b.options["image"])
        self.assertEqual("normal", button_b.options["state"])
        self.assertEqual(before_a, ui.target_profile_from_ui("target_A"))
        self.assertEqual(before_b, ui.target_profile_from_ui("target_B"))
        self.assertEqual(before_text_a, ui.quick_text_target_A.get())
        self.assertEqual(before_text_b, ui.quick_text_target_B.get())

    def test_28_clear_target_b_does_not_change_target_a_image_state(self):
        ui = make_ui()
        image_a = object()
        image_b = object()
        ui.target_preview_labels = {
            "target_A": FakeWidget(image="a", text=""),
            "target_B": FakeWidget(image="b", text=""),
        }
        ui.target_preview_images = {"target_A": image_a, "target_B": image_b}
        ui.target_image_apply_buttons = {
            "target_A": FakeWidget(state="normal"),
            "target_B": FakeWidget(state="normal"),
        }
        ui.target_image_paths = {"target_A": "/tmp/a.jpg", "target_B": "/tmp/b.jpg"}
        ui.target_image_display_vars["target_A"].set("a.jpg")
        ui.target_image_display_vars["target_B"].set("b.jpg")

        ui.clear_target_image("target_B")

        self.assertEqual("/tmp/a.jpg", ui.target_image_paths["target_A"])
        self.assertIs(image_a, ui.target_preview_images["target_A"])
        self.assertEqual("", ui.target_image_paths["target_B"])
        self.assertNotIn("target_B", ui.target_preview_images)
        self.assertEqual("未選択", ui.target_image_display_vars["target_B"].get())
        self.assertEqual(
            "disabled", ui.target_image_apply_buttons["target_B"].options["state"]
        )

    def test_29_apply_button_tracks_select_clear_and_reselect(self):
        ui = make_ui()
        button = FakeWidget(state="disabled")
        ui.target_image_apply_buttons = {"target_A": button}

        ui.set_target_image_path("target_A", "/tmp/first.jpg")
        self.assertEqual("normal", button.options["state"])

        ui.clear_target_image("target_A")
        self.assertEqual("disabled", button.options["state"])

        ui.set_target_image_path("target_A", "/tmp/second.jpg")
        self.assertEqual("normal", button.options["state"])


if __name__ == "__main__":
    unittest.main()
