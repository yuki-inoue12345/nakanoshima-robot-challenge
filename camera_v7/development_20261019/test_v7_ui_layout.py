"""Static and pure-function checks for the Camera V7 vertical layout."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"
LOADER = importlib.machinery.SourceFileLoader("daikoudai_v7_layout_test", str(UI_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
V7 = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = V7
LOADER.exec_module(V7)


class UiLayoutTests(unittest.TestCase):
    def test_initial_size_keeps_desktop_panel_margin(self):
        self.assertEqual((1220, 1000), V7.calculate_initial_ui_size(1920, 1080))
        self.assertEqual((1220, 698), V7.calculate_initial_ui_size(1366, 768))
        self.assertEqual((720, 530), V7.calculate_initial_ui_size(800, 600))

    def test_settings_content_has_vertical_scrollbar(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn("content_canvas = tk.Canvas(content_host", source)
        self.assertIn('orient="vertical"', source)
        self.assertIn("command=content_canvas.yview", source)

    def test_bottom_action_bar_is_outside_scroll_content(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn("buttons = ttk.Frame(shell)", source)
        self.assertIn("self.bottom_action_bar = buttons", source)
        self.assertNotIn("buttons = ttk.Frame(root)", source)

    def test_header_keeps_lm_state_and_prepare_action_fixed(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn("header = ttk.Frame(shell", source)
        self.assertIn('value="● 未準備"', source)
        self.assertIn('text="LM Studio 起動・準備"', source)
        self.assertIn("command=self.prepare_lm_studio", source)
        self.assertLess(source.index("header = ttk.Frame(shell"), source.index("content_host = ttk.Frame(shell)"))

    def test_required_action_buttons_remain_in_fixed_bar(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        for label in (
            "人物探索を実行（LM Studio）",
            "人物探索サーバー起動",
            "手動で探索開始 false→true",
            "停止",
            "カメラ解放",
        ):
            self.assertIn(label, source)

    def test_target_frame_titles_are_stable_and_simple(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn('target_notebook.add(target_a, text="Target A")', source)
        self.assertIn('target_notebook.add(target_b, text="Target B")', source)
        self.assertNotIn("Target A: 白紺ボーダーのマネキン", source)
        self.assertNotIn("Target B: チェックシャツの人間", source)

    def test_target_a_and_b_use_notebook_instead_of_side_by_side_grid(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn("target_notebook = ttk.Notebook(target)", source)
        self.assertIn("self.target_notebook = target_notebook", source)
        self.assertIn("target_a = ttk.Frame(target_notebook)", source)
        self.assertIn("target_b = ttk.Frame(target_notebook)", source)
        self.assertNotIn("target_a.grid(", source)
        self.assertNotIn("target_b.grid(", source)

    def test_show_command_preserves_server_or_direct_selection(self):
        original_askyesno = V7.messagebox.askyesno
        original_showerror = V7.messagebox.showerror
        try:
            for show_server, expected in ((True, "server"), (False, "direct")):
                with self.subTest(show_server=show_server):
                    calls = []
                    ui = object.__new__(V7.Milestone2TargetUI)
                    ui.ros_command = lambda: calls.append("ros") or ["server"]
                    ui.direct_command = lambda: calls.append("direct") or ["direct"]
                    ui.log = lambda message: calls.append(("log", message))
                    ui.show_command_window = lambda title, text: calls.append(("window", title, text))
                    V7.messagebox.askyesno = lambda *_args, **_kwargs: show_server
                    V7.messagebox.showerror = lambda *_args, **_kwargs: self.fail("unexpected error")

                    V7.Milestone2TargetUI.show_command(ui)

                    self.assertIn("ros" if show_server else "direct", calls)
                    self.assertNotIn("direct" if show_server else "ros", calls)
                    self.assertIn(("window", "人物探索サーバー コマンド" if show_server else "人物探索 Direct コマンド", expected), calls)
        finally:
            V7.messagebox.askyesno = original_askyesno
            V7.messagebox.showerror = original_showerror

    def test_command_window_is_read_only_scrollable_and_copyable(self):
        source = inspect.getsource(V7.Milestone2TargetUI.show_command_window)
        self.assertIn('wrap="none"', source)
        self.assertIn('orient="vertical"', source)
        self.assertIn('orient="horizontal"', source)
        self.assertIn('state="disabled"', source)
        self.assertIn('bind("<Control-c>"', source)
        self.assertIn('text="全コピー"', source)
        self.assertIn('text="閉じる"', source)

    def test_diagnostic_log_button_and_window_controls_exist(self):
        build_source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        window_source = inspect.getsource(V7.Milestone2TargetUI.show_log_window)
        self.assertIn('text="実行ログ・診断"', build_source)
        self.assertIn("終了原因・録画・ROS通信・人物探索ログを確認", build_source)
        self.assertIn("LOG_CATEGORIES", window_source)
        self.assertIn('text="検索:"', window_source)
        self.assertIn('text="全コピー"', window_source)
        self.assertIn('text="最新ログファイルを開く"', window_source)
        self.assertIn('text="ログ保存先を開く"', window_source)
        self.assertIn('text="閉じる"', window_source)

    def test_target_image_controls_are_inside_scrollable_content(self):
        build_source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        photo_source = inspect.getsource(V7.Milestone2TargetUI._build_target_photo_panel)
        self.assertIn('self._build_target_photo_panel(target_a, "target_A")', build_source)
        self.assertIn('self._build_target_photo_panel(target_b, "target_B")', build_source)
        self.assertIn('text="対象写真・プレビュー"', photo_source)
        self.assertIn('text="写真を選択"', photo_source)
        self.assertIn('text="写真から反映"', photo_source)
        self.assertIn('text="写真を解除"', photo_source)
        self.assertIn('state="disabled"', photo_source)
        self.assertIn("self.clear_target_image(selected)", photo_source)
        self.assertIn("self.target_preview_labels[target_id] = preview", photo_source)
        self.assertLess(build_source.index("form = ttk.Frame(root)"), build_source.index('buttons = ttk.Frame(shell)'))

    def test_each_target_card_contains_its_own_japanese_text_input(self):
        build_source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        text_source = inspect.getsource(V7.Milestone2TargetUI._build_target_text_panel)
        self.assertIn('self._build_target_text_panel(\n            target_a,', build_source)
        self.assertIn('self._build_target_text_panel(\n            target_b,', build_source)
        self.assertIn('text="日本語指定"', text_source)
        self.assertIn("self.apply_quick_text(selected)", text_source)

    def test_runtime_is_split_into_normal_and_detail_tabs(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn("runtime_notebook = ttk.Notebook(runtime)", source)
        self.assertIn('text="通常設定・状態"', source)
        self.assertIn('text="詳細・ROS"', source)

    def test_bottom_actions_are_grouped_by_purpose(self):
        source = inspect.getsource(V7.Milestone2TargetUI._build_ui)
        self.assertIn('ttk.LabelFrame(primary_buttons, text="人物探索")', source)
        self.assertIn('ttk.LabelFrame(primary_buttons, text="写真だけ")', source)


if __name__ == "__main__":
    unittest.main()
