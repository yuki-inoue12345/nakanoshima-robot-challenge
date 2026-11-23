#!/usr/bin/env python3
"""V8 GUI keeping the *exact* 2026-11-23 desktop V7 UI layout.

Loads the known SHA-256-matched V7 .new UI without editing it, redirects UI
settings/output to V8, and substitutes ONLY the live-search server with V8's
ROS image subscriber.  No navigation publisher or camera opener is added here.
"""
from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import os
import signal
from pathlib import Path
import subprocess
import sys

from v8_ui_routes import (
    V8RoutingError, build_live_search_command,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
V7_DIR = REPO / 'camera_v7' / 'development_20261019'
V7_SOURCE = V7_DIR / 'daikoudai_v7.py.new'
V8_ENGINE = HERE / 'daikoudai_v8.py'
EXPECTED_V7_SHA256 = '720a2968563cf58b0a972f9034dc10747b430582ef12c6f9af2b41c7cfaf2ee9'


def load_v7_ui():
    if hashlib.sha256(V7_SOURCE.read_bytes()).hexdigest() != EXPECTED_V7_SHA256:
        raise RuntimeError('V7 desktop UI source hash mismatch; refuse uncertain UI migration')
    if not V8_ENGINE.is_file():
        raise RuntimeError('V8 recognition backend not found')
    # Resolve camera_video_recording and other existing UI-only dependencies
    # without modifying either V7 or its original desktop entry.
    if str(V7_DIR) not in sys.path:
        sys.path.insert(0, str(V7_DIR))
    loader = SourceFileLoader('kokochan_v7_desktop_layout', str(V7_SOURCE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError('V7 UI loader unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    prior_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior_bytecode

    # UI-generated config, evidence paths and logs belong only to the V8 tree.
    module.APP_DIR = HERE
    module.THIS_FILE = Path(__file__).resolve()
    module.SIM_SCRIPT = V7_DIR / 'person_search_vlm_milestone2_v7.py'
    module.LAST_CONFIG_PATH = HERE / '.daikoudai_v8_target_ui_last.json'
    module.DEFAULT_UI_LOG_DIR = HERE / 'outputs' / 'ui_logs'
    module.PERSON_SEARCH_FOUND_SOUND = V7_DIR / 'sounds' / 'zundamon_found.wav'
    module.PERSON_SEARCH_NOT_FOUND_SOUND = V7_DIR / 'sounds' / 'zundamon_not_found.wav'
    original_create = module.create_ui_log_file

    def create_v8_ui_log_file(*args, **kwargs):
        if not args:
            kwargs.setdefault('directory', module.DEFAULT_UI_LOG_DIR)
        return original_create(*args, **kwargs)

    module.create_ui_log_file = create_v8_ui_log_file
    return module


v7 = load_v7_ui()


def shared_camera_owner_exists():
    try:
        proc = subprocess.run(['ros2', 'node', 'list'], capture_output=True,
                              text=True, timeout=4, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and '/koko_v8_camera_source' in proc.stdout.splitlines()


def c920_in_use():
    devices = sorted(Path('/dev/v4l/by-id').glob('*C920*video-index0'))
    if not devices:
        raise V8RoutingError('C920が見つかりません')
    device = devices[0]
    if not device.exists():
        raise V8RoutingError('C920デバイスが利用できません')
    try:
        paths = [str(device), str(device.resolve())]
        p = subprocess.run(['fuser', *paths], capture_output=True,
                           timeout=4, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Refuse direct-camera preview when camera ownership cannot be verified.
        raise V8RoutingError('C920所有者を確認できないため直接起動を拒否しました') from exc
    if p.returncode not in (0, 1):
        raise V8RoutingError('C920使用状況の確認に失敗したため直接起動を拒否しました')
    return p.returncode == 0


class CameraV8PreservedUI(v7.Milestone2TargetUI):
    def _build_ui(self):
        super()._build_ui()
        # Remove ONLY the video-recording control. The original .new layout,
        # photo input, LM Studio prep, diagnostic UI and photo buttons survive.
        matched = []

        def visit(widget):
            for child in widget.winfo_children():
                if isinstance(child, v7.ttk.LabelFrame) and child.cget('text') == '実走行の動画保存':
                    matched.append(child)
                else:
                    visit(child)

        visit(self)
        if len(matched) != 1:
            raise RuntimeError(f'Unexpected recording UI count: {len(matched)}')
        matched[0].destroy()
        self.record_run_video_var.set(False)

    def _load_last_config(self, silent=False):
        super()._load_last_config(silent=silent)
        # Old V7 saved choices must not revive recording in V8.
        self.record_run_video_var.set(False)

    def current_config(self):
        cfg = super().current_config()
        cfg['record_run_video'] = False
        return cfg

    def ros_command(self):
        cfg = self.current_config()
        common = []
        self._append_common_args(common, cfg)
        command = build_live_search_command(
            sys.executable, V8_ENGINE, cfg, common, self.ms2_runtime_tuning_args())
        return v7.ros_wrapped_command(command)

    def run_ros_node(self):
        try:
            if self.start_process(self.ros_command()):
                self.log('V8: 人物探索サーバー起動。/person_area_arrived を待機します。')
                self.log('V8: C920を直接開かず、探索中のROS生映像のみ受信します。')
                self.log('V8: 動画録画は行いません。ARが録画を担当します。')
                self.show_capture_notice('V8探索待機中', kind='camera', duration_ms=900)
        except Exception as exc:
            v7.messagebox.showerror('実行できません', str(exc))

    def direct_command(self):
        if self.current_config().get('input_mode') == 'video':
            return super().direct_command()
        raise V8RoutingError('実機カメラの直接実行はV8で禁止。人物探索サーバー起動を使用してください。')

    def run_direct(self):
        if self.current_config().get('input_mode') == 'video':
            # Existing V7 offline video test behavior is intentionally unchanged.
            return super().run_direct()
        v7.messagebox.showinfo(
            'V8 実機人物探索',
            '実機人物探索は「人物探索サーバー起動」→到着イベントで開始します。'
            '単独テストは「手動で探索開始 false→true」を押してください。'
        )

    def check_camera_devices(self):
        if shared_camera_owner_exists():
            self.log('V8共有カメラが起動中です。C920を直接開かずROS映像を使用します。')
            self.show_capture_notice('共有カメラOK', kind='done', duration_ms=1100)
            return
        try:
            if c920_in_use():
                self.log('C920は他プロセスが使用中です。カメラを直接開きません。')
                return
        except V8RoutingError as exc:
            self.log(str(exc))
            return
        # V8 cannot open the device even for a one-frame camera probe.
        # Standalone preview starts a separate ROS camera-owner process.
        devices = sorted(Path('/dev/v4l/by-id').glob('*C920*video-index0'))
        devices = [p for p in devices if p.exists()]
        if not devices:
            self.log('C920の接続が確認できません。映像は開いていません。')
            return
        self.vars['camera_device'].set(str(devices[0]))
        self.log(f'C920のデバイスを確認: {devices[0]}（カメラは開いていません）')
        self.show_capture_notice('C920接続OK', kind='done', duration_ms=1100)

    def topic_capture_command(self):
        # The photo path ALWAYS uses ROS raw images, even when AR is absent.
        # The session launcher owns/cleans up only the camera source it starts.
        cfg = self.current_config()
        return [sys.executable, str(HERE / 'v8_photo_session.py'),
                '--request-topic', cfg['capture_request_topic'],
                '--done-topic', cfg['capture_done_topic'],
                '--out-dir', cfg['capture_out_dir']]

    def release_camera_processes(self):
        # Unlike V7's global pkill, only stop the V8 UI's own child process.
        self.log('V8: このUIが起動した処理のみ停止します（他プロセスは停止しません）。')
        self.stop_process()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--capture-server':
        raise SystemExit('V8は直接カメラ撮影サーバーを実行しません。写真だけサーバー起動を使用してください。')
    if len(sys.argv) > 1:
        raise SystemExit('V8 UI entrypoint accepts no CLI server mode other than --capture-server')
    app = CameraV8PreservedUI()
    app.title('中チャレ MS2 人物探索 V8（V7 UI継承・AR映像共有）')

    def on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    prior = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.on_close()
    finally:
        signal.signal(signal.SIGTERM, prior)


if __name__ == '__main__':
    main()
