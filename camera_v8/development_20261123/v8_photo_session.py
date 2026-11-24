#!/usr/bin/env python3
"""V8 still-photo session: ROS-only preview, optional isolated C920 owner.

When a shared owner exists, attach read-only.  Without one, start a separate
camera-owner ROS process only if the C920 is not already in use.  Own, terminate
and reap exactly the process this launcher starts; never stop AR/Nav2/Follower.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
NODE = '/koko_v8_camera_source'


def owner_exists() -> bool:
    try:
        r = subprocess.run(['ros2', 'node', 'list'], capture_output=True,
                           text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and NODE in r.stdout.splitlines()


def locate_c920() -> Path:
    devices = sorted(Path('/dev/v4l/by-id').glob('*C920*video-index0'))
    for device in devices:
        if device.exists() and device.resolve().is_char_device():
            return device
    raise RuntimeError('C920 video-index0 が見つかりません。')


def require_camera_free(device: Path) -> None:
    try:
        result = subprocess.run(['fuser', str(device.resolve())],
                                check=False, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('C920使用状況を確認できないため起動を拒否しました。') from exc
    if result.returncode == 0:
        raise RuntimeError('C920は既に使用中ですが共有ROSノードが見つかりません。二重起動を拒否しました。')
    if result.returncode != 1:
        raise RuntimeError('C920使用状況の検査に失敗しました。')


def stop_owned_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def handle_exit(signum, _frame) -> None:
    raise SystemExit(128 + int(signum))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request-topic', required=True)
    ap.add_argument('--done-topic', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    if any(not x.startswith('/') or any(c.isspace() for c in x)
           for x in (args.request_topic, args.done_topic)):
        raise SystemExit('ROSトピック名が不正です')
    if not (HERE / 'v8_shared_photo_preview.py').is_file():
        raise SystemExit('共有写真プレビューがありません')
    # Ensure ROS access and the camera-driver imports before trying hardware.
    try:
        import rclpy  # noqa: F401
        from ar_path_viewer.latest_image_source import LatestImageSource  # noqa: F401
    except ImportError as exc:
        raise SystemExit('V8写真撮影にはROSとar_wsのinstall/setup.bashが必要です: '+str(exc))

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGHUP, handle_exit)
    lock_file = None
    log_file = None
    owner = None
    preview = None
    started_owner = False
    try:
        if not owner_exists():
            # This flock serializes multiple standalone V8 photo sessions.
            runtime = Path(os.environ.get('XDG_RUNTIME_DIR') or '/tmp')
            lock_path = runtime / f'kokochan-v8-camera-{os.getuid()}.lock'
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            lock_file = os.fdopen(fd, 'w')
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('別のCamera V8がC920起動中です。') from exc
            if not owner_exists():
                device = locate_c920()
                require_camera_free(device)
                log_dir = HERE / 'outputs' / 'camera_owner'
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f'v8_photo_owner_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.log'
                log_file = log_path.open('x', encoding='utf-8')
                owner = subprocess.Popen(
                    [sys.executable, '-u', str(HERE / 'v8_camera_source.py'),
                     '--device', str(device), '--fps', '30',
                     '--expected-parent-pid', str(os.getpid())],
                    cwd=str(HERE), stdout=log_file, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                started_owner = True
                print(f'V8写真撮影: C920専用ROS所有プロセス起動 PID={owner.pid}; log={log_path}', flush=True)
                ready = False
                for _ in range(60):
                    if owner.poll() is not None:
                        break
                    if owner_exists():
                        ready = True
                        break
                    time.sleep(0.25)
                if not ready:
                    raise RuntimeError(f'C920共有所有ノード起動失敗: {log_path} を確認してください')
        if not started_owner:
            print('V8写真撮影: 既存共有カメラへ購読接続（AR/所有プロセスは停止しません）', flush=True)
        preview = subprocess.Popen(
            [sys.executable, str(HERE / 'v8_shared_photo_preview.py'),
             '--request-topic', args.request_topic,
             '--done-topic', args.done_topic,
             '--out-dir', args.out_dir],
            cwd=str(HERE),
        )
        while preview.poll() is None:
            time.sleep(0.15)
        return int(preview.returncode or 0)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Camera V8 photo session error: {exc}', file=sys.stderr, flush=True)
        return 2
    finally:
        stop_owned_process(preview)
        # Only terminate a camera source created by THIS session.
        stop_owned_process(owner)
        if log_file is not None:
            log_file.close()
        if lock_file is not None:
            lock_file.close()


if __name__ == '__main__':
    sys.exit(main())
