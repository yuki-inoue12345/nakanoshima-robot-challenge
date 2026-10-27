#!/usr/bin/env python3
"""Fail-soft raw camera recording and exclusive drive/search ownership."""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


CV_F10_KEYS = {65479, 0xFFC7, 16777273, 0x01000039}


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _local_datetime(value: Optional[dt.datetime] = None) -> dt.datetime:
    current = value if value is not None else dt.datetime.now().astimezone()
    return current.astimezone()


def timestamped_video_path(
    directory: Path,
    role_stem: str,
    *,
    started_at: Optional[dt.datetime] = None,
    reserved_paths: Optional[Set[str]] = None,
) -> Path:
    """Return a readable start-time filename without overwriting a collision."""
    directory = Path(directory)
    timestamp = _local_datetime(started_at).strftime("%Y%m%d_%H%M%S")
    base_name = f"{role_stem}_{timestamp}"
    suffix = 1
    while True:
        suffix_text = "" if suffix == 1 else f"_{suffix:02d}"
        candidate = directory / f"{base_name}{suffix_text}.mp4"
        key = str(candidate.resolve())
        if not candidate.exists() and (
            reserved_paths is None or key not in reserved_paths
        ):
            if reserved_paths is not None:
                reserved_paths.add(key)
            return candidate
        suffix += 1


def _is_stop_key(key: Any) -> bool:
    try:
        key_int = int(key)
    except (TypeError, ValueError):
        return False
    low_key = key_int & 0xFF
    return (
        key_int in CV_F10_KEYS
        or (key_int & 0xFFFF) in {65479, 0xFFC7}
        or low_key in (ord("q"), 27)
    )


class _PreviewCommand:
    def __init__(self, action: str, **payload: Any) -> None:
        self.action = str(action)
        self.payload = payload
        self.done = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None


class HighGuiPreviewManager:
    """Own every OpenCV HighGUI call on one persistent thread."""

    def __init__(
        self,
        *,
        cv2_module: Any,
        logger: Optional[Callable[[str], None]] = None,
        display_available: Optional[bool] = None,
        command_timeout_sec: float = 2.0,
    ) -> None:
        self.cv2 = cv2_module
        self._logger = logger or (lambda _message: None)
        self.display_available = (
            bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            if display_available is None
            else bool(display_available)
        )
        self.command_timeout_sec = max(0.1, float(command_timeout_sec))
        self._commands: "queue.Queue[_PreviewCommand]" = queue.Queue()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._accepting = True
        self._active_owner = ""
        self._active_window = ""
        self._owner_thread_id: Optional[int] = None

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass

    @property
    def owner_thread_id(self) -> Optional[int]:
        return self._owner_thread_id

    def _ensure_thread(self) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(
                target=self._run,
                name="opencv-highgui-owner",
                daemon=False,
            )
            self._thread.start()
            return True

    def _submit(self, action: str, default: Any, **payload: Any) -> Any:
        if action != "shutdown":
            if not self.display_available or not self._ensure_thread():
                return default
        with self._lock:
            thread = self._thread
        if thread is None:
            return default
        command = _PreviewCommand(action, **payload)
        self._commands.put(command)
        if not command.done.wait(self.command_timeout_sec):
            self._log(f"preview {action} timed out; preview disabled")
            return default
        if command.error is not None:
            self._log(
                f"preview {action} failed: "
                f"{type(command.error).__name__}: {command.error}"
            )
            return default
        return command.result

    def activate(
        self,
        owner: str,
        *,
        window_name: str,
        size: Tuple[int, int],
        position: Tuple[int, int],
        topmost: bool,
    ) -> bool:
        return bool(
            self._submit(
                "activate",
                False,
                owner=str(owner),
                window_name=str(window_name),
                size=tuple(int(value) for value in size),
                position=tuple(int(value) for value in position),
                topmost=bool(topmost),
            )
        )

    def show(self, owner: str, frame: Any) -> Optional[bool]:
        return self._submit("show", None, owner=str(owner), frame=frame)

    def deactivate(self, owner: str) -> bool:
        return bool(self._submit("deactivate", False, owner=str(owner)))

    def shutdown(self, timeout_sec: float = 2.0) -> bool:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._accepting = False
                return True
            self._accepting = False
        if thread is threading.current_thread():
            return False
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        command = _PreviewCommand("shutdown")
        self._commands.put(command)
        command.done.wait(max(0.0, deadline - time.monotonic()))
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        stopped = not thread.is_alive()
        if not stopped:
            self._log("preview owner thread did not stop before timeout")
        return stopped

    def _stage_call(
        self, owner: str, operation: str, callback: Callable[[], Any]
    ) -> Any:
        self._log(
            f"[HIGHGUI_STAGE] owner={owner} {operation}_begin "
            f"monotonic={time.perf_counter():.3f}"
        )
        result = callback()
        self._log(
            f"[HIGHGUI_STAGE] owner={owner} {operation}_done "
            f"monotonic={time.perf_counter():.3f}"
        )
        return result

    def _close_active(self) -> None:
        if not self._active_window:
            return
        owner = self._active_owner
        window_name = self._active_window
        try:
            self._stage_call(
                owner,
                "destroyWindow",
                lambda: self.cv2.destroyWindow(window_name),
            )
        except Exception as exc:
            self._log(
                f"preview destroyWindow failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self._active_owner = ""
            self._active_window = ""

    def _activate(self, command: _PreviewCommand) -> bool:
        owner = str(command.payload["owner"])
        window_name = str(command.payload["window_name"])
        self._close_active()
        try:
            self._stage_call(
                owner,
                "namedWindow",
                lambda: self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL),
            )
            self._active_owner = owner
            self._active_window = window_name
            self._stage_call(
                owner,
                "resizeWindow",
                lambda: self.cv2.resizeWindow(window_name, *command.payload["size"]),
            )
            self._stage_call(
                owner,
                "moveWindow",
                lambda: self.cv2.moveWindow(
                    window_name, *command.payload["position"]
                ),
            )
            if command.payload["topmost"]:
                prop = getattr(self.cv2, "WND_PROP_TOPMOST", None)
                if prop is not None:
                    self._stage_call(
                        owner,
                        "setWindowProperty",
                        lambda: self.cv2.setWindowProperty(window_name, prop, 1),
                    )
            return True
        except Exception:
            self._close_active()
            raise

    def _show(self, command: _PreviewCommand) -> Optional[bool]:
        owner = str(command.payload["owner"])
        if owner != self._active_owner or not self._active_window:
            return None
        try:
            self.cv2.imshow(self._active_window, command.payload["frame"])
            key = self.cv2.waitKeyEx(1)
            if _is_stop_key(key):
                self._log(
                    f"[SHUTDOWN_ORIGIN] preview_stop_key key={int(key)} "
                    f"low_key={int(key) & 0xFF}"
                )
                return False
            return True
        except Exception:
            self._close_active()
            raise

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        while True:
            command = self._commands.get()
            try:
                if command.action == "activate":
                    command.result = self._activate(command)
                elif command.action == "show":
                    command.result = self._show(command)
                elif command.action == "deactivate":
                    if str(command.payload["owner"]) == self._active_owner:
                        self._close_active()
                    command.result = True
                elif command.action == "shutdown":
                    self._close_active()
                    command.result = True
                    return
                else:
                    raise ValueError(f"unknown preview command: {command.action}")
            except BaseException as exc:
                command.error = exc
            finally:
                command.done.set()


class CameraVideoManifest:
    """Append camera segments without making metadata failure operational."""

    def __init__(
        self,
        path: Path,
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.path = Path(path)
        self._logger = logger or (lambda _message: None)
        self._lock = threading.RLock()
        self._segments: List[Dict[str, Any]] = []
        self.last_error = ""
        self._reload_from_disk()

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass

    def _reload_from_disk(self) -> None:
        try:
            if not self.path.is_file():
                return
            document = json.loads(self.path.read_text(encoding="utf-8"))
            segments = document.get("segments") if isinstance(document, dict) else None
            if isinstance(segments, list):
                self._segments = [copy.deepcopy(row) for row in segments if isinstance(row, dict)]
        except Exception as exc:
            self.last_error = f"manifest_read_failed:{type(exc).__name__}:{exc}"
            self._log(self.last_error)

    def add_segment(self, segment: Dict[str, Any]) -> bool:
        with self._lock:
            # A search recorder can share this path with the drive recorder.
            # Reload here so neither side overwrites a segment written by the other.
            self._reload_from_disk()
            self._segments.append(copy.deepcopy(segment))
            return self.flush()

    def flush(self) -> bool:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 1,
                    "updated_at": _now_iso(),
                    "segments": self._segments,
                }
                temporary = self.path.with_name(self.path.name + ".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(self.path)
                self.last_error = ""
                return True
            except Exception as exc:
                self.last_error = f"manifest_write_failed:{type(exc).__name__}:{exc}"
                self._log(self.last_error)
                return False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._reload_from_disk()
            return {
                "manifest_path": str(self.path),
                "segment_count": len(self._segments),
                "segments": copy.deepcopy(self._segments),
                "error": self.last_error,
            }


class FailSoftVideoSegment:
    """Lazily write raw frames and always turn writer failures into metadata."""

    def __init__(
        self,
        *,
        path: Path,
        kind: str,
        requested_fps: float,
        camera_source: Any,
        cv2_module: Any,
        manifest: Optional[CameraVideoManifest] = None,
        area_index: Optional[int] = None,
        target_id: str = "",
        writer_factory: Optional[Callable[..., Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        started_at: str = "",
    ) -> None:
        self.path = Path(path)
        self.kind = str(kind)
        self.requested_fps = max(0.1, float(requested_fps or 0.0))
        self.camera_source = str(camera_source)
        self.area_index = area_index
        self.target_id = str(target_id or "")
        self.cv2 = cv2_module
        self.manifest = manifest
        self._writer_factory = writer_factory or cv2_module.VideoWriter
        self._logger = logger or (lambda _message: None)
        self._writer: Any = None
        self._closed = False
        self._manifest_added = False
        self.started_at = str(started_at or _now_iso())
        self.ended_at = ""
        self.frame_count = 0
        self.resolution: List[int] = []
        self.error = ""

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass

    def mark_error(self, message: str) -> None:
        if not self.error:
            self.error = str(message)
            self._log(f"camera recording disabled for {self.path.name}: {self.error}")

    def _open_for_frame(self, frame: Any) -> bool:
        if self.error or self._closed:
            return False
        if self._writer is not None:
            return True
        try:
            height, width = frame.shape[:2]
            self.resolution = [int(width), int(height)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            writer = self._writer_factory(
                str(self.path), fourcc, self.requested_fps, (int(width), int(height))
            )
            if writer is None or not bool(writer.isOpened()):
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
                self.mark_error("video_writer_open_failed")
                return False
            self._writer = writer
            return True
        except Exception as exc:
            self.mark_error(f"video_writer_open_failed:{type(exc).__name__}:{exc}")
            return False

    def write(self, raw_frame: Any) -> bool:
        if not self._open_for_frame(raw_frame):
            return False
        try:
            outcome = self._writer.write(raw_frame)
            if outcome is False:
                raise RuntimeError("VideoWriter.write returned false")
            self.frame_count += 1
            return True
        except Exception as exc:
            self.mark_error(f"video_write_failed:{type(exc).__name__}:{exc}")
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
            return False

    def summary(self) -> Dict[str, Any]:
        status = "recording"
        if self._closed:
            if self.error:
                status = "failed"
            elif self.frame_count <= 0:
                status = "no_frames"
            else:
                status = "completed"
        return {
            "kind": self.kind,
            "area_index": self.area_index,
            "target_id": self.target_id or None,
            "video_path": str(self.path),
            "started_at": self.started_at,
            "ended_at": self.ended_at or None,
            "frame_count": int(self.frame_count),
            "requested_fps": self.requested_fps,
            "resolution": list(self.resolution),
            "camera_source": self.camera_source,
            "status": status,
            "error": self.error,
        }

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return self.summary()
        self._closed = True
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:
                self.mark_error(f"video_writer_release_failed:{type(exc).__name__}:{exc}")
        self.ended_at = _now_iso()
        summary = self.summary()
        if self.manifest is not None and not self._manifest_added:
            self._manifest_added = True
            self.manifest.add_segment(summary)
        return summary


class DriveRecorder:
    """Own Camera, raw writer, and large preview during drive segments."""

    def __init__(
        self,
        *,
        video_dir: Path,
        manifest: CameraVideoManifest,
        capture_opener: Callable[[], Tuple[Any, str, List[str]]],
        cv2_module: Any,
        preview_manager: HighGuiPreviewManager,
        requested_fps: float,
        camera_source: Any,
        camera_width: int = 0,
        camera_height: int = 0,
        read_fail_tolerance: int = 30,
        preview: bool = True,
        preview_topmost: bool = True,
        preview_size: Tuple[int, int] = (1440, 810),
        preview_position: Tuple[int, int] = (180, 80),
        window_name: str = "drive_camera_recording",
        writer_factory: Optional[Callable[..., Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
        shutdown_callback: Optional[Callable[[], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        display_available: Optional[bool] = None,
        wall_clock: Callable[[], dt.datetime] = lambda: dt.datetime.now().astimezone(),
    ) -> None:
        self.video_dir = Path(video_dir)
        self.manifest = manifest
        self.capture_opener = capture_opener
        self.cv2 = cv2_module
        self.preview_manager = preview_manager
        self.requested_fps = max(0.1, float(requested_fps or 0.0))
        self.camera_source = camera_source
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.read_fail_tolerance = max(1, int(read_fail_tolerance or 1))
        self.preview = bool(preview)
        self.preview_topmost = bool(preview_topmost)
        self.preview_size = tuple(int(value) for value in preview_size)
        self.preview_position = tuple(int(value) for value in preview_position)
        self.window_name = str(window_name)
        self.writer_factory = writer_factory
        self._logger = logger or (lambda _message: None)
        self.shutdown_callback = shutdown_callback
        self.monotonic = monotonic
        self.sleep = sleep
        self.wall_clock = wall_clock
        if display_available is not None:
            self.preview_manager.display_available = bool(display_available)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._released_event = threading.Event()
        self._opened_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._segment_index = 0
        self._reserved_video_paths: Set[str] = set()
        self._last_summary: Optional[Dict[str, Any]] = None
        self._camera_release_confirmed = True

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        except Exception:
            pass

    @property
    def opened_event(self) -> threading.Event:
        return self._opened_event

    @property
    def released_event(self) -> threading.Event:
        return self._released_event

    @property
    def last_summary(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self._last_summary)

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._segment_index += 1
            segment_index = self._segment_index
            self._stop_event = threading.Event()
            self._released_event = threading.Event()
            self._opened_event = threading.Event()
            self._camera_release_confirmed = False
            self._thread = threading.Thread(
                target=self._run,
                args=(segment_index,),
                name=f"drive-camera-recorder-{segment_index:03d}",
                daemon=False,
            )
            self._thread.start()
            return True

    def stop(self, timeout_sec: float = 8.0) -> bool:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.1, float(timeout_sec)))
        released = self._released_event.is_set() and self._camera_release_confirmed
        if thread.is_alive() or not released:
            self._log("drive recorder did not confirm Camera release before timeout")
            return False
        return True

    def _configure_capture(self, capture: Any) -> None:
        if self.camera_width > 0:
            capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        if self.requested_fps > 0:
            capture.set(self.cv2.CAP_PROP_FPS, self.requested_fps)

    def _init_preview(self) -> bool:
        if not self.preview or not self.preview_manager.display_available:
            return False
        return self.preview_manager.activate(
            "drive",
            window_name=self.window_name,
            size=self.preview_size,
            position=self.preview_position,
            topmost=self.preview_topmost,
        )

    def _show_preview(
        self, raw_frame: Any, segment_name: str, started_mono: float
    ) -> Optional[bool]:
        try:
            preview = raw_frame.copy()
            elapsed = max(0.0, self.monotonic() - started_mono)
            lines = ("REC", f"{elapsed:08.1f}s", segment_name)
            colors = ((0, 0, 255), (255, 255, 255), (255, 255, 255))
            for index, (text, color) in enumerate(zip(lines, colors)):
                y = 54 + index * 42
                self.cv2.putText(
                    preview, text, (20, y), self.cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 if index == 0 else 0.8, (0, 0, 0), 6, self.cv2.LINE_AA,
                )
                self.cv2.putText(
                    preview, text, (20, y), self.cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 if index == 0 else 0.8, color, 2, self.cv2.LINE_AA,
                )
            return self.preview_manager.show("drive", preview)
        except Exception as exc:
            self._log(f"drive preview disabled: {type(exc).__name__}: {exc}")
            return None

    def _close_preview(self) -> None:
        self.preview_manager.deactivate("drive")

    def _run(self, segment_index: int) -> None:
        segment_started_at = _local_datetime(self.wall_clock())
        with self._lock:
            segment_path = timestamped_video_path(
                self.video_dir,
                "drive",
                started_at=segment_started_at,
                reserved_paths=self._reserved_video_paths,
            )
        segment_name = segment_path.name
        segment = FailSoftVideoSegment(
            path=segment_path,
            kind="drive",
            requested_fps=self.requested_fps,
            camera_source=self.camera_source,
            cv2_module=self.cv2,
            manifest=self.manifest,
            writer_factory=self.writer_factory,
            logger=self._logger,
            started_at=segment_started_at.isoformat(timespec="seconds"),
        )
        capture: Any = None
        preview_active = False
        read_failures = 0
        started_mono = self.monotonic()
        try:
            capture, opened_source, tried = self.capture_opener()
            segment.camera_source = str(opened_source)
            self._configure_capture(capture)
            self._opened_event.set()
            preview_active = self._init_preview()
            self._log(
                f"drive recording started: {segment_name} source={opened_source} tried={tried}"
            )
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures > self.read_fail_tolerance:
                        self._log(
                            "[SHUTDOWN_ORIGIN] camera_read_failure "
                            f"count={read_failures} "
                            f"tolerance={self.read_fail_tolerance}"
                        )
                        segment.mark_error("camera_read_failure_tolerance_exceeded")
                        break
                    self.sleep(0.03)
                    continue
                read_failures = 0
                # The writer receives the untouched frame. Overlay is preview-only.
                segment.write(frame)
                if preview_active:
                    preview_result = self._show_preview(
                        frame, segment_name, started_mono
                    )
                    if preview_result is None:
                        preview_active = False
                        self._close_preview()
                    elif not preview_result:
                        self._stop_event.set()
                        if self.shutdown_callback is not None:
                            try:
                                self.shutdown_callback()
                            except Exception as exc:
                                self._log(f"preview shutdown callback failed: {exc}")
                        break
        except Exception as exc:
            segment.mark_error(f"camera_open_or_loop_failed:{type(exc).__name__}:{exc}")
        finally:
            # Ordering is deliberate: finalize mp4, release Camera, then close UI.
            self._last_summary = segment.close()
            camera_release_confirmed = capture is None
            if capture is not None:
                try:
                    capture.release()
                    camera_release_confirmed = True
                except Exception as exc:
                    self._log(f"drive Camera release failed: {type(exc).__name__}: {exc}")
            if preview_active:
                self._close_preview()
            self._camera_release_confirmed = camera_release_confirmed
            if camera_release_confirmed:
                self._released_event.set()
            self._log(f"drive recording stopped: {segment_name}")


class CameraOwnershipCoordinator:
    """Serialize drive recording and person search Camera ownership."""

    def __init__(
        self,
        recorder: Optional[DriveRecorder],
        *,
        shutdown_event: threading.Event,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.recorder = recorder
        self.shutdown_event = shutdown_event
        self._logger = logger or (lambda _message: None)
        self._paused_for_search = False

    def start_drive(self) -> bool:
        if self.recorder is None or self.shutdown_event.is_set():
            return False
        return self.recorder.start()

    def pause_for_search(self, timeout_sec: float = 8.0) -> bool:
        if self.recorder is None:
            self._paused_for_search = True
            return True
        released = self.recorder.stop(timeout_sec=timeout_sec)
        self._paused_for_search = released
        if not released:
            try:
                self._logger("Camera ownership transfer blocked: drive release unconfirmed")
            except Exception:
                pass
        return released

    def resume_after_search(self) -> bool:
        was_paused = self._paused_for_search
        self._paused_for_search = False
        if (
            not was_paused
            or self.recorder is None
            or self.shutdown_event.is_set()
        ):
            return False
        return self.recorder.start()

    def run_search(self, callback: Callable[[], Any]) -> Any:
        if not self.pause_for_search():
            raise RuntimeError("drive recorder Camera release was not confirmed")
        try:
            return callback()
        finally:
            self.resume_after_search()

    def shutdown(self, timeout_sec: float = 8.0) -> bool:
        self.shutdown_event.set()
        if self.recorder is None:
            return True
        return self.recorder.stop(timeout_sec=timeout_sec)
