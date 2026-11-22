"""OpenCV preview manager for V8. NO recording implementation."""
from __future__ import annotations
import os
import queue
import threading
import time
from typing import Any, Callable, Optional, Tuple
CV_F10_KEYS = {65479, 0xFFC7, 16777273, 0x01000039}

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
