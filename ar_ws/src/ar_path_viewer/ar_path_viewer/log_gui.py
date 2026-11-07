"""Optional Tk log viewer and clipboard integration, created only on demand."""

from typing import Callable


class LogGui:
    """Keep Tk failures isolated from the camera, recorder, and ROS callbacks."""

    def __init__(self) -> None:
        self._root = None
        self._window = None
        self._text = None
        self._refresh_source: Callable[[], str] | None = None
        self.closed = False

    def _ensure_root(self):
        if self._root is not None:
            return self._root
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        self._root = root
        return root

    def show(
        self, content: str, refresh_source: Callable[[], str] | None = None
    ) -> tuple[bool, str]:
        try:
            import tkinter as tk
            from tkinter.scrolledtext import ScrolledText

            root = self._ensure_root()
            if self._window is None or not bool(self._window.winfo_exists()):
                window = tk.Toplevel(root)
                window.title("ココちゃんナビ 診断ログ")
                window.geometry("1000x650")
                text = ScrolledText(window, wrap=tk.NONE, font=("monospace", 10))
                text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
                button = tk.Button(window, text="最新ログへ更新", command=self.refresh)
                button.pack(pady=(0, 8))
                window.protocol("WM_DELETE_WINDOW", window.withdraw)
                self._window = window
                self._text = text
            self._refresh_source = refresh_source
            self._replace_text(content)
            self._window.deiconify()
            self._window.lift()
            self.poll()
            return True, "ログウィンドウを開きました"
        except Exception as exc:  # Tk errors vary by backend and display server.
            return False, f"ログ表示失敗: {type(exc).__name__}: {exc}"

    def refresh(self) -> None:
        if self._refresh_source is None:
            return
        try:
            self._replace_text(self._refresh_source())
        except Exception:
            pass

    def _replace_text(self, content: str) -> None:
        if self._text is None:
            return
        import tkinter as tk

        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", content)
        self._text.configure(state=tk.DISABLED)
        self._text.see(tk.END)

    def copy(self, content: str) -> tuple[bool, str]:
        try:
            root = self._ensure_root()
            root.clipboard_clear()
            root.clipboard_append(content)
            root.update_idletasks()
            root.update()
            return True, f"ログをコピーしました ({len(content)}文字)"
        except Exception as exc:  # Clipboard ownership differs across X11/Wayland/Tk.
            return False, f"ログコピー失敗: {type(exc).__name__}: {exc}"

    def poll(self) -> None:
        if self._root is None or self.closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            # Closing the auxiliary window must never terminate AR processing.
            self._root = None
            self._window = None
            self._text = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._window = None
        self._text = None
        self._refresh_source = None
