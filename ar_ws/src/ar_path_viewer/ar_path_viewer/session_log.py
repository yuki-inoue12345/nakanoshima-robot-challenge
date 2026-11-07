"""Bounded in-memory diagnostics with collision-safe session file output."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Callable, Iterable


@dataclass(frozen=True)
class SessionPaths:
    video: Path | None
    log: Path


def _unique_path(path: Path, reserved: Iterable[Path] = ()) -> Path:
    reserved_paths = {candidate.absolute() for candidate in reserved}
    candidate = path
    counter = 1
    while candidate.exists() or candidate.absolute() in reserved_paths:
        candidate = path.with_name(f"{path.stem}_{counter:03d}{path.suffix}")
        counter += 1
    return candidate


def _unique_video_log_pair(video: Path) -> tuple[Path, Path]:
    candidate_video = video
    candidate_log = video.with_suffix(".log")
    counter = 1
    while candidate_video.exists() or candidate_log.exists():
        candidate_video = video.with_name(
            f"{video.stem}_{counter:03d}{video.suffix}"
        )
        candidate_log = candidate_video.with_suffix(".log")
        counter += 1
    return candidate_video, candidate_log


def resolve_session_paths(
    output_video: str,
    log_output: str,
    log_output_dir: str,
    now: datetime | None = None,
) -> SessionPaths:
    """Resolve paired output names without overwriting an existing recording or log."""
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    video = None
    paired_log = None
    requested_video = str(output_video).strip()
    if requested_video:
        video, paired_log = _unique_video_log_pair(Path(requested_video).expanduser())

    requested_log = str(log_output).strip()
    if requested_log:
        log = _unique_path(Path(requested_log).expanduser(), (() if video is None else (video,)))
    elif video is not None:
        log = paired_log
    else:
        directory = Path(log_output_dir).expanduser()
        log = _unique_path(directory / f"kokochan_navi_{timestamp}.log")
    return SessionPaths(video=video, log=log)


class SessionLog:
    """Keep recent lines for the GUI while streaming every line to a file."""

    _PRIORITY_MARKERS = (
        "session_start",
        "camera_configuration",
        "output_files",
        "viewer_state",
        "plan_rx",
        "plan_adopted",
        "plan_updated",
        "plan_discarded",
        "path_projection",
        "path_out_of_view",
        "localization_state",
        "shutdown",
        "session_end",
    )

    def __init__(
        self,
        path: Path,
        max_entries: int = 4000,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if max_entries < 100:
            raise ValueError("log_max_entries must be at least 100")
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self._now = now or (lambda: datetime.now().astimezone())
        self._entries: deque[tuple[str, str]] = deque(maxlen=self.max_entries)
        self._lock = threading.Lock()
        self._file = None
        self.file_error = ""
        self.closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("x", encoding="utf-8", buffering=1)
        except (OSError, ValueError) as exc:
            self.file_error = f"{type(exc).__name__}: {exc}"
        self.record(
            "INFO",
            f"session_start started_at={self._timestamp()} log_file={str(self.path)!r}",
        )

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="milliseconds")

    def record(self, level: str, message: str) -> None:
        clean_level = str(level).upper().strip() or "INFO"
        clean_message = str(message).replace("\r\n", "\n").replace("\r", "\n")
        lines = clean_message.splitlines() or [""]
        timestamp = self._timestamp()
        formatted = [f"{timestamp} [{clean_level}] {line}" for line in lines]
        with self._lock:
            for line in formatted:
                self._entries.append((clean_level, line))
                if self._file is not None:
                    try:
                        self._file.write(line + "\n")
                    except OSError as exc:
                        self.file_error = f"{type(exc).__name__}: {exc}"
                        try:
                            self._file.close()
                        except OSError:
                            pass
                        self._file = None

    def text(self) -> str:
        with self._lock:
            return "\n".join(line for _, line in self._entries) + "\n"

    def copy_text(self, max_chars: int = 100_000) -> str:
        """Prefer startup context, latest diagnostics, and errors when truncating."""
        if max_chars < 2000:
            raise ValueError("log_copy_max_chars must be at least 2000")
        with self._lock:
            entries = list(self._entries)
        all_text = "\n".join(line for _, line in entries) + "\n"
        if len(all_text) <= max_chars:
            return all_text

        selected: set[int] = set()
        if entries:
            selected.add(0)
        for marker in self._PRIORITY_MARKERS:
            for index in range(len(entries) - 1, -1, -1):
                if marker in entries[index][1]:
                    selected.add(index)
                    break
        used = sum(len(entries[index][1]) + 1 for index in selected)
        for index in range(len(entries) - 1, -1, -1):
            level, line = entries[index]
            if level not in {"WARNING", "WARN", "ERROR", "FATAL"}:
                continue
            line_size = len(line) + 1
            if index not in selected and used + line_size <= max_chars - 120:
                selected.add(index)
                used += line_size
        for index in range(len(entries) - 1, -1, -1):
            if index in selected:
                continue
            line_size = len(entries[index][1]) + 1
            if used + line_size > max_chars - 120:
                continue
            selected.add(index)
            used += line_size
        lines = [entries[index][1] for index in sorted(selected)]
        header = f"# log truncated for clipboard: retained {len(lines)}/{len(entries)} lines\n"
        text = header + "\n".join(lines) + "\n"
        return text[:max_chars]

    def close(self) -> None:
        if self.closed:
            return
        self.record("INFO", f"session_end log_file={str(self.path)!r}")
        with self._lock:
            self.closed = True
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except OSError as exc:
                    self.file_error = f"{type(exc).__name__}: {exc}"
                self._file = None
