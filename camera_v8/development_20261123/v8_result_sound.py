"""Camera V8 result sounds, using the original V7 WAV files without camera ownership.

Runs an independent non-blocking audio player after successful terminal
ROS publication.  Never publishes navigation messages or opens the camera.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

V7_DIR = Path(__file__).resolve().parents[2] / "camera_v7" / "development_20261019"
PERSON_SEARCH_NOTIFICATION_PLAYER = Path("/usr/bin/cvlc")
PERSON_SEARCH_FOUND_SOUND = V7_DIR / "sounds" / "zundamon_found.wav"
PERSON_SEARCH_NOT_FOUND_SOUND = V7_DIR / "sounds" / "zundamon_not_found.wav"


def play_person_search_notification(status: Any, logger: Any) -> bool:
    """Mirror the V7 found/not-found sound contract; no sound for ERROR/BUSY."""
    normalized_status = str(status or "").strip().upper()
    if normalized_status == "FOUND_FRONT_PHOTO":
        sound_path = PERSON_SEARCH_FOUND_SOUND
    elif normalized_status == "NOT_FOUND_OR_NOT_FRONT":
        sound_path = PERSON_SEARCH_NOT_FOUND_SOUND
    else:
        return False

    def warn(message: str) -> None:
        try:
            logger.warning(message)
        except Exception:
            pass

    if not sound_path.is_file():
        warn(f"person search notification sound is missing: {sound_path}")
        return False
    if not PERSON_SEARCH_NOTIFICATION_PLAYER.is_file() or not os.access(
        PERSON_SEARCH_NOTIFICATION_PLAYER, os.X_OK
    ):
        warn(
            "person search notification player is unavailable: "
            f"{PERSON_SEARCH_NOTIFICATION_PLAYER}"
        )
        return False
    try:
        subprocess.Popen(
            [
                str(PERSON_SEARCH_NOTIFICATION_PLAYER),
                "--play-and-exit",
                "--no-video",
                "--quiet",
                str(sound_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        warn(f"person search notification playback failed: {exc}")
        return False
    return True
