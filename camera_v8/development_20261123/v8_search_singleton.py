"""Fail-closed per-ROS-domain singleton for Camera V8 live search servers.

A normal flock is released by the OS when the server process exits, including
SIGKILL. The lock only covers V8's search server; camera ownership and
navigation remain the responsibility of their existing programs.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat

LOCK_ROOT = Path('/tmp')  # The same stable location for desktop and shell launchers.


class V8ServerAlreadyRunning(RuntimeError):
    pass


def acquire_v8_person_search_lock() -> int:
    domain = os.environ.get('ROS_DOMAIN_ID', '0')
    if not domain.isascii() or not domain.isdecimal() or len(domain) > 4:
        raise RuntimeError(f'Invalid ROS_DOMAIN_ID for V8 singleton: {domain!r}')
    uid = os.getuid()
    path = LOCK_ROOT / f'kokochan-v8-person-search-uid{uid}-domain{int(domain)}.lock'
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid:
            raise RuntimeError(f'Unsafe V8 singleton lock file: {path}')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V8ServerAlreadyRunning(
                f'V8_SEARCH_SERVER_ALREADY_RUNNING domain={domain}; '
                'another live search server is already active. '
                'Close its own UI/server normally before retrying.'
            ) from exc
        # Content is diagnostic only; flock, not PID text, defines ownership.
        os.ftruncate(fd, 0)
        os.write(fd, f'pid={os.getpid()} domain={domain}\n'.encode('ascii'))
        return fd  # Keep fd referenced for the ENTIRE server function lifetime.
    except BaseException:
        os.close(fd)
        raise
