"""Linux-only opt-in owner-lifetime guard for the C920 camera process.

The camera source is permitted to open C920 only while its invoking launcher
is alive. PR_SET_PDEATHSIG handles even abrupt parent termination (SIGKILL).
The expected-parent check before AND after prctl closes the startup race.
This module never touches ROS navigation or the camera device directly.
"""
from __future__ import annotations

import ctypes
import os
import platform
import signal

_PR_SET_PDEATHSIG = 1


def arm_parent_death_guard(expected_parent_pid: int) -> None:
    expected = int(expected_parent_pid)
    if expected <= 1:
        raise RuntimeError('refusing unsafe C920 parent PID')
    if os.name != 'posix' or platform.system() != 'Linux':
        raise RuntimeError('C920 parent-death guard requires Linux')
    if os.getppid() != expected:
        raise RuntimeError('C920 launching parent is already gone')

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        prctl = libc.prctl
    except AttributeError as exc:
        raise RuntimeError('Linux prctl is unavailable') from exc
    prctl.argtypes = (
        ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_ulong, ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, 'cannot register C920 parent-death SIGTERM')

    # Parent could have died between initial PID check and PR_SET_PDEATHSIG.
    # In that case Linux would not retroactively send SIGTERM: reject startup.
    if os.getppid() != expected:
        raise RuntimeError('C920 launching parent exited during guard setup')
