"""Thread-safe, route-scoped measured odometry travel (display-only)."""
import math
import threading
import time


class OdomTravelDistanceMeter:
    """Add genuine robot/odom pose increments; never fabricate missed motion.

    The result is explicitly ODOM TRAVELED, not a Nav2 route length. Missing
    observations and jumps are dropped. Only authenticated Follower activity
    can enable accumulation, so unrelated robot movement is not included.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._route = None
        self._active = False
        self._last = None
        self._total = 0.0
        self._accepted_steps = 0
        self._follower_seen_monotonic = None

    def set_follower(self, route_sha256, waypoint_count, activity):
        route = str(route_sha256)
        count = int(waypoint_count)
        activity = int(activity)
        if (len(route) != 64 or any(c not in '0123456789abcdef' for c in route)
                or count < 1 or not 0 <= activity <= 10):
            return
        with self._lock:
            identity = (route, count)
            if self._route != identity:
                self._route = identity
                self._total = 0.0
                self._accepted_steps = 0
                self._last = None
            was_active = self._active
            self._follower_seen_monotonic = time.monotonic()
            # Both 1 (NAVIGATING) and 2 (UPDATING ROUTE) can include actual
            # robot motion. Count ONLY adjacent measured odom positions for
            # the same verified route. Other activities remain excluded.
            self._active = activity in (1, 2)
            if not self._active or not was_active:
                self._last = None

    def observe(self, x, y, stamp):
        try:
            x, y, t = float(x), float(y), float(stamp)
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in (x, y, t)) or t <= 0.0:
            return
        with self._lock:
            if not self._active or self._route is None:
                return
            if (self._follower_seen_monotonic is None or
                    time.monotonic() - self._follower_seen_monotonic > 2.5):
                self._active = False
                self._last = None
                return
            last = self._last
            self._last = (t, x, y)
            if last is None:
                return
            dt = t - last[0]
            step = math.hypot(x - last[1], y - last[2])
            # Disallow clock resets, missing segments, improbable pose jumps.
            if not 0.0 < dt <= 0.25 or step > 0.035 + 1.5 * dt:
                return
            self._total += step
            self._accepted_steps += 1

    def total_m(self):
        with self._lock:
            return self._total if self._accepted_steps >= 2 else None
