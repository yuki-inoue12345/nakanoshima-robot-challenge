"""Strict final waypoint threshold evidence from the verified Follower."""
import math
import re

_ARRIVAL = re.compile(
    r"\[WP:(\d+)\] threshold reached at ([0-9]+(?:\.[0-9]+)?) m"
)


def verified_final_arrival(node_name, message, stamp_age_sec,
                           verified_number, verified_count,
                           *, expected_node_name='koko_waypoint_follower'):
    """Return Follower arrival distance or None; never assert route complete."""
    if (not str(expected_node_name).strip()
            or str(node_name).lstrip('/') !=
            str(expected_node_name).lstrip('/')):
        return None
    try:
        age = float(stamp_age_sec)
        number = int(verified_number)
        count = int(verified_count)
    except (TypeError, ValueError):
        return None
    if (not math.isfinite(age) or age < -0.25 or age > 1.5
            or number < 0 or count < 1):
        return None
    match = _ARRIVAL.fullmatch(str(message).strip())
    if match is None or int(match.group(1)) != number:
        return None
    distance = float(match.group(2))
    if not math.isfinite(distance) or not 0.0 <= distance <= 2.0:
        return None
    return distance
