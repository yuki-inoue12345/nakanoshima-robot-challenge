"""Conservative /plan and Costmap pairing for display-only alerts.

Costmap message arrival establishes a bound to *one immutable real Path*
including its header stamp, viewer generation, and follower-goal identity.
The binding is checked again at render time. A newer /plan or follower goal
cannot silently change which route old Costmap cells are evaluated against.
This module does not publish or change Nav2 goals.
"""

from dataclasses import dataclass
import math
from typing import Optional

from .path_adapter import PathSnapshot


@dataclass(frozen=True)
class CostmapPathBinding:
    costmap_stamp_sec: float
    frame_id: str
    path: Optional[PathSnapshot]
    follower_identity: Optional[tuple]


def follower_goal_identity(gate, now_ros_sec: float) -> tuple | None:
    """Read validated Follower navigation identity without mutating the gate."""
    view = getattr(gate, "_view", None)
    if view is None:
        return None
    age = float(now_ros_sec) - float(view.stamp_sec)
    timeout = float(getattr(gate, "timeout_sec", 0.0))
    if not math.isfinite(age) or age < -0.5 or age > timeout:
        return None
    return (
        view.route_sha256, view.next_index, view.next_number,
        view.navigation_mode, view.goal_generation,
    )


def capture_costmap_path(
    costmap_stamp_sec: float, frame_id: str,
    path: PathSnapshot | None, follower_identity: tuple | None,
) -> CostmapPathBinding:
    return CostmapPathBinding(
        costmap_stamp_sec=float(costmap_stamp_sec),
        frame_id=str(frame_id), path=path,
        follower_identity=follower_identity,
    )


def check_costmap_path_pair(
    binding: CostmapPathBinding | None,
    *, costmap_stamp_sec: float, costmap_frame_id: str,
    current_path: PathSnapshot | None,
    current_follower_identity: tuple | None,
    max_path_age_sec: float = 3.0,
    future_tolerance_sec: float = 0.05,
) -> tuple[PathSnapshot | None, str]:
    """Return immutable bound Path only when its time/goal epoch remains valid.

    Path header stamps describe plan generation, not the exact geometry of
    moving costmap cells. The limited age window is an alert policy, *not* a
    promise that /plan and OccupancyGrid have one atomic Nav2 generation.
    """
    if binding is None:
        return None, "costmap_path_binding_missing"
    if (
        binding.frame_id != costmap_frame_id
        or not math.isclose(binding.costmap_stamp_sec, costmap_stamp_sec,
                            rel_tol=0.0, abs_tol=1e-6)
    ):
        return None, "costmap_path_binding_mismatch"
    bound = binding.path
    if bound is None or current_path is None:
        return None, "real_path_missing"
    if current_path.generation != bound.generation:
        return None, "real_path_generation_changed"
    if bound.header_stamp_sec is None or not math.isfinite(bound.header_stamp_sec):
        return None, "real_path_header_stamp_unknown"
    delta = float(costmap_stamp_sec) - bound.header_stamp_sec
    if (
        not math.isfinite(delta) or delta < -future_tolerance_sec
        or delta > max_path_age_sec
    ):
        return None, f"path_costmap_stamp_mismatch delta={delta:.3f}s"
    if binding.follower_identity is None or current_follower_identity is None:
        return None, "follower_goal_unverified"
    if binding.follower_identity != current_follower_identity:
        return None, "follower_goal_generation_changed"
    return bound, "path_costmap_pair_matched"
