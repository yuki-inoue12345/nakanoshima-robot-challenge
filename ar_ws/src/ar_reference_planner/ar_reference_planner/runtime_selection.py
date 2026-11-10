"""
Resolve Launcher2 map and live Follower route selections fail-closed.

This module contains no ROS process control.  It turns the two existing
sources of truth into one immutable selection that a display-only planner may
use: Launcher2's current-map state file and the Follower's existing
``waypoint_file`` ROS parameter, verified against its typed route hash.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Mapping

import yaml

from .follower_state import FollowerSnapshot
from .keepout_config import (
    KeepoutConfigurationError,
    KeepoutSelection,
    resolve_keepout_selection,
)
from .route import load_route, Route


@dataclass(frozen=True)
class MapSelection:
    """One validated Launcher2 map and its exact Keepout identity."""

    state_path: Path
    keepout: KeepoutSelection

    @property
    def map_yaml(self) -> Path:
        """Return the canonical selected map YAML."""
        return self.keepout.map_signature.yaml_path

    @property
    def identity(self) -> tuple:
        """Return values that require a reference Planner restart."""
        mask = self.keepout.mask_signature
        return (
            str(self.map_yaml),
            self.keepout.map_signature.data_sha256,
            self.keepout.enabled,
            '' if mask is None else mask.data_sha256,
        )


@dataclass(frozen=True)
class WaypointSelection:
    """One live Follower route resolved from its existing parameter."""

    route: Route

    @property
    def waypoint_yaml(self) -> Path:
        """Return the canonical waypoint YAML path."""
        return self.route.path


@dataclass(frozen=True)
class RuntimeSelection:
    """Exact map/route pair used by one reference Planner subprocess."""

    map_selection: MapSelection
    waypoint_selection: WaypointSelection

    @property
    def identity(self) -> tuple:
        """Return content identities, never filenames alone."""
        return (
            *self.map_selection.identity,
            str(self.waypoint_selection.waypoint_yaml),
            self.waypoint_selection.route.sha256,
        )


@dataclass(frozen=True)
class SelectionStatus:
    """A fail-closed result with a stable diagnostic reason."""

    value: object | None
    reason: str

    @property
    def valid(self) -> bool:
        """Return whether a verified selection is available."""
        return self.value is not None


def runtime_transition(
    active: RuntimeSelection | None,
    desired: RuntimeSelection | None,
    process_running: bool,
) -> str:
    """Return the minimal safe lifecycle operation for two selections."""
    if desired is None:
        return 'stop' if active is not None or process_running else 'wait'
    if active is None:
        return 'start'
    if active.identity != desired.identity:
        return 'restart'
    return 'keep' if process_running else 'restart'


def default_launcher_map_state_path(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Match the existing Launcher2 state filename without fixing a robot ID."""
    values = os.environ if environment is None else environment
    home = Path(values.get('HOME', str(Path.home()))).expanduser()
    launcher_id = str(values.get('ROBOTBASE_LAUNCHER_ID', '')).strip()
    if not launcher_id:
        robot_id = str(values.get('ROBOTBASE_ID', '')).strip()
        launcher_id = f'{robot_id}_launcher' if robot_id else 'koko_launcher2'
    return home / f'.{launcher_id}_nav2_current_map.yaml'


class LauncherMapStateReader:
    """Read state changes and reject an old selection on initial startup."""

    def __init__(
        self,
        state_path: str | Path,
        *,
        maximum_initial_age_sec: float = 86400.0,
        wall_clock=time.time,
    ) -> None:
        self.state_path = Path(state_path).expanduser()
        self.maximum_initial_age_sec = float(maximum_initial_age_sec)
        if (
            not math.isfinite(self.maximum_initial_age_sec)
            or self.maximum_initial_age_sec < 0.0
        ):
            raise ValueError(
                'maximum_initial_age_sec must be finite and non-negative'
            )
        self._wall_clock = wall_clock
        self._accepted_stat_key = None
        self._cached_status = None

    def read(self) -> SelectionStatus:
        """Return the current verified map or an explicit unavailable reason."""
        try:
            stat = self.state_path.stat()
        except FileNotFoundError:
            self._accepted_stat_key = None
            self._cached_status = None
            return SelectionStatus(None, 'launcher_map_state_missing')
        except OSError as error:
            self._accepted_stat_key = None
            self._cached_status = None
            return SelectionStatus(
                None, f'launcher_map_state_stat_failed {error}'
            )
        stat_key = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stat_key == self._accepted_stat_key and self._cached_status is not None:
            return self._cached_status
        if self._accepted_stat_key is None and self.maximum_initial_age_sec > 0.0:
            age = self._wall_clock() - stat.st_mtime
            if age > self.maximum_initial_age_sec:
                return SelectionStatus(
                    None,
                    'launcher_map_state_stale '
                    f'age_sec={age:.1f} '
                    f'limit_sec={self.maximum_initial_age_sec:.1f}',
                )
        try:
            document = yaml.safe_load(
                self.state_path.read_text(encoding='utf-8')
            )
        except (OSError, yaml.YAMLError) as error:
            self._accepted_stat_key = None
            self._cached_status = None
            return SelectionStatus(
                None, f'launcher_map_state_invalid {error}'
            )
        if not isinstance(document, dict):
            return SelectionStatus(None, 'launcher_map_state_not_mapping')
        raw_map = str(document.get('current_map', '')).strip()
        if not raw_map:
            return SelectionStatus(None, 'launcher_current_map_missing')
        map_path = Path(raw_map).expanduser()
        if not map_path.is_absolute():
            return SelectionStatus(None, 'launcher_current_map_not_absolute')
        try:
            keepout = resolve_keepout_selection(map_path)
        except (KeepoutConfigurationError, OSError, ValueError) as error:
            return SelectionStatus(
                None, f'launcher_map_validation_failed {error}'
            )
        status = SelectionStatus(
            MapSelection(self.state_path.resolve(), keepout),
            'launcher_map_verified',
        )
        self._accepted_stat_key = stat_key
        self._cached_status = status
        return status


def resolve_follower_waypoint(
    parameter_value: str,
    follower: FollowerSnapshot,
) -> SelectionStatus:
    """Verify the Follower parameter against its independently published hash."""
    raw_path = str(parameter_value).strip()
    if not raw_path:
        return SelectionStatus(None, 'follower_waypoint_parameter_empty')
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return SelectionStatus(None, 'follower_waypoint_path_not_absolute')
    if not path.is_file():
        return SelectionStatus(None, f'follower_waypoint_missing path={path}')
    try:
        route = load_route(str(path))
    except (OSError, ValueError, yaml.YAMLError) as error:
        return SelectionStatus(None, f'follower_waypoint_invalid {error}')
    if route.sha256 != follower.route_sha256:
        return SelectionStatus(None, 'follower_waypoint_route_hash_mismatch')
    if len(route.waypoints) != follower.waypoint_count:
        return SelectionStatus(None, 'follower_waypoint_count_mismatch')
    if not 0 <= follower.next_index < len(route.waypoints):
        return SelectionStatus(None, 'follower_waypoint_index_out_of_range')
    if route.waypoints[follower.next_index].number != follower.next_number:
        return SelectionStatus(None, 'follower_waypoint_number_mismatch')
    return SelectionStatus(
        WaypointSelection(route), 'follower_waypoint_verified'
    )
