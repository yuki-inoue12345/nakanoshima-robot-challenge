"""
Supervise the display-only Planner from Launcher2 and Follower state.

The supervisor never sends navigation goals or velocity commands.  It starts
the already-isolated reference Planner only after both existing selections are
verified, invalidates the Viewer contract before every stop/change, and stops
the child when the live Follower heartbeat disappears.
"""

import math
import os
from pathlib import Path as FilePath
import shutil
import signal
import subprocess
from time import monotonic

from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robotbase_waypoint_interfaces.msg import (
    ReferencePathState,
    WaypointFollowerState,
)

from .follower_state import FollowerSnapshot
from .runtime_selection import (
    default_launcher_map_state_path,
    LauncherMapStateReader,
    resolve_follower_waypoint,
    runtime_transition,
    RuntimeSelection,
    SelectionStatus,
)


def reference_state_qos() -> QoSProfile:
    """Return the Follower's reliable transient-local snapshot contract."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def reference_output_qos() -> QoSProfile:
    """Match the existing request node's volatile output contract."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def make_reference_process_command(
    selection: RuntimeSelection,
    *,
    namespace: str,
    robot_base_frame: str,
    use_sim_time: bool,
    params_file: str = '',
    ros2_executable: str = 'ros2',
) -> list[str]:
    """Build one argument-safe planning-only launch command."""
    command = [
        ros2_executable,
        'launch',
        'ar_reference_planner',
        'reference_planner.launch.py',
        f'namespace:={namespace}',
        f'use_sim_time:={str(bool(use_sim_time)).lower()}',
        f'map:={selection.map_selection.map_yaml}',
        f'waypoint_file:={selection.waypoint_selection.waypoint_yaml}',
        f'robot_base_frame:={robot_base_frame}',
    ]
    if params_file:
        command.append(f'params_file:={params_file}')
    return command


class ReferencePlannerProcess:
    """Own exactly one descendant launch process and its process group."""

    def __init__(self, popen=subprocess.Popen) -> None:
        self._popen = popen
        self.process = None
        self.command = None

    @property
    def running(self) -> bool:
        """Return whether the owned launch process is still alive."""
        return self.process is not None and self.process.poll() is None

    def start(self, command: list[str]) -> None:
        """Start a new isolated reference launch without a shell."""
        if self.running:
            raise RuntimeError('reference Planner process is already running')
        self.command = list(command)
        self.process = self._popen(command, start_new_session=True)

    def poll(self):
        """Return the child result, or None while it remains active."""
        return None if self.process is None else self.process.poll()

    def stop(self, timeout_sec: float = 8.0) -> None:
        """Terminate only the process group created by :meth:`start`."""
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout_sec)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=2.0)


class ReferencePlannerAutoSupervisor(Node):
    """Bind Launcher2/Follower selections to the isolated Planner lifecycle."""

    def __init__(self) -> None:
        super().__init__('reference_planner_auto_supervisor')
        default_prefix = os.environ.get('ROBOTBASE_TF_PREFIX', 'robot')
        defaults = {
            'launcher_map_state_file': '',
            'map_state_max_initial_age_sec': 86400.0,
            'follower_node_name': '/koko_waypoint_follower',
            'follower_state_topic': '/waypoint_follower/reference_state',
            'follower_state_timeout_sec': 2.5,
            # Process-only grace, never extends path/heartbeat validity.
            'follower_process_grace_sec': 5.0,
            'planner_activation_timeout_sec': 20.0,
            'reference_path_topic': '/ar/reference_path',
            'reference_path_state_topic': '/ar/reference_path_state',
            'reference_namespace': 'ar_reference',
            'robot_base_frame': f'{default_prefix}/base_footprint',
            'reference_params_file': '',
            'use_sim_time': False,
            'poll_period_sec': 0.25,
            'parameter_retry_sec': 1.0,
            'process_retry_sec': 3.0,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

        self._follower_timeout = self._positive_parameter(
            'follower_state_timeout_sec'
        )
        self._follower_process_grace = self._positive_parameter(
            'follower_process_grace_sec'
        )
        self._planner_activation_timeout = self._positive_parameter(
            'planner_activation_timeout_sec'
        )
        poll_period = self._positive_parameter('poll_period_sec')
        self._parameter_retry = self._positive_parameter('parameter_retry_sec')
        self._process_retry = self._positive_parameter('process_retry_sec')
        state_path_value = str(
            self.get_parameter('launcher_map_state_file').value
        ).strip()
        state_path = (
            FilePath(state_path_value).expanduser()
            if state_path_value else default_launcher_map_state_path()
        )
        maximum_state_age = float(
            self.get_parameter('map_state_max_initial_age_sec').value
        )
        if not math.isfinite(maximum_state_age) or maximum_state_age < 0.0:
            raise ValueError(
                'map_state_max_initial_age_sec must be finite and non-negative'
            )
        self._map_reader = LauncherMapStateReader(
            state_path,
            maximum_initial_age_sec=maximum_state_age,
        )
        self._follower_node_name = str(
            self.get_parameter('follower_node_name').value
        )
        self._parameter_client = AsyncParameterClient(
            self, self._follower_node_name
        )
        # Only a read-only GetParameters service is required here.
        self._parameter_get_service = self.create_client(
            GetParameters,
            self._follower_node_name.rstrip('/') + '/get_parameters',
        )
        # Observe only the isolated reference Planner's lifecycle.
        namespace = str(self.get_parameter('reference_namespace').value)
        planner_name = '/' + namespace.strip('/') + '/planner_server'
        self._planner_state_client = self.create_client(
            GetState, planner_name + '/get_state'
        )
        self._follower_subscription = self.create_subscription(
            WaypointFollowerState,
            str(self.get_parameter('follower_state_topic').value),
            self._follower_callback,
            reference_state_qos(),
        )
        output_qos = reference_output_qos()
        self._path_publisher = self.create_publisher(
            Path,
            str(self.get_parameter('reference_path_topic').value),
            output_qos,
        )
        self._state_publisher = self.create_publisher(
            ReferencePathState,
            str(self.get_parameter('reference_path_state_topic').value),
            output_qos,
        )
        self._process = ReferencePlannerProcess()
        self._follower = None
        self._follower_received_at = None
        self._waypoint_status = SelectionStatus(
            None, 'waiting_for_follower_waypoint_parameter'
        )
        self._parameter_future = None
        self._parameter_identity = None
        self._next_parameter_query_at = monotonic()
        self._next_process_retry_at = monotonic()
        self._planner_started_at = None
        self._planner_active_seen = False
        self._planner_state_future = None
        self._planner_query_started_at = None
        self._planner_last_query_at = float('-inf')
        self._planner_epoch = 0
        self._active_selection = None
        self._blocked_map_identity = None
        self._blocked_route_sha256 = None
        self._last_invalid_reason = None
        self._timer = self.create_timer(poll_period, self._tick)
        self._invalidate('auto_selection_startup')
        self.get_logger().info(
            'reference_auto_supervisor_started '
            f'map_state_file={state_path} '
            f'follower_node={self._follower_node_name!r} '
            'manual_map_or_waypoint_arguments=false'
        )

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def _follower_callback(self, message: WaypointFollowerState) -> None:
        try:
            snapshot = FollowerSnapshot.from_message(message)
        except ValueError as error:
            self._clear_follower(f'invalid_follower_state {error}')
            return
        changed = self._follower is None or (
            snapshot.route_sha256 != self._follower.route_sha256
            or snapshot.waypoint_count != self._follower.waypoint_count
        )
        self._follower = snapshot
        self._follower_received_at = monotonic()
        if changed:
            self._waypoint_status = SelectionStatus(
                None, 'waiting_for_follower_waypoint_parameter'
            )
            self._parameter_identity = None
            self._next_parameter_query_at = monotonic()

    def _clear_follower(self, reason: str) -> None:
        self._follower = None
        self._follower_received_at = None
        self._waypoint_status = SelectionStatus(None, reason)
        self._parameter_identity = None
        # A confirmed heartbeat loss separates two Follower lifetimes.  A
        # later Follower may intentionally reuse the same route on a new map.
        self._blocked_map_identity = None
        self._blocked_route_sha256 = None
        self._reconcile(None, reason)

    def _request_waypoint_parameter(self, now: float) -> None:
        if self._follower is None:
            return
        if self._parameter_future is not None:
            started = getattr(self, '_parameter_future_started_at', None)
            if started is None or now - started < 2.5:
                return
            # Discard an RPC stranded by a Follower restart; never reuse it.
            self._parameter_future = None
            self._waypoint_status = SelectionStatus(
                None, 'follower_waypoint_parameter_timeout'
            )
        if now < self._next_parameter_query_at:
            return
        get_service = getattr(self, '_parameter_get_service', None)
        ready = (
            get_service.service_is_ready()
            if get_service is not None
            else self._parameter_client.services_are_ready()
        )
        if not ready:
            self._next_parameter_query_at = now + self._parameter_retry
            self._waypoint_status = SelectionStatus(
                None, 'follower_parameter_get_service_unavailable'
            )
            return
        identity = (
            self._follower.route_sha256,
            self._follower.waypoint_count,
        )
        if get_service is None:
            future = self._parameter_client.get_parameters(['waypoint_file'])
        else:
            request = GetParameters.Request()
            request.names = ['waypoint_file']
            future = get_service.call_async(request)
        self._parameter_future = future
        self._parameter_future_started_at = now
        self._parameter_identity = identity
        future.add_done_callback(
            lambda completed, expected=identity: self._parameter_result(
                completed, expected
            )
        )

    def _parameter_result(self, future, expected_identity: tuple) -> None:
        if future is not self._parameter_future:
            return  # Timed-out, superseded response: do not revive stale route.
        self._parameter_future = None
        self._next_parameter_query_at = monotonic() + self._parameter_retry
        follower = self._follower
        current_identity = None if follower is None else (
            follower.route_sha256,
            follower.waypoint_count,
        )
        if current_identity != expected_identity:
            return
        try:
            response = future.result()
            values = list(response.values)
        except Exception as error:  # noqa: BLE001 - ROS service transport
            self._waypoint_status = SelectionStatus(
                None, f'follower_parameter_query_failed {error}'
            )
            return
        if len(values) != 1 or values[0].type != ParameterType.PARAMETER_STRING:
            self._waypoint_status = SelectionStatus(
                None, 'follower_waypoint_parameter_not_string'
            )
            return
        self._waypoint_status = resolve_follower_waypoint(
            values[0].string_value, follower
        )

    def _desired_selection(self, now: float):
        map_status = self._map_reader.read()
        if not map_status.valid:
            return None, map_status.reason
        if self._follower is None or self._follower_received_at is None:
            return None, 'waiting_for_follower_state'
        follower_age = now - self._follower_received_at
        if follower_age > self._follower_timeout:
            # Stale heartbeat is ALWAYS invalid for the Viewer's Path.
            # Preserve only a verified, unchanged child process during a
            # brief dropout to avoid expensive lifecycle reactivation.
            active = self._active_selection
            same_map = (
                active is not None
                and map_status.value.identity
                == active.map_selection.identity
            )
            same_route = (
                active is not None
                and self._follower.route_sha256
                == active.waypoint_selection.route.sha256
                and self._follower.waypoint_count
                == len(active.waypoint_selection.route.waypoints)
            )
            if (
                follower_age <= (
                    self._follower_timeout + self._follower_process_grace
                )
                and self._process.running and same_map and same_route
            ):
                return None, (
                    f'follower_state_stale_grace age_sec={follower_age:.3f}'
                )
            self._clear_follower(
                f'follower_state_stale age_sec={follower_age:.3f}'
            )
            return None, self._waypoint_status.reason
        if not self._waypoint_status.valid:
            self._request_waypoint_parameter(now)
            return None, self._waypoint_status.reason
        candidate = RuntimeSelection(
            map_status.value, self._waypoint_status.value
        )
        return self._guard_map_route_transition(candidate)

    def _guard_map_route_transition(
        self, candidate: RuntimeSelection
    ) -> tuple[RuntimeSelection | None, str]:
        """Do not pair a newly selected map with the old live route."""
        map_identity = candidate.map_selection.identity
        route_sha256 = candidate.waypoint_selection.route.sha256
        if self._blocked_map_identity is not None:
            if map_identity != self._blocked_map_identity:
                self._blocked_map_identity = None
                self._blocked_route_sha256 = None
            elif route_sha256 == self._blocked_route_sha256:
                return None, 'waiting_for_follower_route_after_map_change'
            else:
                self._blocked_map_identity = None
                self._blocked_route_sha256 = None

        active = self._active_selection
        if (
            active is not None
            and active.map_selection.identity != map_identity
            and active.waypoint_selection.route.sha256 == route_sha256
        ):
            self._blocked_map_identity = map_identity
            self._blocked_route_sha256 = route_sha256
            return None, 'waiting_for_follower_route_after_map_change'
        return candidate, 'selection_verified'

    def _check_planner_activation(self, now: float) -> None:
        """
        Retry an isolated Planner stuck during initial lifecycle activation.

        This only queries this reference Planner's GetState; it cannot issue
        commands to the real Nav2 planner or the robot.
        """
        started = self._planner_started_at
        if (started is None or not self._process.running
                or self._active_selection is None
                or self._planner_active_seen):
            return
        if now - started >= self._planner_activation_timeout:
            self._invalidate('reference_planner_activation_timeout')
            self._process.stop()
            self._active_selection = None
            self._planner_started_at = None
            self._planner_state_future = None
            self._planner_epoch += 1
            self._next_process_retry_at = now + self._process_retry
            return
        if self._planner_state_future is not None:
            queried = self._planner_query_started_at
            if queried is not None and now - queried <= 3.0:
                return
            # Ignore an RPC that got stranded during a lifecycle transition.
            self._planner_state_future = None
        if now - self._planner_last_query_at < 1.0:
            return
        self._planner_last_query_at = now
        if not self._planner_state_client.service_is_ready():
            return
        future = self._planner_state_client.call_async(GetState.Request())
        self._planner_state_future = future
        self._planner_query_started_at = now
        epoch = self._planner_epoch
        future.add_done_callback(
            lambda completed, tag=epoch: self._on_planner_state(
                completed, tag
            )
        )

    def _on_planner_state(self, future, epoch: int) -> None:
        if (future is not self._planner_state_future
                or epoch != self._planner_epoch):
            return
        self._planner_state_future = None
        try:
            state = future.result().current_state
            active = int(state.id) == State.PRIMARY_STATE_ACTIVE
        except Exception:
            return  # Retry; the watchdog has an independent bounded deadline.
        if active:
            self._planner_active_seen = True
            self.get_logger().info('reference_planner_lifecycle_active')

    def _tick(self) -> None:
        now = monotonic()
        self._check_planner_activation(now)
        if self._process.process is not None and not self._process.running:
            result = self._process.poll()
            self._process.process = None
            self._active_selection = None
            self._planner_started_at = None
            self._planner_state_future = None
            self._planner_epoch = getattr(self, '_planner_epoch', 0) + 1
            self._invalidate(f'reference_planner_exited returncode={result}')
            self._next_process_retry_at = now + self._process_retry
        desired, reason = self._desired_selection(now)
        self._reconcile(desired, reason, now)

    def _reconcile(
        self,
        desired: RuntimeSelection | None,
        reason: str,
        now: float | None = None,
    ) -> None:
        timestamp = monotonic() if now is None else now
        if (
            desired is None
            and reason.startswith('follower_state_stale_grace ')
            and self._active_selection is not None
            and self._process.running
        ):
            # Fail closed immediately; do not extend the 2.5s Path gate.
            # The child also enforces its own follower staleness checks.
            self._invalidate(reason)
            return
        transition = runtime_transition(
            self._active_selection, desired, self._process.running
        )
        if transition in ('wait', 'keep'):
            if desired is None:
                self._invalidate(reason)
            return
        self._invalidate(
            reason if desired is None else 'selection_change_in_progress'
        )
        self._process.stop()
        self._active_selection = None
        self._planner_started_at = None
        self._planner_state_future = None
        self._planner_epoch = getattr(self, '_planner_epoch', 0) + 1
        # A child that was already publishing can race with the first
        # invalidation while it is shutting down.  Publish the invalid state
        # again after the process is gone so a late valid sample cannot remain
        # latched for the viewer.
        self._invalidate(
            reason if desired is None else 'selection_change_in_progress',
            force=True,
        )
        if desired is None or timestamp < self._next_process_retry_at:
            return
        command = make_reference_process_command(
            desired,
            namespace=str(
                self.get_parameter('reference_namespace').value
            ),
            robot_base_frame=str(
                self.get_parameter('robot_base_frame').value
            ),
            use_sim_time=bool(self.get_parameter('use_sim_time').value),
            params_file=str(
                self.get_parameter('reference_params_file').value
            ).strip(),
            ros2_executable=shutil.which('ros2') or 'ros2',
        )
        try:
            self._process.start(command)
        except (OSError, RuntimeError) as error:
            self._invalidate(f'reference_planner_start_failed {error}')
            self._next_process_retry_at = timestamp + self._process_retry
            return
        self._active_selection = desired
        self._planner_started_at = timestamp
        self._planner_active_seen = False
        self._planner_state_future = None
        self._planner_last_query_at = float('-inf')
        self._planner_epoch = getattr(self, '_planner_epoch', 0) + 1
        self._next_process_retry_at = timestamp
        self.get_logger().info(
            'reference_planner_auto_started '
            f'map={desired.map_selection.map_yaml} '
            f'waypoint_file={desired.waypoint_selection.waypoint_yaml} '
            f'route_sha256={desired.waypoint_selection.route.sha256}'
        )

    def _invalidate(self, reason: str, *, force: bool = False) -> None:
        reason = str(reason)
        if reason == self._last_invalid_reason and not force:
            return
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_publisher.publish(path)
        state = ReferencePathState()
        state.header = path.header
        state.schema_version = ReferencePathState.SCHEMA_VERSION
        state.valid = False
        state.invalid_reason = reason
        state.path = path
        self._state_publisher.publish(state)
        self._last_invalid_reason = reason
        self.get_logger().warning(
            f'reference_auto_unavailable reason={reason}'
        )

    def destroy_node(self):
        """Invalidate first, then stop only the owned planning subprocess."""
        try:
            if rclpy.ok(context=self.context):
                self._invalidate('auto_supervisor_shutdown')
        finally:
            self._process.stop()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the Launcher2/Follower auto-selection supervisor."""
    rclpy.init(args=args)
    node = ReferencePlannerAutoSupervisor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            # ros2 launch forwards SIGINT to the child while it is already
            # unwinding from rclpy.spin().  The owned planner subprocess has
            # been stopped by our destroy_node() override before rclpy tears
            # down its internal parameter services, so a repeated interrupt
            # here is safe to absorb and should not produce a crash report.
            pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass
