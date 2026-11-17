"""Fail-closed Local Costmap sampling and camera projection.

The input is Nav2's ``nav_msgs/OccupancyGrid`` view of the local costmap.
Values are kept semantic instead of being described as generic danger:

* ``-1``: unknown
* ``high_cost_min .. lethal_cost_min - 1``: high navigation cost
* ``lethal_cost_min .. 100``: lethal-cost cells in the Nav2 grid

The module has no ROS node dependency, which keeps geometry and freshness
tests reproducible without connecting to a robot.
"""

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from .projection import CameraIntrinsics, CameraMount, world_to_optical
from .tf_adapter import quaternion_matrix_xyzw, transform_message_points


UNKNOWN_CLASS = 0
HIGH_COST_CLASS = 1
LETHAL_COST_CLASS = 2


@dataclass(frozen=True)
class CostmapOverlaySettings:
    """Validation, classification, and rendering workload limits."""

    timeout_sec: float = 1.5
    stamp_max_age_sec: float = 1.5
    future_tolerance_sec: float = 0.5
    high_cost_min: int = 60
    lethal_cost_min: int = 100
    max_points_per_class: int = 450
    # The on-robot local grid is 120x120. Bound all NumPy allocations before
    # reading message.data; reject unexpected map sizes instead of blocking UI.
    max_grid_width_cells: int = 512
    max_grid_height_cells: int = 512
    max_grid_cells: int = 65536
    plane_height_m: float = 0.02
    tf_wait_timeout_sec: float = 1.5
    max_pending_messages: int = 8

    def validate(self) -> None:
        values = (
            self.timeout_sec,
            self.stamp_max_age_sec,
            self.future_tolerance_sec,
            self.plane_height_m,
            self.tf_wait_timeout_sec,
        )
        if not np.isfinite(values).all():
            raise ValueError("costmap overlay settings must be finite")
        if self.timeout_sec <= 0.0 or self.stamp_max_age_sec <= 0.0:
            raise ValueError("costmap overlay timeouts must be positive")
        if self.future_tolerance_sec < 0.0:
            raise ValueError("costmap future tolerance must be non-negative")
        if not 1 <= self.high_cost_min < self.lethal_cost_min <= 100:
            raise ValueError(
                "costmap thresholds must satisfy 1 <= high < lethal <= 100"
            )
        if self.max_points_per_class < 1:
            raise ValueError("costmap max_points_per_class must be positive")
        for name in (
            "max_grid_width_cells", "max_grid_height_cells", "max_grid_cells"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"costmap {name} must be a positive integer")
        if self.tf_wait_timeout_sec <= 0.0:
            raise ValueError("costmap TF wait timeout must be positive")
        if self.max_pending_messages < 1:
            raise ValueError("costmap pending message limit must be positive")


@dataclass(frozen=True)
class CostmapCameraSyncSettings:
    """Conservative limits for relating a Costmap to one camera frame.

    OpenCV ``VideoCapture.read`` is not trusted as a camera timestamp.  The
    opt-in V4L2 input exposes a kernel monotonic timestamp for each decoded
    frame.  It still does not guarantee precise C920 exposure time.
    """

    max_costmap_age_sec: float = 0.40
    max_capture_duration_sec: float = 0.12
    max_translation_m: float = 0.20
    max_rotation_deg: float = 15.0
    clock_tolerance_sec: float = 0.02
    max_frame_transport_age_sec: float = 0.12
    future_tolerance_sec: float = 0.02
    exact_tf_tolerance_sec: float = 0.005

    def validate(self) -> None:
        values = (
            self.max_costmap_age_sec,
            self.max_capture_duration_sec,
            self.max_translation_m,
            self.max_rotation_deg,
            self.clock_tolerance_sec,
            self.max_frame_transport_age_sec,
            self.future_tolerance_sec,
            self.exact_tf_tolerance_sec,
        )
        if not np.isfinite(values).all():
            raise ValueError("costmap/camera synchronization limits must be finite")
        if self.max_costmap_age_sec <= 0.0:
            raise ValueError("costmap/camera maximum age must be positive")
        if self.max_capture_duration_sec <= 0.0:
            raise ValueError("camera capture duration limit must be positive")
        if self.max_frame_transport_age_sec <= 0.0:
            raise ValueError("camera transport age limit must be positive")
        if self.max_translation_m < 0.0 or self.max_rotation_deg < 0.0:
            raise ValueError("costmap/camera motion limits must be non-negative")
        if (
            self.clock_tolerance_sec < 0.0
            or self.future_tolerance_sec < 0.0
            or self.exact_tf_tolerance_sec < 0.0
        ):
            raise ValueError("costmap/camera time tolerances must be non-negative")


@dataclass(frozen=True)
class ImageFrameTiming:
    """Host-side timestamp estimate attached to one returned image frame."""

    stamp_sec: float
    ros_started_sec: float
    ros_finished_sec: float
    monotonic_started_sec: float
    monotonic_finished_sec: float
    source_kind: str
    usable: bool
    reason: str

    @property
    def capture_duration_sec(self) -> float:
        return self.monotonic_finished_sec - self.monotonic_started_sec


@dataclass(frozen=True)
class CameraTimeCostmap:
    """One Costmap expressed in the robot frame at the image timestamp."""

    points_in_base: np.ndarray
    analysis_points_in_base: np.ndarray
    costmap_age_sec: float
    translation_m: float
    rotation_deg: float


def estimate_image_frame_timing(
    *,
    ros_started_sec: float,
    ros_finished_sec: float,
    monotonic_started_sec: float,
    monotonic_finished_sec: float,
    source_kind: str,
    settings: CostmapCameraSyncSettings,
    previous_stamp_sec: float | None = None,
    previous_timing: ImageFrameTiming | None = None,
    source_timestamp_monotonic_sec: float | None = None,
    timestamp_source: str = "",
) -> ImageFrameTiming:
    """Create a fail-closed host timestamp estimate for an acquired frame."""
    settings.validate()
    values = (
        ros_started_sec,
        ros_finished_sec,
        monotonic_started_sec,
        monotonic_finished_sec,
    )
    normalized_source = str(source_kind).strip().lower()
    reason = "host_receive_time_estimate"
    usable = True
    selected_stamp = float(ros_finished_sec)
    if not np.isfinite(values).all():
        usable = False
        reason = "image_time_non_finite"
    else:
        monotonic_elapsed = monotonic_finished_sec - monotonic_started_sec
        ros_elapsed = ros_finished_sec - ros_started_sec
        if normalized_source not in ("camera", "camera_timestamped", "simulation"):
            usable = False
            reason = f"image_time_untrusted_source source={normalized_source or 'unknown'}"
        elif ros_started_sec <= 0.0 or ros_finished_sec <= 0.0:
            usable = False
            reason = "ros_clock_uninitialized"
        elif monotonic_elapsed < 0.0:
            usable = False
            reason = "monotonic_clock_reversed"
        elif ros_elapsed < 0.0:
            # A backwards clock is not an approximate image timestamp,
            # even when the jump is smaller than the usual clock jitter
            # tolerance.  Fail closed until the next valid acquisition.
            usable = False
            reason = "ros_clock_reversed_during_capture"
        elif (
            previous_stamp_sec is not None
            and ros_finished_sec < float(previous_stamp_sec)
        ):
            usable = False
            reason = "ros_clock_reset_between_frames"
        elif monotonic_elapsed > settings.max_capture_duration_sec:
            usable = False
            reason = (
                f"image_capture_too_slow duration={monotonic_elapsed:.3f}s "
                f"limit={settings.max_capture_duration_sec:.3f}s"
            )
        elif abs(ros_elapsed - monotonic_elapsed) > settings.clock_tolerance_sec:
            usable = False
            reason = (
                f"ros_monotonic_clock_mismatch ros_elapsed={ros_elapsed:.3f}s "
                f"monotonic_elapsed={monotonic_elapsed:.3f}s"
            )
        elif previous_timing is not None and previous_timing.usable:
            # An ROS clock jump between captures is invisible to a within-read
            # check.  Compare clock progress between successive host samples.
            ros_progress = ros_finished_sec - previous_timing.ros_finished_sec
            mono_progress = (
                monotonic_finished_sec - previous_timing.monotonic_finished_sec
            )
            if (
                mono_progress < 0.0
                or abs(ros_progress - mono_progress) > settings.clock_tolerance_sec
            ):
                usable = False
                reason = "ros_monotonic_clock_jump_between_frames"

        if usable and normalized_source == "camera":
            # A fast read can still return a frame recorded long before this
            # call.  Do not mistake call latency for image age.
            usable = False
            reason = "camera_timestamp_unverified_opencv"
        elif usable and normalized_source == "camera_timestamped":
            if timestamp_source == "v4l2_monotonic_eof":
                # EOF is the end of frame transfer, not the exposure instant.
                # Its offset is not characterized on the real C920; reject.
                usable = False
                reason = "camera_eof_exposure_offset_unverified"
            elif timestamp_source != "v4l2_monotonic_soe":
                usable = False
                reason = "camera_timestamp_source_unverified"
            elif source_timestamp_monotonic_sec is None or not math.isfinite(
                source_timestamp_monotonic_sec
            ):
                usable = False
                reason = "camera_timestamp_missing"
            else:
                transport_age = (
                    monotonic_finished_sec - source_timestamp_monotonic_sec
                )
                if transport_age < -0.005:
                    usable = False
                    reason = "camera_timestamp_from_future"
                elif transport_age > settings.max_frame_transport_age_sec:
                    usable = False
                    reason = "camera_buffer_stale"
                else:
                    # The V4L2 timestamp is CLOCK_MONOTONIC, not ROS time.
                    # Convert using the host ROS/monotonic pair sampled after
                    # acquisition.  Do not use a wallclock Unix epoch.
                    selected_stamp = ros_finished_sec - transport_age
                    reason = timestamp_source
                    if selected_stamp <= 0.0:
                        usable = False
                        reason = "camera_converted_timestamp_invalid"
                    elif (
                        previous_timing is not None
                        and previous_timing.usable
                        and selected_stamp <= previous_timing.stamp_sec
                    ):
                        usable = False
                        reason = "camera_timestamp_not_advancing"
    return ImageFrameTiming(
        stamp_sec=float(selected_stamp),
        ros_started_sec=float(ros_started_sec),
        ros_finished_sec=float(ros_finished_sec),
        monotonic_started_sec=float(monotonic_started_sec),
        monotonic_finished_sec=float(monotonic_finished_sec),
        source_kind=normalized_source,
        usable=usable,
        reason=reason,
    )


def _transform_components(transform_message):
    transform = transform_message.transform
    translation = np.asarray(
        (
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ),
        dtype=np.float64,
    )
    quaternion = (
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    if not np.isfinite(translation).all():
        raise ValueError("TF translation contains NaN or Inf")
    return translation, quaternion_matrix_xyzw(quaternion)


def _transform_stamp_sec(transform_message) -> float:
    return _stamp_to_sec(transform_message.header.stamp)


def align_costmap_to_image_time(
    snapshot: "CostmapSnapshot",
    *,
    image_timing: ImageFrameTiming,
    transform_at_costmap,
    transform_at_image,
    settings: CostmapCameraSyncSettings,
) -> CameraTimeCostmap:
    """Express fixed-frame Costmap cells in base at the image timestamp.

    Both transforms are target-base-from-the-same-Costmap-frame transforms.
    Using the image-time transform directly is essential: reusing the
    Costmap-time base coordinates leaves an obstacle in front of the camera
    after the robot has turned.
    """
    settings.validate()
    if not image_timing.usable:
        raise ValueError(image_timing.reason)
    if not math.isfinite(image_timing.stamp_sec) or image_timing.stamp_sec <= 0.0:
        raise ValueError("image_timestamp_invalid")
    if not math.isfinite(snapshot.stamp_sec) or snapshot.stamp_sec <= 0.0:
        raise ValueError("costmap_timestamp_invalid")
    if snapshot.frame_id == "":
        raise ValueError("costmap frame_id is empty")
    age = image_timing.stamp_sec - snapshot.stamp_sec
    if age > settings.max_costmap_age_sec:
        raise ValueError(
            f"costmap_camera_age_exceeded age={age:.3f}s "
            f"limit={settings.max_costmap_age_sec:.3f}s"
        )
    if age < -settings.future_tolerance_sec:
        raise ValueError(f"costmap_newer_than_image offset={-age:.3f}s")

    costmap_tf_stamp = _transform_stamp_sec(transform_at_costmap)
    image_tf_stamp = _transform_stamp_sec(transform_at_image)
    if costmap_tf_stamp <= 0.0 or image_tf_stamp <= 0.0:
        raise ValueError("costmap_camera_tf_zero_stamp")
    if abs(costmap_tf_stamp - snapshot.stamp_sec) > settings.exact_tf_tolerance_sec:
        raise ValueError(
            "costmap_tf_not_exact "
            f"offset={costmap_tf_stamp - snapshot.stamp_sec:.3f}s"
        )
    if abs(image_tf_stamp - image_timing.stamp_sec) > settings.exact_tf_tolerance_sec:
        raise ValueError(
            "image_tf_not_exact "
            f"offset={image_tf_stamp - image_timing.stamp_sec:.3f}s"
        )

    old_translation, old_rotation = _transform_components(transform_at_costmap)
    new_translation, new_rotation = _transform_components(transform_at_image)
    # T(base_image <- base_costmap) = T(base_image <- source) *
    # inverse(T(base_costmap <- source)).
    relative_rotation = new_rotation @ old_rotation.T
    relative_translation = new_translation - relative_rotation @ old_translation
    # Check full 3-D motion, not just xy/yaw: a large roll, pitch or
    # vertical displacement must not slip past the image-time safety gate.
    translation_m = float(np.linalg.norm(relative_translation))
    cos_angle = float((np.trace(relative_rotation) - 1.0) * 0.5)
    rotation_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
    if translation_m > settings.max_translation_m:
        raise ValueError(
            f"costmap_camera_translation_exceeded distance={translation_m:.3f}m "
            f"limit={settings.max_translation_m:.3f}m"
        )
    if rotation_deg > settings.max_rotation_deg:
        raise ValueError(
            f"costmap_camera_rotation_exceeded angle={rotation_deg:.3f}deg "
            f"limit={settings.max_rotation_deg:.3f}deg"
        )

    combined = np.concatenate((snapshot.points, snapshot.analysis_points), axis=0)
    transformed = transform_message_points(combined, transform_at_image)
    split = snapshot.points.shape[0]
    return CameraTimeCostmap(
        points_in_base=transformed[:split],
        analysis_points_in_base=transformed[split:],
        costmap_age_sec=float(age),
        translation_m=translation_m,
        rotation_deg=rotation_deg,
    )


@dataclass(frozen=True)
class CostmapSnapshot:
    """Validated and bounded costmap points in the message frame."""

    frame_id: str
    stamp_sec: float
    received_at: float
    points: np.ndarray
    classes: np.ndarray
    analysis_points: np.ndarray
    analysis_costs: np.ndarray
    unknown_cells: int
    high_cost_cells: int
    lethal_cost_cells: int

    @property
    def sampled_cells(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class ProjectedCostmap:
    """Image pixels grouped by the exact cost class used for rendering."""

    unknown_pixels: np.ndarray
    high_cost_pixels: np.ndarray
    lethal_cost_pixels: np.ndarray
    input_points: int
    visible_points: int


@dataclass(frozen=True)
class CostmapVisualDisplay:
    """Display-only selection of one exact-time Costmap projection.

    ``mode == "hold"`` never authorizes alerts, explanations, or any other
    semantic decision.  It only lets rendering identify a short, explicitly
    labelled visual continuation of a previously confirmed projection.
    """

    projection: ProjectedCostmap | None
    mode: str
    age_sec: float


class CostmapVisualHold:
    """Short monotonic-time hold for already projected image pixels.

    The class intentionally stores image pixels, not Costmap geometry.  It
    cannot reproject with latest TF and therefore cannot weaken exact-time
    synchronization.  Callers must clear it on frame/epoch or localization
    invalidation and must never feed a held value into safety decisions.
    """

    def __init__(
        self,
        max_age_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(max_age_sec) or max_age_sec < 0.0:
            raise ValueError("costmap visual hold must be finite and non-negative")
        self.max_age_sec = float(max_age_sec)
        self.clock = clock
        self._projection: ProjectedCostmap | None = None
        self._context: str | None = None
        self._confirmed_at: float | None = None

    def clear(self) -> None:
        self._projection = None
        self._context = None
        self._confirmed_at = None

    def confirm(
        self, projection: ProjectedCostmap, context: str
    ) -> CostmapVisualDisplay:
        """Replace any held pixels with a newly exact-time-confirmed result."""
        now = float(self.clock())
        if not math.isfinite(now) or not context:
            self.clear()
            return CostmapVisualDisplay(None, "unavailable", 0.0)
        self._projection = projection
        self._context = str(context)
        self._confirmed_at = now
        return CostmapVisualDisplay(projection, "live", 0.0)

    def held(self, context: str) -> CostmapVisualDisplay:
        """Return a labelled visual hold, or fail closed after its deadline."""
        if (
            self._projection is None
            or self._confirmed_at is None
            or not context
            or context != self._context
            or self.max_age_sec <= 0.0
        ):
            if context and self._context is not None and context != self._context:
                self.clear()
            return CostmapVisualDisplay(None, "unavailable", 0.0)
        now = float(self.clock())
        age = now - self._confirmed_at
        if not math.isfinite(age) or age < 0.0 or age > self.max_age_sec:
            self.clear()
            return CostmapVisualDisplay(None, "unavailable", max(0.0, age))
        return CostmapVisualDisplay(self._projection, "hold", float(age))


@dataclass(frozen=True)
class SynchronizedCostmap:
    """One Costmap transformed by TF at its exact header timestamp."""

    snapshot: CostmapSnapshot
    points_in_base: np.ndarray
    analysis_points_in_base: np.ndarray
    synchronized_at: float


class CostmapTransformQueue:
    """Bounded exact-time synchronization without latest-TF fallback.

    Costmap normally reaches the Viewer shortly before the matching odom TF.
    Newer messages supersede older unresolved messages immediately.  Only the
    newest accepted pending sample may become the next ready sample.  A
    previously synchronized snapshot may remain visible while it independently
    passes Costmap freshness limits; untransformed data is never rendered.
    """

    def __init__(
        self,
        settings: CostmapOverlaySettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.clock = clock
        self._pending: list[CostmapSnapshot] = []
        self._ready: SynchronizedCostmap | None = None
        self._frame_epoch: str | None = None
        # A Costmap frame change must never erase the global stamp watermark.
        # Retired frame IDs cannot return through delayed callbacks, even if
        # their header stamp is greater than the most recent accepted stamp.
        self._retired_frames: set[str] = set()
        self._newest_accepted_stamp = -math.inf
        self.resolved_count = 0
        self.expired_count = 0
        self.superseded_count = 0
        self.ignored_out_of_order_count = 0
        self.reason = "waiting_for_local_costmap"

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def clear(self, reason: str, *, reset_epoch: bool = False) -> None:
        self._pending.clear()
        self._ready = None
        # Invalid input/TF expiry keeps ordering history.  reset_epoch=True
        # is reserved for an explicitly verified clock/configuration reset;
        # a new frame_id arriving in a message is NOT such a reset.
        if reset_epoch:
            self._newest_accepted_stamp = -math.inf
            self._frame_epoch = None
            self._retired_frames.clear()
        self.reason = str(reason)

    def submit(self, snapshot: CostmapSnapshot) -> bool:
        if snapshot.stamp_sec <= self._newest_accepted_stamp:
            self.ignored_out_of_order_count += 1
            self.reason = "out_of_order_costmap_ignored"
            return False
        if snapshot.frame_id in self._retired_frames:
            self.ignored_out_of_order_count += 1
            self.reason = "out_of_order_costmap_ignored retired_frame"
            return False
        if (
            self._frame_epoch is not None
            and self._frame_epoch != snapshot.frame_id
        ):
            # A strictly newer frame transition invalidates all old geometry,
            # but preserves timestamp ordering and marks the old frame retired.
            self._retired_frames.add(self._frame_epoch)
            self.clear("costmap_frame_changed")
        self._newest_accepted_stamp = snapshot.stamp_sec
        self._frame_epoch = snapshot.frame_id
        # Never resolve an older *unseen* Costmap after a newer one arrived.
        # Keeping old entries for TF retries is actively unsafe here; only a
        # previously resolved _ready may remain visible while still fresh.
        self.superseded_count += len(self._pending)
        self._pending = [snapshot]
        self.reason = "waiting_for_exact_tf"
        return True

    def _fresh(self, snapshot: CostmapSnapshot, ros_now_sec: float) -> bool:
        receipt_age = max(0.0, self.clock() - snapshot.received_at)
        stamp_age = float(ros_now_sec) - snapshot.stamp_sec
        return (
            receipt_age <= self.settings.timeout_sec
            and stamp_age <= self.settings.stamp_max_age_sec
            and stamp_age >= -self.settings.future_tolerance_sec
        )

    def resolve(self, ros_now_sec: float, transform_snapshot):
        """Return ``(state, synchronized, reason)`` after nonblocking retries."""
        now = self.clock()
        retained = []
        for snapshot in self._pending:
            if (
                now - snapshot.received_at <= self.settings.tf_wait_timeout_sec
                and self._fresh(snapshot, ros_now_sec)
            ):
                retained.append(snapshot)
            else:
                self.expired_count += 1
        self._pending = retained

        last_error = ""
        if self._pending:
            snapshot = self._pending[0]
            try:
                drawing, analysis = transform_snapshot(snapshot)
            except Exception as error:  # tf2 exception classes vary by distro
                last_error = f"{type(error).__name__}: {error}"
            else:
                synchronized = SynchronizedCostmap(
                    snapshot=snapshot,
                    points_in_base=np.asarray(drawing, dtype=np.float64),
                    analysis_points_in_base=np.asarray(
                        analysis, dtype=np.float64
                    ),
                    synchronized_at=now,
                )
                self._ready = synchronized
                self._pending.clear()
                self.resolved_count += 1
                self.reason = "exact_tf_ready"
                return "ACTIVE", synchronized, self.reason

        if self._ready is not None and self._fresh(
            self._ready.snapshot, ros_now_sec
        ):
            self.reason = "using_previous_exact_sync_while_waiting"
            return "ACTIVE", self._ready, self.reason
        self._ready = None
        if self._pending:
            self.reason = "waiting_for_exact_tf"
            if last_error:
                self.reason += f" {last_error}"
            return "WAITING_TF", None, self.reason
        self.reason = "exact_tf_wait_expired"
        if last_error:
            self.reason += f" {last_error}"
        return "UNAVAILABLE", None, self.reason


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _bounded_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministically retain spatially distributed row-major cells."""
    if indices.size <= maximum:
        return indices
    selected = np.linspace(0, indices.size - 1, maximum, dtype=np.int64)
    return indices[selected]


def occupancy_grid_to_snapshot(
    message,
    settings: CostmapOverlaySettings,
    *,
    ros_now_sec: float,
    received_at: float,
) -> CostmapSnapshot:
    """Validate and sample one ``nav_msgs/OccupancyGrid`` message."""
    settings.validate()
    frame_id = str(message.header.frame_id).strip()
    if not frame_id:
        raise ValueError("costmap frame_id is empty")
    stamp_sec = _stamp_to_sec(message.header.stamp)
    if not math.isfinite(stamp_sec) or stamp_sec <= 0.0:
        raise ValueError("costmap timestamp is zero or invalid")
    ros_now = float(ros_now_sec)
    if not math.isfinite(ros_now):
        raise ValueError("ROS time must be finite")
    stamp_age = ros_now - stamp_sec
    if stamp_age > settings.stamp_max_age_sec:
        raise ValueError(
            f"costmap timestamp stale age={stamp_age:.3f}s "
            f"limit={settings.stamp_max_age_sec:.3f}s"
        )
    if stamp_age < -settings.future_tolerance_sec:
        raise ValueError(
            f"costmap timestamp future offset={-stamp_age:.3f}s"
        )

    width = int(message.info.width)
    height = int(message.info.height)
    resolution = float(message.info.resolution)
    if width < 1 or height < 1:
        raise ValueError("costmap dimensions must be positive")
    # Size limits MUST precede np.asarray: an invalid OccupancyGrid must not
    # allocate multiple arrays proportional to an arbitrary ROS message size.
    if width > settings.max_grid_width_cells:
        raise ValueError(
            f"costmap grid width limit exceeded width={width} "
            f"limit={settings.max_grid_width_cells}"
        )
    if height > settings.max_grid_height_cells:
        raise ValueError(
            f"costmap grid height limit exceeded height={height} "
            f"limit={settings.max_grid_height_cells}"
        )
    count = width * height
    if count > settings.max_grid_cells:
        raise ValueError(
            f"costmap grid cells limit exceeded cells={count} "
            f"limit={settings.max_grid_cells}"
        )
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("costmap resolution must be finite and positive")
    length = len(message.data)
    if length != count:
        raise ValueError(
            f"costmap data length mismatch expected={count} actual={length}"
        )
    data = np.asarray(message.data, dtype=np.int16)
    if np.any(data < -1) or np.any(data > 100):
        raise ValueError("OccupancyGrid costs must be -1 or within [0, 100]")

    origin = message.info.origin
    translation = np.asarray(
        (origin.position.x, origin.position.y, origin.position.z),
        dtype=np.float64,
    )
    quaternion = (
        origin.orientation.x,
        origin.orientation.y,
        origin.orientation.z,
        origin.orientation.w,
    )
    rotation = quaternion_matrix_xyzw(quaternion)
    # OccupancyGrid is a horizontal 2-D plane.  Reject tilted origins instead
    # of silently projecting cells at invented heights.
    if not np.allclose(rotation[:, 2], (0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("costmap origin is not a horizontal plane")
    if not np.isfinite(translation).all():
        raise ValueError("costmap origin contains NaN or Inf")

    flat_all = np.arange(data.size, dtype=np.int64)
    rows_all, columns_all = np.divmod(flat_all, width)
    local_all = np.column_stack((
        (columns_all.astype(np.float64) + 0.5) * resolution,
        (rows_all.astype(np.float64) + 0.5) * resolution,
        np.full(data.size, settings.plane_height_m, dtype=np.float64),
    ))
    analysis_points = (rotation @ local_all.T).T + translation

    masks = (
        data == -1,
        (data >= settings.high_cost_min) & (data < settings.lethal_cost_min),
        data >= settings.lethal_cost_min,
    )
    class_counts = tuple(int(np.count_nonzero(mask)) for mask in masks)
    points = []
    classes = []
    for class_id, mask in enumerate(masks):
        flat = _bounded_indices(np.flatnonzero(mask), settings.max_points_per_class)
        if flat.size == 0:
            continue
        points.append(analysis_points[flat])
        classes.append(np.full(flat.size, class_id, dtype=np.uint8))

    point_array = (
        np.concatenate(points, axis=0)
        if points else np.empty((0, 3), dtype=np.float64)
    )
    class_array = (
        np.concatenate(classes, axis=0)
        if classes else np.empty((0,), dtype=np.uint8)
    )
    return CostmapSnapshot(
        frame_id=frame_id,
        stamp_sec=stamp_sec,
        received_at=float(received_at),
        points=point_array,
        classes=class_array,
        analysis_points=analysis_points,
        analysis_costs=data.copy(),
        unknown_cells=class_counts[UNKNOWN_CLASS],
        high_cost_cells=class_counts[HIGH_COST_CLASS],
        lethal_cost_cells=class_counts[LETHAL_COST_CLASS],
    )


class CostmapStore:
    """Keep only a fresh, valid costmap snapshot; invalid input clears it."""

    def __init__(
        self,
        settings: CostmapOverlaySettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.clock = clock
        self.snapshot = None
        self._history: list[CostmapSnapshot] = []
        self.reason = "waiting_for_local_costmap"
        self.receive_count = 0
        self._newest_accepted_stamp = -math.inf
        self._active_frame: str | None = None
        self._retired_frames: set[str] = set()

    @property
    def history_count(self) -> int:
        return len(self._history)

    def update(self, message, ros_now_sec: float) -> bool:
        self.receive_count += 1
        try:
            snapshot = occupancy_grid_to_snapshot(
                message,
                self.settings,
                ros_now_sec=ros_now_sec,
                received_at=self.clock(),
            )
        except (AttributeError, TypeError, ValueError) as error:
            self.snapshot = None
            self._history.clear()
            self.reason = f"invalid_local_costmap {error}"
            return False
        if snapshot.stamp_sec <= self._newest_accepted_stamp:
            # Ordering history survives stale/invalid clear and frame change.
            # Reject without clearing the newer synchronized queue snapshot.
            self.reason = (
                "out_of_order_local_costmap_ignored "
                f"stamp={snapshot.stamp_sec:.9f} "
                f"newest={self._newest_accepted_stamp:.9f}"
            )
            return False
        if snapshot.frame_id in self._retired_frames:
            self.reason = (
                "out_of_order_local_costmap_ignored "
                f"retired_frame={snapshot.frame_id!r}"
            )
            return False
        if (
            self._active_frame is not None
            and self._active_frame != snapshot.frame_id
        ):
            self._retired_frames.add(self._active_frame)
            self.snapshot = None
            self._history.clear()
        self._newest_accepted_stamp = snapshot.stamp_sec
        self._active_frame = snapshot.frame_id
        self.snapshot = snapshot
        self._history.append(snapshot)
        minimum_stamp = snapshot.stamp_sec - self.settings.stamp_max_age_sec
        self._history = [
            item for item in self._history
            if item.frame_id == snapshot.frame_id
            and item.stamp_sec >= minimum_stamp
        ][-16:]
        self.reason = "local_costmap_ready"
        return True

    def state_for_image(self, image_stamp_sec: float, ros_now_sec: float):
        """Return the newest Costmap whose stamp does not follow the image."""
        image_stamp = float(image_stamp_sec)
        if not math.isfinite(image_stamp) or image_stamp <= 0.0:
            return "UNAVAILABLE", None, "image_timestamp_invalid"
        if not self._history:
            return "UNAVAILABLE", None, self.reason
        frame_id = self._history[-1].frame_id
        candidates = [
            item for item in self._history
            if item.frame_id == frame_id and item.stamp_sec <= image_stamp
        ]
        if not candidates:
            offset = self._history[0].stamp_sec - image_stamp
            return (
                "WAITING_IMAGE_TIME", None,
                f"costmap_history_newer_than_image offset={offset:.3f}s",
            )
        snapshot = candidates[-1]
        image_age = image_stamp - snapshot.stamp_sec
        if image_age > self.settings.stamp_max_age_sec:
            return (
                "STALE", None,
                f"costmap_history_stale_for_image age={image_age:.3f}s "
                f"limit={self.settings.stamp_max_age_sec:.3f}s",
            )
        receipt_age = max(0.0, self.clock() - snapshot.received_at)
        if receipt_age > self.settings.timeout_sec:
            return (
                "STALE", None,
                f"costmap_history_receive_stale age={receipt_age:.3f}s "
                f"limit={self.settings.timeout_sec:.3f}s",
            )
        now_age = float(ros_now_sec) - snapshot.stamp_sec
        if now_age < -self.settings.future_tolerance_sec:
            return (
                "UNAVAILABLE", None,
                f"costmap_history_future offset={-now_age:.3f}s",
            )
        return "ACTIVE", snapshot, "costmap_history_matched_image"

    def state(self, ros_now_sec: float):
        snapshot = self.snapshot
        if snapshot is None:
            return "UNAVAILABLE", None, self.reason
        receipt_age = max(0.0, self.clock() - snapshot.received_at)
        if receipt_age > self.settings.timeout_sec:
            self.snapshot = None
            self._history.clear()
            self.reason = (
                f"local_costmap_receive_stale age={receipt_age:.3f}s "
                f"limit={self.settings.timeout_sec:.3f}s"
            )
            return "STALE", None, self.reason
        stamp_age = float(ros_now_sec) - snapshot.stamp_sec
        if stamp_age > self.settings.stamp_max_age_sec:
            self.snapshot = None
            self._history.clear()
            self.reason = (
                f"local_costmap_stamp_stale age={stamp_age:.3f}s "
                f"limit={self.settings.stamp_max_age_sec:.3f}s"
            )
            return "STALE", None, self.reason
        if stamp_age < -self.settings.future_tolerance_sec:
            self.snapshot = None
            self._history.clear()
            self.reason = (
                f"local_costmap_stamp_future offset={-stamp_age:.3f}s"
            )
            return "UNAVAILABLE", None, self.reason
        return "ACTIVE", snapshot, "local_costmap_ready"

    def clear(self, reason: str) -> None:
        self.snapshot = None
        self._history.clear()
        self.reason = str(reason)


def project_costmap_points(
    points_in_base: np.ndarray,
    classes: np.ndarray,
    mount: CameraMount,
    intrinsics: CameraIntrinsics,
    near: float,
) -> ProjectedCostmap:
    """Project classified base-frame points into the calibrated image."""
    points = np.asarray(points_in_base, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("costmap points must have shape (N, 3)")
    if class_values.shape != (points.shape[0],):
        raise ValueError("costmap classes must match point count")
    if not math.isfinite(float(near)) or near <= 0.0:
        raise ValueError("near clip must be finite and positive")
    if points.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int32)
        return ProjectedCostmap(empty, empty, empty, 0, 0)
    optical = world_to_optical(points, (0.0, 0.0, 0.0), mount)
    depth = optical[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * optical[:, 0] / depth + intrinsics.cx
        v = intrinsics.fy * optical[:, 1] / depth + intrinsics.cy
    visible = (
        np.isfinite(u)
        & np.isfinite(v)
        & (depth >= near)
        & (u >= 0.0)
        & (u <= intrinsics.width - 1.0)
        & (v >= 0.0)
        & (v <= intrinsics.height - 1.0)
    )
    pixels = np.column_stack((u, v))

    def group(class_id: int) -> np.ndarray:
        selected = visible & (class_values == class_id)
        return np.rint(pixels[selected]).astype(np.int32)

    return ProjectedCostmap(
        unknown_pixels=group(UNKNOWN_CLASS),
        high_cost_pixels=group(HIGH_COST_CLASS),
        lethal_cost_pixels=group(LETHAL_COST_CLASS),
        input_points=int(points.shape[0]),
        visible_points=int(np.count_nonzero(visible)),
    )
