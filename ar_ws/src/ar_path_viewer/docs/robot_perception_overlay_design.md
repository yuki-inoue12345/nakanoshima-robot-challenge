# Robot perception overlay

## Safety boundary

This is an opt-in, display-only layer. The Viewer subscribes to an existing
Nav2 costmap and never publishes goals, velocity, costmap data, filters,
recovery commands, or parameter updates. A missing, invalid, stale, or
untransformable message suppresses only this layer. The real green Path,
cyan reference Path, orange guidance, recording, and Nav2 operation remain
independent.

Both tracked base profiles keep `perception_costmap_enabled: false`. The
review overlay is the only tracked profile that enables it:

```bash
ros2 run ar_path_viewer ar_path_viewer --ros-args \
  --params-file /path/to/ar_path_viewer.yaml \
  --params-file /path/to/perception_costmap_review_overlay.yaml
```

This command is for a stationary review after the live topic contract below
has been checked. It is not a production default.

## Camera-frame time alignment (P1-① review)

Costmap's stamped `robot/odom` coordinates are re-expressed in the robot base
frame **at the camera frame's estimated acquisition time**, rather than
projecting the previous Costmap-time base coordinates onto a later image.
The complete relative 3-D translation and rotation between these two TFs
are bounded; yaw-only and XY-only checks are insufficient for a tilting robot.
An interpolated tf2 lookup is permitted only when its returned stamp matches
the *requested* time within 5 ms. This does not authorize a lookup of the
latest available TF in place of the requested timestamp.

The normal OpenCV source has no trusted acquisition timestamp and never
authorizes this overlay.  The opt-in V4L2 source associates one dequeued MJPEG
buffer with its kernel monotonic SOE timestamp and converts that timestamp to
ROS time using the host ROS/monotonic pair sampled after acquisition.  The
timestamp is still a kernel/driver observation rather than independently
measured C920 exposure time, so NUC stationary and motion acceptance tests
remain mandatory.

The V4L2 review pipeline holds an image for 120 ms, and for at most 300 ms,
without changing its image or timestamp.  An exact image-time TF that has not
arrived causes `FRAME SYNC WAIT`; the same frame is retried nonblockingly and
is discarded at the finite deadline.  A two-thread executor lets tf2's
reentrant subscriptions receive 30 Hz TF while camera/rendering remains in
the Viewer's mutually-exclusive callback group.  This replaces the former
one-shot behavior, which consumed the delayed image on its first failed TF
lookup and immediately tried a newer image.

The Viewer retains a bounded Costmap history and selects the newest Costmap
whose header stamp is not newer than the held image.  It never pairs a future
Costmap with an older image.  Green and cyan Paths are also transformed at the
held image timestamp; a Path received after that image is hidden rather than
drawn with a future route generation.  Historical/replayed MP4 sources without
a trusted timestamp must not silently be treated as live camera frames.

Failures suppress Costmap-derived image overlays, minimap, and cost-based
explanations/alerts. They do not alter the real Nav2 Path or navigation.
Pending-order and Path-goal-generation audit items P1-②/P1-③ are independent
and **not fixed** by this timestamp alignment.

## Pending Costmap ordering (P1-② review)

The Costmap store and exact-time TF queue both require **globally strictly
increasing header stamps**, including across `frame_id` changes. Each newly
accepted sample supersedes all older unresolved pending samples; late TF
arrival cannot revive an older unseen Costmap. A previously synchronized
(`ready`) sample may be retained only while its independent freshness tests
pass, and never across an accepted frame change.

A frame transition with a newer stamp invalidates ready/pending geometry and
retires the previous frame ID. A delayed callback from any retired frame is
ignored even if its header stamp is later than the current watermark; frame
A→B→A is **not** treated as an implicit epoch reset. A subsequent distinct
frame C with a strictly newer stamp can still be accepted. The store performs
the same ordering checks before replacing its snapshot, and the Viewer treats
these rejections as non-invalidating (preserving a newer ready pair).

The watermark and retired-frame set survive TF expiry, stale snapshots, and
invalid-input clears. An actual ROS clock reset or a deliberate return to a
retired frame therefore requires an **externally verified new session**, such
as a Viewer restart; merely receiving another `frame_id` does not authorize
resetting history. This sacrifices automatic recovery from an unverified
clock/configuration change to prevent stale overlays. P1-③ (real Path/Goal
generation correspondence) is separate and remains unresolved.

The pending queue now holds at most one unresolved Costmap.  The legacy
`max_pending_messages` configuration remains accepted for compatibility,
but its configured limit is no longer used to justify retaining superseded
pending samples.

## Source selection

The first implementation uses `/local_costmap/costmap` as a
`nav_msgs/msg/OccupancyGrid`:

- It represents the 2-D navigation evidence that Nav2 currently uses locally,
  including merged observation and inflation layers.
- Its grid geometry makes a bounded ground-plane visualization practical.
- It must still be labelled as costmap information, not camera object
  recognition, a person detector, or a general-purpose danger verdict.

Direct LiDAR is deliberately deferred. A `sensor_msgs/msg/LaserScan` such as
`/hokuyo_scan` or `/scan3` is closer to a raw sensor return, but it needs scan
time interpolation, per-beam range validation, sensor-frame TF, and an honest
statement that a 2-D scan provides neither object height nor image occlusion.
The Global Costmap is also deferred because it is less useful for a
camera-local view and can add a much larger sampling workload.

The read-only Renya source snapshot (`fb8b808`) configures the Local Costmap
as a 6 x 6 m rolling grid at 0.05 m, with 5 Hz update and 2 Hz publish rates.
Its layers consume `/scan3` and `/hokuyo_scan_filtered`, then add inflation.
The configured local frame is rewritten with the robot TF prefix at launch.
These are source-code observations, **not confirmation of the running NUC**.

## Message semantics and rendering

The standard Nav2 OccupancyGrid publication uses values in `[-1, 100]`.
This Viewer preserves three explicit classes:

- `-1`: unknown, muted grey-blue dot
- configurable `60..99`: high navigation cost, amber dot
- configurable `100`: lethal-cost cell, magenta diamond

The thresholds describe the received grid values only. In particular,
inflation cost is not called an obstacle or danger, and a lethal-cost cell is
not asserted to be a visually identifiable object. Live value histograms and
the running Nav2 version still need checking on the NUC.

Dense filled polygons were considered, but rejected for this first phase:
they obscure the green and cyan Paths at 640 x 480 and cost more to render.
The selected design samples at most 450 cells per class, projects their real
cell centres, draws low-alpha semantic markers below the Path layers, and
labels the layer `LOCAL COSTMAP`. Sampling is deterministic and retains the
full per-class counts in diagnostics.

## Costmap + real-Path explanation

The optional explanation card uses the same fresh Local Costmap and the
already-authorized real `/plan`.  It does **not** claim access to Nav2's
internal planner objective or identify a camera-visible object.  The compact
top-left card therefore uses neutral, directly supported wording:

- `FRONT LOW COST | PATH STRAIGHT`
- `RIGHT COST | PATH LEFT` / `LEFT COST | PATH RIGHT`
- `PATH CHANGING | COST AHEAD`
- `COSTMAP BLOCKED AHEAD`

The first label requires low cost in a 0.4--2.5 m front region and a stable
straight real-Path decision.  Side labels require both a stable Path turn and
a configured cost asymmetry on the opposite side.  Replanning/HOLD labels
require lethal-cost evidence in the front region.  Unknown-heavy or
undersampled regions, stale messages, invalid TF, unstable localization,
missing real Path, and ambiguous cost/path relationships produce no card.
The diagnostics event is `perception_path_explanation` and records the exact
fractions used for the decision.

Two alternatives were rejected for the first implementation.  A dense heat
map hid the two routes at 640 x 480.  A free-form statement such as "Nav2
chose left because of an obstacle" overstated causality and object identity
that cannot be established from the published OccupancyGrid and Path alone.
Both `perception_costmap_enabled` and `perception_explanation_enabled` remain
false in base profiles; the review overlay enables them together.

## Transform and freshness contract

For every accepted OccupancyGrid, the Viewer:

1. validates frame, non-zero timestamp, dimensions, resolution, data length,
   values, and a horizontal grid origin;
2. converts cell centres using the OccupancyGrid origin translation and yaw;
3. requests the transform from the message frame to `base_footprint` at the
   message timestamp;
4. rejects zero, stale, or future TF stamps;
5. uses the same `CameraMount`, `CameraIntrinsics`, optical-axis conversion,
   near clipping, and image bounds as the Path projection;
6. clears invalid input and suppresses stale data instead of retaining old
   markers.

The marker plane defaults to 0.02 m above the costmap plane only to avoid
numeric overlap. It does not infer object height or image occlusion. Camera
calibration is still provisional, so visual alignment cannot be claimed until
the stationary NUC review is complete.

## NUC contract audit (read-only)

Before enabling the review overlay, record the actual type, QoS, frame, rate,
geometry, and values without changing the running system:

```bash
ros2 topic info /local_costmap/costmap --verbose
ros2 topic hz /local_costmap/costmap
ros2 topic echo /local_costmap/costmap --once --field header
ros2 topic echo /local_costmap/costmap --once --field info
ros2 topic type /hokuyo_scan
ros2 topic type /scan3
ros2 run tf2_ros tf2_echo robot/base_footprint robot/odom
```

The exact prefixed frames must be taken from the messages and graph; do not
substitute these example names blindly. Confirm that the Publisher offers
Reliable + Transient Local if the tracked subscription is to match it. If the
live type, QoS, value encoding, or frames differ, leave the feature OFF and
update the review implementation/test contract first.

## Acceptance status

- Completed offline: class decoding, grid-origin geometry, bounded sampling,
  base/camera projection, left/right signs, stale/invalid/TF suppression,
  recovery, conservative cost/path explanation, coexistence with both Paths
  and guidance, and MP4 pixel checks.
- Not yet completed: connection to the NUC topic, live QoS/type/frame/value
  verification, real C920 alignment, full Viewer CPU/RSS, and approved-run
  visual comparison.

No production YAML should enable this layer until those live checks pass.


## Camera source timestamp gate (P1-1; pre-NUC review)

The standard `image_input_type: camera` uses OpenCV `VideoCapture.read()`,
which supplies no per-frame acquisition timestamp.  Even a fast `read()` can
return a stale queued image.  Consequently **camera frames from this source
never authorize Costmap-derived AR, minimap, path explanation or costmap
alerts**.  Green `/plan`, cyan reference, guidance, live camera and recording
continue unchanged. This is intentional fail-closed behavior, not a missing
Costmap subscription.

An **experimental, explicitly opt-in** `image_input_type: camera_timestamped`
uses Linux V4L2 streaming with `VIDIOC_DQBUF` on the same camera file
descriptor that supplies MJPEG image bytes.  It requires the kernel monotonic
buffer timestamp flag; metadata and image can therefore be associated with
one specific dequeued buffer.  Age since the timestamp must be <=0.12s;
ROS/monotonic clock drift within/between acquisitions must be <=0.02s; the
existing Costmap age/motion/TF gates still apply. A non-monotonic/absent
kernel timestamp, old queued frame, or decode failure never triggers a
fallback to OpenCV timestamps. **Only kernel SOE (start-of-exposure) source
flags authorize AR**. An EOF timestamp is accepted as provenance for
diagnostics but does not authorize camera/Costmap alignment until its
exposure-to-EOF offset is separately measured and a reviewed policy is added.

**V4L2 buffer timestamp is not guaranteed to be the true C920 exposure time.**
The camera driver may report SOE or EOF (end-of-frame) time. In particular,
the interval from exposure to an EOF timestamp depends on the camera and USB
transport. Until that uncertainty is characterized on the NUC, do not claim
that the kernel timestamp has proven image-to-world alignment or that the
P1-1 real-robot release gate has passed. Keep normal YAML defaults OFF.
The opt-in review file
`config/perception_costmap_v4l2_review_overlay.yaml` does not replace the
robot's installed camera device URI/recording paths. Use only after checking
C920 MJPEG 640x480 support, V4L2 timestamp source flags, image/Costmap/TF
clocks and static/rotational test footage.  The OpenCV preview remains
available if the V4L2 source is not supported, but no perception overlay is
shown from unverified capture timestamps.

### P2: Costmap input budgets and minimap class priority

- OccupancyGrid dimensions are checked **before** NumPy conversion and before
  generating row/column/world-coordinate arrays. Configurable limits:
  `perception_costmap_max_grid_width_cells=512`,
  `perception_costmap_max_grid_height_cells=512`,
  `perception_costmap_max_grid_cells=65536` (the observed local grid is 120x120).
  Over-limit input is invalid and clears the current Costmap snapshot; it must
  not be displayed or used for alerts. This bounds allocation at the callback,
  **not** the ROS middleware's message receive/deserialization itself.
- Minimap sampling is class-aware. Lethal-cost cells are retained first, then
  high-cost cells, then unknown cells, up to 2500 plotted cells. If lethal alone
  exceeds the budget, lethal is spatially downsampled. Free cells are not
  plotted. The minimap is not an exhaustive occupancy visualization: several
  cells may share one pixel, and later path/robot strokes may cover a point.
- These protections do not change the full-resolution data passed to the
  separate alert evaluator, which remains independently gated.
