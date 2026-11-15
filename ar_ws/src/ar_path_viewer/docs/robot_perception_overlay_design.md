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
