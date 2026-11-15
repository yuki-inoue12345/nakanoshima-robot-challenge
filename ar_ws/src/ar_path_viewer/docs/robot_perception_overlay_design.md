# Robot perception overlay design (review only)

## Scope and safety boundary

This future overlay is display-only. It may subscribe to the robot's existing
costmap or LiDAR topics, but it must never publish a goal, velocity, costmap,
filter, recovery command, or parameter update. Its absence or failure must not
affect Nav2. The feature remains disabled by default until recorded-data tests
and stationary NUC tests have passed.

## Runtime facts to verify on the NUC

Before implementing a decoder, record the exact type, QoS, `frame_id`, update
rate, dimensions, resolution, origin, and a value histogram for:

- `/local_costmap/costmap`
- `/global_costmap/costmap`
- `/hokuyo_scan`

Do not assume that an `OccupancyGrid` exposes Nav2's internal cost bytes
unchanged. In particular, unknown, free, inflated, inscribed, and lethal values
must be identified from the actual message encoding and the running Nav2
version. A cost is not automatically a collision verdict or a general-purpose
"danger" label.

## Proposed offline-first pipeline

1. Read a recorded costmap message without changing its producer or QoS.
2. Reject stale data, unsupported encodings, invalid geometry, missing TF, and
   timestamps that cannot be aligned with the image.
3. Transform only cell centres within a configured range from the message
   frame to `robot/base_footprint` at the message time.
4. Project a downsampled ground-plane grid through the same camera model and
   clipping code used by the Nav2 paths. Do not shift off-screen cells into the
   image and do not claim image-based object recognition.
5. Render a low-alpha, separately labelled legend such as `COSTMAP COST`; keep
   unknown, inflated, inscribed, and lethal classes visually distinct only
   after their live encoding has been verified.
6. Measure callback age, processed cells, render time, dropped frames, CPU, and
   memory. Automatically suppress the overlay when its budget or freshness
   limit is exceeded.

LiDAR points may be a later independent layer. They require the scan-time TF,
range validity checks, angular geometry, camera calibration, and an explicit
warning that a 2-D scan does not provide camera-image occlusion or object
height. Costmap and raw scan observations must not be merged under one label.

## Acceptance phases

- Phase P1: read-only metadata/value audit and rosbag-based rendering.
- Phase P2: synthetic and recorded tests for transforms, timestamps, clipping,
  value classes, and fail-closed behaviour.
- Phase P3: stationary NUC test with the feature flag enabled, including frame
  rate and resource measurements.
- Phase P4: visual comparison during an approved run; still no control-side
  interfaces.

Until these phases pass, no production YAML should enable this overlay.
