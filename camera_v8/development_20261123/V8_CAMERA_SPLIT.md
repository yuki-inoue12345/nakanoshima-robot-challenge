# Camera V8 / AR separate camera process (2026-11-23)

## Architecture

```text
C920 USB (opened by ONE process ONLY)
  v8_camera_source.py (independent ROS process)
    V4L2TimestampCameraSource -> bounded LatestImageSource
    ├── /koko/c920/raw_image             sensor_msgs/Image BGR8 640x480
    │     └── AR Viewer ROS Subscriber only -> AR overlay/MP4 NEVER stops
    └── V8PersonFrameShare node (SAME independent camera-owner process)
          /person_area_arrived true -> /koko/person_search/raw_image
          /person_search_done_detail terminal=true or false arrival -> stops
          └── V8 search server (always waiting; runs recognition when triggered)
```

V7, the original V7 recognition algorithm, Renya Launcher2, Follower, Nav2,
control topics, physical C920 mounting and regular AR-only mode are unchanged.
V8 video recording/UI remain removed, and only the AR Viewer saves MP4.
V8 still retains its static evidence photographs and JSON result logs.

## Source-time safety

The independent camera owner gets kernel V4L2 SOE or EOF timestamps.
It sends a ROS image header with the original acquisition time converted from
CLOCK_MONOTONIC to ROS time, labelled with `koko_c920_v4l2_soe` only when the
camera driver explicitly provided SOE. The AR subscriber accepts only new
640x480 BGR8 frames, with queue depth 1, within 0.45 sec. It reconstructs
local monotonic time from ROS time on the SAME NUC, then reuses the existing
`estimate_image_frame_timing(..., source_kind='camera_timestamped', ...)`
checks. Untrusted/EOF/late stamps fail closed: raw preview may remain, but
AR/Costmap overlays requiring trusted exposure timing must not use them.

This does NOT claim perfect sensor synchronization. The exposure-time and
transport/latency behavior must still be measured on the physical NUC.

## Development PC checks (do these BEFORE Git)

1. From feature branch, apply the ZIP script with `--check`, then `--apply`.
2. Compile all new modules; `bash -n` shell runner; run included mock tests.
3. Build/test ar_path_viewer and ar_reference_planner with colcon.
4. Verify `ar_ws/src/ar_path_viewer/ar_path_viewer/` has only subscriber APIs.
5. Verify no tracked modifications to Camera V7, MS2 algorithm, Nav2/Follower.

## Stationary real-hardware validation after Git transfer

1. The robot should be stationary and physically supervised; no test here sends
   driving commands.
2. Start existing Nav2/Launcher2 as normal. Start V8 server separately (NOT V7).
3. Start `camera_v8/development_20261123/run_ar_with_v8_share.sh`.
4. Check exactly one C920 owner with `fuser /dev/v4l/by-id/*C920*video-index0`
   and `/koko_v8_camera_source` in `ros2 node list`.
5. Observe continuous `/koko/c920/raw_image` and continuous AR display/MP4.
6. Without firing a search, `/koko/person_search/raw_image` should have no
   active published frames (Topic exists; silence is expected).
7. Trigger only with proper existing Follower person_area arrival / safe test
   trigger; V8 must start detection, use only unannotated source, and send
   existing completion status with fail-closed safety.
8. After search terminal or false arrival, search image messages stop; AR and
   AR recording do NOT stop.
9. Check actual frames/s, CPU, memory, lighting and overlay timestamp gates.
10. Verify failure cases: missing shared owner => AR refuses startup; missing
   V8 frames => search must fail closed, not report safe-to-advance.

**Not production verified.** No physical C920/ROS2 end-to-end validation has
been run from this ZIP. Preserve the existing stable `main` branch until
stationary testing passes. If timestamps are untrusted, remove the experimental
share and use the normal AR launcher only — don't bypass Costmap gates.

Desktop icon installation is intentionally deferred until NUC E2E succeeds.
