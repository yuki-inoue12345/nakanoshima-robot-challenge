# AR reference planner

This package runs a Nav2 Planner Server that is separate from the robot-control
Planner Server. It only computes a display-only reference path. It contains no
controller, navigator, velocity publisher, or robot-control action client.

## Topic and action separation

- Planning action: `/ar_reference/compute_path_through_poses`
- Valid reference result: `/ar/reference_path`
- Atomic Viewer contract: `/ar/reference_path_state`
- Read-only follower input: `/waypoint_follower/reference_state`
- Planner debug output: `/ar_reference/planner_internal_plan`
- Robot control path `/plan`: never published by this package

The request node publishes an empty `nav_msgs/Path` when planning fails, the
result is invalid, the request times out, or the last successful result expires.
The Phase 3 Viewer does not trust that Path alone. `ReferencePathState` carries
the Path, route SHA-256, next index/number, navigation mode, follower goal
generation, safe boundary, source follower timestamp, planner generation, and
declared path length in one atomic message. The Viewer separately subscribes to
the current transient-local `WaypointFollowerState` and draws only when those
identities match exactly. Pause/person-search, route/index/generation changes,
an invalid planner result, or either timeout clears the display. This topic is
never substituted for the control `/plan`.

## Selected-map and Keepout safety gate

Production launch requires the exact selected map YAML. The launch validates
the map image, geometry, `.keepout/<map>/settings.json`, source-map identity,
and mask image before it derives parameters containing `KeepoutFilter`.
Forbidden cells are never embedded in this package.

The robot-control launch remains the sole owner of the root
`filter_mask_server` and `costmap_filter_info_server`. The reference planner
consumes `/costmap_filter_info` and `/keepout_filter_mask` read-only, avoiding
duplicate root node/topic ownership. Before sending any action request, its
request node independently requires all of the following:

- the live `/map` geometry and complete occupancy-data SHA-256 match the
  selected map;
- when Keepout is enabled, the live mask geometry and complete data SHA-256
  match the selected map's saved mask;
- filter type, mask topic, base and multiplier match the expected Keepout
  contract;
- both shared filter servers report lifecycle `ACTIVE`.

Map changes, stale/wrong masks, missing servers, or a lifecycle failure clear
the reference Path and reject late results. A map whose saved Keepout setting
is absent or explicitly disabled uses ordinary reference planning, but its live
map must still match. This gate protects the display-only result; it does not
modify or control the robot Nav2 stack.

## Follower state contract (Phase 2)

`robotbase_waypoint_interfaces/msg/WaypointFollowerState` is reliable and
transient-local. Its index is zero-based; `safe_section_end_index` is inclusive.
It carries the exact waypoint-YAML SHA-256, next index and waypoint number,
`to_pose`/`through_poses` mode, the follower's existing `goal_serial`, activity,
route size, and ROS creation timestamp. The planner rejects a different route,
inconsistent index/number, stale or future timestamps, unsupported schema,
invalid safety boundary, and every activity except `NAVIGATING`.

The Follower computes the safety boundary with its existing
`navigation_segment_end()` logic. The reference planner does not reproduce its
progress, person-search, pause, or waiting decisions. A state/index/generation
change clears the old path immediately. An outstanding request is canceled and
remains the sole owned request until its result or timeout; a late result whose
identity differs is rejected.

The Follower publishes changes within 0.2 seconds and a 1-second heartbeat.
The reference planner replans every 3 seconds while state remains unchanged.
This is inside the requested 2--5 second range, follows obstacle changes, and
keeps planning load below the prior 5-second diagnostic cadence. Phase 2 does
not suppress planning based on the control `/plan`: doing that safely would add
freshness/route-identity coupling to a control topic and is deferred.
The request node also checks its own namespaced Planner Server lifecycle service
and sends no action request until that server reports `ACTIVE`.

## Build

```bash
cd /home/inoue-yuki/care_patrol_ws/local_repositories/nakanoshima-robot-challenge/robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to robotbase_waypoint --symlink-install
source install/setup.bash
cd ../ar_ws
colcon build --packages-select ar_reference_planner --symlink-install
source install/setup.bash
```

## Development-PC integration test

This uses real Nav2 Map Server, Filter Info Server, Lifecycle Manager,
KeepoutFilter, Planner Server, and Costmap components with deterministic map
files generated under `/tmp`. It uses no sensors, a fixed test TF, and an
existing repository waypoint YAML. A development-only typed mock Follower
state starts at zero-based index 1. It is not an equivalent test of the real
Velodyne/Hokuyo costmap or the NUC map files.

```bash
export ROS_DOMAIN_ID=77
export ROS_LOG_DIR=/tmp/ar_reference_ros_logs
ros2 launch ar_reference_planner reference_planner_dev_test.launch.py \
  waypoint_file:=/home/inoue-yuki/care_patrol_ws/local_repositories/nakanoshima-robot-challenge/robot_ws/maps_waypoints/waypoints/0919_nakanosima_way.yaml \
  keepout_enabled:=true
```

The synthetic Keepout rectangle crosses the direct WP4-to-WP5 line. A
successful Navfn result must go around it instead of reproducing the waypoint
polyline. Use `keepout_enabled:=false` to test the normal-map path. Setting
`keepout_enabled:=true start_keepout_servers:=false` is a fail-closed test: no
reference action request or valid Path may be produced.

## Production-shaped launch (not yet approved for the NUC)

The root map server, TF, `/scan3`, and `/hokuyo_scan` must already exist.

```bash
ros2 launch ar_reference_planner reference_planner.launch.py \
  namespace:=ar_reference \
  map:=/absolute/path/to/selected-map.yaml \
  waypoint_file:=/absolute/path/to/route.yaml
```

When the selected map has Keepout enabled, start the robot-control Nav2 launch
first so its root filter servers are active. Keep the shared filter-info/mask
topics and lifecycle services absolute. If the selected map has Keepout
disabled, those filter servers are not required. The AR Viewer setting
`reference_path_enabled: false` remains the production default until this is
verified against the actual NUC map and sensor Costmap.

## CPU and memory measurement

Measure after lifecycle activation and exclude the development-only synthetic
map and static-TF processes:

```bash
ps -C planner_server -o pid,pcpu,rss,vsz,etimes,args
ps -eo pid,pcpu,rss,vsz,etimes,args | \
  grep -E 'reference_path_request|lifecycle_manager_reference_planner'
```

For periodic reference planning, measure a multi-second CPU window and record
Planner Server `planning_time_sec` separately from end-to-end request latency.

## Future follower interface

Phase 2 still requires the same waypoint file on both processes. The file hash
prevents accidental cross-route results. The existing Follower does not reload
routes at runtime; changing routes means starting a new Follower whose hash
invalidates the previous reference path.
