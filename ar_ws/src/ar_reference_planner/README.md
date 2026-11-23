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

The state also carries the final Waypoint position only when the active
Follower route hash and Waypoint count exactly match the YAML loaded by this
Planner. The Viewer may use it for an explicitly labelled map-plane `DIRECT`
distance; it is not a Nav2 route length and is never used for control.

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

## Launcher2 automatic integration

The automatic supervisor deliberately reuses the two existing sources of
truth instead of adding another map/waypoint selection UI:

- map: Launcher2's
  `$HOME/.${ROBOTBASE_LAUNCHER_ID}_nav2_current_map.yaml` `current_map`;
- route: the running `/koko_waypoint_follower` node's existing
  `waypoint_file` ROS parameter, accepted only when its file SHA-256 and
  waypoint count/index/number agree with `WaypointFollowerState`.

The Viewer may start before the Follower. Until a fresh Follower heartbeat and
the parameter service are present, the supervisor publishes an invalid atomic
reference state and starts no Planner. A missing, relative, unreadable, or
initially stale map state; a missing route; a route-hash mismatch; or an invalid
Keepout bundle behaves the same way. The initial map-state age limit defaults
to 24 hours and is configurable. Once a state written during this supervisor
run is accepted, it remains selected until Launcher2 changes or removes it;
the live map/Keepout gate still compares the complete OccupancyGrid content.

Changing the map or route invalidates the old Path before stopping the owned
reference launch process. After a map change, the still-running old route hash
is rejected until the Follower publishes a different route or its old lifetime
has ended; this prevents a transient new-map/old-route pair. A new isolated
Planner is then launched from the verified pair. Follower
index/generation/pause changes do not restart Nav2;
the existing request node handles those fast state transitions atomically.
When the Follower heartbeat stops, the reference launch is stopped and the
water-blue path is cleared. The supervisor contains no control action client
and publishes neither `/plan` nor a velocity topic.

The checked-in Viewer YAML remains `reference_path_enabled: false`. This
dedicated launch is the only place that explicitly overrides it for testing:

```bash
ros2 launch ar_reference_planner launcher2_reference_ar.launch.py \
  viewer_config_file:=/home/koko-chan-nuc/ar_navigation_project/config/ar_path_viewer_launcher2.yaml
```

固定し直したC920との比較走行だけは、本番YAMLを変更せずreview-only
overlayを2つ目のparameter fileとして明示できます。`fx/fy`と
`camera_pitch_up_deg=11.70`は確定値ではなく比較候補です。

```bash
ros2 launch ar_reference_planner launcher2_reference_ar.launch.py \
  viewer_config_file:=/home/koko-chan-nuc/ar_navigation_project/config/ar_path_viewer_launcher2.yaml \
  viewer_calibration_overlay_file:=$(ros2 pkg prefix ar_path_viewer)/share/ar_path_viewer/config/c920_fixed_20261111_review_overlay.yaml
```

通常比較は`viewer_calibration_overlay_file`を省略して既存設定を使い、同じ
ルート・停止位置・照明条件で録画を分けます。overlayはC920デバイス、録画先、
Launcher2自動選択、参考Path既定OFFという所有関係を変更しません。

No `map:=` or `waypoint_file:=` argument is accepted. Start order is flexible:
the command can run first, then Launcher2 can select/start Nav2 and later start
the Follower. To use an already-running Viewer, add `start_viewer:=false`.
`ROS_DOMAIN_ID` and `ROBOTBASE_TF_PREFIX` must already match Launcher2.

### Launcher2旧Follower互換

`launcher2_reference_ar.launch.py`は、走行制御へ接続しない
`legacy_follower_reference_bridge`も起動します。実行中Followerに
`reference_state_topic`パラメータが存在する新実装では、このbridgeはpublisherを
作らず、Follower本来の厳密な状態をそのまま使用します。

ROS 2 Jazzyでは、未宣言パラメータを含む`GetParameters`要求は、同時に指定した
宣言済みパラメータも含めて空の応答になります。そのためBridgeは、最初に
`ListParameters`で新実装専用の`reference_state_topic`の有無だけを調べ、旧実装には
宣言済みの`waypoint_file`、`navigation_mode`、`map_frame`だけを別要求します。
Followerより先に起動した場合も1秒周期で再試行し、サービス出現後に復旧します。

旧実装では、次の全条件を満たしたときだけ既存の
`/waypoint_follower/reference_state`型へ読み取り専用状態を出します。

- Followerの絶対`waypoint_file`と`navigation_mode`をROSパラメータから取得できる
- Nav2 Action statusにactiveなgoal UUIDが1つだけ存在する
- Action statusはROS 2 Action標準のtransient-local QoSで受信し、既知の同一goal
  UUIDを持つfeedbackだけをstatus heartbeat間の生存確認に利用できる
- goal変更後に届いた`/plan`終点が、選択ルートのWaypointへ0.40 m以内で一意に対応する
- 2番目のWaypoint候補との距離差が0.20 m以上ある
- `through_poses`では同じgoal UUIDのfeedbackから残pose数を確認できる

互換状態の`goal_generation`はNav2 goal UUIDの変更ごとに更新します。参考経路の
安全終端は、照合済みの次indexから最初の`stop`、`wait_time`、`person_area`までに
制限します。bridgeはAction client、Navigation goal、`cmd_vel` publisherを
持ちません。

statusがinactiveになった時点で`HOLDING_STOP`を出して緑線由来の方向・
`PLAN LEFT`・水色線を消します。status/plan/feedbackが古い、終点が曖昧、
ルート外、複数goalがactiveの場合もfail closedです。旧Followerはpauseと人物探索を
区別して公開しないため、画面表示は詳細理由ではなく`STOPPED`になります。また、
Nav2 goal UUIDの発行元そのものはAction statusだけでは証明できないため、終点が
選択ルートの一意なWaypointと一致する条件を追加しています。この条件を満たさない
走行では互換表示を推測で有効にしません。

feedbackはstatusで一度確認済みのgoalを新規作成したり、異なるUUIDのgoalを許可したり
しません。statusも同一UUIDのfeedbackも2.5秒を超えて古い場合は、従来どおり
`legacy_action_status_stale`で非表示にします。Waypoint切替後は新goalのstatusと、
そのgoal発行後の新しい`/plan`終点照合が完了するまで`UPDATING ROUTE`を維持します。
これは通信揺れによる偽HOLDを減らしつつ、旧経路を新goalへ流用しないためです。

次回走行では起動ログの`reference_path_configuration enabled=True`、
supervisorのmap/waypoint選択、`reference_path`の`valid=true`、Viewerの
`reference_projection ... reason=visible`を順に保存します。水色線が出ない場合は
`waiting_for_follower_state`、Keepout不一致、route/generation/index不一致、鮮度、
AMCL/TF、画角外をこの順で切り分けます。緑線の`REPLANNING`または
`NAVIGATION HOLD`は実走行Pathの表示保留であり、水色Pathを緑線へ代用する状態では
ありません。

旧Followerで正常に互換状態へ移行したことは、次のログで確認します。

- `legacy_follower_compatibility reason=legacy_follower_verified ...`
- `legacy_follower_compatibility reason=legacy_progress_verified`
- `reference_planner_auto_started map=... waypoint_file=...`

`follower_parameter_response_incomplete expected=3 actual=...`は必須3項目の応答数を、
`follower_parameter_service_unavailable`はFollower起動待ちまたは通信断を表します。
通信断時は最後に検証済みの状態を表示許可falseの`HOLDING_STOP`として再送し、古い
案内を復活させません。新しい起動スクリプトの`*.launch.log`にはBridge、Supervisor、
Viewerを含むlaunch全体の遷移が残ります。

Keepout有効地図では、マスク内容・geometry・FilterInfoに加え、共有する
`filter_mask_server`と`costmap_filter_info_server`のLifecycle `ACTIVE`を1秒周期で
確認します。明示的なnon-active応答は即時に参考経路を無効化します。ROS graphの
瞬間的なservice未検出や応答失敗だけは、直近3.5秒以内に取得したACTIVE応答を
上限付きで維持し、それを超えればfail closedに戻ります。次のログを順に確認します。

- `keepout_lifecycle_verified server=mask ... state_id=3`
- `keepout_lifecycle_verified server=info ... state_id=3`
- `reference_plan_success ...`
- Viewerの`reference_path_state ... valid=true`と`reason=visible`

`keepout_lifecycle_not_active`はサーバーから明示的にACTIVE以外が返った状態、
`keepout_lifecycle_transport_uncertain`はservice通信の不確実性です。両者を同じ
「inactive」として診断しないでください。

実機ではLauncher2/Nav2/Followerを従来手順で起動し、別ターミナルから次を実行します。

```bash
cd /home/koko-chan-nuc/inoueyuki_ws/nakanoshima-robot-challenge
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
source ar_ws/install/setup.bash
export ROS_DOMAIN_ID=54
export ROBOTBASE_TF_PREFIX=robot
export ROBOTBASE_ID=koko
export ROBOTBASE_LAUNCHER_ID=koko_launcher2
ros2 launch ar_reference_planner launcher2_reference_ar.launch.py \
  viewer_config_file:=/home/koko-chan-nuc/ar_navigation_project/config/ar_path_viewer_launcher2.yaml
```

### NUCデスクトップ起動用runner

ビルド後の次のrunnerは、上記launchを同じ環境変数で起動します。

```bash
/home/koko-chan-nuc/inoueyuki_ws/nakanoshima-robot-challenge/ar_ws/install/\
ar_reference_planner/lib/ar_reference_planner/run_kokochan_navi_launcher2
```

既存NUC YAMLを変更せず、一時overlayで同じタイムスタンプの`.mp4`、Viewer `.log`、
launch全体の`.launch.log`を`~/ar_navigation_project/outputs`へ生成します。一時YAMLは
終了時に削除します。AR 3ノードの重複、runnerの二重起動、C920を使用中の別プロセス
を検出した場合は、既存プロセスを終了せず起動を拒否します。Launcher2、Follower、
Nav2を起動・停止する処理や制御topic/actionは含みません。
ROS 2の`setup.bash`は未定義の任意`AMENT_*`変数を確認することがあるため、runnerは
setupをsourceする区間だけ`nounset`を解除し、直後に`set -u`を戻します。`set -e`と
`pipefail`、録画名、ロック、C920競合検査、一時設定の終了時削除は維持します。
q/Q/F10でViewerが終了した場合は、launchの`OnProcessExit`がBridgeとSupervisorも
終了させ、Supervisorは自身が起動した参考Plannerのprocess groupを停止します。
runnerは`ros2 launch`を前景で実行するため、SIGINTをAR子プロセスへ通常どおり伝え、
終了後に一時overlayを削除します。Viewerログには判定したraw `key_code`も残るため、
意図したキーとGUI backend由来の値を次回検証で区別できます。

本番YAMLを編集せず20/30 fpsを比較する場合は、runnerだけに次の環境変数を渡します。
比較モードは`camera_timestamped`を明示し、Viewer timer、V4L2の要求fps、MP4の固定fpsを
同じ値にします。通常起動では設定ファイルの値をそのまま使用します。

```bash
# 20 fps基準
KOKO_NAVI_FPS=20 KOKO_NAVI_PERCEPTION_REVIEW=1 \
  ar_ws/install/ar_reference_planner/lib/ar_reference_planner/\
run_kokochan_navi_launcher2

# 30 fps比較（20 fps試験終了後、同じ条件で別セッションとして実行）
KOKO_NAVI_FPS=30 KOKO_NAVI_PERCEPTION_REVIEW=1 \
  ar_ws/install/ar_reference_planner/lib/ar_reference_planner/\
run_kokochan_navi_launcher2
```

出力名には`_20fps`または`_30fps`が付きます。ログの
`camera_rate_configuration`で`requested_capture_fps`と`effective_capture_fps`、
`camera_pipeline`でSOE由来の`source_interval_ms`、HighGUI API呼出しまでの
`gui_new_interval_ms`/`display_age_ms`、間引き、GUI重複、録画補間数を比較します。
`display_age_ms`はモニターのscanout完了時刻ではありません。ドライバが要求fpsを
受理しない場合は`capture_rate_reason=negotiated_mismatch`等を警告し、30 fps達成とは
扱いません。120 msのexact-time TF待機は両モードで維持します。

同梱の`desktop/kokochan_navi_launcher2.desktop`は、インストール済みrunnerだけを
呼ぶテンプレートです。NUC反映時は既存Desktopファイルを時刻付きでバックアップし、
このテンプレートをコピーしてから内容を確認します。現在の実機ファイルを開発PCから
直接編集したり、Launcher2用ショートカットを置き換えたりはしません。

ビルド後は次のinstallerをユーザー権限で1回実行すると、現在のDesktop entryを
時刻付きでバックアップし、インストール済みrunnerの絶対パスを持つentryへ更新します。
`--restore`は最新バックアップへ戻します。Launcher2のdesktop entryには触れません。

```bash
source /opt/ros/jazzy/setup.bash
source robot_ws/install/setup.bash
source ar_ws/install/setup.bash
ar_ws/install/ar_reference_planner/lib/ar_reference_planner/\
install_kokochan_navi_desktop

# 復元
ar_ws/install/ar_reference_planner/lib/ar_reference_planner/\
install_kokochan_navi_desktop --restore
```

Costmap認識レイヤーは実機受入まで既定OFFです。静止レビュー時だけrunnerへ
`KOKO_NAVI_PERCEPTION_REVIEW=1`を与えると、明示的なreview overlayが追加されます。
通常のdesktop起動ではこの環境変数を設定しません。runnerはC920のby-idデバイスを
発見できない場合や別processが使用中の場合、cameraを開かず通知して終了します。

開発PCの保存動画・bagレビューはproduction domain 54を拒否する専用runnerを使います。
bagを指定した場合も観測topicだけを再生し、制御topic/actionを含めません。録画動画と
bagに共通時刻がない場合、画面とROSデータが同期した検証とは扱えません。

```bash
export KOKO_NAVI_REVIEW_DOMAIN_ID=154
ar_ws/install/ar_reference_planner/lib/ar_reference_planner/\
run_kokochan_navi_offline_review \
  --video /absolute/path/to/recording.mp4 \
  --bag /absolute/path/to/rosbag_directory
```

## Manual production-shaped launch (diagnostics only)

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
