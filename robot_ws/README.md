# Robot workspace selection

参照元`renya_20261025`から、中之島で使用するWaypoint連携とその起動依存を
コピーしたsnapshotです。元のRobot workspaceは変更していません。

- `robotbase_waypoint`: Waypoint YAML読込、Nav2 goal、`person_area`連携
- `robotbase_bt_nodes`: ordered waypoint passage用BT node
- `robotbase_bringup`: Robot/Nav2 bringup資産
- `params/real`, `bash`, `rviz`: 関連設定と起動補助
- `maps_waypoints`: backupを除くWaypoint YAMLと選定map

`1014-15_True.yaml`には`person_area: true`のWaypointが含まれます。実機で使う
route、map、`navigation_mode`は起動processとparameterをREAD-ONLYで確認し、
repository上の候補から推測しないでください。

開発PCではROS 2 Jazzyをsourceして別のbuild/install/logへbuildします。実機では
既存Robot本番workspaceへ上書きせず、まずclone先でbuildと静的確認を行います。
走行やROS service/actionへの要求は、現場の安全手順と個別許可を得た試験だけで
実施してください。
