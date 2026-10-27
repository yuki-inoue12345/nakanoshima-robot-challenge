# Source inventory

調査日: 2026-10-27

## Repositoryへの割当

| 機能 | Git内の配置 | READ-ONLY参照元 | 選定 |
|---|---|---|---|
| AR経路表示 | `ar_ws/src/ar_path_viewer` | `/home/inoue-yuki/ar_navigation_project/ar_path_viewer` | 実際のAR package一式。build/install/log/outputsを除外 |
| Camera V7 runtime snapshot | `camera_v7/snapshots/runtime_20260919` | `/home/inoue-yuki/nuc_audit_20260919/camera_v7` | 監査時runtime比較用の3 file |
| Camera V7 production snapshot | `camera_v7/snapshots/production_20260928` | `/home/inoue-yuki/nuc_audit_20260919/camera_v7_production_20260928` | production比較用の5 file |
| Camera V7 development | `camera_v7/development_20261019` | `/home/inoue-yuki/nuc_audit_20260919/camera_v7_fix` | UI、MS2、認識core、launcher、preflight、test、音声asset |
| Robot連携 | `robot_ws` | `/home/inoue-yuki/robot_archives/renya_20261025` | Waypoint、BT、bringup、設定、起動script、route/map |

各copyのsource pathとSHA256は`ORIGIN_MANIFEST.tsv`に記録し、Git内全fileのhashは
`SOURCE_MANIFEST.sha256`に記録します。

## 重要な版識別

- `development_20261019/run_daikoudai_v7_ui.sh`は`daikoudai_v7.py`を起動する。
- `run_daikoudai_v7_ui_dev.sh`は`daikoudai_v7.py.new`を起動する。
- `.new`は開発候補であり、production実行版として扱わない。
- archive/snapshot名は保存時点の由来であり、現在の実機processを保証しない。
- Robot route `1014-15_True.yaml`は候補であり、実機active routeは未確認。

## 除外

- 元workspaceのbuild/install/log、venv、cache、outputs
- backup、editor一時file、core dump
- runtime生成の`.daikoudai_v7_target_ui_last.json`（対象画像のlocal pathを含み得る）
- 実機固有の`robot.env`（安全なtemplateだけを収録）
- YOLO/Gemma model binary
- 録画、ROS bag、実験時画像
- 認証情報とlocal environment file
- 卒研専用`care_patrol` package（別repository `care-patrol-thesis`）

## Copy漏れの判断

AR packageはsource/config/launch/test/test_dataを一式収録しています。Camera V7は
launcherから直接参照されるPython、preflight、MS2、認識core、音声asset、testを
収録しています。Robot側は3 packageと実環境の関連設定・起動補助・Waypoint/mapを
収録しています。元workspace全体の複製ではなく、機能境界に必要なfileを選定して
います。実機固有のudev、OS service、外部driver、LM Studio/modelはsetup依存です。

## 卒研repositoryとの共通由来

`care-patrol-thesis`はCamera V7のYOLO入力・person class判定の考え方とRobot側
`set_paused` interfaceを参照しますが、このrepositoryのCamera/Robot sourceを
直接importしません。卒研側の詳細なsource hashは同repositoryの
`docs/source_manifest.md`で固定します。
