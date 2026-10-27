# nakanoshima-robot-challenge

中之島ロボットチャレンジ用のAR表示、Camera V7人物探索、Robot側Waypoint連携を、
元の開発環境と本番環境を変更せず保存したPrivate repositoryです。

## 構成

```text
ar_ws/src/ar_path_viewer/        AR経路表示ROS 2 package
camera_v7/                       実行snapshotと開発候補を分離したCamera V7
robot_ws/src/                    Waypoint、BT node、bringup package
robot_ws/{params,bash,rviz}/     関連設定と起動補助
robot_ws/maps_waypoints/         選定したmapとWaypoint YAML
models/                          非Git modelの取得・照合方針
docs/                            出典、setup、運用、安全、license
```

元ソース、選定理由、SHA256は[Source inventory](docs/SOURCE_INVENTORY.md)と
`docs/ORIGIN_MANIFEST.tsv`に記録します。

## Camera V7の版区分

- `snapshots/runtime_20260919`: 監査時にruntime側から採取した小規模snapshot
- `snapshots/production_20260928`: 2026-09-28のproduction snapshot
- `development_20261019`: 2026-10-19の開発候補。通常launcherは
  `daikoudai_v7.py`、dev launcherだけが`daikoudai_v7.py.new`を起動

これらは由来を表すsnapshot名です。実機NUCで現在稼働中の版を意味しません。
実機の実行版はREAD-ONLY確認後、hashが一致したcommit/tagを選択してください。

## 開発開始

- 開発PC: [SETUP_DEVELOPMENT_PC.md](docs/SETUP_DEVELOPMENT_PC.md)
- 実機NUC: [SETUP_NUC.md](docs/SETUP_NUC.md)
- Camera V7: [camera_v7/README.md](camera_v7/README.md)
- Robot連携: [robot_ws/README.md](robot_ws/README.md)

実機ではtest済みcommitまたはtagを独立workspaceへcloneします。既存Robot本番
workspaceを`git pull`で更新しません。ROS接続、camera open、走行、service callは
手順確認と現場許可を分けて実施してください。

## 公開・artifact方針

このrepositoryはPrivate専用です。YOLO/Gemma model、録画、ROS bag、build結果、
venv、認証情報は含めません。第三者assetの再配布条件と施設・経路情報を再監査
するまではPublicにしないでください。
