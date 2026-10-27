# Local validation — 2026-10-27

対象はこのrepositoryへコピーしたfileです。元のAR、Camera、Robot directoryは
READ-ONLYのままです。camera open、ROS publish/service call、実機走行は行っていません。

## Source同一性

- originからのcopy: 189 files
- `tools/generate_manifests.py`によるsource/destination SHA256: 全件一致
- repository checksum: `docs/SOURCE_MANIFEST.sha256`で照合可能

## ROS 2 Jazzy

- isolated `/tmp` build: 4 packages PASS
- `ar_path_viewer`: 51 tests PASS
- `robotbase_waypoint`: 77 tests PASS、既知のfork warning 8件
- `robotbase_bt_nodes`: 3 assertionsを含むCTest 1件 PASS
- `robotbase_bringup`: build PASS、testなし
- colcon集計: 132 tests、0 errors、0 failures、0 skipped

## Camera V7

- 主要6 Python files: `py_compile` PASS
- 参照Camera venvのPython 3.12.3で`test_v7_*.py`: 213 tests PASS
- 初回はtestだけを`tests/`へ移したため相対path参照が壊れ88 errorsとなった。
  元sourceと同じ階層へ戻した後に全213件PASSを確認した。これは配置修正であり、
  Camera/Robot本体sourceは変更していない。

## 未検証

- 現在の実機NUCで実行中のCamera V7、Waypoint Follower、routeとのhash一致
- C920実映像、Gemma実model、実ROS graphとの統合
- 実機走行、`person_area`到着、物理的な停止・安全動作
