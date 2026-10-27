# Camera V7

## 実行版と開発版

`development_20261019/run_daikoudai_v7_ui.sh`の対象は`daikoudai_v7.py`です。
`run_daikoudai_v7_ui_dev.sh`の対象だけが`daikoudai_v7.py.new`です。`.new`を
実行版へ自動昇格させません。付属handoffもE2E確認前のdeployを認めていません。

`snapshots/`は過去監査snapshotです。現在のNUC実行版は、process command line、
launcher、対象fileのSHA256をREAD-ONLYで照合して確定します。

## 主な連携

- C920をCamera V7が直接openする
- YOLO person検出とbest frame選択を行う
- Gemma互換local APIは`127.0.0.1:1234`を使用する
- MS2はWaypoint側`person_area`到着eventを受けて探索を開始する
- model identifierは`gemma-4-e2b-it-qat`

C920をARまたは別processと同時openしないでください。共有が必要な場合は単一所有者
からROS image topic等へ配信する設計を別途検証します。

通常launcherは既定で`~/robotbase_ws_renya/robot.env`をREAD-ONLYに読みます。
独立配置では[robot.env.example](robot.env.example)をGit対象外の`robot.env`へコピーし、
実機で確認した数値のdomain IDを設定して`ROBOT_ENV=/absolute/path/robot.env`を渡します。
値をrepositoryへcommitしません。

## Python

Camera V7専用venvを使用し、ROS Pythonを必要とする場合は
`python3 -m venv --system-site-packages venv`で作成します。参照環境では
Ultralytics 8.4.47、OpenCV 4.13.0.92、Torch 2.11.0が確認されています。
固定requirementsは[requirements.txt](requirements.txt)を参照してください。

起動前に`v7_startup_preflight.py --dependencies-only`、保存入力でのtest、
C920占有processがないこと、`ROS_DOMAIN_ID`を確認します。実機での起動は
[NUC手順](../docs/SETUP_NUC.md)に従います。
