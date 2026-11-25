# 人物探索 Camera V8 — AR映像共用・探索時のみ・録画なし（開発版）

## 目的（2026-11-23）

- Camera V7と`person_search_vlm_milestone2_v7.py`は**一切変更しない**。
- V8のUIで「実走行の動画保存」ON/OFFを廃止。V8にmission/search動画録画を実装しない。
- C920の取得と映像録画を**ARナビだけ**が担当。V8は`/dev/video*`を開かない。
- V8はサーバーを先に立ち上げて待機し、既存の`/person_area_arrived`の**false→true**でだけ人物探索を実行。
- AR側の**別ノード**が、探索イベント中のみ未加工のBGR8元画像を`/koko/person_search/raw_image`へ流す。V8はこのTopicから読み込む。
- 人物探索エンジンの`run_milestone2_search`には既存の`external_capture`引数だけを渡す。**認識・マッチング・閾値・モデル・探索結果処理に変更なし**。
- V8の`/person_search_done`と`/person_search_done_detail`、到着ACKは従来の安全契約を維持。
- ARは人物探索中も画面表示・録画を続ける。ARのNav2・Follower・motor制御には一切変更しない。
- 人物特定の**静止画像証拠とJSONログは保持**。撤廃するのはCamera V8の**動画録画**。

## 構成

```
C920（ARのみが所有）
  └─ AR Viewer（表示・経路重畳・AR録画を継続）
        └─ 別ROS Node: koko_ar_v8_camera_frame_share
             ↓ /person_area_arrived=true で開始
             ↓ /koko/person_search/raw_image (sensor_msgs/Image, BGR8, 生映像)
             ↓ 走行中・非探索中は配信しない
              Camera V8 server（通常は待機）
                └─ 既存V7人物照合エンジン（external_capture）
                      ↓ /person_search_done_detail, /person_search_done
             ↑ terminal結果で送信停止
```

### 安全方針

- V8を起動してもカメラ機器は開かない。ARが起動・接続していない状態で探索イベントを受け取ると、映像待受8秒後にエラーとして**安全側で失敗**（`safe_to_advance=false`）。
- ROS画像は最大640×480程度を想定。購読は深さ1・最新画像のみで無制限キューなし。
- AR画像送信失敗でもAR本体の表示・録画・Nav2動作を止めない。ただし探索は失敗扱いになり得る。
- 同一イベント再送・忙しい間の再入、終了時のfail-closed処理はV7由来の状態管理を保持する。
- V8が起動していない場合もARは通常通り動く。AR側V8送信は環境変数`KOKO_NAVI_V8_SHARE=1`で明示的に有効化するまで無効。
- V8とV7の人物探索サーバーを**同時起動しない**（既存のトピック・ノード名が同一）。

## 開発PCへの適用

GitHub `main`に上書きせず、必ずfeatureブランチを作る。

```bash
cd ~/care_patrol_ws/local_repositories/nakanoshima-robot-challenge
git status --short
# ?? で表示される未追跡build/log等は消さない。追跡対象の変更がないことを確認
git switch feature/camera-v8-ar-search-only-20261123  # 既存ブランチの場合

# ZIPを展開した場所から（パスは変更）
python3 /path/to/koko_v8_dev_bundle_20261123/apply_v8_to_dev_repo.py \
  --repo "$PWD" --check

python3 /path/to/koko_v8_dev_bundle_20261123/apply_v8_to_dev_repo.py \
  --repo "$PWD" --apply

python3 -m py_compile \
  camera_v8/development_20261123/daikoudai_v8.py \
  camera_v8/development_20261123/v8_shared_camera.py \
  camera_v8/development_20261123/camera_preview.py \
  ar_ws/src/ar_path_viewer/ar_path_viewer/viewer_node.py \
  camera_v8/development_20261123/v8_person_frame_share.py

git diff --check
```

**注意**: V8生成時の`--check`は構文チェックであり、ROS実機E2E合格を示さない。必ず開発PCで既存ARテストを実行し、統合テストを追加してから実機へ持ち込む。

## 実機NUCでの統合検証（無走行が先）

1. 開発PCのfeatureブランチでレビュー・単体テスト・依存関係検査を済ませる。
2. GitHubへのpush後、NUCの**別テスト用配置**に同branchをcheckoutしてビルドする。本番ワークスペース・V7・Renyaは上書きしない。
3. まず保存映像/シミュレーションでAR生映像配信の再現試験をする。
4. V8起動時にC920がV8から開かれていないことを `fuser` などで確認する。
5. ARの起動時に `KOKO_NAVI_V8_SHARE=1` を付与するか、検証用 `camera_v8/development_20261123/run_ar_with_v8_share.sh` を使用する（AR自体の映像取得は今まで通り）。V8 UIを別プロセスで起動。
6. `/person_area_arrived=false→true`イベント発火時のみ`/koko/person_search/raw_image`画像が流れることを停止状態で確認。`false`時および`person_search_done_detail.terminal=true`のあと送信が止まることを確認。
7. V8が異常終了・画像を受信できない・モデル応答不可のとき、**誤って`safe_to_advance=true`を送らない**ことを確認。
8. ARのMP4録画が探索中も継続し、V8がMP4ファイルを生成しないことを確認。
9. NUC負荷・カメラ遅延・画像欠落・人物探索精度・AR同期と録画・音声・Follower ACK/DoneのE2Eを確認してから走行へ進む。

## 未完成/注意

- **実機の同時動作試験は未実施**。このバンドルは開発ソース作成用で、本番即投入用ではない。
- V8はV7のMS2エンジンを外部依存として参照する（*エンジン本体を改変しない*）。認識エンジンを変更したくない場合は当該ファイルのハッシュを固定する。
- カメラハードウェアタイムスタンプは現状のROS Imageヘッダには保証されない。V8の探索用であり、ARのCostmap安全判定へ流用しない。
- UIの「手動発火」は**テスト用**。本番ではFollowerから既存のperson_areaイベントを受け取る。
- GUIの古い「写真だけ撮影」内部ヘルパーはV7由来でソース内に残るが、V8のUIからは呼び出せない。後日完全整理予定。
- Desktopアイコンのインストールは、実機NUCの検証用クローンから `install_v8_desktop.sh` を明示実行したときのみ。既存のV7.desktopを上書きしない。
## AR/V8 pipeline diagnostics (next field run)

The diagnostics are scalar counters/timings inside the existing camera owner,
AR subscriber and V8 subscriber. They do **not** add an image subscriber and
do not retain image pixels. Every five seconds the logs contain:

- `camera_owner_pipeline`: C920 handoff, AR/V8 publish rates, serialization,
  owner CPU/RSS;
- `camera_pipeline`: AR ROS callback/copy, Viewer tick/render, GUI-new-frame
  interval, capture-to-display age, recorder and AR process CPU/RSS;
- `v8_image_pipeline`: V8 receive/overwrite/consume rates, YOLO/VLM calls and
  durations, V8 process CPU/RSS.

`camera_fps_bookkeeping=15.0` in the V8 line is explicitly **not** a measured
input rate. V8 continues to consume the latest shared frames and the unchanged
V7 engine invokes YOLO according to `process_every=5`. The measured
`receive_fps`, `consume_fps`, and `yolo_fps` show the actual rates. This change
does not alter V8 resolution, publication rate, recognition thresholds, or
person-search completion behavior.

After AR, camera owner, and the V8 person-search server are running, start the
read-only collector in a fourth terminal:

```bash
cd ~/v8_camera_eval/nakanoshima-robot-challenge
export ROS_DOMAIN_ID=54
source /opt/ros/jazzy/setup.bash
source ~/robotbase_ws_renya/install/setup.bash
source robot_ws/install/setup.bash
source ar_ws/install/setup.bash
bash camera_v8/development_20261123/collect_ar_v8_diagnostics.sh
```

The collector requires exactly one PID for each real command line used by the
current launchers: `v8_camera_source.py`, the installed `ar_path_viewer`
console script, and `daikoudai_v8.py --person-search-server`. It also verifies
the corresponding ROS nodes (`/koko_v8_camera_source`, `/ar_path_viewer`, and
`/camera_all_in_one_person_search`) and resolves the three persistent logs. It
aborts with the candidates/reason on zero, duplicate, or incomplete startup.
The V8 server is expected to exist while waiting for a person-search event;
start the preserved V8 UI server at least five seconds before the collector.

`pidstat -H -t -w` records epoch time, process/thread CPU and RSS, plus
voluntary/involuntary context switches. The MCAP intentionally
records only low-bandwidth state/TF/path/costmap/person-search topics; both raw
image topics are excluded so diagnosis does not duplicate camera traffic.

Run one route containing these three segments, without restarting the
processes: normal navigation, active person search, and navigation after the
terminal search result. Press Ctrl+C only after the post-search segment. The
collector writes the exact source-log paths to `process_manifest.txt` and an
escaped, ready-to-run `analyze_this_run.sh`; it does not copy a still-open log
and thereby lose its tail. The generated command also bounds analysis to the
collector start/end epoch, so an earlier V8-server run in the same UI log is
not mixed into this run. Run:

```bash
<diagnostic-dir>/analyze_this_run.sh
```

The analyzer accepts rclpy epoch lines and the Viewer's ISO-8601 session lines.
It uses the owner's exact enable/stop transition lines when present, with the
five-second `v8_active` sample only as a fallback. It prints `INSUFFICIENT DATA`
instead of guessing when a process, rate field, or any of normal/search/post
segments is absent. It separately reports bounded versus accumulating display
latency, so low FPS is not automatically mislabeled as stale video.
Its first branch is selected as follows:

1. owner AR publish below 27 fps: camera owner/capture/serialization;
2. owner healthy but AR receive low: owner-to-AR DDS/callback;
3. AR receive healthy but GUI new-frame rate low: Viewer/render/GUI;
4. owner healthy but V8 receive low: DDS/V8 callback contention;
5. V8 receive healthy but consumption low: YOLO/VLM consumption contention;
6. degradation limited to active search: system contention;
7. increasing or high capture-to-display age: latency/jitter before FPS.

The exact five-second rows and `pidstat.log` remain the source evidence; the
automatic branch is a triage result, not a claim about the robot.

For a hardware-free instrumentation overhead comparison:

```bash
source ar_ws/install/setup.bash
python3 ar_ws/src/ar_path_viewer/scripts/benchmark_pipeline_metrics.py
```

To return to the pre-diagnostic runtime without reverting source, pass
`--no-pipeline-diagnostics` to `v8_camera_source.py` and
`--no-pipeline_diagnostics` to the V8 server. This disables only scalar
collection; no navigation or recognition setting changes. In an emergency,
use the previously built/committed revision through the normal Git deployment
workflow—never patch the NUC working tree directly.
