# Camera V7 Deployment Handoff

作成日: 2026-10-19  
状態: **E2E待ち / production未反映 / DEPLOY NOW: NO**

## 1. 監査範囲と制約

このhandoffは開発PC上のREAD-ONLY監査、既存tests再実行、deployment差分整理の結果です。
Robot NUCへの接続・scp・rsync、production同期、Camera/LM Studio/ROS node起動、git mutationは
行っていません。既存コードとtestsは変更せず、新規Markdown 2本だけを作成しました。

## 2. Current development hashes

| ファイル | SHA256 | 指定値との照合 |
|---|---|---|
| `camera_video_recording.py` | `b5f1e41e86890c796b1eb1c26ade77accc1e6ddb6d1c7e65a69a74a9eb1f48b8` | 一致 |
| `daikoudai_v7.py.new` | `720a2968563cf58b0a972f9034dc10747b430582ef12c6f9af2b41c7cfaf2ee9` | 一致 |
| `person_search_vlm_milestone2_v7.py` | `c5061b82ce2a8cfcc11b5b2bc51faeb3b19b1d0e9591349829e50df7ae675d89` | 一致 |
| 通常版`daikoudai_v7.py` | `252cdee857e50932cddbcf0ea990d6e1a9ef7af1ad0bf34f7c1ec9d5bbcdd0a5` | 一致・未変更 |
| `person_search_vlm_v3_9_7_final_wait_reduce_v3.py` | `7f49b5b2c3af3f1649ffc92816d6399ce5050e3750f28b93007acc9fd611ba5b` | **指定の「認識core 6258…」と不一致** |
| `run_daikoudai_v7_ui.sh` | `b4401c9b46923e87f829aa7475eb979a447726d9aabfdd465f6cb70df882cc9c` | 参考 |
| `v7_startup_preflight.py` | `8d87f1b88cd34fcff43d572efa2259e8af93a04a4d751d33f97896b01a202f4f` | 参考 |
| `sounds/zundamon_found.wav` | `b2db4c7d09011e7cdd9ff11a3e21a9213123121a836e6e91a9140a2025778018` | 実在 |
| `sounds/zundamon_not_found.wav` | `d451ba79cfb3cdf782671ef9ac119e3b5b4ff30d6d9e5ba2ab070addb7570c3c` | 実在 |

不一致の`6258c207...`は、
`backups/continuous_camera_20261019_170512/person_search_vlm_milestone2_v7.py`のhashと一致します。
つまり依頼文中の「認識core」というラベルと実ファイルの対応が一致していません。
本監査では修正せず、coreはCONDITIONAL扱いとします。

## 3. Local production/normal snapshots

### `/home/inoue-yuki/nuc_audit_20260919/camera_v7`

| ファイル | SHA256 | current devとの比較 |
|---|---|---|
| `daikoudai_v7.py` | `252cdee857e50932cddbcf0ea990d6e1a9ef7af1ad0bf34f7c1ec9d5bbcdd0a5` | `daikoudai_v7.py.new`とDIFF |
| `person_search_vlm_milestone2_v7.py` | `59de1a100c467c5bad82b503df03cd9da647646dd9ba9dc7186f9df90d1138e1` | DIFF |
| `run_daikoudai_v7_ui.sh` | `c9051353c512971ddd2231585e7ba5a8289226c0227d177db245da0a08e7f54e` | DIFF |

このsnapshotには`camera_video_recording.py`、`v7_startup_preflight.py`、VOICEVOX sounds、
認識coreは含まれていません。

### `/home/inoue-yuki/nuc_audit_20260919/camera_v7_production_20260928`

| ファイル | SHA256 | current devとの比較 |
|---|---|---|
| `camera_video_recording.py` | `5bc574be66cf241df18814cbfc9bef7c27973600ebd4a6dbeaca20fa4ca29266` | DIFF |
| `daikoudai_v7.py` | `a48fb8db80beede83865e3e874917f6dd5356bc075f5983e05d6a896b3031b86` | DIFF |
| `person_search_vlm_milestone2_v7.py` | `fb10e3979e9309e4c20511c67f193dd76a3e545935cf265e02165acf415d6d84` | DIFF |
| `run_daikoudai_v7_ui.sh` | `5ebb6fbb60afaac89ba9d68384310c309bb060ff19a36bff5d7914372fe53480` | DIFF |
| `v7_startup_preflight.py` | `45c275e1fec152eaabc9e1f36fe4a93256f2bf8079feda7782ed92a0217ed4a0` | DIFF |

これはローカルsnapshot比較であり、現在のRobot/Camera NUC実体を示すものではありません。
実機接続は禁止されているため、deployment前に実機側hashを別途取得する必要があります。

### Route/Domainのローカル確認範囲

- Renya snapshotの`robot.env`では`ROBOTBASE_ROS_DOMAIN_ID="54"`を確認した。
- 開発PCには実機の`~/bin/person_area`が無く、その参照先変更は直接確認できなかった。
- Renya snapshot内には予定route `1014_15_True.yaml` / `1014-15_True.yaml`が見つからなかった。

したがってroute名、WP3/WP7、`stop:false`、`wait_time:0`は提供された本番予定値として
手順書へ記載しましたが、実機開始前に実ファイルで確認する必要があります。

## 4. Final READ-ONLY audit

### A. VOICEVOX

- terminal `detail.status`を使用し、UIログ文字列では判定しない。
- `FOUND_FRONT_PHOTO`は`zundamon_found.wav`。
- `NOT_FOUND_OR_NOT_FRONT`は`zundamon_not_found.wav`。
- `ERROR` / `ERROR_*`を含むその他statusは無音。
- terminal detail/done publish成功後に`notification_status`を確定する。
- notification callはterminal lockの外。
- `/usr/bin/cvlc`を`subprocess.Popen`で非同期起動し、stdout/stderrは`DEVNULL`。
- 音源/player欠如やPopen例外はwarningだけで、結果を変更しない。
- `safe_to_advance`とdone値はnotificationより前に確定・publishされる。
- 音源2本はPCM 16-bit mono 24000 Hz WAVEとして実在。

判定: **コード・unit test上PASS、実スピーカーE2E待ち**。

### B. Continuous mission recording

- 録画ON + live Cameraでは`DriveRecorder`がpersistent ownerになる。
- capture openerはserver lifetime中1回で、選択したphysical Captureを保持する。
- `mission_<timestamp>.mp4`、`kind=mission`のsegmentを1本作る。
- producerはraw frameをmission writerへ書いた後、sequence付きlatest frameを公開する。
- shared readerは新sequenceだけを待ち、consumerにはcopyを返す。
- reader releaseはdetachだけでphysical Captureをreleaseしない。
- search境界ではdrive previewだけを切り替え、producer/writerは停止しない。
- VLM consumerが遅くてもunbounded queueを作らずproducerをblockしない。
- 正常時はserver shutdownでmission writer close、physical Capture release。
- Camera open/read failure時はfail-closed cleanupし、waiting readerをwakeする。
- writer failureだけでは人物探索を自動ERRORにしない。
- 録画OFFはMS2 direct Camera open/release経路を維持。
- 動画入力は`cv2.VideoCapture(video_path)`を維持。
- search動画はshared frameから生成できるが、latest-frame方式のため全実時間frame保証はない。

判定: **unit test PASS、探索を挟む実C920 E2E待ち**。

### C. Recognition contract

- exact modelは`gemma-4-e2b-it-qat`。
- continuous recording導入前backupからMS2の変更はoptional external capture経路だけ。
- prompt/scoring/thresholdの意図しない変更はdiff上なし。
- prompt hash保護test PASS。
- area winnerは`ms2_success=true`の結果だけをFOUNDとして扱う。
- capture_best保存失敗は`CAPTURE_BEST_SAVE_FAILED`、`status=ERROR`、`ms2_success=false`。
- Target stateはA/B独立。
- `AREA_TARGET_ID_BY_INDEX = {1: target_A, 2: target_B}`。

判定: **PASS**。ただし認識coreの依頼記載hash不一致は未解決。

### D. ROS contract

| interface | 契約 |
|---|---|
| `/person_area_arrived` | `Bool`, RELIABLE, TRANSIENT_LOCAL, depth 1 |
| `/person_search/arrival_received` | `Bool`, RELIABLE, VOLATILE, depth 10 |
| `/person_search_done` | `Bool`, depth 10 |
| `/person_search_done_detail` | `String`, depth 10 |

- true callback受付時にACKをpublishする既存契約を維持。
- duplicate/stale trueは新workerを開始しない。
- detailの後にdoneをpublishする。
- done値は`safe_to_advance`と同じ。
- `ERROR`は`safe_to_advance=false`。
- VOICEVOX/continuous recording変更でtopic名、QoS、ACK、done契約を変更していない。

判定: **PASS**。

## 5. Tests executed

| command | result |
|---|---|
| `python3 -m py_compile daikoudai_v7.py.new camera_video_recording.py person_search_vlm_milestone2_v7.py person_search_vlm_v3_9_7_final_wait_reduce_v3.py v7_startup_preflight.py` | PASS |
| `python3 test_v7_voice_notification.py` | 6/6 PASS |
| `python3 test_v7_continuous_camera.py` | 8/8 PASS |
| `python3 test_v7_arrival_observability.py` | 10/10 PASS |
| `python3 test_v7_ms2_success_policy.py` | 9/9 PASS |
| `python3 test_v7_capture_best.py` | 4/4 PASS |
| `python3 -m unittest discover` | **213/213 PASS** |

`arrival_observability`実行時にmock subprocessの`ResourceWarning`が出ましたが、FAIL/ERRORは
ありません。tests中に実Camera、LM Studio、network、Robot、ROS nodeは起動していません。

## 6. Completed smoke tests

実在確認済み:

```text
outputs/mission_20261019_172819/camera_video/mission_20261019_172819.mp4
outputs/mission_20261019_172819/camera_video/camera_video_manifest.json
```

manifest実測:

- `kind=mission`
- `status=completed`
- `error=""`
- 272 frames
- 640×480
- requested 15 fps
- 2026-10-19 17:28:19 → 17:28:41
- MP4 size 1,221,095 bytes

これは「実C920・探索なし・mission MP4 1本」のPASS証拠です。開発PCに`ffprobe`が無いため、
MP4 container metadataの追加検査は未実施です。

## 7. Deployment classification

### DEPLOY候補（E2E PASS後のみ）

| development source | production destination | 理由 |
|---|---|---|
| `daikoudai_v7.py.new` | `daikoudai_v7.py` | 開発UI、VOICEVOX、persistent Camera連携。開発名のまま置かない |
| `camera_video_recording.py` | 同名 | persistent owner、mission writer、shared reader |
| `person_search_vlm_milestone2_v7.py` | 同名 | optional external/shared capture対応 |
| `sounds/zundamon_found.wav` | `sounds/zundamon_found.wav` | FOUND通知 |
| `sounds/zundamon_not_found.wav` | `sounds/zundamon_not_found.wav` | NOT_FOUND通知 |

### CONDITIONAL

| ファイル | 条件 |
|---|---|
| `run_daikoudai_v7_ui.sh` | 実機hashと比較し、Renya `robot.env` domain参照・dependency preflightを持たない場合だけ、別テーマの承認済み差分として反映 |
| `v7_startup_preflight.py` | 実機hashと比較し、AppImage cold-start/daemon readiness/exact model collision対応が未反映の場合だけ反映 |
| `person_search_vlm_v3_9_7_final_wait_reduce_v3.py` | 実機依存fileの存在/hashを確認し、正しいexpected hashを確定してから判断。現時点ではdeploy不可 |

launcher/preflight/coreは今回変更していません。production依存上必要でも、今回の5ファイルと
無条件に混ぜてはいけません。

### DEPLOY不要

- `test_*.py`
- `outputs/`
- `backups/`
- `venv/`, `.venv/`
- `__pycache__/`, `*.pyc`
- UI log、runtime log
- rosbag
- mission/search動画
- target画像
- `.daikoudai_v7_target_ui_last.json`などruntime config
- 新規手順書・handoff（運用資料として別保管可。runtimeには不要）
- workspace全体

## 8. Backup locations

- continuous Camera導入前: `backups/continuous_camera_20261019_170512/`
- VOICEVOX導入前: `backups/voicevox_notification_20261019_163653/`
- UI/cold-start作業前: `backups/ui_cold_start_20260930_213946/`

production反映前には、実機側でも対象ファイルとsoundsを同一timestamp directoryへ保存し、
hash一覧を残してください。既存backupを上書きしません。

## 9. Remaining E2E

1. LM Link復旧後、実人物探索FOUNDから実スピーカーで「見つけたよー！」。
2. 実人物探索NOT_FOUNDから実スピーカーで「見つからないよー！」。
3. 実C920 + person_area探索中も同じmission MP4への記録が継続。
4. Area 1 + Area 2探索後もmission MP4が1本だけ。
5. production反映後、WP3 Target A / WP7 Target Bの実走行E2E。

## 10. Exact selective deployment plan

この計画は上記1～4の開発PC/実Camera E2EがPASSした後にだけ実施します。

1. 実機側の対象5ファイル、CONDITIONAL 3ファイル、dependency fileのhashを取得する。
2. route、person_area数、ROS domain、exact model、C920、disk空き容量を確認する。
3. 実機側対象fileをtimestamp付きbackupへコピーし、backup hashを保存する。
4. DEPLOY候補5ファイルだけをstaging directoryへ転送する。
5. staging hashをこのhandoffのdevelopment hashと照合する。
6. `daikoudai_v7.py.new`だけはproduction destinationで`daikoudai_v7.py`として配置する。
7. runtime owner/permissionとsounds directoryを確認する。
8. 実機で`py_compile`とdependency-only preflightを行う。Robotはまだ起動しない。
9. Camera-only smoke、LM exact model/chat、VOICEVOX、continuous recordingの順で確認する。
10. 最後にCamera/Robotをfresh起動し、`start_index=1`からWP3/WP7実走行E2Eを行う。

workspace丸ごとのコピー、outputs/backups/tests/venvの転送は禁止です。

## 11. Rollback policy

- rollbackはdeployment直前に作成した実機timestamp backupだけを使用する。
- 5ファイルの一部だけを旧版へ戻す混在rollbackは禁止する。
- rollback後はhashをbackup記録と照合する。
- Camera/Robotを両方停止し、両方fresh起動する。
- `start_index=1`から再確認する。
- failure evidence（Camera log、Robot log、manifest、result JSON）を削除しない。

## 12. 「まだdeployしてはいけない」条件

次のいずれか1つでも該当すればDEPLOY NOWはNOです。

- Remaining E2E 1～4が未完了またはFAIL
- current dev hashが本handoffから変化
- 認識coreの正しいexpected hashが未確定
- 実機側hash/backup未取得
- route YAMLまたはWP3/WP7設定未確認
- Robot/CameraのROS domain不一致
- exact model/chat疎通未確認
- LM Linkを使うのにpeer未確認
- C920 ownershipまたは録画保存先に問題
- unit test/py_compileにFAIL/ERROR

現在は複数条件が未完了なので、**DEPLOY NOW: NO**です。
