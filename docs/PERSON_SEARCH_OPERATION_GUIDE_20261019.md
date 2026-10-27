# 人物探索 Camera V7 本番運用手順書

作成日: 2026-10-19  
対象: ココちゃんランチャー2 / Renya workspace / Camera V7

> この版はproduction未反映です。pre-deployment E2EをPASSし、
> `DEPLOYMENT_HANDOFF_20261019.md`のgateを満たすまでは本番投入しないでください。

## 1. 本番構成

- Robot workspace: `/home/koko-chan-nuc/robotbase_ws_renya`
- Camera UI production名: `daikoudai_v7.py`
- exact LM model: `gemma-4-e2b-it-qat`
- person area数: 2
- `start_index=1`
- `loop=false`
- `navigation_only=false`
- CameraとRobotは毎回fresh起動する

RobotとCameraは同じ`ROS_DOMAIN_ID`でなければ通信できません。Camera launcherは
Robot側`robot.env`の`ROBOTBASE_ROS_DOMAIN_ID`を正本として読み取る構成です。
本番直前に両側の値が一致していることを確認してください。値が欠落・不正な場合は
fallbackせず中止します。

## 2. person_area route設定

実機の`~/bin/person_area`は次のdirectoryを見る運用です。

```text
/home/koko-chan-nuc/robotbase_ws_renya/maps_waypoints/waypoints
```

新規routeもこの一覧から選択します。本番予定値は次のとおりです。

| 項目 | 値 |
|---|---|
| map | `1014_15_True.yaml` |
| waypoint | `1014-15_True.yaml` |
| person_area | `WP3`, `WP7` |
| WP3 | Target A |
| WP7 | Target B |
| person_area stop | `false` |
| wait_time | `0` |

開始前に実機のroute一覧とYAML実体を確認してください。ファイルが無い、WP番号や
person_area数が違う、`stop`や`wait_time`が違う場合は走行を開始しません。

## 3. LM Studio / LM Link

使用モデルは必ず次のexact modelです。

```text
gemma-4-e2b-it-qat
```

別モデルへのfallback、新規download、自動installは行いません。LM Link利用時は、
開発PCで次を実行してpeerが見えていることを確認します。

```bash
lms link status
```

peerが見えない場合、この情報だけで原因を断定しません。LM Studio、LM Link、network、
peer側の状態を個別に確認し、exact modelでchat疎通が確認できるまで本番を開始しません。

## 4. Camera V7起動

1. Robotを起動する前に、前回のCamera/Robot processが残っていないことを確認する。
2. C920を接続し、他のアプリケーションが使用していないことを確認する。
3. Camera V7をfresh起動する。
4. UIでTarget A/B、Camera入力、録画設定を確認する。
5. 「LM Studio起動・準備」でexact modelの準備とchat疎通を確認する。
6. 人物探索サーバーを起動する。
7. Camera側のROS domainがRobot側`robot.env`と一致することをログで確認する。

録画ON + live Cameraでは、人物探索サーバー開始からshutdownまで次の通し動画を
1本保存する新仕様です。

```text
outputs/mission_日時/camera_video/mission_日時.mp4
```

人物探索中もphysical Camera、producer、mission recordingは停止しません。

## 5. VOICEVOX通知

構造化された最終statusに応じて次の通知を非同期再生します。

| status | 音声 |
|---|---|
| `FOUND_FRONT_PHOTO` | 「見つけたよー！」 |
| `NOT_FOUND_OR_NOT_FRONT` | 「見つからないよー！」 |
| `ERROR` / `ERROR_*` | 無音 |

音声再生失敗は人物探索結果、`safe_to_advance`、`/person_search_done`を変更しません。
ただし、実人物探索結果から実スピーカーで鳴るE2Eはまだ残テストです。

## 6. Robot起動条件

- Renya workspaceを使う
- person_areaは2個
- `start_index=1`
- `loop=false`
- `navigation_only=false`
- routeは本番予定map/waypointと一致
- Camera人物探索サーバーとACK受信準備が完了

途中でCameraまたはRobotがrestart/crashした場合、片方だけを使い回しません。
CameraとRobotを両方freshにし、`start_index=1`からやり直します。前sessionのACK、done、
target state、mission directoryを次sessionへ引き継ぎません。

## 7. 正常シーケンス

```text
走行
  ↓
WP3
  ↓
停止
  ↓
/person_area_arrived = true
  ↓
Camera ACK: /person_search/arrival_received = true
  ↓
Target A探索
  ↓
FOUND または NOT_FOUND
  ↓
/person_search_done
  ↓
再走行
  ↓
WP7
  ↓
停止、ACK、Target B探索、done
  ↓
走行継続
```

`ERROR`は`NOT_FOUND`と同じ扱いにしません。`ERROR`は正常探索完了ではなく、
`safe_to_advance=false`のfail-closed状態として原因確認とfresh restartを行います。

## 8. 録画の扱い

### mission動画

- server lifetimeの通し録画1本
- `mission_<timestamp>.mp4`
- 人物探索中やVLM待機中もproducerが書き続ける
- 正常shutdown時にwriterをcloseし、Cameraをreleaseする
- 本番後の解析ではmission動画を正本とする

### search動画

- area/target単位で別途生成される場合がある
- persistent Cameraのshared latest-frame readerから保存する
- VLM待機中はlatest-frame方式により実時間の全frameを保持しない可能性がある
- search動画は補助資料であり、mission動画の代替ではない

既知の探索なしC920 smokeでは次を確認済みです。

```text
outputs/mission_20261019_172819/camera_video/mission_20261019_172819.mp4
kind=mission, status=completed, error="", 272 frames, 640x480,
requested_fps=15.0, 17:28:19 → 17:28:41
```

## 9. 本番前チェックリスト

- [ ] Camera V7 production反映版のhashをhandoffと照合
- [ ] CameraとRobotを両方fresh起動
- [ ] C920接続、単一Camera ownership、他アプリ未使用
- [ ] exact model `gemma-4-e2b-it-qat`
- [ ] LM Studio API/chat疎通
- [ ] LM Link使用時はpeer確認
- [ ] map `1014_15_True.yaml`
- [ ] waypoint `1014-15_True.yaml`
- [ ] person_area数が2（WP3、WP7）
- [ ] WP3→Target A、WP7→Target B
- [ ] `start_index=1`
- [ ] `loop=false`
- [ ] `navigation_only=false`
- [ ] person_area `stop:false`、`wait_time:0`
- [ ] 録画ON/OFFの選択を確認
- [ ] Camera/Robotの`ROS_DOMAIN_ID`一致
- [ ] Camera ACK待受、done/detail publish準備
- [ ] mission保存先の空き容量

## 10. 終了後確認

1. Camera人物探索サーバーを正常shutdownする。
2. `camera_video_manifest.json`のmission segmentを確認する。
3. mission MP4が1本で、`status=completed`、`error=""`であることを確認する。
4. Area 1/2の結果、capture_best、search動画、VOICEVOX結果を保存する。
5. Robot側ログとCamera側stage logの時刻を対応付ける。

## 11. 未完了E2E

以下はまだPASS扱いにしません。

1. LM Link復旧後、実人物探索のFOUNDで「見つけたよー！」が実スピーカーから鳴る。
2. 実人物探索のNOT_FOUNDで「見つからないよー！」が実スピーカーから鳴る。
3. 実C920 + person_area探索中も同じmission MP4が継続する。
4. Area 1 + Area 2を挟んでもmission MP4が増えず1本である。
5. production反映後、WP3 Target A / WP7 Target Bの実走行E2Eが完了する。

上記1～4がpre-deployment gateです。すべてPASSするまで最初のproduction反映を行いません。
5は反映後に行うproduction validationであり、これがPASSするまで本番作業完了とはしません。
