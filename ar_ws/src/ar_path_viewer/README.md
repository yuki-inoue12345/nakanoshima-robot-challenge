# ar_path_viewer

ROS 2 Jazzy向けの、走行制御から独立した読み取り専用AR Path Viewerです。`nav_msgs/msg/Path`（初期値 `/plan`）を購読し、TFで現在の`robot/base_footprint`座標へ変換して、カメラ画像へ緑色の経路線を重ねます。Publisher、Nav2 Action client、走行制御処理はありません。実機モードは地図名やWaypointファイル名を参照せず、最新PathとTFだけで動作します。

## 座標系と精度上の制約

- ROS body: x=前、y=左、z=上
- camera body: x=レンズ前方、y=左、z=上
- OpenCV optical: x=右、y=下、z=前
- 変換は `(x_opt, y_opt, z_opt) = (-y_cam, -z_cam, x_cam)`
- `camera_pitch_up_deg`は正値がレンズ上向きです。標準右手回りの+y回転とは符号が逆になります。
- 経路は地面平面`z=0`として投影します。Path自体にzがある場合はその値を使用します。
- 初期値`camera_z=0.6967 m`、`pitch_up=7.5°`は測定値/候補値です。
- `camera_x=0.20 m`、`camera_y=0.0 m`は明示的な仮値です。筐体端からの測定値をROS座標へ流用した値ではありません。
- `fx=fy=525 px`等は未校正の仮内部パラメータです。実写との位置精度は保証しません。
- 画角外、カメラ後方、near plane内の経路を飾り線で代替しません。

## ビルドとテスト

ワークスペース直下で実行します。

```bash
colcon --log-base log build \
  --base-paths ar_path_viewer \
  --build-base build \
  --install-base install

colcon --log-base log-test test \
  --build-base build \
  --install-base install \
  --packages-select ar_path_viewer

colcon test-result --test-result-base build --verbose
```

## 実機ARモード

デフォルトは安全な合成背景、画面表示なしです。既存ROS graphで実行する前にdomainとTopicの運用確認が必要です。

```bash
source install/setup.bash
ros2 launch ar_path_viewer ar_path_viewer.launch.py
```

設定は`config/ar_path_viewer.yaml`です。画像入力は次を選べます。

- `simulation`: 合成背景
- `image`: 静止画ファイル
- `video`: 動画ファイル（末尾でループ）
- `camera`: OpenCV VideoCapture。URIにはデバイスパスまたは数値index

VideoCaptureはC920を排他的に掴む可能性があります。Camera V7等が同じデバイスを使用中か確認し、競合時は起動しないでください。入力層は`ImageSource` interfaceで分離してあり、将来のROS Image Topic入力を追加できます。

C920ライブ表示・録画の設定例です（実機での実行は未実施）。

```yaml
image_input_type: camera
image_input_uri: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_66F7D3AF-video-index0
display: true
output_video: /home/inoue-yuki/ar_navigation_project/outputs/c920_static_test.mp4
```

同じ内容の[ar_path_viewer_c920.yaml](/home/inoue-yuki/ar_navigation_project/ar_path_viewer/config/ar_path_viewer_c920.yaml)を指定する場合：

```bash
ros2 launch ar_path_viewer ar_path_viewer.launch.py \
  config_file:=/path/to/ar_path_viewer_c920.yaml
```

表示状態は`ACTIVE`、`NO PATH`、`PATH STALE`、`TF UNAVAILABLE`、`CAMERA UNAVAILABLE`、`PATH OUT OF VIEW`、`INVALID CONFIG`です。Pathは受信ごとに全体交換し、不正Pathを受けた場合は直ちに旧線を消します。TFが取得できない場合も線を出しません。初期値6.5秒を超えて更新がない場合は描画を消します。

AR ViewerはWaypoint Followerより先に起動できます。起動時にPathがなくても映像表示・録画を継続し、後から`/plan`を受信すると自動的に投影を開始します。ルート変更後の新しいPathはgeneration単位で全体交換されます。

## NO PATH診断

追加診断はROS parameterで切り替えます。

```yaml
diagnostic_logging: true
diagnostic_log_interval_sec: 5.0
```

`false`にすると追加診断を停止します。同じ種類の受信・TF失敗・画角外ログは既定で最大5秒に1回ですが、状態変化は直ちに1回記録します。

主なログ：

- `plan_rx`: 累積受信回数、monotonic受信時刻、Path header stamp、frame_id、点数
- `plan_adopted` / `plan_updated`: 有効Pathのgeneration、frame_id、点数
- `plan_discarded`: 空Path、NaN/Inf、不正frame_id等の破棄理由
- `viewer_state`: `STARTUP -> NO PATH`、`NO PATH -> ACTIVE`、`ACTIVE -> PATH STALE`等
- `tf_unavailable`: source/target frame、例外型、TF lookupの原因本文
- `path_out_of_view`: 3Dクリップ後にカメラ視錐台と交差する線分がないこと

`NO PATH`のdetailが`no Path received since startup`かつ`plan_rx_count=0`なら、Viewer起動後にまだ`/plan`を一度も受信していません。`plan_discarded`があれば空または不正Pathが最後の原因です。`ACTIVE`後に`PATH STALE`へ変化した場合は、最後の有効Pathから設定timeoutを超えています。

## 汎用オフライン動画生成

ROSノードやPublisherを起動せず、指定Waypoint間を直線補間してAR投影を通し実行します。Pythonソース内に地図名やWaypoint名の固定値はありません。

CLIですべて指定する例：

```bash
source install/setup.bash
ros2 run ar_path_viewer simulate_lap \
  --waypoints /path/to/waypoints.yaml \
  --map-yaml /path/to/map.yaml \
  --input-type simulation \
  --output-dir /home/inoue-yuki/ar_navigation_project/outputs/route_name \
  --output-video /home/inoue-yuki/ar_navigation_project/outputs/route_name/route.mp4 \
  --speed-mps 1.0 --fps 20 \
  --fx 525 --fy 525 --cx 319.5 --cy 239.5 \
  --camera-x 0.20 --camera-y 0 --camera-z 0.6967 \
  --camera-yaw-deg 0 --camera-pitch-up-deg 7.5 --camera-roll-deg 0
```

地図画像はMap YAMLの`image`から読み、相対パスはMap YAMLのディレクトリ基準で解決します。`resolution`、`origin`（yawを含む）を使用して世界座標を画像座標へ変換します。互換用の`--map-image` / `--map-pgm`で画像だけを明示上書きすることもできます。

設定ファイルで切り替える例：

```bash
ros2 run ar_path_viewer simulate_lap --config /path/to/offline_route.yaml
```

設定形式は[offline_route_b.yaml](/home/inoue-yuki/ar_navigation_project/ar_path_viewer/config/offline_route_b.yaml)を参照してください。CLI値は設定ファイルより優先されます。相対パスは設定ファイルの場所を基準に解決されます。

入力は`simulation`、`image`、`video`、`camera`から選択できます。Waypointは現行形式の`number`、`x`、`y`、`angle_radians`、`threshold`、`stop`、`wait_time`、`person_area`を検証します。未知の無害な追加属性は保持し、有効な`change_map`や`rotate`は単一地図シミュレーションでは再現できないため明示エラーにします。`stop`と`wait_time`は停止フレームへ反映しますが、`person_area`のCamera検索動作は実行せずログに属性だけを残します。

Waypointが地図画像範囲外なら動画生成前にエラー終了します。地図上に入っていることは障害物回避や通行可能性の証明ではありません。この補間はNav2の経路計画、実カメラ映像、車体運動を再現しません。

## 次回の実機静止テスト（走行なし）

1. C920を単独で使えることとby-idを確認し、Camera V7との競合を避ける。
2. ROS domain 54へ参加する前に運用者の許可を取り、ロボットを停止状態にする。
3. `camera_x/y`を`base_footprint`基準で測定し、C920をチェッカーボードで校正して内部パラメータを更新する。
4. Viewer設定を`image_input_type: camera`、C920のby-idへ変更する。
5. `/plan`と`map -> robot/base_footprint`が存在することを読み取りだけで確認する。
6. Viewerのみ起動し、ゴール送信や制御Topic publishをせず、静止状態で重なりと符号を確認する。
7. 安全な既知位置でyawを手動変更できる場合も、走行指令なし・管理者立会いで画面追従だけを確認する。

実機TFにはカメラframeを追加せず、初期版はViewer内部のmount値を使います。

終了処理は`rclpy.try_shutdown()`を使用し、SIGINT時にROS contextが既にshutdown済みでも二重shutdown例外を発生させない構造です。
