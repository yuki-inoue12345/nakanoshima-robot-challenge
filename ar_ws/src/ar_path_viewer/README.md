# ar_path_viewer

ROS 2 Jazzy向けの、走行制御から独立した読み取り専用AR Path Viewerです。`nav_msgs/msg/Path`（初期値 `/plan`）を購読し、TFで現在の`robot/base_footprint`座標へ変換して、カメラ画像へ緑色の経路線を重ねます。Phase 4では`/amcl_pose`と`map -> robot/base_footprint` TFも購読・参照し、自己位置が不安定な間だけ緑線を隠します。Publisher、Nav2 Action client、走行制御処理はありません。実機モードは地図名やWaypointファイル名を参照せず、Nav2の最新Pathをそのまま使用します。

## 座標系と精度上の制約

- ROS body: x=前、y=左、z=上
- camera body: x=レンズ前方、y=左、z=上
- OpenCV optical: x=右、y=下、z=前
- 変換は `(x_opt, y_opt, z_opt) = (-y_cam, -z_cam, x_cam)`
- `camera_pitch_up_deg`は正値がレンズ上向きです。標準右手回りの+y回転とは符号が逆になります。
- 経路は地面平面`z=0`として投影します。Path自体にzがある場合はその値を使用します。
- 初期値`camera_z=0.6967 m`、`pitch_up=7.5°`は測定値/候補値です。
- `camera_y=+0.094 m`はロボット左側へのオフセットです。CADの上部門型フレーム外端間400 mmと、左外端からレンズ中心までの実測106 mmから、`400 / 2 - 106 = 94 mm`と求めています。
- STEP上では左側部品`shazai_left`がCAD +X、右側部品`shazai_right`がCAD -Xにあり、車体中心面はCAD X=0です。このCAD +XをROSのロボット左、すなわち`base_footprint`の+Yへ対応させています。
- 400 mmの根拠は左右の上部縦材`KHFS5-2020-560`の外端（CAD X=-200 mm、+200 mm）です。下部`alumi_frames`の外端間577 mm（-288.5 mmから+288.5 mm）はカメラ搭載フレームではないため使用していません。
- `camera_x=0.20 m`は引き続き明示的な仮値です。`camera_y`も実機の外端基準がCADの同じ20 mm角縦材であることを静止投影で最終確認してください。
- `fx=fy=525 px`等は未校正の仮内部パラメータです。実写との位置精度は保証しません。
- 画角外、カメラ後方、near plane内の経路を飾り線で代替しません。

### Phase 5の投影幾何検証

地面上の点をカメラ前方距離`d`、カメラ高さ`h`、上向き角`θ`とすると、中央線の画像行は概ね次式です。

```text
v = cy + fy * (d sin(θ) + h cos(θ)) / (d cos(θ) - h sin(θ))
```

初期値`h=0.6967 m`、`fy=525 px`、`cy=239.5 px`、`θ=7.5°`では、地平線は`v=308.6 px`、地面が画像下端`v=479 px`へ入り始めるのはカメラ前方約`2.276 m`です。`camera_x=0.20 m`を含む`base_footprint`前方距離では約`2.476 m`です。コードとは独立した三角関数計算と投影実装が一致しており、正の`camera_pitch_up_deg`が実際に上向きとなる符号も一致しています。

同じ仮内部パラメータでの参考値です。物理角度を変えず設定値だけを変更すると投影が実写から外れるため、実測せずに採用しません。

近距離の床経路をより多く映す必要がある場合、物理取付角を上向き7.5°から0〜3°付近へ下げることが検討候補です。その場合も、実際のレンズ角を測定してから同じ値を`camera_pitch_up_deg`へ設定します。今回は物理取付を確認できないため7.5°を変更していません。

| `camera_pitch_up_deg` | 地平線の画像行 | 地面が下端へ入るカメラ前方距離 |
|---:|---:|---:|
| 0° | 239.5 px | 1.527 m |
| 3° | 267.0 px | 1.767 m |
| 5° | 285.4 px | 1.965 m |
| 7.5° | 308.6 px | 2.276 m |

2026-11-06の実機録画では、ACTIVE時の緑線が画像下端12〜24 px程度に限られるフレームを確認しました。上向きカメラにより近距離の床が物理的に画角外となることは確定しています。一方、受信PathのTF変換後形状が保存されていないため、多数のPath点が後方・左右・下側のどこへ外れたかは録画だけでは確定できません。Phase 5では点数だけでなく、TF変換後のPath全長、ロボット原点から終点までの直線距離、カメラ前方にあるPath実長、視錐台内へ3DクリップされたPath実長、描画線分数、描画ピクセル長を5秒間隔で記録します。次回の実機ログで「Path自体が短い」「多くが後方」「前方だが画角外」「画角内実長はあるが遠近法で短く見える」を区別できます。

メートル単位の長さは隣接Path点間の3次元線分長から求めます。`camera_forward_path_length_m`はカメラ光学座標の前方半空間に含まれる各線分部分、`visible_path_length_m`はnear planeと上下左右の視錐台でクリップ後に残る各線分部分の合計です。両者の差を`nonvisible_forward_path_length_m`として記録します。境界を横切る線分は交点までの割合を使うため、端点が画面外でも画面内部分を失いません。`visible_depth_min_m` / `visible_depth_max_m`は実際に画角へ入った線分のカメラ前方深度範囲です。`endpoint_distance_m`は現在の`base_frame`原点からPath終点までの3次元直線距離で、Path全長とは別の値です。

同録画の32秒・40秒の直線廊下について、廊下方向の直線群から求めた消失点候補は画像行`v≈378〜380 px`でした。仮の`fy=525 px`、`cy=239.5 px`へ当てはめると上向き約14.8〜15.0°相当で、7.5°設定より大きい可能性があります。仮に15°なら地面が画像下端へ入り始める距離はカメラ前方約4.17 m、`camera_x=0.20 m`を含むbase前方約4.37 mです。ただし、廊下線の選択、レンズ歪み、`fx/fy/cx/cy`自体が未校正なので、これは校正値ではありません。この映像推定だけで設定を15°へ変更せず、チェッカーボード内部校正と水平基準を写した静止画で物理ピッチを測定します。実機ログの`camera_configuration`で、NUC固有YAMLから実際に読み込まれた投影値を確認します。

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
display_width: 960
display_height: 720
output_video: /home/inoue-yuki/ar_navigation_project/outputs/c920_static_test.mp4
```

通常ウィンドウはマウスでリサイズでき、映像は黒帯を使って4:3を維持します。`F11`で全画面と通常表示を切り替え、`q`、`Q`、`F10`で正常終了します。HighGUI backendがリサイズ・全画面APIに対応しない場合は通常表示へ留まり、表示自体を作れない場合も録画処理は継続します。表示用の拡大・黒帯はGUIだけに適用され、投影計算とMP4は従来どおり640×480、20fpsです。

同じ内容の[ar_path_viewer_c920.yaml](/home/inoue-yuki/ar_navigation_project/ar_path_viewer/config/ar_path_viewer_c920.yaml)を指定する場合：

```bash
ros2 launch ar_path_viewer ar_path_viewer.launch.py \
  config_file:=/path/to/ar_path_viewer_c920.yaml
```

表示状態は`ACTIVE`、`NO PATH`、`PATH STALE`、`LOCALIZING`、`LOCALIZATION UNSTABLE`、`TF UNAVAILABLE`、`CAMERA UNAVAILABLE`、`PATH OUT OF VIEW`、`INVALID CONFIG`です。Pathは受信ごとに全体交換し、不正Pathを受けた場合は直ちに旧線を消します。TFが取得できない場合も線を出しません。初期値6.5秒を超えてPath更新がない場合は描画を消します。

ACTIVE時は投影済みのNav2 Path線分上に進行方向矢印を描きます。矢印は画面内へクリップされた実線分から生成し、経路座標の移動、変形、Waypoint直線接続は行いません。

AR ViewerはWaypoint Followerより先に起動できます。起動時にPathがなくても映像表示・録画を継続し、後から`/plan`を受信すると自動的に投影を開始します。ルート変更後の新しいPathはgeneration単位で全体交換されます。

## 自己位置ゲート（Phase 4）

緑線を描くには、有効なPathに加えて次の条件が必要です。

- `/amcl_pose`のframeが`localization_global_frame`（初期値`map`）と一致する
- pose、quaternion、ROS timestamp、6x6 covarianceが有限かつ形式上妥当
- AMCL timestampが古すぎず、将来方向のずれも許容値内
- xy共分散行列とyaw分散から求めた標準偏差が暫定上限内
- 前回の妥当なAMCL poseからの差が、固定ジャンプ余裕と経過時間に応じた速度余裕の範囲内
- `localization_global_frame -> base_frame`のTFが取得でき、時刻が新しい
- 正常条件が`localization_recovery_sec`以上継続する

共分散が小さいことは真の位置誤差が小さいことの保証ではありません。このゲートは、受信鮮度、座標系、timestamp、TF、共分散、時系列ジャンプを組み合わせて明白な異常を保守的に除外するもので、地図との実位置一致を証明するものではありません。

Nav2 AMCLは、静止してparticle filter更新条件を満たさない間、`/amcl_pose`を毎LaserScanで再発行しないことがあります。このため、最後のAMCL poseが正常で、`map <- base_frame` TFが新しく、最後のposeから位置・yawが静止保持範囲内の場合は、短い受信間隔だけを理由に非表示へ戻しません。TF上で保持範囲を超えて移動した場合や、AMCL pose自体に古いtimestamp、不正値、高い共分散が届いた場合は、この保持を使わず非表示にします。

初期値は実機ログ取得前の暫定値です。

```yaml
amcl_pose_topic: /amcl_pose
localization_global_frame: map
localization_pose_timeout_sec: 3.0
localization_stamp_max_age_sec: 3.0
localization_future_tolerance_sec: 1.5
localization_tf_max_age_sec: 3.0
localization_max_position_stddev_m: 0.75
localization_max_yaw_stddev_rad: 0.75
localization_jump_distance_m: 1.5
localization_jump_yaw_rad: 1.2
localization_max_linear_speed_mps: 1.5
localization_max_angular_speed_radps: 2.0
localization_stationary_hold_distance_m: 0.5
localization_stationary_hold_yaw_rad: 0.5
localization_recovery_samples: 1
localization_recovery_sec: 2.0
```

許容移動量は概ね`jump_distance + max_speed * 受信間隔`です。固定余裕に速度余裕を加えるため、通常走行を単純な座標差だけでジャンプ扱いしません。ジャンプ検出後は新しい位置を復帰候補の基準にするため、一度の正当な大補正で永久に復帰不能にはなりません。

状態の優先順位は、カメラ/設定異常、`NO PATH` / `PATH STALE`、`LOCALIZING` / `LOCALIZATION UNSTABLE`、`TF UNAVAILABLE`、`PATH OUT OF VIEW`、`ACTIVE`の順です。最初のWaypointへの到達は表示条件ではありません。現在位置から最初のWaypointへ向かうNav2 Pathも、同じ条件を満たせば表示します。自己位置ゲートで緑線を隠している間も、C920映像の表示とMP4録画は継続し、録画フレームにも緑線を描きません。

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
- `path_projection`: TF変換後点数、入力線分数、Path全長、ロボットから終点までの距離、カメラ前方部分の実長、視錐台内部分と前方だが非表示の実長、前方全体と画角内線分それぞれの深度範囲、画像内点数、後方・near内・左右・上下の点数、描画線分数、可視画素長・表示範囲
- `camera_configuration`: 実際に読み込まれた画像サイズ、内部パラメータ、カメラ位置・姿勢、near clip（起動時に1回）
- `localization_state`: AMCL受信数、`LOCALIZING` / `LOCALIZATION UNSTABLE` / `LOCALIZATION OK`、判定理由

`path_out_of_view`の理由は、全点がカメラ後方、全点がnear clip内、前方点がすべて画像の下・上・左・右、または複数境界の組合せで視錐台と交差しない場合に分類します。点がすべて画像外でも、隣接点間の線分が画角を横切れば3Dクリッピング後に描画します。投影診断は`diagnostic_log_interval_sec`（初期値5秒）で集約されます。

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
  --camera-x 0.20 --camera-y 0.094 --camera-z 0.6967 \
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

## 実機NUCへのcamera_y反映

既存cloneの絶対パスだけを`REPO_DIR`へ設定して実行します。作業ツリーに変更があれば停止し、新規clone、reset、stash、force pullは行いません。

```bash
set -euo pipefail
REPO_DIR=/path/to/existing/nakanoshima-robot-challenge
test -d "$REPO_DIR/.git"
cd "$REPO_DIR"
if test -n "$(git status --porcelain)"; then
  echo "STOP: existing Git changes must be reviewed first" >&2
  exit 1
fi
git pull --ff-only origin main

cd "$REPO_DIR/ar_ws"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ar_path_viewer
colcon test --packages-select ar_path_viewer
colcon test-result --test-result-base build/ar_path_viewer --verbose
```

Git管理外のNUC固有YAMLは、他の値をGitHub版で置換せず、`camera_y`の1行だけを更新します。

```bash
set -euo pipefail
NUC_CONFIG=/home/koko-chan-nuc/ar_navigation_project/ar_path_viewer/config/ar_path_viewer_nav2_test.yaml
test -f "$NUC_CONFIG"
test "$(grep -Ec '^[[:space:]]*camera_y:' "$NUC_CONFIG")" -eq 1
BACKUP="${NUC_CONFIG}.bak.$(date +%Y%m%d_%H%M%S)"
cp -a -- "$NUC_CONFIG" "$BACKUP"
sed -i -E 's/^([[:space:]]*camera_y:)[[:space:]]*[^#[:space:]]+([[:space:]]*(#.*)?)$/\1 0.094\2/' "$NUC_CONFIG"
test "$(grep -Ec '^[[:space:]]*camera_y:[[:space:]]*0\.094([[:space:]]|$)' "$NUC_CONFIG")" -eq 1
diff -u -- "$BACKUP" "$NUC_CONFIG" || true
```

`diff`が`camera_y`以外も示した場合は、バックアップを保持したまま起動せず原因を確認します。既存のC920デバイスパス、MP4録画先、診断設定、デスクトップアイコン、起動スクリプトは変更しません。

## 次回の実機静止テスト（走行なし）

1. C920を単独で使えることとby-idを確認し、Camera V7との競合を避ける。
2. ROS domain 54へ参加する前に運用者の許可を取り、ロボットを停止状態にする。
3. `camera_y=+0.094 m`の左右符号を静止投影で確認し、未確定の`camera_x`を`base_footprint`基準で測定する。C920をチェッカーボードで校正して内部パラメータも更新する。
4. Viewer設定を`image_input_type: camera`、C920のby-idへ変更する。
5. `/plan`、`/amcl_pose`、`map -> robot/base_footprint`が存在することを読み取りだけで確認する。
6. Viewerのみ起動し、ゴール送信や制御Topic publishをせず、静止状態で重なりと符号を確認する。
7. 安全な既知位置でyawを手動変更できる場合も、走行指令なし・管理者立会いで画面追従だけを確認する。

実機TFにはカメラframeを追加せず、初期版はViewer内部のmount値を使います。

画面表示中は`F11`で全画面を切り替え、`q`、`Q`、`F10`で終了できます。OpenCVのWin32、GTK、Qtで異なるF10/F11コードを扱い、終了要求とC920/MP4/Windowの解放は一度だけ実行します。Ctrl+Cも引き続き`rclpy.try_shutdown()`で正常終了します。

開発PCの自動テストでは状態遷移、録画フレームの緑線有無、キーコード、リソース解放を確認しています。実機NUCでのAMCL共分散分布、実際の発行間隔、TF遅延、C920とMP4 codec、各HighGUI backendの物理キー入力は未確認であり、走行なしのログ取得後に閾値を調整してください。
