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

このC920は人物検知と共用するため、ARを理由に物理取付位置・角度を変更しません。近距離床が物理的に写らない範囲は緑線を画面内へ移動・伸長せず、正確な`PATH OUT OF VIEW`を維持します。見やすさの補助が必要なら、実Pathと明確に区別した画面端方向表示、または走行制御とは分離された将来経路Topicを別設計として検討します。

| `camera_pitch_up_deg` | 地平線の画像行 | 地面が下端へ入るカメラ前方距離 |
|---:|---:|---:|
| 0° | 239.5 px | 1.527 m |
| 3° | 267.0 px | 1.767 m |
| 5° | 285.4 px | 1.965 m |
| 7.5° | 308.6 px | 2.276 m |

2026-11-06の実機録画では、ACTIVE時の緑線が画像下端12〜24 px程度に限られるフレームを確認しました。上向きカメラにより近距離の床が物理的に画角外となることは確定しています。一方、受信PathのTF変換後形状が保存されていないため、多数のPath点が後方・左右・下側のどこへ外れたかは録画だけでは確定できません。Phase 5では点数だけでなく、TF変換後のPath全長、ロボット原点から終点までの直線距離、カメラ前方にあるPath実長、視錐台内へ3DクリップされたPath実長、描画線分数、描画ピクセル長を5秒間隔で記録します。次回の実機ログで「Path自体が短い」「多くが後方」「前方だが画角外」「画角内実長はあるが遠近法で短く見える」を区別できます。

メートル単位の長さは隣接Path点間の3次元線分長から求めます。`camera_forward_path_length_m`はカメラ光学座標の前方半空間に含まれる各線分部分、`visible_path_length_m`はnear planeと上下左右の視錐台でクリップ後に残る各線分部分の合計です。両者の差を`nonvisible_forward_path_length_m`として記録します。境界を横切る線分は交点までの割合を使うため、端点が画面外でも画面内部分を失いません。`visible_depth_min_m` / `visible_depth_max_m`は実際に画角へ入った線分のカメラ前方深度範囲です。`endpoint_distance_m`は現在の`base_frame`原点からPath終点までの3次元直線距離で、Path全長とは別の値です。

同録画の32秒・40秒の直線廊下について、廊下方向の直線群から求めた消失点候補は画像行`v≈378〜380 px`でした。仮の`fy=525 px`、`cy=239.5 px`へ当てはめると上向き約14.8〜15.0°相当で、7.5°設定より大きい可能性があります。仮に15°なら地面が画像下端へ入り始める距離はカメラ前方約4.17 m、`camera_x=0.20 m`を含むbase前方約4.37 mです。ただし、廊下線の選択、レンズ歪み、`fx/fy/cx/cy`自体が未校正なので、これは校正値ではありません。この映像推定だけで設定を15°へ変更せず、チェッカーボード内部校正と水平基準を写した静止画で物理ピッチを測定します。実機ログの`camera_configuration`で、NUC固有YAMLから実際に読み込まれた投影値を確認します。

### 2026-11-09実走データでの確認

2走行分のMCAP、AR録画、診断ログ、RViz録画を時刻照合しました。1走目の127 Pathは全長中央値13.768 m、最大72.067 m、2走目の180 Pathは中央値1.810 m、最大3.065 mでした。2走目はWaypoint Followerが`navigation_mode=to_pose`で約2 m間隔の30地点を1地点ずつNav2へ渡しており、180 Path中175件の終点がWaypointから0.05 m以内でした。したがって点数が100点を超えても、実Path自体が1〜3 mしかない場面があります。

2走目の代表例ではPath全長2.907 m、カメラ前方2.601 mに対し、画角内は0.415 m・約35 pxでした。`visible_depth_min_m=2.165`が繰り返され、非表示の前方部分は約2.186 mでした。設定上の下端交点と一致するため、クリッピングによる遠方Path欠落ではなく、近距離床が画像下側の画角外にあることが主因です。1走目の72.067 m Pathは終点直線距離8.408 mですが、地図上で建物外周を大きく迂回するNav2の実Pathでした。AR側が架空に引き伸ばした値ではありません。

1走目の右折Pathをbagの`/plan`と同時刻TFから再投影すると、録画された緑線へ数px程度で重なりました。ROS右（base Y負）が画像右へ出ており、map→baseのTF方向、ROS/OpenCV軸対応、左右・進行方向の反転は確認されませんでした。実データには同等に判定できる明瞭な左折表示がないため、左側は軸対称の単体テストでも維持します。

直線廊下の長辺から得た11月9日録画の消失点候補は`v≈404 px`（複数フレームで約403.5〜405.0 px）でした。仮の`fy=525 px`、`cy=239.5 px`なら約17.4°上向き相当で、現設定7.5°の地平線`v=308.6 px`と約95 pxずれます。これは壁・扉の高さへ緑線が重なる現象を説明しますが、内部パラメータと歪みが未校正であり、廊下線の平行性も前提なので校正値ではありません。`camera_pitch_up_deg`、`fx/fy/cx/cy`は変更せず、チェッカーボード校正と水平基準を使った外部姿勢校正を次の必須確認とします。

起動時の`camera_configuration`には、設定値から求めた`ground_horizon_v_px`、`ground_bottom_center_camera_distance_m`、`ground_bottom_center_base_x_m`、`ground_bottom_center_base_y_m`も記録します。これらは診断値であり、投影位置やPathを変更しません。

### 短い実Path向けの表示高さ

固定C920では1〜3 m先の床が画像下へ外れるため、AR表示は次の2方式をROS parameterで切り替えられます。

```yaml
path_display_mode: elevated  # elevated または ground
path_display_height_m: 0.50  # elevated時だけ使用、0〜2 m
```

`ground`は従来どおりTF変換後のNav2 Pathを`base_footprint`のZ=0へ置く床面表示です。`elevated`は全点のX/Yと点順序を一切変えず、描画時のZだけを一定値へ置く仮想的な浮遊表示です。Nav2経路の延長、補間、画面座標への貼り付けではありません。TF、AMCL、Path鮮度、視錐台クリッピングも従来と同じです。橙色方向案内の判定には高さ変更前の実Pathを使います。

2026-11-09の2走目bag（180 Path）を本番仮設定`fx=fy=525`、pitch 7.5°で再投影すると、Z=0は43/180（23.9%）のみ描画可能でした。Z=0.3/0.5/0.7/1.0 mではそれぞれ108/166/168/168 Pathが描画可能でした。0.7 mはカメラ高さ付近へ長い横線として出やすく、1.0 mは近傍が画像上へ外れやすいため、映像内で低めの遠近感を残しつつ1〜3 m Pathの94.3%（全長1 m以上）を描画できた0.50 mを初期値としました。これは床面位置を示す校正値ではなく、進行方向を見やすくする表示値です。画面には`AR FLOAT 0.50m`または`AR GROUND`を表示し、ログには`path_display_configuration`を記録します。

2026-11-10走行ログではPath全長2.641〜3.060 mに対して床面の可視部分が0.014〜0.549 m（中央値0.420 m）、描画長が3.2〜43.6 px（中央値33.1 px）でした。画角外診断38回中35回が「前方Path全体が画像下」で、TF失敗はありません。したがって浮遊表示は校正誤差を解消するものではありませんが、この短い実Pathを架空に延ばさず視認可能にする目的に適合します。厳密な床面重畳を確認するときは`ground`へ戻します。

## 固定したC920の校正

校正中もC920の取付位置・向きは動かしません。内部校正ではカメラではなくチェッカーボードを動かし、外部校正では停止中のロボットと、位置を実測した目印を撮影します。`camera_pitch_up_deg=7.5`や録画から推定した約17.4°は、以下の実測と再投影確認が完了するまで変更しません。

校正CLIはROS graphへ接続せず、Topic、Publisher、Action、Nav2サービスを一切使用しません。結果は新しいYAMLへだけ書き、既存設定を上書きしません。

### 用意するもの

- 9×6個の**内側交点**を持つ10×7マスのチェッカーボード
- 1マス実測20.0 mm、パターン部約200×140 mm、用紙全体約295×210 mm（A4相当）の実物ボード
- 用紙を平らで反りにくい板へ固定したもの（C920とロボット側は動かさない）
- 1 m以上を測れるレーザー距離計または巻尺、下げ振り、水平器
- 外部校正用の巻尺、細い糸またはレーザー墨出し器、水平器（気泡式で可）
- 10～12個の丸印は、自動チェッカーボード方式が使えない場合だけの予備

この実物では`--inner-columns 9 --inner-rows 6 --square-size-m 0.020`を使用します。`square-size`は用紙寸法や白黒パターン全体ではなく、隣接する内側交点間、すなわち白または黒1マスの一辺です。複数マスをまとめて測り、実寸が20.0 mmから明確に異なる場合だけ、測定値をm単位へ換算して指定します。

### 走行なしで行う内部校正

1. ロボットを停止・固定し、AR ViewerとCamera V7を終了してC920の排他競合を避ける。人物検知の設定やCamera V7のファイルは変更しない。
2. 実運用と同じデバイス、640×480、20 fps、同じフォーカス・露出条件を使用する。
3. カメラには触れず、ボードを中央、四隅、上下左右端へゆっくり移動し、正面だけでなくyaw/pitchを約±20～35°、時計・反時計方向にも傾ける。20 mm・A4ボードは遠距離では交点が小さくなるため、レンズから概ね0.35、0.5、0.7、0.9、1.2 mを混ぜ、1.5 mは交点が明瞭な場合だけ補助的に使用する。上向きC920の画面内へ入るようボードの高さを変える。
4. 推奨方式では約90秒の動画を撮影し、自動抽出で30～40枚を選ぶ。ボードを急に振らず、各構図で一瞬静止するとブレ除外後の候補を確保しやすい。

#### 推奨：動画撮影と自動選択

実機NUCではC920の生フレームだけを640×480・20 fpsのMP4へ保存します。画面の`REC`文字はpreviewだけで動画には入りません。90秒に達するか、`q`、`Q`、`F10`で正常終了します。既存ファイルは上書きしません。

```bash
set -euo pipefail
source /opt/ros/jazzy/setup.bash
source /home/koko-chan-nuc/ar_navigation_project/install/setup.bash

CALIB_ROOT=/home/koko-chan-nuc/ar_navigation_project/calibration
mkdir -p "$CALIB_ROOT"

ros2 run ar_path_viewer calibrate_camera record-video \
  --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_66F7D3AF-video-index0 \
  --output "$CALIB_ROOT/c920_640x480_20fps_board_10x7_20mm.mp4" \
  --fps 20 --duration-sec 90
```

保存済み動画を開発PCへコピーした後、次の1コマンドで自動選択と内部校正を実行します。`extract-video`は初期値4 fpsで候補を調べ、交点検出失敗、Laplacian sharpnessが低いブレ画像、内側交点領域が画面の1.5%未満の小さすぎるボード、構図descriptorが近い重複画像を除外します。位置、見かけの大きさ、回転、台形歪みから多様な36枚を選び、同程度に多様な候補では面積が大きく鮮明な画像を優先します。遠距離画像だけで焦点距離や歪みを決めないための下限であり、`--minimum-area-ratio`で明示的に変更できます。

```bash
set -euo pipefail
source /opt/ros/jazzy/setup.bash
source /home/inoue-yuki/care_patrol_ws/local_repositories/nakanoshima-robot-challenge/ar_ws/install/setup.bash

CALIB_ROOT=/home/inoue-yuki/ar_navigation_project/calibration
VIDEO="$CALIB_ROOT/c920_640x480_20fps_board_10x7_20mm.mp4"

ros2 run ar_path_viewer calibrate_camera extract-video \
  --video "$VIDEO" \
  --output-dir "$CALIB_ROOT/selected_10x7_20mm" \
  --report "$CALIB_ROOT/selection_10x7_20mm.yaml" \
  --target-images 36 --min-images 30 \
  --analysis-fps 4 \
  --intrinsics-output "$CALIB_ROOT/intrinsics_10x7_20mm.yaml" \
  --preview-dir "$CALIB_ROOT/corner_previews_10x7_20mm"
```

別の保存済みMP4/AVIも`--video`へ指定できますが、フレームは640×480でなければ拒否します。出力画像ディレクトリ、report、内部校正YAMLは新しいパスを使い、既存データや別セッションへ混在させません。

`selection_10x7_20mm.yaml`には、解析・検出・ブレ・小さすぎるボード・採用候補の件数、選択フレーム時刻、sharpness、画面位置、面積、回転、台形歪みを保存します。さらに次の不足を`missing_compositions`へ列挙します。

- 左・右・上・下・中央にボードがない
- 近・中・遠に相当する見かけの大きさが不足
- 正面または台形になる傾斜構図がない
- 時計・反時計方向の傾きが揃っていない
- 類似除外後の画像が30枚未満

不足がある場合、PNGとreportは保存しますが、`--intrinsics-output`を指定した自動校正は停止します。reportに書かれた不足構図だけを追加撮影し、新しい出力先で再実行してください。閾値を下げて不十分な撮影を通すより、撮り直しを優先します。

#### 従来方式：手動でPNGを保存

従来の`capture`も維持しています。Spaceまたは`C`で保存し、`q`、`Q`、`F10`で終了します。

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/ar_ws/install/setup.bash

ros2 run ar_path_viewer calibrate_camera capture \
  --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_XXXXXXXX-video-index0 \
  --output-dir ~/ar_navigation_project/calibration/c920_640x480_board_10x7_20mm \
  --fps 20

ros2 run ar_path_viewer calibrate_camera intrinsics \
  --images "$HOME/ar_navigation_project/calibration/c920_640x480_board_10x7_20mm/*.png" \
  --inner-columns 9 --inner-rows 6 --square-size-m 0.020 \
  --min-images 20 \
  --preview-dir ~/ar_navigation_project/calibration/corner_previews_10x7_20mm \
  --output ~/ar_navigation_project/calibration/intrinsics_10x7_20mm.yaml
```

`capture`はボード寸法を計算に使わず、取得フレームを必ず640×480 PNGとして保存します。寸法の混同を避けるため保存先名に`10x7_20mm`を含め、別寸法のボード画像を同じディレクトリへ混在させないでください。`intrinsics`の出力YAMLには、実際に使用した`inner_columns`、`inner_rows`、`square_size_m`が記録されます。

RMSだけで採否を決めず、全previewで順序正しく交点を検出していることを目視します。目安は平均再投影誤差0.5 px以下、最大1 px以下ですが、超えた場合は画像構成やボードの平面性を見直し、値を無理に採用しません。

### 走行なしで行う外部校正（チェッカーボード内部校正とは別作業）

内部校正はチェッカーボードからC920の`fx/fy/cx/cy`とレンズ歪みを求める工程です。以下の外部校正は、その結果を使い、別途実測したロボット座標の3D基準点からC920の位置・姿勢を求める独立工程です。20 mmチェッカーボードの寸法を3D基準点の代用にはしません。

#### 推奨：既知配置のチェッカーボード54交点を共同推定

硬い板へ固定した同じ10×7マスのボードを4か所へ置き、各画像の9×6交点を自動検出します。ボードの20 mm寸法だけから配置を推測するのではなく、各配置の原点と軸を`base_footprint`基準で実測します。実測姿勢から54交点すべての3D座標を生成し、全配置の216点を一つのPnP問題として共同推定します。単一画像ごとの結果を平均する方式ではありません。

`base_footprint`はx前、y左、z上です。現在のURDFでは左右駆動輪の中心が`x=0`、左右輪中心の中点が`y=0`、床面が`z=0`です。したがって実機では、左右車軸中心を結んだ線を床へ下ろした点の中点を原点候補にします。ただしURDF値だけで外観上の中心を決めず、左右の車軸中心と床面が実機の`base_footprint`に対応していることを最初に確認してください。確認できなければ座標を入力せず、Robotbase担当者へ基準位置を確認します。

ボードは複雑な角度へ傾けません。ロボットへ正対した垂直面として置きます。

- ロボットから見た左上の**内側交点**をボード原点とする。白黒パターン外周の左上ではなく、そこから右20 mm・下20 mmの交点である。
- ボード原点から画像右方向の9交点目へ向かう軸は`[0, -1, 0]`。
- ボード原点から画像下方向の6交点目へ向かう軸は`[0, 0, -1]`。
- 水平器で板を垂直にし、上段左右の内側交点の床からの高さを測って2～3 mm以内に揃える。
- ボード面の左右端から`x=0`車軸線までの前後距離を測り、2～3 mm以内に揃えてロボットへ正対させる。揃えられなければ軸を上記の値に固定してはいけない。

推奨する4配置の範囲です。数値は置き場所の目安であり、YAMLには必ず実測値を記入します。

1. 近・中央：ボード中心`x=1.2～1.4 m、y≈0 m`
2. 中・左：`x=1.5～1.8 m、y=+0.30～+0.40 m`
3. 中・右：左配置とは異なるxで、`y=-0.30～-0.40 m`
4. 遠・中央：`x=1.9～2.2 m、y≈0 m`

内側交点領域は幅160 mm、高さ100 mmです。ボード中心を実測した場合、垂直・正対配置の左上内側交点は`[center_x, center_y + 0.080, center_z + 0.050]`です。各配置で実際に画面内へ明瞭に写る高さへ置き、原点のx/y/zを個別に測ります。必要写真は各配置1枚、合計4枚です。同じ静止配置の連写を多数入れると、その配置だけを過大評価するので使用しません。

水平器なしでも、3個の内側交点の3D座標を別々に正確に測れば軸を算出できますが、作業量と誤差が増えます。通常は小型水平器または垂直を保証できる直角治具を使う方が簡単です。

実機NUCで、Camera V7、人物検知、AR ViewerがC920を使用していないことを確認してからPNGを保存します。オートフォーカス、ズーム、解像度は変更しません。各配置を測定後、Spaceまたは`C`を1回押します。

```bash
set -euo pipefail
source /opt/ros/jazzy/setup.bash
source /home/koko-chan-nuc/ar_navigation_project/install/setup.bash

EXT_ROOT=/home/koko-chan-nuc/ar_navigation_project/calibration/extrinsics
mkdir -p "$EXT_ROOT/images"

# 読み取り確認だけ。値を変更しない。
v4l2-ctl -d /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_66F7D3AF-video-index0 --all \
  > "$EXT_ROOT/v4l2_before_capture.txt"

ros2 run ar_path_viewer calibrate_camera capture \
  --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_66F7D3AF-video-index0 \
  --output-dir "$EXT_ROOT/images" --fps 20
```

4枚を開発PCへコピーし、[`checkerboard_extrinsics_layout_example.yaml`](config/checkerboard_extrinsics_layout_example.yaml)を別名でコピーします。例の座標をすべて実測値へ置換し、写真パスを直し、最後にだけ`survey_confirmed: true`へ変更します。未確認のテンプレートはツールが拒否します。

```bash
set -euo pipefail
source /opt/ros/jazzy/setup.bash
source /home/inoue-yuki/care_patrol_ws/local_repositories/nakanoshima-robot-challenge/ar_ws/install/setup.bash

EXT_ROOT=/home/inoue-yuki/ar_navigation_project/calibration/extrinsics
PKG_ROOT=/home/inoue-yuki/care_patrol_ws/local_repositories/nakanoshima-robot-challenge/ar_ws/src/ar_path_viewer

cp "$PKG_ROOT/config/checkerboard_extrinsics_layout_example.yaml" \
  "$EXT_ROOT/measured_layout.yaml"
# measured_layout.yamlを編集し、実測後にsurvey_confirmed: trueとする。

ros2 run ar_path_viewer calibrate_camera extrinsics-board \
  --intrinsics "$PKG_ROOT/config/c920_intrinsics_development_candidate.yaml" \
  --layout "$EXT_ROOT/measured_layout.yaml" \
  --output "$EXT_ROOT/extrinsics_review.yaml" \
  --preview-dir "$EXT_ROOT/previews" \
  --review-overlay "$EXT_ROOT/runtime_overlay_review_only.yaml" \
  --monte-carlo-iterations 500
```

各previewでは緑丸が検出交点、紫十字が共同推定、橙十字が現行7.5°設定、赤丸がYAML原点へ対応する交点0です。`corner_order: auto`は180°の検出順序を全配置の整合性から選びますが、赤丸が実際に測った左上内側交点に対応することを必ず目視します。違う場合はその配置だけ`normal`または`reversed`へ固定して新しい出力先で再実行します。

ツールは次を報告します。

- 全配置共同RMS、最大誤差、配置ごとのRMS
- 1配置ずつ除いたときのcamera x/y/z、yaw/pitch/rollの変動
- YAMLの位置・角度測定誤差を加えたMonte Carlo感度
- 代替内部校正YAMLを`--sensitivity-intrinsics`へ渡した場合の外部姿勢差
- 配置が全て同一平面、前後差不足、左右差不足、単位・軸長・直交の誤り

全配置が同じxの垂直面にある場合は、54点が何枚あっても全体が一平面なので拒否します。配置数ではなく、異なる前後距離と左右位置を持つことが必要です。

#### 予備方式：個別に実測した10～12点

座標は`base_footprint`原点を基準に、x=前、y=左、z=上のm単位です。近距離床が写らないため床点だけに依存せず、遠方床と壁・自立板の高さ違いを組み合わせ、非共面の10～12点を使います。例としてx=4、6、8 m、y=-0.6、+0.6 m付近の床点、およびx=4、6 m、y=-0.6、+0.6 m、z=0.6、1.2 m付近の垂直面上の点を、実際の安全な廊下に合わせて配置します。数字をそのまま仮定せず、各点を`base_footprint`基準で実測してください。

基準点YAMLは次の形式です。

```yaml
points:
  - {name: floor_far_left, base_xyz_m: [6.002, 0.604, 0.000]}
  - {name: floor_far_right, base_xyz_m: [6.001, -0.597, 0.000]}
  - {name: wall_left_low, base_xyz_m: [4.010, 0.601, 0.603]}
  - {name: wall_left_high, base_xyz_m: [4.011, 0.602, 1.198]}
# 実測点を合計10～12点記入する
```

ロボットを動かさず撮った静止画で、YAML順に印の中心をクリックします。Backspaceで1点戻り、Enterで全点を保存します。

```bash
ros2 run ar_path_viewer calibrate_camera mark-points \
  --image ~/ar_navigation_project/calibration/extrinsics/reference.png \
  --reference-points ~/ar_navigation_project/calibration/reference_points.yaml \
  --output ~/ar_navigation_project/calibration/correspondences.yaml

ros2 run ar_path_viewer calibrate_camera extrinsics \
  --intrinsics ~/ar_navigation_project/calibration/intrinsics.yaml \
  --correspondences ~/ar_navigation_project/calibration/correspondences.yaml \
  --output ~/ar_navigation_project/calibration/extrinsics.yaml

ros2 run ar_path_viewer calibrate_camera export \
  --intrinsics ~/ar_navigation_project/calibration/intrinsics.yaml \
  --extrinsics ~/ar_navigation_project/calibration/extrinsics.yaml \
  --output ~/ar_navigation_project/calibration/review_only_overlay.yaml
```

位置・姿勢を独立に撮り直した2～3組でも求め、sample RMSが概ね2 px以下、yaw/pitch/rollと位置の組間差が測定誤差に収まることを確認します。壁の平面だけなど共面点のみで得た解や、左右符号が実景と合わない解は採用しません。

### 校正前後の比較と設定反映

`path_points`は実Pathをbase座標へ変換して保存したCSV（x,y,z列）または同じ点列を持つYAMLです。比較画像は左が現設定、右が歪み補正後の校正候補で、走行制御には接続しません。

```bash
ros2 run ar_path_viewer calibrate_camera compare \
  --image reference.png --path-points path_base.csv \
  --before-config ar_path_viewer_nav2_test.yaml \
  --after-overlay review_only_overlay.yaml \
  --output projection_before_after.png
```

採用時もNUC固有YAML全体を置換しません。バックアップを作り、review overlayの`fx/fy/cx/cy`、歪み係数、undistort、確認済みの外部パラメータだけを手作業で反映し、C920デバイス、MP4出力、ログ、診断設定を維持します。歪み補正は初期値`false`であり、実測結果を採用した場合だけ有効にします。補正後も出力と録画は640×480です。

## 進行方向とPlan残長

橙色の進行方向表示は、実Pathの現在位置に最も近い点から1.5 m先までをたどって`GO STRAIGHT`、`TURN LEFT`、`TURN RIGHT`を判定します。緑線が見える`ACTIVE`時も、緑線を描画できる線分がない`PATH OUT OF VIEW`時も、状態バー直下の固定ナビカードとして表示します。LEFT/RIGHTは斜め前進と誤読しにくい90度の折れ矢印です。床経路が集中する画面下部を避けるため、実映像・緑線・水色線を従来より隠しません。いずれも床へ投影した架空線ではなく、受信した実`/plan`だけが根拠です。Pathを延長、移動、変形しません。

次の場合は誤案内防止のため矢印も出しません。

- `NO PATH`、`PATH STALE`、`LOCALIZING`、`LOCALIZATION UNSTABLE`、TF/カメラ/設定異常
- 終点まで0.6 m以内、残りPathが0.6 m未満
- 実Pathがロボットから1.0 m以上離れて始まる
- 進行方位が前方±80°を超える、1.5 m内で方位が65°以上急変する、または左右をまたいで曖昧
- Followerがpause・待機・人物探索中、状態が古い、またはgoal世代変更後の新しい`/plan`をまだ受信していない

正しく投影できる部分は従来どおり緑線を維持し、その形状と線上矢印を優先します。橙色は緑と明確に区別し、MP4にも案内として記録されます。LEFT/RIGHT/STRAIGHTが変わった直後は旧矢印を即座に消し、既定0.35秒にわたり同じ最新判断が続くまで`DIRECTION CHECK`として案内を保留します。古い経路を補間したり、見かけだけ滑らかにつなぐ処理ではありません。判定根拠は`offscreen_guidance`ログへ集約して残します。閾値は`offscreen_guidance_*`、`guidance_direction_hold_sec` ROS parametersで調整できます。

水色参考Pathの破線間隔・色・座標は維持し、明るい壁でも見失いにくい暗色の細い縁取りだけを描きます。緑の実Pathは後から描画されるため、重なった場所でも走行用経路が前面です。短い`/plan`から「曲がり角までの距離」を安定して識別できないため、その距離は推測表示しません。

大きな再計画では、旧Pathを保持したり新旧を補間せず、新しい実`/plan`を既定0.35秒だけ非表示にして`REPLANNING` / `PATH CHANGE CHECK`を表示します。通常の走行進捗と、既にPathに含まれていた曲がり角への進入はmap座標上で旧Pathの対応位置へ合わせて比較するため、再計画とは扱いません。既定閾値は先頭2 m内の形状差0.60 mまたは方向差35度で、`path_change_*`パラメータから変更できます。

Followerがpause、人物探索、task wait、holding stopを明示している間は、古い緑線、方向案内、`PLAN LEFT`を消し、`NAVIGATION HOLD`と実際の状態を表示します。goal/Waypoint世代の切替中は、技術用語の`SWITCHING GOAL`ではなく`UPDATING ROUTE`と表示します。Follower状態がない通常Viewerでは従来どおり`/plan`だけで動作します。緑線は`path_overlay_alpha`（既定0.82）で半透明にし、人物や周辺映像を完全には覆いません。これは深度や人物マスクに基づく遮蔽ではなく、線の前後関係を保証する機能ではありません。

`waypoint_distance_enabled: true`では、Followerの読み取り専用`/waypoint_follower/reference_state`を安全状態の確認に使い、画面左下に`PLAN LEFT: 2.4 m`を表示します。距離は直線距離でも次Waypointまでの距離でもなく、ロボットに最も近いPath位置からPath終点までの実`/plan`ポリライン残長です。1 m未満はcm表示です。`to_pose`と`through_poses`のどちらでも意味は「現在受信しているPlanの残長」に限定します。

Followerのpause・待機・人物探索、状態期限切れ、AMCL/TF不良、Path開始位置がロボットから離れすぎている場合は表示しません。route/index/goal generationが変わると直ちに旧距離を消し、その変更後に受信した新しい`/plan`まで再表示を待ちます。Waypoint座標やWaypointまでの距離を推測しないため、現インターフェースでは`NEXT WP`という表記を使用しません。

Launcher2の旧Followerを使う場合、専用Launchの読み取り専用compatibility bridgeが
Followerパラメータ、Nav2 goal UUID、goal変更後の`/plan`終点を選択ルートへ一意に
照合できたときだけ同じ状態契約を供給します。照合後の緑線・橙色案内・
`PLAN LEFT`の意味は新Follower時と同一です。pause/person searchの種別は旧実装から
読めないため`STOPPED`までしか表示せず、終点が曖昧、route外、通信断の場合は方向・
距離・水色線をすべて非表示にします。単にFollowerゲートを解除する動作ではありません。

`/navigate_to_pose/_action/feedback`には`distance_remaining`がありますが、現行の読み取り専用`WaypointFollowerState`はNav2 Action goal UUIDを公開していません。そのためViewerだけでは、同じROS graph上のfeedbackが現在のFollower goalと厳密に一致することを証明できません。誤ったWaypoint距離を表示しないため、goal UUIDを安全に照合できるインターフェースが導入されるまでは`PLAN LEFT`を維持します。

通常画面はカメラ映像、緑実線、水色破線、方向、`PLAN LEFT`、必須状態だけを表示し、点数・TF・投影統計などの詳細を上部へ出しません。`D`キーでデバッグ表示を切り替えられます。`Q/F10`終了、`F11`全画面、ログ表示・コピーは従来どおりです。通常/デバッグのどちらも、GUI操作バーを除いた描画結果がMP4へ記録されます。

## 長い参考経路の分離設計

2走目の約2 m PathはWaypoint Followerの`to_pose`が次の1地点へ使う実Pathです。Viewerがその先を緑線として足すことはしません。Phase 1・2ではAR Viewerとは別の読み取り専用参考計画ノードが、Followerの安全な進行状態と今後のWaypoint列を受け、独立Nav2の`ComputePathThroughPoses`を3秒周期で呼び、その結果だけを`/ar/reference_path`へ出す構成を実装しました。Viewer自身は購読専用を維持します。

### Phase 3 長距離参考Path表示

走行用`/plan`は従来どおり緑実線（進行矢印付き）です。独立PlannerがNav2で計算した参考Pathは、水色破線と`REFERENCE`表示で重ねます。参考線は同じTF・AMCLゲート・3Dクリッピング・カメラモデルを通り、画角内へ実際に入る線分だけを描画します。画面へ移動、直線延長、走行用緑線への代用はしません。橙色の画角外案内は走行用`/plan`だけを根拠とする従来仕様です。

Viewerは生の`/ar/reference_path`だけでは採用しません。Pathとroute SHA-256、Waypoint index、navigation mode、goal generation、安全区間終端、Follower状態時刻、Planner generationを一体化した`/ar/reference_path_state`を購読し、別途受信した最新`/waypoint_follower/reference_state`と完全一致した場合だけ表示します。pause、人物探索、待機、route/index/generation変更、Plannerの空/失敗通知、Follower状態2.5秒超、参考Path 6.5秒超では即座に破線を消します。更新順序の異なるPathとメタデータを組み合わせることはありません。

本番の参考CostmapとKeepoutの一致はPhase 4で確認するため、`reference_path_enabled`の既定値は`false`です。開発PCの分離環境だけで次のように明示的に有効化します。

```bash
ros2 run ar_path_viewer ar_path_viewer --ros-args \
  --params-file ar_ws/src/ar_path_viewer/config/ar_path_viewer.yaml \
  -p reference_path_enabled:=true
```

診断ログの`reference_path_state`には有効性、route/index/generation、点数、全長を、`reference_path_projection`には画角内実長、描画線分数、非表示理由を残します。録画はGUI操作バーを含まない640×480フレームへ、実際に描いた緑実線・水色破線・橙案内をそのまま保存します。

ロボットが実際に認識しているcostmap/LiDAR情報を将来AR表示するための、安全な
読み取り専用設計は[docs/robot_perception_overlay_design.md](docs/robot_perception_overlay_design.md)
に分離しています。現段階では実装・既定有効化しておらず、「高コスト」を未検証の
まま「危険」と表示しません。

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

Qt HighGUIでは自由比率の簡素GUIを使用し、Qt側のツールバーと二重レターボックスを無効にします。4:3調整はAR Viewerが取得した実際の画像表示領域全体に対して1回だけ行います。

### Phase 6 ログ操作と自動保存

OpenCV画面の下部に表示専用の操作バーを追加します。ボタンは「ログを見る」と「ログをコピー」です。OpenCV FreeTypeまたは日本語フォントが利用できない環境では、操作を失わないよう`LOG VIEW`と`LOG COPY`へフォールバックします。操作バー、ボタン、コピー結果表示は画面表示用キャンバスだけに描き、640×480のMP4フレームには入りません。リサイズ・F11全画面中も映像領域だけを4:3で維持し、その下に操作バーを確保します。

ボタン文字はフォントのベースライン値だけに依存せず、実際に描画したグリフの外接矩形を水平・垂直中央に配置します。日本語と英語フォールバックのどちらも同じ方法で中央揃えします。

- `LOG VIEW`: 選択・スクロール可能な別ウィンドウを開きます。`最新ログへ更新`で現在の内容へ更新できます。閉じてもAR表示・録画・ROS処理は継続します。
- `LOG COPY`: ChatGPT/Codexへ貼り付けられるテキストをOSクリップボードへコピーします。成功・失敗は操作バーへ約3秒表示します。
- Tk/Display/Clipboardが利用できない場合は警告ログを残し、カメラ表示・録画・Nav2購読を停止しません。

ログは起動時からメモリとUTF-8ファイルへ同時記録します。`output_video`が設定されていれば同じディレクトリ・同じstemの`.log`を使用します。

```text
kokochan_navi_20261106_182206.mp4
kokochan_navi_20261106_182206.log
```

動画またはログがすでに存在する場合は、両方に`_001`、`_002`等を付けて既存データを上書きしません。動画を記録しない場合は`log_output_dir`へ`kokochan_navi_YYYYMMDD_HHMMSS.log`を作成します。ファイル保存に失敗してもメモリ上のログ表示・コピーは継続します。

```yaml
log_output: ""  # 空ならoutput_videoと同じstem
log_output_dir: ~/ar_navigation_project/outputs
log_max_entries: 4000
log_copy_max_chars: 100000
```

ファイルには全ての集約済み診断を保存します。コピー内容が上限を超える場合は、起動情報、`camera_configuration`、最新の`viewer_state`・`plan_rx`・`path_projection`、警告・エラー・終了理由を優先し、直近ログで残りを埋めます。ファイル側はこのコピー上限では切り詰めません。別ログウィンドウとクリップボードには`python3-tk`が必要です。

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

`diagnostic_logging: false`の場合も、起動日時、出力先、実効カメラ設定、直接検出したエラー、終了理由はセッションログへ残ります。Path・TF・AMCLの追加診断を取得する実機調査では`true`を使用します。

主なログ：

- `plan_rx`: 累積受信回数、monotonic受信時刻、Path header stamp、frame_id、点数
- `plan_adopted` / `plan_updated`: 有効Pathのgeneration、frame_id、点数
- `plan_discarded`: 空Path、NaN/Inf、不正frame_id等の破棄理由
- `viewer_state`: `STARTUP -> NO PATH`、`NO PATH -> ACTIVE`、`ACTIVE -> PATH STALE`等
- `tf_unavailable`: source/target frame、例外型、TF lookupの原因本文
- `path_out_of_view`: 3Dクリップ後にカメラ視錐台と交差する線分がないこと
- `path_projection`: TF変換後点数、入力線分数、Path全長、ロボットから終点までの距離、カメラ前方部分の実長、視錐台内部分と前方だが非表示の実長、前方全体と画角内線分それぞれの深度範囲、画像内点数、後方・near内・左右・上下の点数、描画線分数、可視画素長・表示範囲
- `camera_configuration`: 実際に読み込まれた画像サイズ、内部パラメータ、カメラ位置・姿勢、near clip（起動時に1回）
- `path_display_configuration`: `ground` / `elevated`、表示高さ、Nav2 XY保持（起動時に1回）
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

## 実機NUCへの反映

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

Phase 6ではTkログウィンドウ、クリップボード、日本語ボタン表示を使うため、NUCへ一度だけ依存パッケージを導入します。

```bash
sudo apt-get update
sudo apt-get install --yes python3-tk fonts-noto-cjk
python3 -c 'import tkinter; print("tkinter OK")'
mkdir -p /home/koko-chan-nuc/ar_navigation_project/outputs
test -w /home/koko-chan-nuc/ar_navigation_project/outputs
```

既存のNUC固有YAMLにPhase 6パラメータがなくても、コード既定値で自動保存と操作バーは動作します。既存`output_video`が設定されていれば同名stemの`.log`を自動作成します。保存先だけを明示したい場合は、NUC固有YAMLをバックアップして`ros__parameters`内へ次を追加します。

```yaml
log_output: ""
log_output_dir: /home/koko-chan-nuc/ar_navigation_project/outputs
log_max_entries: 4000
log_copy_max_chars: 100000
```

既存のC920デバイスパス、`output_video`、診断設定、デスクトップアイコン、起動スクリプトは維持します。デスクトップアイコンの再作成は不要です。

Git管理外のNUC固有YAMLは、他の値をGitHub版で置換せず、`camera_y`の1行だけを更新します。

```bash
set -euo pipefail
NUC_CONFIG=/home/koko-chan-nuc/ar_navigation_project/config/ar_path_viewer_launcher2.yaml
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
