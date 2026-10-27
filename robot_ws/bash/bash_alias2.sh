# ココちゃん（robotbase_ws）専用ショートカット。
# Sirius側と同時に読み込めるよう、公開名はすべて koko_* にする。

# 旧名は再source時にも残さない。
unalias koko_nav2_sim 2>/dev/null || true

# PRESET: 実機基本
# PRESET_ITEMS: koko_roboteq,koko_velodyne,koko_hokuyo,koko_imu,koko_sf_real,koko_twist_mux

# PRESET: 地図生成（sim）
# PRESET_ITEMS: koko_sim,koko_slamtoolbox_sim,koko_rviz_sim

# PRESET: 自律移動（sim）
# PRESET_ITEMS: koko_sim,koko_nav2_sim_map,koko_rviz_sim

# PRESET: SLAMしながら自律移動（sim）
# PRESET_ITEMS: koko_sim,koko_nav2_sim_slam,koko_rviz_sim

# PRESET: SLAMしながら自律移動（実機）
# PRESET_ITEMS: koko_roboteq,koko_velodyne,koko_hokuyo,koko_imu,koko_sf_real,koko_twist_mux,koko_nav2_real_slam,koko_rviz_real

# PRESET: 地図＋ウェイポイント記録（実機）
# PRESET_ITEMS: koko_roboteq,koko_velodyne,koko_hokuyo,koko_imu,koko_sf_real,koko_twist_mux,koko_nav2_real_slam,koko_rviz_real,koko_keyop2,koko_slam_waypoint_record

# PRESET: 地図＋ウェイポイント記録（sim）
# PRESET_ITEMS: koko_sim,koko_nav2_sim_slam,koko_rviz_sim,koko_keyop2_sim,koko_slam_waypoint_record_sim

# TAB: センサー・ハードウェア
# GROUP: センサー・ハードウェア

# Roboteq起動(udevルール設定済み前提)
alias koko_roboteq='koko_src && ros2 launch robotbase_bringup roboteq.launch.py pub_odom_tf:=false tf_prefix:=${ROBOTBASE_TF_PREFIX}'

alias koko_roboteq_no_sf='koko_src && ros2 launch robotbase_bringup roboteq.launch.py pub_odom_tf:=true tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# Velodyne起動
alias koko_velodyne='koko_src && ros2 launch robotbase_bringup velodyne.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# Ethernet Hokuyo 2D LiDAR起動
# TFはrobotbase.urdfから <tf_prefix>/base_footprint -> <tf_prefix>/hokuyo_link を配信する。
alias koko_hokuyo='koko_src && ros2 launch urg_node2 urg_node2_ether.launch.py ip_address:=${ROBOTBASE_HOKUYO_IP} ip_port:=${ROBOTBASE_HOKUYO_PORT} frame_id:=${ROBOTBASE_TF_PREFIX}/hokuyo_link scan_topic_name:=hokuyo_scan'

# IMU起動(udevルール設定済み前提)
alias koko_imu='koko_src && ros2 launch robotbase_bringup imu.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# TAB: シミュレーション
# GROUP: シミュレーション

# Gazebo Sim本体（VLP16 + IMU、RViz/SLAM/Nav2は起動しない）
alias koko_sim='koko_sim_src && bash "${_KOKO_WS}/bash/startup_bash/start_sim.sh" tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# RViz2のみ起動（Gazeboクロック、SLAM/Nav2は起動しない）
alias koko_rviz_sim='koko_sim_src && ros2 launch robotbase_sim rviz.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# 地図作成用SLAM Toolboxのみ起動（koko_simを先に起動）
alias koko_slamtoolbox_sim='koko_sim_src && ros2 launch robotbase_sim mapping.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# 一覧から既存地図を選ぶNav2（koko_simを先に起動）
alias koko_nav2_sim_map='koko_sim_env && bash "${_KOKO_WS}/bash/startup_bash/nav2_bringup_sim.sh"'

# 地図なし: SLAM ToolboxとNav2を同時起動（koko_simを先に起動）
alias koko_nav2_sim_slam='koko_sim_src && ros2 launch robotbase_sim navigation_slam.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# シミュレーションROS Domainへ手動速度を送る。
alias koko_keyop2_sim='koko_sim_src && ros2 run robotbase_keyop robotbase_keyop_v2 --ros-args --params-file "${ROBOTBASE_PARAMS_DIR}/sim/keyop.yaml"'

# シミュレーションの/mapを保存する。
alias koko_map_save_sim='koko_sim_env && bash "${_KOKO_WS}"/bash/startup_bash/map_save.sh'

# YAMLを選択してシミュレーションでウェイポイント追従（Nav2起動後に使用）
alias koko_waypoint_follow_sim='koko_sim_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_follow.sh sim'

# 現在位置をウェイポイントへ1点保存 / 走行距離ごとに連続保存
alias koko_waypoint_save_sim='koko_sim_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_record.sh once sim'
alias koko_waypoint_record_sim='koko_sim_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_record.sh distance sim'
alias koko_slam_waypoint_record_sim='koko_sim_src && bash "${_KOKO_WS}"/bash/startup_bash/slam_waypoint_record.sh sim'

# シミュレーションのrosbag記録（sim時刻・/clock対応、MCAP内部Zstd圧縮）
alias koko_record_rosbag_sim='koko_sim_env && bash "${_KOKO_WS}"/bash/startup_bash/record_rosbag.sh sim'

# TAB: ユーティリティ
# GROUP: ユーティリティ

alias koko_install_packages='sudo apt update && sudo apt install xterm -y && \
sudo apt install ros-jazzy-rqt-tf-tree -y && \
sudo apt install ros-jazzy-foxglove-bridge -y && \
sudo apt-get install libqt5serialport5-dev'

# 実機ROS Domain用Foxglove WebSocketサーバー（Siriusの8765と分離）
alias koko_foxglove='koko_src && bash "${_KOKO_WS}/bash/startup_bash/foxglove_server.sh"'

# シミュレーションROS Domainを見るFoxgloveサーバー（実機用とは同時起動しない）。
alias koko_foxglove_sim='koko_sim_src && bash "${_KOKO_WS}/bash/startup_bash/foxglove_server.sh"'

# Behavior Tree 可視化ツール Groot2 起動
alias koko_groot2='$HOME/Groot2/groot2.sh'

# Behavior Tree Docker 操作
alias koko_bt_start='xhost +local:docker > /dev/null 2>&1 && cd "${_KOKO_WS}"/bt_jazzy_docker && docker compose run --rm --name koko_bt_dev_container bt_dev'
alias koko_bt_enter='docker exec -it koko_bt_dev_container bash'

# PCL 3D点群物体検出ノード起動 (Velodyne等の点群をリアルタイムクラスタリング・バウンディングボックス化します)
alias koko_pcl_detect='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/pcl_detect.sh'

# TAB: ナビゲーション
# GROUP: ナビゲーション

# 手動操作 V2（優先順位対応版）
alias koko_keyop2='koko_src && ros2 run robotbase_keyop robotbase_keyop_v2 --ros-args --params-file "${ROBOTBASE_PARAMS_DIR}/common/keyop.yaml"'

# YAMLを選択して実機でウェイポイント追従（Nav2起動後に使用）
alias koko_waypoint_follow='koko_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_follow.sh real'

# 現在位置をウェイポイントへ1点保存 / 走行距離ごとに連続保存
alias koko_waypoint_save='koko_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_record.sh once real'
alias koko_waypoint_record='koko_src && bash "${_KOKO_WS}"/bash/startup_bash/waypoint_record.sh distance real'
alias koko_slam_waypoint_record='koko_src && bash "${_KOKO_WS}"/bash/startup_bash/slam_waypoint_record.sh real'

# ココちゃんランチャー起動
alias koko_launcher2='bash "${_KOKO_WS}/bash/start_launcher2.sh"'

# TAB: リアル実験
# GROUP: リアル実験

# 実機用の速度指令優先順位制御（シミュレータではkoko_sim内で起動済み）
alias koko_twist_mux='koko_src && ros2 launch robotbase_bringup twist_mux.launch.py'

# Nav2起動(任意MAP、実時間)
alias koko_nav2_real='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/nav2_bringup_real.sh'

# SLAMToolbox起動(実時間)
alias koko_slamtoolbox_real='koko_src && ros2 launch robotbase_bringup slam.launch.py use_sim_time:=false tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# 地図を生成・更新しながらNav2を実行（実時間、map server / AMCLなし）
alias koko_nav2_real_slam='koko_src && ros2 launch robotbase_bringup navigation_slam.launch.py tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# Sensor Fusion起動(実時間)
alias koko_sf_real='koko_src && ros2 launch robotbase_bringup sensor_fusion.launch.py use_sim_time:=false tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# マップ保存起動
alias koko_map_save='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/map_save.sh'

# RViz2のみ起動（実時間）
alias koko_rviz_real='koko_src && ros2 launch robotbase_bringup rviz.launch.py use_sim_time:=false publish_description:=false tf_prefix:=${ROBOTBASE_TF_PREFIX}'

# マップ切り替え（Nav2実行中に地図を変更）
alias koko_change_map='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/change_map.sh'

# ROS2 bag記録起動
alias koko_record_rosbag='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/record_rosbag.sh'

# 好きなタイミングで実験メモを送信するエイリアス
# 使い方: koko_pub_memo "送信したい内容"
alias koko_pub_memo='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/pub_memo.sh'

# # マップ切り替え（プログラム呼び出し用、引数に地図名を指定）
# # 例: koko_change_map_simple 1202-15f
# alias change_map_simple='koko_env && bash "${_KOKO_WS}"/bash/startup_bash/change_map_simple.sh'
