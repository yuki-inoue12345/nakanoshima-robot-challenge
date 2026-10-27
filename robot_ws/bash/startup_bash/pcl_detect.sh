#!/bin/bash

_ROBOTBASE_SCRIPT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# PCL Object Detection Node Startup Script
# -------------------------------------------------------------
trap 'echo ""; echo "PCL Object Detection を終了しました。"' SIGINT

cd "${_ROBOTBASE_SCRIPT_WS}"
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "=========================================================="
echo "  PCL Object Detection (点群ベースの3D物体検出・クラスタリング)"
echo "=========================================================="
echo "このノードは、入力された3D点群(PointCloud2)を距離情報に基づいて"
echo "クラスタ(最大6個)に分割し、それぞれの点群をトピック(cluster_0〜5)に"
echo "出力するほか、Rviz2で可視化するための境界ボックス(viz)を出力します。"
echo "----------------------------------------------------------"
INPUT_TOPIC="/velodyne_points"
echo "----------------------------------------------------------"
echo "起動中..."
echo "  入力トピック: $INPUT_TOPIC"
echo "  出力バウンディングボックス: /viz (MarkerArray)"
echo "  出力個別クラスタ: /cluster_0 〜 /cluster_5"
echo "  (Rviz2で確認可能です)"
echo "----------------------------------------------------------"

ros2 run pcl_object_detection pcl_object_detection_node --ros-args --remap filtered_clouds:=$INPUT_TOPIC
