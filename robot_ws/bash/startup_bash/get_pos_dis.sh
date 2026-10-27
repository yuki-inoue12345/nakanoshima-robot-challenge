#!/bin/bash

_ROBOTBASE_SCRIPT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
trap 'echo ""; echo "Ctrl + Cが押されましたが、ウィンドウは閉じません"' 2
cd "${_ROBOTBASE_SCRIPT_WS}"
while : ;do
    read -p "Press [Enter] key to start get_position_distance..."
    source install/setup.bash
    ros2 run sirius_navigation get_position_distance
done