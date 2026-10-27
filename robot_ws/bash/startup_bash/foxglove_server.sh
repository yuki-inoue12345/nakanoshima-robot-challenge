#!/bin/bash

_ROBOTBASE_SCRIPT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
    echo "Foxglove Bridgeがインストールされていません。"
    echo "次を一度実行してください:"
    echo "  sudo apt update && sudo apt install -y ros-jazzy-foxglove-bridge"
    exit 1
fi

foxglove_address="${ROBOTBASE_FOXGLOVE_ADDRESS:-0.0.0.0}"
foxglove_port="${ROBOTBASE_FOXGLOVE_PORT:-8766}"
foxglove_profile="${ROBOTBASE_FOXGLOVE_PROFILE:-foxglove.yaml}"
foxglove_params_root="${ROBOTBASE_PARAMS_DIR:-${_ROBOTBASE_SCRIPT_WS}/params}"
foxglove_params_file="${foxglove_params_root}/common/${foxglove_profile}"

echo "========================================="
echo "  ココちゃん Foxglove Bridge"
echo "========================================="
echo "PC IP: $(hostname -I)"
echo "接続先: ws://<上記IP>:${foxglove_port}"
echo "ローカル: ws://localhost:${foxglove_port}"
echo "待受: ${foxglove_address}:${foxglove_port}"
echo "設定: ${foxglove_params_file}"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-未設定}"
echo "========================================="

exec ros2 launch robotbase_bringup foxglove.launch.py \
    address:="${foxglove_address}" \
    port:="${foxglove_port}" \
    params_file:="${foxglove_params_file}"
