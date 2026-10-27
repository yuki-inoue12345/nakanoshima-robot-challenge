#!/usr/bin/env bash
# Recover a missed Nav2 lifecycle transition during simulation startup.
set -u

sleep 8
for attempt in {1..20}; do
    map_state="$(timeout 3s ros2 lifecycle get /map_server 2>/dev/null || true)"
    amcl_state="$(timeout 3s ros2 lifecycle get /amcl 2>/dev/null || true)"

    if [[ "$map_state" == "active [3]" && "$amcl_state" == "active [3]" ]]; then
        exit 0
    fi
    if [[ "$map_state" == "inactive [2]" ]]; then
        echo "map_serverがinactiveのため、地図配信を復旧します。"
        timeout 10s ros2 lifecycle set /map_server activate || true
    fi
    if [[ "$amcl_state" == "inactive [2]" ]]; then
        echo "AMCLがinactiveのため、自己位置推定を復旧します。"
        timeout 10s ros2 lifecycle set /amcl activate || true
    fi
    sleep 1
done

echo "警告: map_serverまたはAMCLがactiveになりませんでした。Nav2ログを確認してください。" >&2
