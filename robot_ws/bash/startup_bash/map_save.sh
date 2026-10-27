#!/bin/bash

trap 'echo ""; echo "Ctrl + Cが押されましたが、ウィンドウは閉じません"' INT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MAPS_DIR="${WORKSPACE_ROOT}/maps_waypoints/maps"

mkdir -p -- "$MAPS_DIR" || {
    echo "エラー: 地図保存先を作成できません: $MAPS_DIR" >&2
    exit 1
}

if [ ! -w "$MAPS_DIR" ]; then
    echo "エラー: 地図保存先に書き込み権限がありません: $MAPS_DIR" >&2
    exit 1
fi

cd -- "$WORKSPACE_ROOT" || exit 1

while :; do
    read -r -p "Press [Enter] key to start map save..."
    # shellcheck disable=SC1091
    source install/setup.bash
    read -r -p "Input map name (without .yaml): " map_name

    if [ -z "$map_name" ] || [ "$map_name" = "." ] || [ "$map_name" = ".." ] ||
       [ "$(basename -- "$map_name")" != "$map_name" ]; then
        echo "エラー: 地図名を1つ入力してください（/ は使用できません）"
        continue
    fi

    map_path="${MAPS_DIR}/${map_name}"
    if ros2 run nav2_map_server map_saver_cli -f "$map_path"; then
        echo "保存完了: ${map_path}.yaml / ${map_path}.pgm"
    else
        echo "保存失敗: /map トピックと上記エラーを確認してください" >&2
    fi
done
