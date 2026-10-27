#!/bin/bash

trap 'echo ""; echo "Ctrl + Cが押されましたが、ウィンドウは閉じません"' INT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SAVED_MAPS_DIR="${WORKSPACE_ROOT}/maps_waypoints/maps"
BUILTIN_MAP="${WORKSPACE_ROOT}/src/robotbase_sim/maps/test_arena.yaml"

map_paths=()
map_labels=()

build_map_list() {
    local map_file
    map_paths=()
    map_labels=()

    if [ -f "$BUILTIN_MAP" ]; then
        map_paths+=("$BUILTIN_MAP")
        map_labels+=("test_arena [同梱シミュレーション地図]")
    fi

    mkdir -p -- "$SAVED_MAPS_DIR"
    shopt -s nullglob
    for map_file in "$SAVED_MAPS_DIR"/*.yaml "$SAVED_MAPS_DIR"/*.yml; do
        map_paths+=("$map_file")
        map_labels+=("$(basename -- "$map_file" | sed -E 's/\.(yaml|yml)$//') [保存済み]")
    done
    shopt -u nullglob
}

print_map_list() {
    local index
    echo "========================================="
    echo "  シミュレーション用地図一覧"
    echo "========================================="
    for index in "${!map_paths[@]}"; do
        echo "  [$((index + 1))] ${map_labels[$index]}"
    done
    echo "========================================="
}

select_map() {
    local selection

    build_map_list
    if [ "${#map_paths[@]}" -eq 0 ]; then
        echo "エラー: 選択可能な地図がありません" >&2
        return 1
    fi

    print_map_list
    read -r -p "番号を入力してください (1-${#map_paths[@]}): " selection
    if [[ ! "$selection" =~ ^[0-9]+$ ]] ||
       [ "$selection" -lt 1 ] || [ "$selection" -gt "${#map_paths[@]}" ]; then
        echo "エラー: 無効な選択です"
        return 1
    fi

    selected_map="${map_paths[$((selection - 1))]}"
    selected_label="${map_labels[$((selection - 1))]}"
    return 0
}

cd -- "$WORKSPACE_ROOT" || exit 1
# shellcheck disable=SC1091
source install/setup.bash

if [ "${1:-}" = "--list" ]; then
    build_map_list
    print_map_list
    exit 0
fi

while :; do
    echo ""
    read -r -p "Press [Enter] key to select a simulation map..."
    selected_map=""
    selected_label=""

    if ! select_map; then
        continue
    fi

    echo "選択された地図: $selected_label"
    echo "パス: $selected_map"
    echo ""
    echo "Gazebo（koko_sim）を先に起動しておいてください。"

    # Recover missed lifecycle transitions while the launch process runs.
    bash "${SCRIPT_DIR}/ensure_sim_map_active.sh" &
    recovery_pid=$!
    ros2 launch robotbase_sim navigation.launch.py \
        map:="$selected_map" \
        tf_prefix:="${ROBOTBASE_TF_PREFIX:-robot}"
    kill "$recovery_pid" 2>/dev/null || true
    wait "$recovery_pid" 2>/dev/null || true
done

