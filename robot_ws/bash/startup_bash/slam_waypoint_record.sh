#!/bin/bash

# Record distance-based or Enter-triggered waypoints during SLAM, then save /map.

set -u

runtime_mode="${1:-real}"
case "${runtime_mode}" in
    real) use_sim_time=false ;;
    sim) use_sim_time=true ;;
    *)
        echo "Usage: $0 real|sim" >&2
        exit 2
        ;;
esac

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
maps_dir="${workspace_root}/maps_waypoints/maps"
waypoints_dir="${workspace_root}/maps_waypoints/waypoints"
mkdir -p "${maps_dir}" "${waypoints_dir}"

read -r -p "地図・ウェイポイント名（.yamlなし） [slam_route]: " route_name
route_name="${route_name:-slam_route}"
route_name="${route_name%.yaml}"
if [ -z "${route_name}" ] || [ "${route_name}" = "." ] || \
   [ "${route_name}" = ".." ] || [ "$(basename -- "${route_name}")" != "${route_name}" ]; then
    echo "名前にはファイル名だけを指定してください。" >&2
    exit 1
fi

read -r -p "記録方法 [distance/enter] [distance]: " record_mode
record_mode="${record_mode:-distance}"
distance_threshold=2.0
continuous=false
enter_to_record=false
case "${record_mode}" in
    distance)
        continuous=true
        read -r -p "ウェイポイント記録間隔[m] [2.0]: " distance_threshold
        distance_threshold="${distance_threshold:-2.0}"
        if ! [[ "${distance_threshold}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
           ! awk -v value="${distance_threshold}" 'BEGIN { exit !(value > 0) }'; then
            echo "記録間隔には0より大きい数値を指定してください。" >&2
            exit 1
        fi
        if [[ "${distance_threshold}" =~ ^[0-9]+$ ]]; then
            distance_threshold="${distance_threshold}.0"
        fi
        ;;
    enter)
        enter_to_record=true
        ;;
    *)
        echo "記録方法は distance または enter を指定してください。" >&2
        exit 1
        ;;
esac

waypoint_file="${waypoints_dir}/${route_name}.yaml"
map_stem="${maps_dir}/${route_name}"
if [ -e "${waypoint_file}" ] || [ -e "${map_stem}.yaml" ] || [ -e "${map_stem}.pgm" ]; then
    read -r -p "同名データがあります。waypointへ追記し、地図を更新しますか [y/N]: " overwrite
    if ! [[ "${overwrite:-}" =~ ^[yY]$ ]]; then
        echo "キャンセルしました。"
        exit 0
    fi
fi

echo "========================================="
echo " SLAM 地図＋ウェイポイント記録 (${runtime_mode})"
echo "========================================="
echo "ウェイポイント: ${waypoint_file}"
echo "地図: ${map_stem}.yaml / ${map_stem}.pgm"
if [ "${continuous}" = true ]; then
    echo "記録間隔: ${distance_threshold} m"
else
    echo "このターミナルでEnterを押すたびに現在位置を記録します。"
fi
echo "前提: SLAM Toolboxが起動し、map -> ${ROBOTBASE_TF_PREFIX:-robot}/base_footprint が必要です。"
echo "走行中に p キーを押すと、その地点を10秒停止ウェイポイントとして記録します。"
echo "走行後に Ctrl+C を押すと、記録を止めて現在の /map を保存します。"
echo ""

record_status=0
ros2 run robotbase_waypoint waypoint_record --ros-args \
    -p use_sim_time:="${use_sim_time}" \
    -p output_file:="${waypoint_file}" \
    -p continuous:="${continuous}" \
    -p enter_to_record:="${enter_to_record}" \
    -p distance_threshold:="${distance_threshold}" \
    -p base_frame:="${ROBOTBASE_TF_PREFIX:-robot}/base_footprint" || record_status=$?

if [ "${record_status}" -ne 0 ] && [ "${record_status}" -ne 130 ]; then
    echo "ウェイポイント記録が異常終了しました（status=${record_status}）。" >&2
    exit "${record_status}"
fi

echo "現在のSLAM地図を保存しています..."
if ros2 run nav2_map_server map_saver_cli -f "${map_stem}" --ros-args \
    -p use_sim_time:="${use_sim_time}"; then
    echo "保存完了:"
    echo "  ${waypoint_file}"
    echo "  ${map_stem}.yaml"
    echo "  ${map_stem}.pgm"
else
    echo "地図保存に失敗しました。SLAMの /map を確認してください。" >&2
    echo "ウェイポイントは保存済みです: ${waypoint_file}" >&2
    exit 1
fi
