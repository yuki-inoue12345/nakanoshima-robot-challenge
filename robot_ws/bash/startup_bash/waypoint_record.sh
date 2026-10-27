#!/bin/bash
set -u

record_mode="${1:-once}"
runtime_mode="${2:-real}"
case "${record_mode}" in
    once) continuous=false ;;
    distance) continuous=true ;;
    *)
        echo "Usage: $0 once|distance real|sim"
        exit 2
        ;;
esac
case "${runtime_mode}" in
    real) use_sim_time=false ;;
    sim) use_sim_time=true ;;
    *)
        echo "Usage: $0 once|distance real|sim"
        exit 2
        ;;
esac

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
waypoints_dir="${workspace_root}/maps_waypoints/waypoints"
mkdir -p "${waypoints_dir}"

read -r -p "保存先名（.yamlなし） [raw_waypoints]: " route_name
route_name="${route_name:-raw_waypoints}"
if [[ "${route_name}" == */* ]] || [[ "${route_name}" == .* ]]; then
    echo "保存先名にはファイル名だけを指定してください。"
    exit 1
fi
route_name="${route_name%.yaml}"
output_file="${waypoints_dir}/${route_name}.yaml"

distance_threshold=2.0
if [ "${continuous}" = true ]; then
    read -r -p "記録間隔[m] [2.0]: " distance_threshold
    distance_threshold="${distance_threshold:-2.0}"
    if ! [[ "${distance_threshold}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
       ! awk -v value="${distance_threshold}" 'BEGIN { exit !(value > 0) }'; then
        echo "記録間隔には0より大きい数値を指定してください: ${distance_threshold}" >&2
        exit 1
    fi
    if [[ "${distance_threshold}" =~ ^[0-9]+$ ]]; then
        distance_threshold="${distance_threshold}.0"
    fi
    echo "Ctrl+Cまで走行軌跡を記録します。"
    echo "走行中に p キーを押すと、その地点を10秒停止ウェイポイントとして記録します。"
fi

echo "保存先: ${output_file}"
exec ros2 run robotbase_waypoint waypoint_record --ros-args \
    -p use_sim_time:="${use_sim_time}" \
    -p output_file:="${output_file}" \
    -p continuous:="${continuous}" \
    -p distance_threshold:="${distance_threshold}" \
    -p base_frame:="${ROBOTBASE_TF_PREFIX:-robot}/base_footprint"
