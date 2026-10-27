#!/bin/bash
set -u

mode="real"
navigation_mode="to_pose"
mode_selected=false
navigation_only=false
for argument in "$@"; do
    case "${argument}" in
        real|sim)
            if [ "${mode_selected}" = true ]; then
                echo "Usage: $0 [real|sim] [--navigation-only]"
                exit 2
            fi
            mode="${argument}"
            mode_selected=true
            ;;
        --navigation-only)
            navigation_only=true
            ;;
        *)
            echo "Usage: $0 [real|sim] [--navigation-only]"
            exit 2
            ;;
    esac
done
case "${mode}" in
    real) use_sim_time=false ;;
    sim) use_sim_time=true ;;
esac

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
waypoints_dir="${workspace_root}/maps_waypoints/waypoints"
mkdir -p "${waypoints_dir}"

mapfile -t waypoint_files < <(find "${waypoints_dir}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [ "${#waypoint_files[@]}" -eq 0 ]; then
    echo "ウェイポイントがありません: ${waypoints_dir}"
    echo "先に koko_waypoint_save または koko_waypoint_save_sim で地点を保存してください。"
    exit 1
fi

echo "========================================="
echo " ココちゃん ウェイポイント選択 (${mode})"
echo "========================================="
for index in "${!waypoint_files[@]}"; do
    printf '  [%d] %s\n' "$((index + 1))" "$(basename "${waypoint_files[index]}")"
done
read -r -p "番号を入力 [1]: " selection
selection="${selection:-1}"
if ! [[ "${selection}" =~ ^[0-9]+$ ]] || \
   [ "${selection}" -lt 1 ] || [ "${selection}" -gt "${#waypoint_files[@]}" ]; then
    echo "無効な選択です: ${selection}"
    exit 1
fi
waypoint_file="${waypoint_files[$((selection - 1))]}"

read -r -p "開始番号 [1]: " start_index
start_index="${start_index:-1}"
if ! [[ "${start_index}" =~ ^[0-9]+$ ]] || [ "${start_index}" -lt 1 ]; then
    echo "開始番号には1以上の整数を指定してください: ${start_index}" >&2
    exit 1
fi
echo "走行方式を選択："
echo "  1: NavigateToPose（従来方式）"
echo "  2: NavigateThroughPoses（連続走行）"
read -r -p "番号 [1]: " navigation_selection
case "${navigation_selection:-1}" in
    1) navigation_mode="to_pose" ;;
    2) navigation_mode="through_poses" ;;
    *)
        echo "無効な走行方式です: ${navigation_selection}" >&2
        exit 1
        ;;
esac
threshold="1.0"
threshold_args=()
if [ "$navigation_mode" = to_pose ]; then
    read -r -p "中間点の切替距離[m] [1.0]: " threshold
    threshold="${threshold:-1.0}"
    threshold_parameter="default_threshold"
else
    read -r -p "通常区間の切替距離[m] [6.0]: " threshold
    threshold="${threshold:-6.0}"
    threshold_parameter="section_switch_distance"
fi
if ! [[ "${threshold}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   ! awk -v value="${threshold}" 'BEGIN { exit !(value > 0) }'; then
    echo "切替距離には0より大きい数値を指定してください: ${threshold}" >&2
    exit 1
fi
# ROS 2 infers `2` as INTEGER, but this parameter is declared as DOUBLE.
if [[ "${threshold}" =~ ^[0-9]+$ ]]; then
    threshold="${threshold}.0"
fi
threshold_args=(-p "${threshold_parameter}:=${threshold}")
read -r -p "無限ループしますか [y/N]: " loop_answer
loop_value=false
if [[ "${loop_answer:-}" =~ ^[yY]$ ]]; then
    loop_value=true
fi

echo "ウェイポイント: ${waypoint_file}"
echo "開始番号: ${start_index}, loop: ${loop_value}"
if [ "$navigation_mode" = through_poses ]; then
    echo "走行方式: NavigateThroughPoses（区間上限40 m、通常区間の切替距離: ${threshold} m）"
else
    echo "走行方式: NavigateToPose（中間点の切替距離: ${threshold} m）"
fi
if [ "${navigation_only}" = true ]; then
    echo "人物探索連携: OFF (Navigation-only)"
else
    echo "人物探索連携: ON"
fi

# setup.cfg installs console scripts under <package-prefix>/lib/robotbase_waypoint.
# Keep the sourced launcher/alias environment, but select this workspace's
# executable directly: ros2 run would introduce another parent process.
install_layout=isolated
if [ -f "${workspace_root}/install/.colcon_install_layout" ]; then
    IFS= read -r install_layout < "${workspace_root}/install/.colcon_install_layout" || true
fi
case "${install_layout}" in
    isolated) install_prefix="${workspace_root}/install/robotbase_waypoint" ;;
    merged) install_prefix="${workspace_root}/install" ;;
    *)
        echo "未対応の install layout: ${install_layout}" >&2
        exit 126
        ;;
esac
waypoint_executable="${install_prefix}/lib/robotbase_waypoint/waypoint_follow"
if [ ! -x "${waypoint_executable}" ]; then
    echo "waypoint_follow 実行ファイルがありません: ${waypoint_executable}" >&2
    echo "この workspace の robotbase_waypoint をビルドしてください。" >&2
    exit 127
fi

# Preserve this startup PID across the handoff so TERM/INT reach the follower
# itself and the launcher's existing PID snapshot remains valid after exec.
exec "${waypoint_executable}" --ros-args \
    -p navigation_mode:="${navigation_mode}" \
    -p use_sim_time:="${use_sim_time}" \
    -p waypoint_file:="${waypoint_file}" \
    -p start_index:="${start_index}" \
    "${threshold_args[@]}" \
    -p loop:="${loop_value}" \
    -p navigation_only:="${navigation_only}" \
    -p base_frame:="${ROBOTBASE_TF_PREFIX:-robot}/base_footprint"
