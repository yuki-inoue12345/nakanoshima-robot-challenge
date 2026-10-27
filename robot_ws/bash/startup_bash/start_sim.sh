#!/bin/bash
# Start exactly one Koko Gazebo clock on the simulation ROS domain.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOCK_FILE="/tmp/${ROBOTBASE_ID:-koko}_sim_domain_${ROS_DOMAIN_ID:-unset}.lock"

# Keep this descriptor open for the complete ros2 launch lifetime.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "エラー: koko_simは既に起動中です（ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設定}）。" >&2
    echo "既存のシミュレータ端末を停止してから再実行してください。" >&2
    exit 2
fi

clock_publishers="$(ros2 topic info /clock 2>/dev/null | sed -n 's/^Publisher count: //p')"
if [ "${clock_publishers:-0}" -gt 0 ]; then
    echo "エラー: /clock publisherが既に${clock_publishers}個あります。" >&2
    echo "Gazeboを重ねて起動すると時刻が巻き戻り、TF_OLD_DATAやRViz異常終了の原因になります。" >&2
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設定}の既存シミュレーション一式を終了してください。" >&2
    exit 3
fi

existing_nodes="$(ros2 node list 2>/dev/null || true)"
if printf '%s\n' "${existing_nodes}" | grep -Eq '^/(koko_rviz|robotbase_gz_bridge|robot_state_publisher|slam_toolbox|controller_server)$'; then
    echo "エラー: 前回のシミュレーション用RViz/Nav2/SLAMノードが残っています。" >&2
    echo "時刻巻き戻りを防ぐため、関連端末を終了してからkoko_simを起動してください。" >&2
    exit 4
fi

cd -- "${WORKSPACE_ROOT}" || exit 1
# shellcheck disable=SC1091
source install/setup.bash
exec ros2 launch robotbase_sim sim.launch.py "$@"
