#!/bin/bash

_ROBOTBASE_SCRIPT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 軽量版ROS2 bag記録スクリプト
# 使い方: ./record_rosbag.sh [real|sim]
#         ファイル名を入力して記録開始

ROSBAG_PID=""
CURRENT_BAG_PATH=""
RESULT_SHOWN=false
experiment_memo="No description"
RECORD_STATUS=0
RECORD_MODE=real
RECORD_TIME_OPTIONS=()
RECORD_CLOCK_TOPICS=()
RECORD_STARTING=false
STOP_REQUESTED=false
RECORD_LOG=""
RECORD_LOG_PID=""
RECORD_OUTPUT_FD=""
RECORD_ERROR_FD=""

enable_recording_log() {
    mkdir -p "${ROSBAG_DIR}/.recording-logs" || return 1
    RECORD_LOG="$(mktemp "${ROSBAG_DIR}/.recording-logs/record_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")" || return 1
    # A closed terminal/launcher tee must never kill the shell before it sends
    # SIGINT, or raise BrokenPipeError in the finalizer. This relay keeps its
    # file sink even after the display disappears, and outlives terminal HUP.
    trap '' PIPE
    exec {RECORD_OUTPUT_FD}>&1 {RECORD_ERROR_FD}>&2
    exec > >(trap '' INT TERM HUP PIPE
        exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
            tee --output-error=warn -a "$RECORD_LOG" >&"$RECORD_OUTPUT_FD") 2>&1
    RECORD_LOG_PID=$!
}

finish_recording_log() {
    if [ -z "$RECORD_LOG_PID" ]; then
        return
    fi
    exec 1>&"$RECORD_OUTPUT_FD" 2>&"$RECORD_ERROR_FD"
    exec {RECORD_OUTPUT_FD}>&- {RECORD_ERROR_FD}>&-
    # Closing the pipe lets the relay drain to disk; its display may have gone.
    wait "$RECORD_LOG_PID" || true
    RECORD_LOG_PID=""
}

configure_recording_mode() {
    RECORD_MODE="${1:-real}"
    RECORD_TIME_OPTIONS=()
    RECORD_CLOCK_TOPICS=()
    case "$RECORD_MODE" in
        real)
            source "${_ROBOTBASE_SCRIPT_WS}/bash/activate_koko_env.sh" || return 1
            ;;
        sim)
            source "${_ROBOTBASE_SCRIPT_WS}/bash/activate_koko_sim_env.sh" || return 1
            RECORD_TIME_OPTIONS=(--use-sim-time)
            RECORD_CLOCK_TOPICS=(/clock)
            ;;
        *)
            echo "使い方: $0 [real|sim]" >&2
            return 1
            ;;
    esac
}

stop_recording() {
    # A signal can arrive between spawning the recorder and assigning $!.
    if [ "$RECORD_STARTING" = true ] && [ -z "$ROSBAG_PID" ]; then
        STOP_REQUESTED=true
        return
    fi
    # Do not let repeated Ctrl+C interrupt the writer while it flushes its indexes.
    trap '' INT TERM HUP
    if [ -n "$ROSBAG_PID" ]; then
        echo ""
        echo "記録を停止・保存しています。完了までお待ちください..."
        kill -INT "$ROSBAG_PID" 2>/dev/null
        wait "$ROSBAG_PID"
        RECORD_STATUS=$?
    fi
    ROSBAG_PID=""
    if [ -n "$CURRENT_BAG_PATH" ]; then
        show_result
    fi
    finish_recording_log
    exit "$RECORD_STATUS"
}

ROSBAG_DIR="${ROSBAG_DIR:-${HOME}/rosbag2_data}"

# 保存ディレクトリ作成
mkdir -p "$ROSBAG_DIR"

# 結果表示（1回だけ表示）
has_mcap_files() {
    local path
    for path in "$CURRENT_BAG_PATH"/*.mcap; do
        [ -f "$path" ] && return 0
    done
    return 1
}

show_result() {
    if [ "$RESULT_SHOWN" = true ]; then
        return
    fi
    RESULT_SHOWN=true
    
    echo ""
    echo "========================================="
    echo "  保存先: $CURRENT_BAG_PATH"
    
    if [ "$RECORD_STATUS" -ne 0 ] &&
       ! has_mcap_files; then
        echo "⚠ 記録の起動に失敗しました（終了コード: $RECORD_STATUS、MCAP未作成）"
    elif python3 "${_ROBOTBASE_SCRIPT_WS}/tools/finalize_rosbag.py" "$CURRENT_BAG_PATH"; then
        if [ "$RECORD_STATUS" -eq 0 ]; then
            echo "✓ 保存が完了しました（Foxgloveで開けます）"
        else
            echo "⚠ 記録プロセスは異常終了しました（終了コード: $RECORD_STATUS）。MCAPの保存は確認済みです"
        fi
    else
        if [ "$RECORD_STATUS" -ne 0 ]; then
            echo "⚠ 記録プロセスは異常終了しました（終了コード: $RECORD_STATUS）"
        fi
        echo "⚠ 保存の確認に失敗しました。上のエラーを確認してください"
        RECORD_STATUS=1
    fi
    
    # ファイルサイズ表示
    if [ -d "$CURRENT_BAG_PATH" ]; then
        size=$(du -sh "$CURRENT_BAG_PATH" | cut -f1)
        echo "  サイズ: $size"
    fi
    echo "========================================="
}

# ファイル名を入力
input_filename() {
    echo "========================================="
    echo "  軽量版 ROS2 bag 記録"
    echo "========================================="
    echo ""
    echo "保存先: $ROSBAG_DIR"
    echo "モード: $RECORD_MODE (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})"
    if [ "$RECORD_MODE" = sim ]; then
        echo "※ /clock を記録し、シミュレーション時刻を使用します"
        echo "※ シミュレータの /clock が配信されるまで記録は待機します"
    fi
    echo ""
    echo "記録対象トピック:"
    echo "  - /scan3, /hokuyo_scan (LiDAR)"
    echo "  - /odom, /odom/filtered (オドメトリ)"
    echo "  - /imu (IMU)"
    echo "  - /tf, /tf_static (座標変換)"
    echo "  - /cmd_vel, /cmd_vel_nav, /cmd_vel_smoothed (速度指令)"
    echo "  - FollowPath / NavigateToPose / NavigateThroughPoses の status・feedback"
    echo "  - /spin/_action/status, /spin/_action/feedback (Spinアクション)"
    echo "  - /parameter_events (パラメータ変更)"
    echo "  - /rosout (ノードログ)"
    echo "  - /amcl_pose, /initialpose (位置推定)"
    echo "  - /map, /plan, /optimal_trajectory (地図・経路)"
    echo "  - /cmd_vel_teleop"
    echo "  - /global_costmap/costmap, /local_costmap/costmap, /local_costmap/published_footprint (コストマップ・フットプリント)"
    echo ""
    echo "※ /velodyne_points (3D点群) は除外されます"
    echo "※ ROS 2サービスも記録します（--all-services）"
    echo ""
    echo "========================================="
    echo -n "ファイル名を入力してください (例: outdoor_test): "
    read bag_name
    
    if [ -z "$bag_name" ]; then
        # デフォルト名: 日時
        bag_name="rosbag_$(date +%Y%m%d_%H%M%S)"
        echo "デフォルト名を使用: $bag_name"
    fi

    echo -n "実験内容のメモを入力してください (例: SLAM精度検証): "
    read input_memo
    if [ -n "$input_memo" ]; then
        experiment_memo="$input_memo"
    else
        experiment_memo="No description"
    fi
    
    # 既存チェック
    if [ -d "$ROSBAG_DIR/$bag_name" ]; then
        echo ""
        echo "警告: $bag_name は既に存在します"
        echo -n "上書きしますか? (y/N): "
        read overwrite
        if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
            echo "キャンセルしました"
            return 1
        fi
        rm -rf "$ROSBAG_DIR/$bag_name"
    fi
    
    selected_name="$bag_name"
    echo ""
}

# 記録開始
start_recording() {
    CURRENT_BAG_PATH="$ROSBAG_DIR/$1"
    RESULT_SHOWN=false
    RECORD_STATUS=0
    # Mark descendants created later during recovery/reindex for sim cleanup.
    export ROBOTBASE_ROSBAG_SAVE=1
    STOP_REQUESTED=false
    RECORD_STARTING=true
    if ! enable_recording_log; then
        RECORD_STARTING=false
        RECORD_STATUS=1
        echo "記録ログを作成できないため開始しません" >&2
        return
    fi
    
    echo "========================================="
    echo "記録を開始します..."
    echo "  保存先: $CURRENT_BAG_PATH"
    echo "  保存確認ログ: $RECORD_LOG"
    echo "  実験メモ: $experiment_memo"
    echo "  停止するには Ctrl+C を押してください"
    echo "========================================="
    echo ""
    
    # 録画開始の瞬間にのみ実験メタデータを1回パブリッシュ
    ros2 topic pub --once /experiment_metadata std_msgs/msg/String "data: '$experiment_memo'" >/dev/null 2>&1 &

    # Bash asynchronous commands inherit SIGINT=ignored. Reset it before exec,
    # and isolate the recorder from terminal signals so only this shell stops it.
    python3 -c 'import os, signal, sys; os.setsid(); signal.signal(signal.SIGINT, signal.SIG_DFL); signal.signal(signal.SIGHUP, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
        ros2 bag record -o "$CURRENT_BAG_PATH" \
        --storage mcap \
        --storage-config-file "${_ROBOTBASE_SCRIPT_WS}/params/common/rosbag_mcap.yaml" \
        --disable-keyboard-controls \
        "${RECORD_TIME_OPTIONS[@]}" \
        --all-services \
        --include-hidden-topics \
        --topics \
        "${RECORD_CLOCK_TOPICS[@]}" \
        /experiment_metadata \
        /scan3 \
        /hokuyo_scan \
        /rosout \
        /odom \
        /odom/filtered \
        /imu \
        /tf \
        "/cmd_vel_direct" \
        "/stop" \
        "/person_area_arrived" \
        "/person_search_done" \
        "/person_search_done_detail" \
        "/person_search/arrival_received" \
        /tf_static \
        /cmd_vel \
        /cmd_vel_nav \
        /cmd_vel_smoothed \
        /follow_path/_action/status \
        /follow_path/_action/feedback \
        /navigate_to_pose/_action/status \
        /navigate_to_pose/_action/feedback \
        /navigate_through_poses/_action/status \
        /navigate_through_poses/_action/feedback \
        /spin/_action/status \
        /spin/_action/feedback \
        /parameter_events \
        /amcl_pose \
        /initialpose \
        /map \
        /plan \
        /optimal_trajectory \
        /global_costmap/costmap \
        /local_costmap/costmap \
        /local_costmap/published_footprint \
        /roboteq/odom \
        /cmd_vel_teleop \
        /joint_states &
    
    ROSBAG_PID=$!
    RECORD_STARTING=false
    if [ "$STOP_REQUESTED" = true ]; then
        stop_recording
    fi
    wait "$ROSBAG_PID"
    RECORD_STATUS=$?
    # Protect validation/recovery too when the recorder exits on its own.
    trap '' INT TERM HUP
    ROSBAG_PID=""
    
    show_result
    finish_recording_log
    trap stop_recording INT TERM HUP
}

# メイン処理（ループ）
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return
fi
if [ "$#" -gt 1 ]; then
    echo "使い方: $0 [real|sim]" >&2
    exit 1
fi
configure_recording_mode "${1:-real}" || exit 1
trap stop_recording INT TERM HUP
cd "${_ROBOTBASE_SCRIPT_WS}"
source install/setup.bash || exit 1

while : ; do
    echo ""
    read -r -p "Press [Enter] key to start recording..." || break
    
    selected_name=""
    input_filename
    
    if [ $? -eq 0 ] && [ -n "$selected_name" ]; then
        start_recording "$selected_name"
    fi
    
    echo ""
    echo "-----------------------------------------"
done
exit "$RECORD_STATUS"
