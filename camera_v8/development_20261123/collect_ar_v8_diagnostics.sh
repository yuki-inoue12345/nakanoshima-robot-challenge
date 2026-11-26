#!/usr/bin/env bash
# Read-only field diagnostics. Run only after AR camera owner and V8 server exist.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$HOME/ar_navigation_project/outputs/diagnostics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p -- "$OUT_DIR"

for command_name in pgrep pidstat ros2 readlink ps; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required diagnostic command is unavailable: $command_name" >&2
        exit 2
    }
done

find_one_pid() {
    local label="$1" pattern="$2" matches count
    matches="$(pgrep -f -- "$pattern" || true)"
    count="$(wc -w <<<"$matches")"
    if [[ "$count" != 1 ]]; then
        echo "$label PID discovery expected 1, found $count: ${matches:-none}" >&2
        echo "Candidate command lines:" >&2
        pgrep -af -- "$pattern" >&2 || true
        return 1
    fi
    printf '%s' "$matches"
}

OWNER_PID="$(find_one_pid camera_owner '/v8_camera_source[.]py([[:space:]]|$)')"
VIEWER_PID="$(find_one_pid ar_viewer '/ar_path_viewer/lib/ar_path_viewer/ar_path_viewer([[:space:]]|$)')"
V8_PID="$(find_one_pid v8_server '/daikoudai_v8[.]py[[:space:]]+--person-search-server([[:space:]]|$)')"
PIDS="${OWNER_PID},${VIEWER_PID},${V8_PID}"

node_list="$(ros2 node list 2>/dev/null || true)"
for required_node in \
    /koko_v8_camera_source \
    /ar_path_viewer \
    /camera_all_in_one_person_search; do
    node_count="$(grep -Fxc -- "$required_node" <<<"$node_list" || true)"
    if [[ "$node_count" != 1 ]]; then
        echo "ROS node verification failed for $required_node: expected 1, found $node_count" >&2
        echo "Start AR camera owner, Viewer, and the V8 person-search server first." >&2
        exit 3
    fi
done

fd_log_path() {
    local pid="$1" basename_pattern="$2" fd target
    for fd in /proc/"$pid"/fd/*; do
        target="$(readlink -f -- "$fd" 2>/dev/null || true)"
        [[ -n "$target" && -f "$target" ]] || continue
        if [[ "$(basename -- "$target")" == $basename_pattern ]]; then
            printf '%s' "$target"
            return 0
        fi
    done
    return 1
}

OWNER_LOG="$(readlink -f -- "/proc/$OWNER_PID/fd/1" 2>/dev/null || true)"
if [[ -z "$OWNER_LOG" || ! -f "$OWNER_LOG" ]]; then
    echo "Camera-owner stdout is not a persistent regular log file (pid=$OWNER_PID)." >&2
    exit 4
fi
VIEWER_LOG="$(fd_log_path "$VIEWER_PID" 'kokochan_navi_*.log' || true)"
if [[ -z "$VIEWER_LOG" ]]; then
    echo "Viewer session log cannot be resolved from /proc/$VIEWER_PID/fd." >&2
    exit 4
fi
V8_LOG="$(find "$HERE/outputs/ui_logs" -maxdepth 1 -type f \
    -name 'camera_v7_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -n 1 | cut -d' ' -f2- || true)"
if [[ -z "$V8_LOG" || ! -f "$V8_LOG" ]]; then
    echo "V8 UI/server persistent log was not found under $HERE/outputs/ui_logs." >&2
    exit 4
fi

STARTED_EPOCH="$(date +%s.%N)"
if ! grep -Fq 'v8_image_pipeline' "$V8_LOG"; then
    echo "Latest V8 UI log has no pipeline sample yet: $V8_LOG" >&2
    echo "Wait at least five seconds after starting the V8 server, then retry." >&2
    exit 4
fi

{
    echo "started_epoch=$STARTED_EPOCH"
    echo "camera_owner_pid=$OWNER_PID"
    echo "ar_viewer_pid=$VIEWER_PID"
    echo "v8_server_pid=$V8_PID"
    echo "ros_domain_id=${ROS_DOMAIN_ID:-54}"
    printf 'camera_owner_log=%q\n' "$OWNER_LOG"
    printf 'ar_viewer_log=%q\n' "$VIEWER_LOG"
    printf 'v8_server_log=%q\n' "$V8_LOG"
    echo "pidstat_mode=epoch_process_and_threads_cpu_rss_context_switch"
    ps -o pid=,ppid=,lstart=,cmd= -p "$OWNER_PID" -p "$VIEWER_PID" -p "$V8_PID"
} >"$OUT_DIR/process_manifest.txt"

write_analyzer_command() {
    local ended_epoch="${1:-}"
    {
        echo '#!/usr/bin/env bash'
        printf 'python3 %q --owner-log %q --viewer-log %q --v8-log %q --start-epoch %q' \
            "$HERE/analyze_ar_v8_diagnostics.py" "$OWNER_LOG" "$VIEWER_LOG" \
            "$V8_LOG" "$STARTED_EPOCH"
        if [[ -n "$ended_epoch" ]]; then
            printf ' --end-epoch %q' "$ended_epoch"
        fi
        printf ' --output %q\n' "$OUT_DIR/pipeline_summary.json"
    } >"$OUT_DIR/analyze_this_run.sh"
    chmod +x "$OUT_DIR/analyze_this_run.sh"
}
write_analyzer_command

PIDSTAT_PID=""
BAG_PID=""

# A background job started by non-interactive Bash may inherit SIGINT=ignored.
# Therefore kill -INT followed by an unbounded wait can leave rosbag running
# and metadata.yaml unwritten. Stop only our verified children with SIGTERM.
stop_owned_child() {
    local label="$1" pid="$2" required_pattern="$3"
    local cmd state attempt
    [[ -n "$pid" ]] || return 0
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
    fi
    cmd="$(ps -ww -p "$pid" -o args= 2>/dev/null || true)"
    if [[ -z "$cmd" || "$cmd" != *"$required_pattern"* ]]; then
        echo "STOP: $label PID=$pid command changed; not signaling: $cmd" >&2
        return 1
    fi
    if ! kill -TERM "$pid" 2>/dev/null; then
        echo "STOP: SIGTERM failed for $label PID=$pid" >&2
        return 1
    fi
    # Never wait forever: large MCAP recordings can need time to flush.
    for ((attempt=0; attempt<150; attempt++)); do
        state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
        state="${state//[[:space:]]/}"
        if [[ -z "$state" || "$state" == Z* ]]; then
            wait "$pid" 2>/dev/null || true
            echo "Stopped $label PID=$pid"
            return 0
        fi
        sleep 0.2
    done
    echo "STOP: $label PID=$pid did not exit within 30s; no SIGKILL sent" >&2
    return 1
}

cleanup() {
    local original_status="$?" bag_ok=0 pidstat_ok=0 ended_epoch
    # A second Ctrl+C during MCAP finalization must not interrupt cleanup.
    trap '' INT TERM
    trap - EXIT
    set +e
    stop_owned_child rosbag "$BAG_PID" "ros2 bag record --storage mcap -o $OUT_DIR/ros_state" || bag_ok=1
    stop_owned_child pidstat "$PIDSTAT_PID" 'pidstat -H -h -t -u -r -w' || pidstat_ok=1
    ended_epoch="$(date +%s.%N)"
    echo "ended_epoch=$ended_epoch" >>"$OUT_DIR/process_manifest.txt"
    write_analyzer_command "$ended_epoch"
    if (( bag_ok == 0 )) && [[ -s "$OUT_DIR/ros_state/metadata.yaml" ]]; then
        echo 'bag_finalized=true' >>"$OUT_DIR/process_manifest.txt"
        echo "MCAP finalized: $OUT_DIR/ros_state/metadata.yaml"
    else
        echo 'bag_finalized=false' >>"$OUT_DIR/process_manifest.txt"
        echo 'STOP: MCAP metadata missing or recorder still running; inspect rosbag.log' >&2
        (( original_status == 0 )) && original_status=5
    fi
    if (( pidstat_ok == 0 )); then
        echo 'pidstat_stopped=true' >>"$OUT_DIR/process_manifest.txt"
    else
        echo 'pidstat_stopped=false' >>"$OUT_DIR/process_manifest.txt"
        (( original_status == 0 )) && original_status=5
    fi
    echo "Diagnostics stopped. Analyze with: $OUT_DIR/analyze_this_run.sh"
    exit "$original_status"
}
# SIGINT/TERM exits once; EXIT always finalizes both owned children.
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

# -H uses epoch time. -t adds per-thread rows; -w then records voluntary and
# involuntary context switches for both each process and its worker threads.
LC_ALL=C pidstat -H -h -t -u -r -w -p "$PIDS" 1 \
    >"$OUT_DIR/pidstat.log" 2>&1 &
PIDSTAT_PID=$!

# Deliberately exclude both raw image topics. Existing process logs contain the
# image-stage timing; duplicating 640x480 BGR traffic would perturb the test.
ros2 bag record --storage mcap -o "$OUT_DIR/ros_state" \
    /plan /amcl_pose /tf /tf_static /local_costmap/costmap \
    /waypoint_follower/reference_state /ar/reference_path_state \
    /person_area_arrived /person_search_done /person_search_done_detail \
    >"$OUT_DIR/rosbag.log" 2>&1 &
BAG_PID=$!

echo "Diagnostics started: $OUT_DIR"
echo "PIDs: owner=$OWNER_PID viewer=$VIEWER_PID v8=$V8_PID"
echo "No image topic is recorded. Press Ctrl+C after the post-search segment."
wait "$BAG_PID"
