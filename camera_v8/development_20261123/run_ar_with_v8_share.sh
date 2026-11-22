#!/usr/bin/env bash
# One-process camera owner + ROS subscriber-only AR; V8 search stream on demand.
# This wrapper neither starts nor commands Nav2/Follower/robot motion.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${HERE}/../.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-54}"
export ROBOTBASE_TF_PREFIX="${ROBOTBASE_TF_PREFIX:-robot}"
export ROBOTBASE_ID="${ROBOTBASE_ID:-koko}"
export ROBOTBASE_LAUNCHER_ID="${ROBOTBASE_LAUNCHER_ID:-koko_launcher2}"
export DISPLAY="${DISPLAY:-:1}"
export KOKO_NAVI_REPOSITORY="$REPO"
export KOKO_NAVI_SHARED_CAMERA=1
export KOKO_NAVI_FPS="${KOKO_NAVI_FPS:-30}"
DEVICE="${KOKO_NAVI_CAMERA_DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
    for candidate in /dev/v4l/by-id/*C920*video-index0; do
        if [[ -e "$candidate" ]]; then DEVICE="$candidate"; break; fi
    done
fi
if [[ -z "$DEVICE" || ! -e "$DEVICE" ]]; then
    echo "C920 device not found. No viewer or camera started." >&2
    exit 2
fi
export KOKO_NAVI_CAMERA_DEVICE="$DEVICE"
set +u
source /opt/ros/jazzy/setup.bash
[[ ! -f "$HOME/robotbase_ws_renya/install/setup.bash" ]] || source "$HOME/robotbase_ws_renya/install/setup.bash"
source "$REPO/robot_ws/install/setup.bash"
source "$REPO/ar_ws/install/setup.bash"
set -u
RUNNER="$REPO/ar_ws/install/ar_reference_planner/lib/ar_reference_planner/run_kokochan_navi_launcher2"
[[ -x "$RUNNER" ]] || { echo "AR runner missing: build ar_ws first." >&2; exit 2; }
LOG_DIR="${KOKO_NAVI_OUTPUT_DIR:-$HOME/ar_navigation_project/outputs}"
mkdir -p "$LOG_DIR"
OWNER_LOG="$LOG_DIR/v8_camera_owner_$(date +%Y%m%d_%H%M%S).log"
# Strict opt-in: don't start a second owner on top of an existing owner.
if ros2 node list 2>/dev/null | grep -Fxq /koko_v8_camera_source; then
    echo "C920 camera owner node is already running; refusing duplicate." >&2
    exit 3
fi
python3 -u "$HERE/v8_camera_source.py" --device "$DEVICE" --fps "$KOKO_NAVI_FPS" \
    >"$OWNER_LOG" 2>&1 &
OWNER_PID=$!
cleanup() {
    if kill -0 "$OWNER_PID" 2>/dev/null; then
        kill "$OWNER_PID" 2>/dev/null || true
        wait "$OWNER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 50); do
    if ! kill -0 "$OWNER_PID" 2>/dev/null; then break; fi
    if ros2 node list 2>/dev/null | grep -Fxq /koko_v8_camera_source; then
        ready=1
        break
    fi
    sleep 0.2
done
if [[ "$ready" != 1 ]]; then
    echo "Shared camera source startup failed; last log lines:" >&2
    tail -n 35 "$OWNER_LOG" >&2 || true
    exit 4
fi
echo "V8 camera owner: $OWNER_PID, log: $OWNER_LOG"
echo "AR will subscribe to /koko/c920/raw_image and record as usual."
echo "V8 search frames will be sent ONLY after /person_area_arrived."
"$RUNNER"
