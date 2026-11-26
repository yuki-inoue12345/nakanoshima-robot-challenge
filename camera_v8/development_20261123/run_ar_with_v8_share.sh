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
# Explicit viewer operation mode. It is not inferred from Launcher2/Follower.
case "${KOKO_NAVI_MODE:-autonomous}" in
    autonomous)
        # The autonomous operator view is expected to include the existing
        # read-only Costmap review profile.  Keep an explicit caller override
        # (0) for diagnostics, but do not depend on a desktop-only wrapper to
        # opt in: the mode chooser is now the canonical entry point.
        export KOKO_NAVI_PERCEPTION_REVIEW="${KOKO_NAVI_PERCEPTION_REVIEW:-1}"
        case "$KOKO_NAVI_PERCEPTION_REVIEW" in
            0|1) ;;
            *) echo "KOKO_NAVI_PERCEPTION_REVIEW must be 0 or 1" >&2; exit 2 ;;
        esac
        ;;
    manual_mapping) RUNNER="$HERE/run_ar_manual_mapping_viewer.sh" ;;
    *) echo "Unsupported KOKO_NAVI_MODE: ${KOKO_NAVI_MODE}" >&2; exit 2 ;;
esac
if [[ "${KOKO_NAVI_MODE:-autonomous}" == "autonomous" ]]; then
    echo "Autonomous perception review: KOKO_NAVI_PERCEPTION_REVIEW=${KOKO_NAVI_PERCEPTION_REVIEW}"
else
    echo "Manual mapping perception review: disabled by manual mapping runtime profile"
fi
[[ -x "$RUNNER" ]] || { echo "AR runner missing: build ar_ws first." >&2; exit 2; }
# Match the standalone photo-session lock across ALL ROS domains.
command -v flock >/dev/null 2>&1 || { echo "flock unavailable" >&2; exit 2; }
CAMERA_LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}"
exec 8>"$CAMERA_LOCK_DIR/kokochan-v8-camera-$(id -u).lock"
flock -n 8 || { echo "Camera V8 session already owns the C920 startup lock" >&2; exit 3; }
LOG_DIR="${KOKO_NAVI_OUTPUT_DIR:-$HOME/ar_navigation_project/outputs}"
mkdir -p "$LOG_DIR"
OWNER_LOG="$LOG_DIR/v8_camera_owner_$(date +%Y%m%d_%H%M%S).log"
# Strict opt-in: don't start a second owner on top of an existing owner.
if ros2 node list 2>/dev/null | grep -Fxq /koko_v8_camera_source; then
    echo "C920 camera owner node is already running; refusing duplicate." >&2
    exit 3
fi
# ROS discovery is domain-scoped: check the physical camera across ALL domains.
command -v fuser >/dev/null 2>&1 || { echo "fuser unavailable; refusing camera startup" >&2; exit 4; }
REAL_DEVICE="$(readlink -f -- "$DEVICE")"
[[ -c "$REAL_DEVICE" ]] || { echo "Not a C920 character device: $REAL_DEVICE" >&2; exit 4; }
if fuser "$REAL_DEVICE" >/dev/null 2>&1; then
    echo "Camera already occupied: $REAL_DEVICE; refusing duplicate C920 open" >&2
    exit 4
else
    case "$?" in
        1) ;; # fuser returns 1 when no process holds the device
        *) echo "C920 ownership cannot be verified; refusing startup" >&2; exit 4 ;;
    esac
fi
CAMERA_PYTHON="${V8_PYTHON:-python3}"
if ! command -v "$CAMERA_PYTHON" >/dev/null 2>&1; then
    echo "Camera V8 Python not available: $CAMERA_PYTHON" >&2
    exit 2
fi
if ! "$CAMERA_PYTHON" -c 'import cv2, numpy, rclpy; from ar_path_viewer.latest_image_source import LatestImageSource' >/dev/null 2>&1; then
    echo "Camera V8 Python dependencies unavailable" >&2
    exit 2
fi
"$CAMERA_PYTHON" -u "$HERE/v8_camera_source.py" --device "$DEVICE" --fps "$KOKO_NAVI_FPS" \
    --expected-parent-pid "$$" \
    >"$OWNER_LOG" 2>&1 &
OWNER_PID=$!
cleanup() {
    if kill -0 "$OWNER_PID" 2>/dev/null; then
        kill "$OWNER_PID" 2>/dev/null || true
        wait "$OWNER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
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
