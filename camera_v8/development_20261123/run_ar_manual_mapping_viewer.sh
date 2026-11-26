#!/usr/bin/env bash
# Viewer-only manual mapping session. The existing V8 camera owner is provided
# by run_ar_with_v8_share.sh; this script must NEVER open C920 or drive robot.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
PREPARE_ONLY=0
if (($#)); then
  if [[ "$#" == 1 && "$1" == '--prepare-only' ]]; then
    PREPARE_ONLY=1
  else
    echo 'Usage: run_ar_manual_mapping_viewer.sh [--prepare-only]' >&2
    exit 2
  fi
fi
BASE_DEFAULT="$HOME/ar_navigation_project/config/ar_path_viewer_launcher2.yaml"
if [[ ! -r "$BASE_DEFAULT" ]]; then
  BASE_DEFAULT="$REPO/ar_ws/src/ar_path_viewer/config/ar_path_viewer_c920.yaml"
fi
BASE="${KOKO_NAVI_VIEWER_CONFIG:-$BASE_DEFAULT}"
OVERLAY="${KOKO_NAVI_MANUAL_MAPPING_OVERLAY:-$REPO/ar_ws/src/ar_path_viewer/config/manual_mapping_overlay.yaml}"
[[ -r "$BASE" ]] || { echo "Viewer base config missing: $BASE" >&2; exit 2; }
[[ -r "$OVERLAY" ]] || { echo "Manual mapping overlay missing: $OVERLAY (keep the Codex-created file and stage it before NUC deploy)" >&2; exit 2; }
FPS="${KOKO_NAVI_FPS:-30}"
case "$FPS" in 20|30) ;; *) echo "KOKO_NAVI_FPS must be 20 or 30" >&2; exit 2 ;; esac
OUTPUT_DIR="${KOKO_NAVI_OUTPUT_DIR:-$HOME/ar_navigation_project/outputs}"
# In prepare-only mode do not create output directories or write log/video.
if [[ "$PREPARE_ONLY" == 1 ]]; then
  SESSION_BASE="${TMPDIR:-/tmp}/kokochan_mapping_preview"
else
  [[ "${KOKO_NAVI_SHARED_CAMERA:-0}" == 1 ]] || {
    echo 'Manual runner requires existing shared-camera owner wrapper; refusing a second C920 owner' >&2
    exit 3
  }
  command -v ros2 >/dev/null 2>&1 || { echo 'ros2 not found' >&2; exit 3; }
  if ! ros2 node list 2>/dev/null | grep -Fxq /koko_v8_camera_source; then
    echo 'C920 shared owner is not running' >&2
    exit 3
  fi
  if ros2 node list 2>/dev/null | grep -Fxq /ar_path_viewer; then
    echo 'AR Viewer already running; refusing duplicate' >&2
    exit 3
  fi
  mkdir -p "$OUTPUT_DIR"
  SESSION_BASE="$OUTPUT_DIR/kokochan_mapping_$(date +%Y%m%d_%H%M%S)_${FPS}fps"
  if [[ -e "${SESSION_BASE}.mp4" || -e "${SESSION_BASE}.log" ]]; then
    SESSION_BASE="${SESSION_BASE}_$$"
  fi
fi
TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/kokochan_mapping_runtime.XXXXXXXX.yaml")"
cleanup() { rm -f -- "$TEMP_CONFIG"; }
trap cleanup EXIT
python3 - "$BASE" "$OVERLAY" "$TEMP_CONFIG" "$SESSION_BASE" "$FPS" <<'PY'
from pathlib import Path
import sys
try:
    import yaml
except ImportError as exc:
    raise SystemExit('PyYAML is required (ROS2 Python environment): '+str(exc))
base_path, overlay_path, dest, session, fps = sys.argv[1:]
def params_from(path):
    doc = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(doc, dict):
        raise SystemExit(f'Invalid ROS YAML: {path}')
    root = doc.get('ar_path_viewer')
    if not isinstance(root, dict) or not isinstance(root.get('ros__parameters'), dict):
        raise SystemExit(f'Missing ar_path_viewer/ros__parameters: {path}')
    return root['ros__parameters']
base = dict(params_from(base_path))
manual = dict(params_from(overlay_path))
# Safety: do not silently launch the default autonomous viewer if the untracked
# Codex overlay is absent, empty, or not actually selecting manual_mapping.
if manual.get('operation_mode') != 'manual_mapping':
    raise SystemExit('Overlay does not explicitly enable manual_mapping; refusing launch')
base.update(manual)
base.update({
    'image_input_type':'ros_image',
    'image_input_uri':'/koko/c920/raw_image',
    'fps':float(fps),
    'output_video':session+'.mp4',
    'log_output':session+'.log',
    'reference_path_enabled':False,
    'waypoint_distance_enabled':False,
    'guidance_require_follower_state':False,
    'perception_costmap_enabled':False,
    'perception_minimap_enabled':False,
    'perception_alert_enabled':False,
    'perception_explanation_enabled':False,
})
Path(dest).write_text(yaml.safe_dump({'ar_path_viewer':{'ros__parameters':base}},
                                     sort_keys=False, allow_unicode=True),encoding='utf-8')
PY
if [[ "$PREPARE_ONLY" == 1 ]]; then
  cat "$TEMP_CONFIG"
  exit 0
fi
printf 'Mapping session: %s\nManual overlay: %s\n' "$SESSION_BASE" "$OVERLAY"
echo 'Controls: R toggles mapping distance session (as implemented by Viewer); q/Q/F10 exits Viewer.'
echo 'No Launcher2, Nav2, Follower, SLAM or teleop process is started/stopped by this runner.'
# Unlike ros2 launch ar_path_viewer.launch.py, ros2 run returns when q/F10
# closes the Viewer. The outer camera-owner wrapper then closes its own owner.
ros2 run ar_path_viewer ar_path_viewer --ros-args --params-file "$TEMP_CONFIG" \
  2>&1 | tee "${SESSION_BASE}.launch.log"
