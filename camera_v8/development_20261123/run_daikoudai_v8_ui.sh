#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
V7="$REPO/camera_v7/development_20261019"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-54}"
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
else
  echo 'ROS 2 Jazzy not found' >&2; exit 2
fi
if [[ -r "$HOME/robotbase_ws_renya/install/setup.bash" ]]; then
  set +u
  source "$HOME/robotbase_ws_renya/install/setup.bash"
  set -u
fi
PYTHON_BIN="${V8_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for p in "$HOME/venvs/person_search/bin/python" "$V7/venv/bin/python" "$V7/.venv/bin/python"; do
    if [[ -x "$p" ]]; then PYTHON_BIN="$p"; break; fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
[[ -f "$V7/person_search_vlm_milestone2_v7.py" ]] || { echo 'V7 algorithm missing' >&2; exit 2; }
cd "$HERE"
exec "$PYTHON_BIN" "$HERE/daikoudai_v8.py"
