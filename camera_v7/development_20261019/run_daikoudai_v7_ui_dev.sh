#!/usr/bin/env bash
set -Eeuo pipefail

WS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="daikoudai_v7.py.new"
VENV_DIR="$WS_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

cd "$WS_DIR"

# Match the production ROS environment without starting ROS communication.
set +u
if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
set -u

export ROS_DOMAIN_ID=57

if [ ! -x "$PYTHON_BIN" ]; then
  echo "DEV_VENV_NOT_FOUND: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -B "$WS_DIR/v7_startup_preflight.py" --dependencies-only; then
  echo "V7 dependency preflight failed; development UI will not start." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$WS_DIR/$APP"
