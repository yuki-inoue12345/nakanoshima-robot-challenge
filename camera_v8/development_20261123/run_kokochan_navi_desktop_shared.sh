#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/v8_camera_eval/nakanoshima-robot-challenge"

test "$(id -un)" = "koko-chan-nuc"
test -d "$REPO/.git"

# 通常のココちゃん走行環境
export ROS_DOMAIN_ID=54
unset ROS_AUTOMATIC_DISCOVERY_RANGE

export ROBOTBASE_TF_PREFIX=robot
export ROBOTBASE_ID=koko
export ROBOTBASE_LAUNCHER_ID=koko_launcher2

# 元の地図アイコンの設定を維持
export KOKO_NAVI_REPOSITORY="$REPO"
export KOKO_NAVI_FPS=30
export KOKO_NAVI_PERCEPTION_REVIEW=1

# 実機テスト済みのV8 Python環境
export V8_PYTHON="$HOME/venvs/person_search/bin/python"
test -x "$V8_PYTHON"

# C920共有カメラ → ARナビ
exec "$REPO/camera_v8/development_20261123/run_ar_with_v8_share.sh"
