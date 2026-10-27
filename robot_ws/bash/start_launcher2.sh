#!/usr/bin/env bash
# Run in a child shell; never change the caller's aliases or ROS environment.
set -e
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${workspace}/bash/activate_koko_env.sh"
cd "${workspace}"
exec /usr/bin/python3 "${workspace}/other_programs/sirius_launcher/robot_launcher.py" "$@"
