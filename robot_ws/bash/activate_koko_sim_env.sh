#!/bin/bash
# Source this file for the isolated Koko simulation ROS graph.

_KOKO_SIM_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${_KOKO_SIM_WS}/bash/activate_koko_env.sh"

export ROS_DOMAIN_ID="${ROBOTBASE_SIM_ROS_DOMAIN_ID}"

unset _KOKO_SIM_WS
