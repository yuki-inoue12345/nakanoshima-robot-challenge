#!/bin/bash
# Source this file to remove Sirius overlay paths and activate koko isolation.

_KOKO_ACTIVATE_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${_KOKO_ACTIVATE_WS}/robot.env"

_koko_remove_sirius_entries() {
    local variable_name="$1"
    local current_value="${!variable_name-}"
    local entry
    local cleaned=""

    IFS=':' read -r -a _koko_entries <<< "${current_value}"
    for entry in "${_koko_entries[@]}"; do
        [ -z "${entry}" ] && continue
        case "${entry}" in
            */sirius_jazzy_ws/*|*/robotbase_ws/*) continue ;;
        esac
        if [ -z "${cleaned}" ]; then
            cleaned="${entry}"
        else
            cleaned="${cleaned}:${entry}"
        fi
    done

    printf -v "${variable_name}" '%s' "${cleaned}"
    export "${variable_name}"
    unset _koko_entries
}

for _koko_path_variable in \
    AMENT_PREFIX_PATH \
    CMAKE_PREFIX_PATH \
    COLCON_PREFIX_PATH \
    GZ_SIM_RESOURCE_PATH \
    IGN_GAZEBO_RESOURCE_PATH \
    LD_LIBRARY_PATH \
    PATH \
    PKG_CONFIG_PATH \
    PYTHONPATH; do
    _koko_remove_sirius_entries "${_koko_path_variable}"
done

export ROS_DOMAIN_ID="${ROBOTBASE_ROS_DOMAIN_ID}"
export GZ_PARTITION="${ROBOTBASE_GZ_PARTITION}"
export ROBOTBASE_DISPLAY_NAME
export ROBOTBASE_ID
export ROBOTBASE_LAUNCHER_NAME
export ROBOTBASE_LAUNCHER_ID
export ROBOTBASE_ROS_DOMAIN_ID
export ROBOTBASE_SIM_ROS_DOMAIN_ID
export ROBOTBASE_TF_PREFIX
export ROBOTBASE_HOKUYO_IP
export ROBOTBASE_HOKUYO_PORT
export ROBOTBASE_FOXGLOVE_ADDRESS
export ROBOTBASE_FOXGLOVE_PORT
export ROBOTBASE_FOXGLOVE_PROFILE
export ROBOTBASE_PARAMS_DIR="${_KOKO_ACTIVATE_WS}/params"

unset _koko_path_variable
unset -f _koko_remove_sirius_entries
unset _KOKO_ACTIVATE_WS
