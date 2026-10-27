#!/bin/bash
# Stop processes belonging to the isolated Koko simulation domain/partition.

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${WORKSPACE_ROOT}/robot.env"

# Overrides are used only by isolated tests; normal UI operation uses robot.env.
TARGET_DOMAIN="${ROBOTBASE_STOP_SIM_DOMAIN:-${ROBOTBASE_SIM_ROS_DOMAIN_ID}}"
PARTITION_PREFIX="${ROBOTBASE_STOP_GZ_PREFIX:-${ROBOTBASE_GZ_PARTITION}_sim_}"
CURRENT_UID="$(id -u)"
PROTECTED_PIDS=" $$ "
ancestor_pid="${PPID}"

# Never terminate this cleanup process, the launcher, or its ancestor shells.
while [ "${ancestor_pid}" -gt 1 ] 2>/dev/null; do
    PROTECTED_PIDS+="${ancestor_pid} "
    if [ -r "/proc/${ancestor_pid}/stat" ]; then
        ancestor_pid="$(awk '{print $4}' "/proc/${ancestor_pid}/stat")"
    else
        break
    fi
done

is_protected() {
    [[ "${PROTECTED_PIDS}" == *" $1 "* ]]
}

is_recording_save_process() {
    local entry
    [ -r "/proc/$1/environ" ] || return 1
    while IFS= read -r -d '' entry; do
        [ "${entry}" = ROBOTBASE_ROSBAG_SAVE=1 ] && return 0
    done < "/proc/$1/environ"
    return 1
}

is_simulation_process() {
    local pid="$1"
    local process_uid
    local entry

    [ -r "/proc/${pid}/environ" ] || return 1
    process_uid="$(stat -c '%u' "/proc/${pid}" 2>/dev/null)" || return 1
    [ "${process_uid}" = "${CURRENT_UID}" ] || return 1

    while IFS= read -r -d '' entry; do
        case "${entry}" in
            "ROS_DOMAIN_ID=${TARGET_DOMAIN}") return 0 ;;
            "GZ_PARTITION=${PARTITION_PREFIX}"*) return 0 ;;
        esac
    done < "/proc/${pid}/environ"
    return 1
}

targets=()
# Recording wrappers own their writer/finalizer. Request their normal save path
# and exclude that tree (and its terminal shell/tee) from timed escalation.
recording_protected="$(python3 - "${TARGET_DOMAIN}" <<'PY'
import os
import signal
import sys
import psutil

domain = sys.argv[1]
wrappers = {}
for process in psutil.process_iter(['pid', 'uids', 'cmdline']):
    try:
        if process.info['uids'].real != os.getuid():
            continue
        args = process.info['cmdline'] or []
        if not any(os.path.basename(arg) == 'record_rosbag.sh' for arg in args[1:]):
            continue
        tree = [process] + process.children(recursive=True)
        for child in tree:
            try:
                if child.environ().get('ROS_DOMAIN_ID') == domain:
                    wrappers[process.pid] = process
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
protected = set()
for wrapper in wrappers.values():
    try:
        protected.update(p.pid for p in [wrapper] + wrapper.children(recursive=True))
        for parent in wrapper.parents():
            protected.add(parent.pid)
            args = parent.cmdline()
            if '--rcfile' in args:
                protected.update(child.pid for child in parent.children(recursive=True))
        wrapper.send_signal(signal.SIGTERM)
        print(f'rosbag PID {wrapper.pid}: 保存終了を待っています。記録端末の結果を確認してください。',
              file=sys.stderr)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
print(' '.join(str(pid) for pid in protected))
PY
)" || exit 1
PROTECTED_PIDS+="${recording_protected} "
for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    is_protected "${pid}" && continue
    is_recording_save_process "${pid}" && continue
    if is_simulation_process "${pid}"; then
        targets+=("${pid}")
    fi
done

echo "ココちゃん シミュレーション一式終了"
echo "対象: ROS_DOMAIN_ID=${TARGET_DOMAIN}, GZ_PARTITION=${PARTITION_PREFIX}*"

if [ "${#targets[@]}" -eq 0 ]; then
    echo "終了対象のプロセスはありません。"
else
    echo "終了対象: ${#targets[@]}プロセス"
    for pid in "${targets[@]}"; do
        command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        printf '  PID %-7s %s\n' "${pid}" "${command_line:0:120}"
    done

    # Let ROS launch, RViz, Nav2, and Gazebo shut down cleanly first.
    kill -INT "${targets[@]}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        alive=()
        for pid in "${targets[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive+=("${pid}")
        done
        [ "${#alive[@]}" -eq 0 ] && break
        sleep 0.1
    done

    if [ "${#alive[@]}" -gt 0 ]; then
        echo "SIGINTで終了しない${#alive[@]}プロセスへSIGTERMを送ります。"
        kill -TERM "${alive[@]}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            remaining=()
            for pid in "${alive[@]}"; do
                kill -0 "${pid}" 2>/dev/null && remaining+=("${pid}")
            done
            [ "${#remaining[@]}" -eq 0 ] && break
            sleep 0.1
        done
    else
        remaining=()
    fi

    if [ "${#remaining[@]}" -gt 0 ]; then
        echo "残った${#remaining[@]}プロセスを強制終了します。"
        kill -KILL "${remaining[@]}" 2>/dev/null || true
    fi
fi

# Close launcher shells for simulation buttons after their ROS children stopped.
# Validate the command line so a stale PID file cannot terminate an unrelated PID.
launcher_id="${ROBOTBASE_LAUNCHER_ID:-${ROBOTBASE_ID}_launcher}"
launcher_prefix="${ROBOTBASE_STOP_LAUNCHER_PREFIX:-/tmp/${launcher_id}_}"
launcher_shells=()
for pid_file in "${launcher_prefix}"*.pid; do
    [ -e "${pid_file}" ] || continue
    button_name="${pid_file#${launcher_prefix}}"
    button_name="${button_name%.pid}"
    # The recording terminal must remain available until its save result appears.
    [ "${button_name}" = koko_record_rosbag_sim ] && continue
    case "${button_name}" in
        koko_sim|koko_*_sim|koko_*_sim_*)
            launcher_pid="$(sed -n '1p' "${pid_file}" 2>/dev/null)"
            if [[ "${launcher_pid}" =~ ^[0-9]+$ ]] && \
               ! is_protected "${launcher_pid}" && \
               [ -r "/proc/${launcher_pid}/cmdline" ]; then
                launcher_command="$(
                    tr '\0' ' ' < "/proc/${launcher_pid}/cmdline" 2>/dev/null || true)"
                if [[ "${launcher_command}" == *"bash --rcfile ${launcher_prefix}"* ]]; then
                    launcher_shells+=("${launcher_pid}")
                fi
            fi
            rm -f -- "${pid_file}"
            ;;
    esac
done
if [ "${#launcher_shells[@]}" -gt 0 ]; then
    kill -TERM "${launcher_shells[@]}" 2>/dev/null || true
fi

# Remove the startup lock only after its owner has been stopped.
lock_file="/tmp/${ROBOTBASE_ID}_sim_domain_${TARGET_DOMAIN}.lock"
exec 8>"${lock_file}"
if flock -n 8; then
    rm -f -- "${lock_file}"
fi

# DDS discovery can retain the old /clock endpoint briefly.
export ROS_DOMAIN_ID="${TARGET_DOMAIN}"
ros2 daemon stop >/dev/null 2>&1 || true
sleep 1

survivors=()
for pid in "${targets[@]}"; do
    kill -0 "${pid}" 2>/dev/null && survivors+=("${pid}")
done
if [ "${#survivors[@]}" -gt 0 ]; then
    echo "エラー: ${#survivors[@]}プロセスが残っています: ${survivors[*]}" >&2
    exit 1
fi

echo "シミュレーション一式を終了しました。koko_simを再起動できます。"
