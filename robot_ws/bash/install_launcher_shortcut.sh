#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${workspace}/robot.env"

template="${workspace}/bash/desktop_shortcut/robot_launcher.desktop.in"
desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
if [[ -z "${desktop_dir}" ]]; then
  desktop_dir="${HOME}/Desktop"
fi
applications_dir="${HOME}/.local/share/applications"
commands_dir="${HOME}/.local/bin"
launcher_id="${ROBOTBASE_LAUNCHER_ID:-${ROBOTBASE_ID}_launcher}"
launcher_name="${ROBOTBASE_LAUNCHER_NAME:-${ROBOTBASE_DISPLAY_NAME} ランチャー}"
[[ "${launcher_id}" =~ ^[a-z0-9_-]+$ ]] || { echo "Invalid launcher ID" >&2; exit 1; }
shortcut_name="${launcher_id}.desktop"

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&|]/\\&/g'
}

display_name="$(escape_sed_replacement "${ROBOTBASE_DISPLAY_NAME}")"
launcher_name="$(escape_sed_replacement "${launcher_name}")"
workspace_value="$(escape_sed_replacement "${workspace}")"
generated="$(mktemp --suffix=.desktop)"
trap 'rm -f "${generated}"' EXIT

sed \
  -e "s|@DISPLAY_NAME@|${display_name}|g" \
  -e "s|@LAUNCHER_NAME@|${launcher_name}|g" \
  -e "s|@WORKSPACE@|${workspace_value}|g" \
  "${template}" > "${generated}"

install -d "${desktop_dir}" "${applications_dir}" "${commands_dir}"
install -m 0755 "${generated}" "${desktop_dir}/${shortcut_name}"
install -m 0644 "${generated}" "${applications_dir}/${shortcut_name}"

# A standalone command avoids sourcing this workspace's koko_* aliases in .bashrc.
printf '#!/usr/bin/env bash\nexec /bin/bash %q "$@"\n' \
  "${workspace}/bash/start_launcher2.sh" > "${generated}"
install -m 0755 "${generated}" "${commands_dir}/${launcher_id}"

# GNOME uses this metadata to allow launching directly from the Desktop.
gio set "${desktop_dir}/${shortcut_name}" metadata::trusted true \
  >/dev/null 2>&1 || true

echo "Desktop shortcut: ${desktop_dir}/${shortcut_name}"
echo "Application entry: ${applications_dir}/${shortcut_name}"
echo "Command: ${commands_dir}/${launcher_id}"
