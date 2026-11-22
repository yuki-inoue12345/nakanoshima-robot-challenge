#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_daikoudai_v8_ui.sh"
[[ -x "$RUNNER" ]] || { echo "V8 runner not found: $RUNNER" >&2; exit 1; }
mkdir -p "$HOME/Desktop"
TARGET="$HOME/Desktop/中チャレ人物探索_V8.desktop"
if [[ -e "$TARGET" ]]; then
  echo "Desktop file already exists, refusing overwrite: $TARGET" >&2; exit 2
fi
cat >"$TARGET" <<EOF
[Desktop Entry]
Type=Application
Name=中チャレ人物探索 V8（AR映像共有）
Comment=V8 waiting for person_area arrival; AR continues showing and recording
Exec=/bin/bash "$RUNNER"
Path=$HERE
Terminal=true
Categories=Science;Education;
Icon=camera-photo
EOF
chmod +x "$TARGET"
echo "V8 desktop launcher installed: $TARGET"
echo "This does NOT change V7, AR desktop, Nav2 or Renya. Integration E2E is still required."
