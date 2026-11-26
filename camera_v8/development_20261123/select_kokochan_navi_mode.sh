#!/usr/bin/env bash
# Interactive mode chooser. Mode is explicit, never inferred from Launcher2.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$HERE/run_ar_with_v8_share.sh"
MODE=""
DRY_RUN=0
usage() {
  cat <<'EOF'
ココちゃんナビ: 起動モード選択
  select_kokochan_navi_mode.sh [--mode autonomous|manual_mapping] [--dry-run]
  引数なし: 画面でモード選択（zenityがなければ端末で番号選択）
  --dry-run: 起動せず設定を表示
EOF
}
while (($#)); do
  case "$1" in
    --mode)
      (($# >= 2)) || { echo "--mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$MODE" ]]; then
  if [[ -n "${DISPLAY:-}" ]] && command -v zenity >/dev/null 2>&1; then
    choice="$(zenity --list --title='ココちゃんナビ：起動モード選択' \
      --text='使用するモードを選んでください' \
      --column='モード' '自律走行' 'キーオペ・地図作成')" || exit 0
    case "$choice" in
      '自律走行') MODE=autonomous ;;
      'キーオペ・地図作成') MODE=manual_mapping ;;
      *) echo "No mode selected" >&2; exit 2 ;;
    esac
  elif [[ -t 0 ]]; then
    printf 'ココちゃんナビ：起動モードを選択\n  1) 自律走行\n  2) キーオペ・地図作成\n' >&2
    read -r -p '番号 [1/2]（Enterだけなら中止）: ' choice
    case "$choice" in
      1) MODE=autonomous ;;
      2) MODE=manual_mapping ;;
      '') echo '中止しました'; exit 0 ;;
      *) echo 'Invalid selection' >&2; exit 2 ;;
    esac
  else
    echo 'No GUI or interactive terminal; use --mode autonomous|manual_mapping' >&2
    exit 2
  fi
fi
case "$MODE" in
  autonomous|manual_mapping) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'MODE=%s\nWRAPPER=%s\nLAUNCHER2_DEPENDENCY=none_for_mode_selection\nCAMERA_OWNER=existing_v8_wrapper\n' "$MODE" "$WRAPPER"
  exit 0
fi
[[ -f "$WRAPPER" ]] || { echo "Wrapper missing: $WRAPPER" >&2; exit 2; }
# Preserve the NUC's validated Camera V8 Python environment when available,
# while keeping development PCs usable with their active ROS Python.
if [[ -z "${V8_PYTHON:-}" && -x "$HOME/venvs/person_search/bin/python" ]]; then
  export V8_PYTHON="$HOME/venvs/person_search/bin/python"
fi
export KOKO_NAVI_MODE="$MODE"
exec bash "$WRAPPER"
