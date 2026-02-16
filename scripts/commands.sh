#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
if [ -f "$ROOT_DIR/scripts/env.sh" ]; then
  source "$ROOT_DIR/scripts/env.sh"
else
  # preflight未実行でも最低限動くように（ただし推奨はpreflight）
  PYTHON=${PYTHON:-python3}
fi

RUN_CMD=("$PYTHON" -m wafer_surrogate.cli)
VERIFY_CMD=("$PYTHON" -m wafer_surrogate.verify)

cmd=${1:-}
shift || true

case "$cmd" in
  run)
    "${RUN_CMD[@]}" "$@"
    ;;
  verify|test)
    "${VERIFY_CMD[@]}" "$@"
    ;;
  verify-full)
    WAFER_SURROGATE_REQUIRE_REAL_ME=1 "${VERIFY_CMD[@]}" --full "$@"
    ;;
  torchmd)
    bash "$ROOT_DIR/scripts/manage_torchmd_env.sh" "$@"
    ;;
  python)
    echo "$PYTHON"
    ;;
  *)
    echo "Usage: ./scripts/commands.sh {run|verify|verify-full|test|torchmd|python} [-- <args>]" >&2
    exit 2
    ;;
esac
