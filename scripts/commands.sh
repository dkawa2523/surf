#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
if [ -f "$ROOT_DIR/scripts/env.sh" ]; then
  source "$ROOT_DIR/scripts/env.sh"
fi

PYTHON_CMD=()
resolve_python_cmd() {
  if [ -n "${PYTHON:-}" ]; then
    if [ -x "${PYTHON}" ] || command -v "${PYTHON}" >/dev/null 2>&1; then
      PYTHON_CMD=("${PYTHON}")
      return
    fi
  fi

  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD=("py" "-3")
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("python3")
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("python")
    return
  fi

  echo "ERROR: python executable not found; run ./scripts/preflight.sh first." >&2
  exit 1
}

resolve_python_cmd
RUN_CMD=("${PYTHON_CMD[@]}" -m wafer_surrogate.cli)
VERIFY_CMD=("${PYTHON_CMD[@]}" -m wafer_surrogate.verify)

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
    printf '%s\n' "${PYTHON_CMD[*]}"
    ;;
  *)
    echo "Usage: ./scripts/commands.sh {run|verify|verify-full|test|torchmd|python} [-- <args>]" >&2
    exit 2
    ;;
esac
