#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="safe"
DRY_RUN=0
REMOVE_TMP_VENV=0
TARGETS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/cleanup_generated.sh [--mode safe|targeted] [--dry-run] [--target <relative-path>] [--remove-tmp-venv]

Modes:
  safe      Remove cache-like artifacts only (default)
  targeted  Remove only explicitly provided --target paths

Examples:
  ./scripts/cleanup_generated.sh --dry-run
  ./scripts/cleanup_generated.sh --mode targeted --target benchmark/tmp --target runs/smoke_tmp --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --remove-tmp-venv)
      REMOVE_TMP_VENV=1
      shift
      ;;
    --target)
      TARGETS+=("${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "cleanup: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "safe" && "$MODE" != "targeted" ]]; then
  echo "cleanup: unsupported --mode '$MODE' (use safe|targeted)" >&2
  exit 2
fi

remove_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "cleanup: skip missing $path"
    return
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "cleanup: [dry-run] rm -rf $path"
    return
  fi
  rm -rf "$path"
  echo "cleanup: removed $path"
}

if [[ "$MODE" == "safe" ]]; then
  echo "cleanup: mode=safe (cache-like artifacts only)"
  while IFS= read -r dir; do
    remove_path "$dir"
  done < <(find src -type d -name '__pycache__' -print)
  while IFS= read -r file; do
    remove_path "$file"
  done < <(find src -type f -name '*.pyc' -print)

  if [[ -d "src/wafer_surrogate.egg-info" ]]; then
    remove_path "src/wafer_surrogate.egg-info"
  fi
  if [[ "$REMOVE_TMP_VENV" -eq 1 && -d ".venv_torchmd_tmp" ]]; then
    remove_path ".venv_torchmd_tmp"
  fi
  echo "cleanup: done (safe)"
  exit 0
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "cleanup: --mode targeted requires at least one --target" >&2
  exit 2
fi

echo "cleanup: mode=targeted"
for rel in "${TARGETS[@]}"; do
  if [[ -z "$rel" ]]; then
    continue
  fi
  case "$rel" in
    /*|[A-Za-z]:*|*'..'*)
      echo "cleanup: target must be a relative path under repository root: '$rel'" >&2
      exit 2
      ;;
  esac
  remove_path "$rel"
done

echo "cleanup: done (targeted)"
