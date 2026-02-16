#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOVE_TMP_VENV=0
if [[ "${1:-}" == "--remove-tmp-venv" ]]; then
  REMOVE_TMP_VENV=1
fi

echo "cleanup: removing Python cache files under src/"
find src -type d -name '__pycache__' -prune -exec rm -rf {} +
find src -type f -name '*.pyc' -delete

if [[ -d "src/wafer_surrogate.egg-info" ]]; then
  echo "cleanup: removing src/wafer_surrogate.egg-info/"
  rm -rf src/wafer_surrogate.egg-info
fi

if [[ "$REMOVE_TMP_VENV" -eq 1 && -d ".venv_torchmd_tmp" ]]; then
  echo "cleanup: removing .venv_torchmd_tmp/"
  rm -rf .venv_torchmd_tmp
fi

echo "cleanup: done"
