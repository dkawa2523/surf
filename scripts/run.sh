#!/usr/bin/env bash
set -euo pipefail

# Matplotlib環境差事故を抑制
export MPLCONFIGDIR=/tmp

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

# 環境確定
./scripts/preflight.sh
# shellcheck disable=SC1091
source ./scripts/env.sh

# autorun
"$PYTHON" ./scripts/codex_autorun.py
