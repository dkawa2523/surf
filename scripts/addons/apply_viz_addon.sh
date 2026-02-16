#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Ensure env.sh exists for PYTHON (recommended)
if [ ! -f "$ROOT_DIR/scripts/env.sh" ]; then
  echo "[INFO] scripts/env.sh not found. Running preflight..."
  ./scripts/preflight.sh
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh"

echo "[INFO] Applying VIZ add-on tasks..."
"$PYTHON" "$ROOT_DIR/scripts/addons/apply_viz_addon.py"
echo "[INFO] Done. You can now run: ./scripts/run.sh"
