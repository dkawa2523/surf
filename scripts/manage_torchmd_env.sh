#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
CMD="${1:-status}"
shift || true

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/manage_torchmd_env.sh status
  bash scripts/manage_torchmd_env.sh isolate [--venv .venv_torchmd] [--source /path/to/torchmd-net] [--skip-install] [--remove-from-current]
  bash scripts/manage_torchmd_env.sh remove-current

Commands:
  status
    Show current interpreter, torch version, and torchmd-net-cpu install status.

  isolate
    Create dedicated torchmd virtualenv and (optionally) install editable torchmd source.
    Use --skip-install to only create isolated env without installing torchmd source.
    If --remove-from-current is set, uninstall torchmd-net-cpu from current interpreter
    after dedicated env setup succeeds.

  remove-current
    Uninstall torchmd-net-cpu from the current interpreter only.
EOF
}

detect_torchmd_source() {
  local src
  src="$("$PYTHON_BIN" <<'PY'
import importlib.metadata as md
from pathlib import Path
from urllib.parse import unquote, urlparse
import json

try:
    dist = md.distribution("torchmd-net-cpu")
except Exception:
    print("")
    raise SystemExit(0)

try:
    txt = dist.read_text("direct_url.json")
    if txt:
        data = json.loads(txt)
        if data.get("dir_info", {}).get("editable"):
            url = data.get("url", "")
            if url.startswith("file://"):
                print(unquote(urlparse(url).path))
                raise SystemExit(0)
except Exception:
    pass

print(str(Path(dist.locate_file("")).resolve()))
PY
)"
  printf '%s' "$src"
}

status_cmd() {
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import json
import sys

payload = {
    "python": sys.executable,
    "torch_version": None,
    "torchmd_net_cpu": False,
    "torchmd_location": None,
    "torchmd_requires": [],
    "compat_warning": None,
}

try:
    import torch
    payload["torch_version"] = torch.__version__
except Exception:
    payload["torch_version"] = None

try:
    dist = md.distribution("torchmd-net-cpu")
    payload["torchmd_net_cpu"] = True
    payload["torchmd_location"] = str(dist.locate_file("").resolve())
    payload["torchmd_requires"] = [str(v) for v in (dist.requires or [])]
except Exception:
    pass

if payload["torchmd_net_cpu"] and payload["torch_version"]:
    requires = payload["torchmd_requires"]
    req_torch = [r for r in requires if r.lower().startswith("torch")]
    if req_torch:
        payload["compat_warning"] = (
            "torchmd-net-cpu requires "
            + ", ".join(req_torch)
            + f" but current torch={payload['torch_version']}"
        )

print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

remove_current_cmd() {
  "$PYTHON_BIN" -m pip uninstall -y torchmd-net-cpu || true
}

isolate_cmd() {
  local venv_path=".venv_torchmd"
  local source_path=""
  local remove_from_current=0
  local skip_install=0
  local install_ok=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --venv)
        venv_path="$2"
        shift 2
        ;;
      --source)
        source_path="$2"
        shift 2
        ;;
      --remove-from-current)
        remove_from_current=1
        shift 1
        ;;
      --skip-install)
        skip_install=1
        shift 1
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        print_usage
        exit 2
        ;;
    esac
  done

  if [[ -z "$source_path" ]]; then
    source_path="$(detect_torchmd_source || true)"
  fi

  "$PYTHON_BIN" -m venv "$venv_path"
  "$venv_path/bin/python" -m pip install --upgrade pip setuptools wheel

  if [[ "$skip_install" -eq 1 ]]; then
    echo "Skip install requested (--skip-install)."
  elif [[ -n "$source_path" && -d "$source_path" ]]; then
    if [[ -f "$source_path/pyproject.toml" || -f "$source_path/setup.py" ]]; then
      if "$venv_path/bin/python" -m pip install -e "$source_path"; then
        echo "Installed torchmd source in isolated env: $source_path"
      else
        install_ok=0
        echo "WARN: Failed to install torchmd source in isolated env: $source_path" >&2
        echo "WARN: You can keep this venv for separation and install manually later." >&2
      fi
    else
      echo "Skip install: source does not look like Python package: $source_path" >&2
    fi
  else
    echo "No torchmd source path detected. Created empty isolated env only." >&2
  fi

  if [[ "$remove_from_current" -eq 1 ]]; then
    if [[ "$install_ok" -ne 1 ]]; then
      echo "WARN: --remove-from-current skipped because isolate install failed." >&2
      echo "WARN: run remove-current manually after confirming isolated env state." >&2
      remove_from_current=0
    fi
  fi

  if [[ "$remove_from_current" -eq 1 ]]; then
    "$PYTHON_BIN" -m pip uninstall -y torchmd-net-cpu || true
    echo "Removed torchmd-net-cpu from current interpreter."
  fi

  cat <<EOF
Isolated env ready: $venv_path
Activate:
  source $venv_path/bin/activate
EOF
}

case "$CMD" in
  status)
    status_cmd
    ;;
  isolate)
    isolate_cmd "$@"
    ;;
  remove-current)
    remove_current_cmd
    ;;
  -h|--help|help)
    print_usage
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    print_usage
    exit 2
    ;;
esac
