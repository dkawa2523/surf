#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is intended for macOS only." >&2
  exit 2
fi

for dep in /opt/homebrew/opt/openblas /opt/homebrew/opt/libomp; do
  if [[ ! -d "$dep" ]]; then
    echo "Missing dependency: $dep" >&2
    echo "Install with: brew install openblas libomp" >&2
    exit 2
  fi
done

echo "[1/4] Pin torch to ME-compatible version (torch==2.2.2)"
"$PYTHON_BIN" -m pip install --upgrade --force-reinstall "torch==2.2.2"
"$PYTHON_BIN" -m pip install --upgrade "numpy<2"

TMP_DIR="$(mktemp -d)"
export TMP_DIR
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[2/4] Fetch MinkowskiEngine source"
git clone --depth=1 https://github.com/NVIDIA/MinkowskiEngine.git "$TMP_DIR/MinkowskiEngine" >/dev/null

echo "[3/4] Patch setup.py for macOS OpenMP flags"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["TMP_DIR"]) / "MinkowskiEngine" / "setup.py"
s = p.read_text(encoding="utf-8")
old = """if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CC_FLAGS += ["/sdl"]
    else:
        CC_FLAGS += ["/sdl", "/permissive-"]
else:
    CC_FLAGS += ["-fopenmp"]

if "darwin" in platform:
    CC_FLAGS += ["-stdlib=libc++", "-std=c++17"]
"""
new = """if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CC_FLAGS += ["/sdl"]
    else:
        CC_FLAGS += ["/sdl", "/permissive-"]
elif "darwin" in platform:
    omp_prefix = os.getenv("LIBOMP_PREFIX", "/opt/homebrew/opt/libomp")
    include_dirs += [str(Path(omp_prefix) / "include")]
    extra_link_args += [f"-L{Path(omp_prefix) / 'lib'}", "-lomp"]
    CC_FLAGS += ["-Xpreprocessor", "-fopenmp"]
else:
    CC_FLAGS += ["-fopenmp"]

if "darwin" in platform:
    CC_FLAGS += ["-stdlib=libc++", "-std=c++17"]
"""
if old not in s:
    raise SystemExit("Expected OpenMP snippet not found in setup.py")
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched setup.py")
PY

echo "[4/4] Install MinkowskiEngine (CPU build)"
TMP_PIP_BIN="$(mktemp -d)"
cat >"$TMP_PIP_BIN/pip" <<'SH'
#!/usr/bin/env bash
exec python3 -m pip "$@"
SH
chmod +x "$TMP_PIP_BIN/pip"

PATH="$TMP_PIP_BIN:$PATH" \
ARCHFLAGS='-arch arm64' \
LIBOMP_PREFIX=/opt/homebrew/opt/libomp \
CC=/usr/bin/clang \
CXX=/usr/bin/clang++ \
OPENBLAS=/opt/homebrew/opt/openblas/lib/libopenblas.dylib \
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
NPY_BLAS_ORDER=openblas \
NPY_LAPACK_ORDER=openblas \
CFLAGS='-I/opt/homebrew/opt/openblas/include' \
LDFLAGS='-L/opt/homebrew/opt/openblas/lib' \
"$PYTHON_BIN" -m pip install --no-build-isolation --no-cache-dir "$TMP_DIR/MinkowskiEngine"

rm -rf "$TMP_PIP_BIN"

echo "Done. Validate with:"
echo "  ./scripts/commands.sh verify -- --full"
