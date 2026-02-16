#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-ataset_3d_test2}"
OUT_ROOT="${OUT_ROOT:-runs/dataset_3d_test2_pilot}"
PREP_DIR="$OUT_ROOT/prepared"
PREP_SECONDARY_DIR="$OUT_ROOT/prepared_secondary"
EVAL_ROOT="$OUT_ROOT/eval"
QUICK_PRIMARY_EVAL_DIR="$EVAL_ROOT/quick_primary"
QUICK_SECONDARY_EVAL_DIR="$EVAL_ROOT/quick_secondary"
REAL_EVAL_DIR="$EVAL_ROOT/real_me_primary"
GATE_JSON="$EVAL_ROOT/gate_primary_run_0006.json"
COMPARE_JSON="$EVAL_ROOT/comparison_quick_vs_real.json"

BASE_CONFIG="${BASE_CONFIG:-configs/dataset_3d_test2_pilot_train.toml}"
VIZ_CONFIG="${VIZ_CONFIG:-configs/visualization.default.yaml}"
TRAIN_RUNS="${TRAIN_RUNS:-run_0000,run_0001,run_0002,run_0003,run_0004,run_0005,run_0006,run_0007,run_0008,run_0009,run_0010,run_0011}"
PRIMARY_HOLDOUT_RUN="${PRIMARY_HOLDOUT_RUN:-run_0006}"
SECONDARY_HOLDOUT_RUN="${SECONDARY_HOLDOUT_RUN:-run_0012}"

VTI_PATTERN="${VTI_PATTERN:-vox_t*.vti}"
BAND_WIDTH="${BAND_WIDTH:-2.0}"
MIN_GRAD_NORM="${MIN_GRAD_NORM:-1e-8}"
VALID_MASK_ARRAY="${VALID_MASK_ARRAY:-ValidMask}"
TARGET_MATERIAL_ID="${TARGET_MATERIAL_ID:-}"
TARGET_SELECTION_MODE="${TARGET_SELECTION_MODE:-}"
TARGET_MAX_RATIO="${TARGET_MAX_RATIO:-}"
TARGET_MAX_BOUNDARY_RATIO="${TARGET_MAX_BOUNDARY_RATIO:-}"
TARGET_MIN_TEMPORAL_DELTA="${TARGET_MIN_TEMPORAL_DELTA:-}"
DOMAIN_BOUNDARY_MARGIN_VOX="${DOMAIN_BOUNDARY_MARGIN_VOX:-1}"
PHI_BOUNDARY_CLIP_VOX="${PHI_BOUNDARY_CLIP_VOX:-3}"
INCLUDE_TERMINAL_STEP_TARGET="${INCLUDE_TERMINAL_STEP_TARGET:-0}"
MASK_ARRAY="${MASK_ARRAY:-}"
MASK_MATERIAL_ID="${MASK_MATERIAL_ID:-}"
ANALYSIS_XY_MARGIN_VOX="${ANALYSIS_XY_MARGIN_VOX:-2}"
ANALYSIS_Z_MARGIN_VOX="${ANALYSIS_Z_MARGIN_VOX:-1}"

RUN_REAL_ME="${RUN_REAL_ME:-0}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-runs/dataset_3d_test2_pilot_h0006/eval/quick/holdout_eval.json}"

remove_run_from_csv() {
  local csv="$1"
  local remove_a="$2"
  local remove_b="$3"
  python3 - <<PY
items=[x.strip() for x in "${csv}".split(",") if x.strip()]
rem={r for r in ["${remove_a}","${remove_b}"] if r.strip()}
out=[x for x in items if x not in rem]
print(",".join(out))
PY
}

echo "[pilot] preflight"
./scripts/preflight.sh
# shellcheck disable=SC1091
source ./scripts/env.sh

EFFECTIVE_TRAIN_RUNS="$(remove_run_from_csv "$TRAIN_RUNS" "$PRIMARY_HOLDOUT_RUN" "$SECONDARY_HOLDOUT_RUN")"
if [[ -z "$EFFECTIVE_TRAIN_RUNS" ]]; then
  echo "ERROR: EFFECTIVE_TRAIN_RUNS became empty after removing holdouts" >&2
  exit 1
fi

mkdir -p "$PREP_DIR" "$PREP_SECONDARY_DIR" "$QUICK_PRIMARY_EVAL_DIR" "$QUICK_SECONDARY_EVAL_DIR" "$REAL_EVAL_DIR"

prepare_common_args=(
  --data-dir "$DATA_DIR"
  --vti-pattern "$VTI_PATTERN"
  --band-width "$BAND_WIDTH"
  --min-grad-norm "$MIN_GRAD_NORM"
  --valid-mask-array "$VALID_MASK_ARRAY"
  --domain-boundary-margin-vox "$DOMAIN_BOUNDARY_MARGIN_VOX"
  --phi-boundary-clip-vox "$PHI_BOUNDARY_CLIP_VOX"
)
if [[ -n "$TARGET_MATERIAL_ID" ]]; then
  prepare_common_args+=(--target-material-id "$TARGET_MATERIAL_ID")
fi
if [[ -n "$TARGET_SELECTION_MODE" ]]; then
  prepare_common_args+=(--target-selection-mode "$TARGET_SELECTION_MODE")
fi
if [[ -n "$TARGET_MAX_RATIO" ]]; then
  prepare_common_args+=(--target-max-ratio "$TARGET_MAX_RATIO")
fi
if [[ -n "$TARGET_MAX_BOUNDARY_RATIO" ]]; then
  prepare_common_args+=(--target-max-boundary-ratio "$TARGET_MAX_BOUNDARY_RATIO")
fi
if [[ -n "$TARGET_MIN_TEMPORAL_DELTA" ]]; then
  prepare_common_args+=(--target-min-temporal-delta "$TARGET_MIN_TEMPORAL_DELTA")
fi
if [[ "$INCLUDE_TERMINAL_STEP_TARGET" == "1" ]]; then
  prepare_common_args+=(--include-terminal-step-target)
fi
if [[ -n "$MASK_ARRAY" ]]; then
  prepare_common_args+=(--mask-array "$MASK_ARRAY")
fi
if [[ -n "$MASK_MATERIAL_ID" ]]; then
  prepare_common_args+=(--mask-material-id "$MASK_MATERIAL_ID")
fi

echo "[pilot] step 1/6 prepare primary holdout (${PRIMARY_HOLDOUT_RUN})"
"$PYTHON" scripts/prepare_dataset_3d_test2.py \
  "${prepare_common_args[@]}" \
  --train-runs "$EFFECTIVE_TRAIN_RUNS" \
  --holdout-runs "$PRIMARY_HOLDOUT_RUN" \
  --out-dir "$PREP_DIR"

for path in \
  "$PREP_DIR/train_narrow_band.h5" \
  "$PREP_DIR/feature_contract.json" \
  "$PREP_DIR/point_level_manifest.json" \
  "$PREP_DIR/sampling_diagnostics.json" \
  "$PREP_DIR/split_manifest.json"
do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing prepared artifact: $path" >&2
    exit 1
  fi
done

echo "[pilot] step 2/6 quick train"
./scripts/commands.sh run -- pipeline run --config "$BASE_CONFIG" --stages train

QUICK_RUN_ID="$("$PYTHON" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from wafer_surrogate.config.loader import load_config

cfg_path = Path("$BASE_CONFIG")
cfg = load_config(cfg_path)
run = cfg.get("run", {})
run_id = ""
if isinstance(run, dict):
    run_id = str(run.get("run_id", "")).strip()
print(run_id)
PY
)"
if [[ -z "${QUICK_RUN_ID}" ]]; then
  QUICK_RUN_ID="dataset_3d_test2_pilot_quick"
fi
QUICK_MODEL_STATE="runs/${QUICK_RUN_ID}/train/outputs/model_state.json"
if [[ ! -f "$QUICK_MODEL_STATE" ]]; then
  echo "ERROR: quick model_state not found: $QUICK_MODEL_STATE" >&2
  exit 1
fi

echo "[pilot] step 3/6 quick eval primary (${PRIMARY_HOLDOUT_RUN})"
"$PYTHON" scripts/eval_dataset_3d_test2_holdout.py \
  --model-state "$QUICK_MODEL_STATE" \
  --holdout-json "$PREP_DIR/holdout_dataset.json" \
  --split-manifest "$PREP_DIR/split_manifest.json" \
  --out-dir "$QUICK_PRIMARY_EVAL_DIR" \
  --viz-config-yaml "$VIZ_CONFIG" \
  --analysis-xy-margin-vox "$ANALYSIS_XY_MARGIN_VOX" \
  --analysis-z-margin-vox "$ANALYSIS_Z_MARGIN_VOX"

echo "[pilot] step 4/6 gate primary holdout"
"$PYTHON" - <<PY
import json
from pathlib import Path

baseline_path = Path("$BASELINE_EVAL_JSON")
current_path = Path("$QUICK_PRIMARY_EVAL_DIR") / "holdout_eval.json"
out_path = Path("$GATE_JSON")
out_path.parent.mkdir(parents=True, exist_ok=True)

def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        obj = json.load(fp)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid payload: {path}")
    return obj

cur = _load(current_path)
cur_summary = cur.get("summary", {})
baseline = _load(baseline_path) if baseline_path.exists() else None
base_summary = baseline.get("summary", {}) if baseline else {}

cur_iou = float(cur_summary.get("top_aperture_iou_final", 0.0))
cur_early = float(cur_summary.get("early_window_error", 0.0))
cur_leak = float(cur_summary.get("aperture_boundary_leak_ratio", 1.0))
base_iou = float(base_summary.get("top_aperture_iou_final", cur_iou))
base_early = float(base_summary.get("early_window_error", cur_early))

checks = {
    "iou_not_worse": bool(cur_iou >= base_iou),
    "early_not_degrade_10pct": bool(cur_early <= (base_early * 1.10 if base_early > 0 else cur_early)),
    "boundary_leak_threshold": bool(cur_leak <= 0.02),
}
passed = bool(all(checks.values()))
payload = {
    "primary_holdout_run": "$PRIMARY_HOLDOUT_RUN",
    "baseline_eval_json": str(baseline_path) if baseline else None,
    "current_eval_json": str(current_path),
    "checks": checks,
    "current": {
        "top_aperture_iou_final": cur_iou,
        "early_window_error": cur_early,
        "aperture_boundary_leak_ratio": cur_leak,
    },
    "baseline": {
        "top_aperture_iou_final": base_iou,
        "early_window_error": base_early,
    } if baseline else None,
    "gate_passed": passed,
}
with out_path.open("w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2)
print(f"gate written: {out_path}")
PY

GATE_PASSED="$("$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$GATE_JSON")
if not p.exists():
    print("0")
else:
    obj = json.loads(p.read_text())
    print("1" if bool(obj.get("gate_passed")) else "0")
PY
)"

if [[ "$GATE_PASSED" != "1" ]]; then
  echo "[pilot] gate failed, skipping secondary holdout and writing partial comparison"
  "$PYTHON" - <<PY
import json
from pathlib import Path
primary = Path("$QUICK_PRIMARY_EVAL_DIR") / "holdout_eval.json"
gate = Path("$GATE_JSON")
sampling = Path("$PREP_DIR/sampling_diagnostics.json")
primary_payload = None
if primary.exists():
    primary_payload = json.loads(primary.read_text())
primary_summary = (primary_payload or {}).get("summary", {})
metric_keys = [
    "top_aperture_iou_final",
    "interface_depth_mae",
    "aperture_boundary_leak_ratio",
    "early_window_error",
]
out = {
    "quick_primary_eval_json": str(primary) if primary.exists() else None,
    "quick_secondary_eval_json": None,
    "real_primary_eval_json": None,
    "sampling_diagnostics_json": str(sampling) if sampling.exists() else None,
    "gate_json": str(gate) if gate.exists() else None,
    "gate_passed": False,
    "metrics": [
        {
            "metric": key,
            "quick_primary": float(primary_summary.get(key)) if isinstance(primary_summary.get(key), (int, float)) else None,
            "real_primary": None,
            "delta_real_minus_quick": None,
            "secondary_quick": None,
        }
        for key in metric_keys
    ],
    "visual_reports": {
        "quick_primary": str(Path("$QUICK_PRIMARY_EVAL_DIR") / "report_index.json"),
        "quick_secondary": None,
        "real_primary": None,
        "train_visual_manifest": str(Path("runs/$QUICK_RUN_ID/train/outputs/viz/visualization_manifest.json")),
    },
}
out_path = Path("$COMPARE_JSON")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as fp:
    json.dump(out, fp, indent=2)
print(f"comparison written: {out_path}")

report_index = {
    "prepare_dir": str(Path("$PREP_DIR")),
    "quick_run_id": "$QUICK_RUN_ID",
    "quick_model_state_json": "$QUICK_MODEL_STATE",
    "comparison_json": str(out_path),
    "quick_primary_report_index": str(Path("$QUICK_PRIMARY_EVAL_DIR") / "report_index.json"),
    "quick_secondary_report_index": None,
    "real_primary_report_index": None,
    "train_viz_manifest_json": str(Path("runs/$QUICK_RUN_ID/train/outputs/viz/visualization_manifest.json")),
}
report_path = Path("$EVAL_ROOT") / "report_index.json"
with report_path.open("w", encoding="utf-8") as fp:
    json.dump(report_index, fp, indent=2)
print(f"report index written: {report_path}")
PY
  exit 2
fi

echo "[pilot] step 5/6 secondary follow-up (${SECONDARY_HOLDOUT_RUN})"
"$PYTHON" scripts/prepare_dataset_3d_test2.py \
  "${prepare_common_args[@]}" \
  --train-runs "$EFFECTIVE_TRAIN_RUNS" \
  --holdout-runs "$SECONDARY_HOLDOUT_RUN" \
  --out-dir "$PREP_SECONDARY_DIR"

"$PYTHON" scripts/eval_dataset_3d_test2_holdout.py \
  --model-state "$QUICK_MODEL_STATE" \
  --holdout-json "$PREP_SECONDARY_DIR/holdout_dataset.json" \
  --split-manifest "$PREP_SECONDARY_DIR/split_manifest.json" \
  --out-dir "$QUICK_SECONDARY_EVAL_DIR" \
  --viz-config-yaml "$VIZ_CONFIG" \
  --analysis-xy-margin-vox "$ANALYSIS_XY_MARGIN_VOX" \
  --analysis-z-margin-vox "$ANALYSIS_Z_MARGIN_VOX"

REAL_RUN_ID="dataset_3d_test2_pilot_real"
REAL_MODEL_STATE="runs/${REAL_RUN_ID}/train/outputs/model_state.json"
if [[ "$RUN_REAL_ME" == "1" ]]; then
  echo "[pilot] optional real ME path"
  REAL_CHECK="$("$PYTHON" - <<'PY'
import json, os, subprocess, sys
cmd = [sys.executable, "-c", "from wafer_surrogate.runtime.capabilities import detect_runtime_capabilities as d; import json; print(json.dumps(d().missing_summary()))"]
env = dict(os.environ)
env.pop("PYTHONPATH", None)
p = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
if p.returncode != 0:
    print('{"ok":false}')
    raise SystemExit(0)
print(p.stdout.strip())
PY
)"
  if ! echo "$REAL_CHECK" | grep -q '"MinkowskiEngine_real": true'; then
    echo "ERROR: real ME path unavailable. capability=$REAL_CHECK" >&2
    exit 1
  fi
  TMP_CFG="$(mktemp -t dataset3d_pilot_real_XXXX.toml)"
  trap 'rm -f "$TMP_CFG"' EXIT
  sed 's/run_id = "dataset_3d_test2_pilot_quick"/run_id = "dataset_3d_test2_pilot_real"/' "$BASE_CONFIG" > "$TMP_CFG"
  env -u PYTHONPATH "$PYTHON" -m wafer_surrogate.cli pipeline run --config "$TMP_CFG" --stages train
  if [[ ! -f "$REAL_MODEL_STATE" ]]; then
    echo "ERROR: real model_state not found: $REAL_MODEL_STATE" >&2
    exit 1
  fi
  env -u PYTHONPATH "$PYTHON" scripts/eval_dataset_3d_test2_holdout.py \
    --model-state "$REAL_MODEL_STATE" \
    --holdout-json "$PREP_DIR/holdout_dataset.json" \
    --split-manifest "$PREP_DIR/split_manifest.json" \
    --out-dir "$REAL_EVAL_DIR" \
    --viz-config-yaml "$VIZ_CONFIG" \
    --analysis-xy-margin-vox "$ANALYSIS_XY_MARGIN_VOX" \
    --analysis-z-margin-vox "$ANALYSIS_Z_MARGIN_VOX"
fi

echo "[pilot] step 6/6 comparison summary"
"$PYTHON" - <<PY
import json
from pathlib import Path

def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        return None
    return payload

primary_quick = _load(Path("$QUICK_PRIMARY_EVAL_DIR") / "holdout_eval.json")
secondary_quick = _load(Path("$QUICK_SECONDARY_EVAL_DIR") / "holdout_eval.json")
real_primary = _load(Path("$REAL_EVAL_DIR") / "holdout_eval.json")
gate = _load(Path("$GATE_JSON"))
sampling_diag = Path("$PREP_DIR/sampling_diagnostics.json")

def _metric(payload: dict | None, key: str) -> float | None:
    if not payload:
        return None
    summary = payload.get("summary", {})
    val = summary.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None

pairs = [
    "top_aperture_iou_final",
    "interface_depth_mae",
    "aperture_boundary_leak_ratio",
    "early_window_error",
]
metrics = []
for key in pairs:
    q = _metric(primary_quick, key)
    r = _metric(real_primary, key)
    metrics.append(
        {
            "metric": key,
            "quick_primary": q,
            "real_primary": r,
            "delta_real_minus_quick": (float(r - q) if (q is not None and r is not None) else None),
            "secondary_quick": _metric(secondary_quick, key),
        }
    )

out = {
    "quick_primary_eval_json": str(Path("$QUICK_PRIMARY_EVAL_DIR") / "holdout_eval.json"),
    "quick_secondary_eval_json": str(Path("$QUICK_SECONDARY_EVAL_DIR") / "holdout_eval.json"),
    "real_primary_eval_json": str(Path("$REAL_EVAL_DIR") / "holdout_eval.json") if real_primary else None,
    "sampling_diagnostics_json": str(sampling_diag) if sampling_diag.exists() else None,
    "gate_json": str(Path("$GATE_JSON")),
    "gate_passed": bool(gate.get("gate_passed")) if gate else None,
    "metrics": metrics,
    "visual_reports": {
        "quick_primary": str(Path("$QUICK_PRIMARY_EVAL_DIR") / "report_index.json"),
        "quick_secondary": str(Path("$QUICK_SECONDARY_EVAL_DIR") / "report_index.json"),
        "real_primary": str(Path("$REAL_EVAL_DIR") / "report_index.json") if real_primary else None,
        "train_visual_manifest": str(Path("runs/$QUICK_RUN_ID/train/outputs/viz/visualization_manifest.json")),
    },
}
out_path = Path("$COMPARE_JSON")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as fp:
    json.dump(out, fp, indent=2)
print(f"comparison written: {out_path}")

report_index = {
    "prepare_dir": str(Path("$PREP_DIR")),
    "quick_run_id": "$QUICK_RUN_ID",
    "quick_model_state_json": "$QUICK_MODEL_STATE",
    "comparison_json": str(out_path),
    "quick_primary_report_index": str(Path("$QUICK_PRIMARY_EVAL_DIR") / "report_index.json"),
    "quick_secondary_report_index": str(Path("$QUICK_SECONDARY_EVAL_DIR") / "report_index.json"),
    "real_primary_report_index": str(Path("$REAL_EVAL_DIR") / "report_index.json") if real_primary else None,
    "train_viz_manifest_json": str(Path("runs/$QUICK_RUN_ID/train/outputs/viz/visualization_manifest.json")),
}
report_path = Path("$EVAL_ROOT") / "report_index.json"
with report_path.open("w", encoding="utf-8") as fp:
    json.dump(report_index, fp, indent=2)
print(f"report index written: {report_path}")
PY

echo "[pilot] done"
echo "  quick primary eval : $QUICK_PRIMARY_EVAL_DIR"
echo "  quick secondary eval: $QUICK_SECONDARY_EVAL_DIR"
echo "  gate file          : $GATE_JSON"
echo "  compare file       : $COMPARE_JSON"
