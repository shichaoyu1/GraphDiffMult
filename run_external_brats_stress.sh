#!/usr/bin/env bash
set -euo pipefail

# External validation / stress test runner for BraTS_GLI-style data.
# Design goal:
# - keep this as external validation, not a full main-experiment rerun
# - run a small matrix first: clip, no_graph, full, graph_shared_only (+optional no_anchor)
# - run 3 seeds by default: 42,43,44
# - produce summary tables and bootstrap CI for this run_id

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-external_brats_$(date +%Y%m%d_%H%M%S)}"

DATA_ROOT="${DATA_ROOT:-}"
METADATA_TSV="${METADATA_TSV:-}"

CORE_SEEDS="${CORE_SEEDS:-42 43 44}"
CORE_VARIANTS_BASE="${CORE_VARIANTS_BASE:-clip no_graph full graph_shared_only}"
INCLUDE_NO_ANCHOR="${INCLUDE_NO_ANCHOR:-0}"  # set 1 only when pathology/molecular anchors are available

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROI_SIZE="${ROI_SIZE:-96}"
Z_SLICES="${Z_SLICES:-7}"
ALIGN_MAX_CASES="${ALIGN_MAX_CASES:-80}"
GRAPH_TOP_K="${GRAPH_TOP_K:-3}"

# For stress-test scale control without changing run_all_pro.sh internals:
# extra args are appended by run_all_pro.sh
MAX_CASES="${MAX_CASES:-180}"
SEMANTIC_EXTRA_ARGS="${SEMANTIC_EXTRA_ARGS:---max_cases ${MAX_CASES}}"

BASE_LAMBDA_ANCHOR="${BASE_LAMBDA_ANCHOR:-0.05}"
BASE_LAMBDA_CONS="${BASE_LAMBDA_CONS:-0.05}"
BASE_LAMBDA_DIFF="${BASE_LAMBDA_DIFF:-0.05}"

N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2026}"

mkdir -p logs results output

log() {
  echo "[$(date '+%F %T')] $*"
}

first_existing_dir() {
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$(first_existing_dir \
    /root/autodl-tmp/dataset/BraTS2021_Training_Data \
    /root/autodl-tmp/BraTS2021_Training_Data \
    /root/autodl-tmp/dataset/BraTS_GLI \
    /root/autodl-tmp/BraTS_GLI \
    /root/autodl-tmp/dataset/UTSW-Glioma \
    /root/autodl-tmp/UTSW-Glioma \
    ./datasetDemo \
  )" || {
    log "[ERROR] DATA_ROOT not set and no default dataset directory found."
    exit 1
  }
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  log "[ERROR] DATA_ROOT not found: $DATA_ROOT"
  exit 1
fi

# Auto-resolve common BraTS layout: parent/BraTS2021_Training_Data/<patient_id>
if [[ -d "$DATA_ROOT/BraTS2021_Training_Data" ]]; then
  DATA_ROOT="$DATA_ROOT/BraTS2021_Training_Data"
  log "[INFO] Auto-switched DATA_ROOT to nested BraTS training folder: ${DATA_ROOT}"
fi

if [[ "$INCLUDE_NO_ANCHOR" == "1" ]]; then
  CORE_VARIANTS="${CORE_VARIANTS_BASE} no_anchor"
else
  CORE_VARIANTS="${CORE_VARIANTS_BASE}"
fi

log "===== EXTERNAL STRESS TEST START ====="
log "RUN_ID=${RUN_ID}"
log "DATA_ROOT=${DATA_ROOT}"
log "METADATA_TSV=${METADATA_TSV:-<none>}"
log "CORE_VARIANTS=${CORE_VARIANTS}"
log "CORE_SEEDS=${CORE_SEEDS}"
log "EPOCHS=${EPOCHS}, MAX_CASES=${MAX_CASES}, ALIGN_MAX_CASES=${ALIGN_MAX_CASES}"

# Preflight: make sure current dataset+metadata can form semantic cases and anchors.
export DATA_ROOT METADATA_TSV PYTHON_BIN INCLUDE_NO_ANCHOR
"$PYTHON_BIN" - <<'PY'
import os
from train_semantic_alignment import discover_semantic_cases, stratified_split, build_anchor_vocab

data_root = os.environ["DATA_ROOT"]
metadata_tsv = os.environ.get("METADATA_TSV") or None

cases = discover_semantic_cases(
    data_root,
    metadata_tsv=metadata_tsv,
    max_cases=64,
    seed=42,
    include_clinical=False,
)
if len(cases) < 2:
    raise SystemExit(
        "Preflight failed: <2 semantic cases. "
        "This dataset likely lacks usable pathology/molecular metadata for current alignment protocol."
    )

splits = stratified_split(cases, train_ratio=0.7, val_ratio=0.1, seed=42)
anchors, _ = build_anchor_vocab(
    splits["train"],
    include_pathology=True,
    include_molecular=True,
    include_clinical=False,
)
if len(anchors) < 2:
    raise SystemExit(
        "Preflight failed: <2 train anchors. "
        "Need harmonized pathology/molecular tags (or revise anchor protocol for this external set)."
    )
print(f"[Preflight OK] cases={len(cases)} train={len(splits['train'])} anchors={len(anchors)}")

# Check whether no_anchor variant is actually feasible on this external dataset.
mol_only_anchors, _ = build_anchor_vocab(
    splits["train"],
    include_pathology=False,
    include_molecular=True,
    include_clinical=False,
)
if os.environ.get("INCLUDE_NO_ANCHOR", "0") == "1" and len(mol_only_anchors) < 2:
    print(
        "[Preflight NOTE] INCLUDE_NO_ANCHOR=1 but molecular-only anchors <2; "
        "no_anchor will be auto-disabled."
    )
    with open(".external_no_anchor_disable.flag", "w", encoding="utf-8") as f:
        f.write("disable_no_anchor=1\n")
PY

if [[ -f ".external_no_anchor_disable.flag" ]]; then
  rm -f ".external_no_anchor_disable.flag"
  CORE_VARIANTS="$CORE_VARIANTS_BASE"
  log "[ADJUST] no_anchor disabled due to insufficient molecular-only anchors on external set."
fi

log "Preflight passed. Launching core stress matrix..."

RUN_ID="$RUN_ID" \
DATA_ROOT="$DATA_ROOT" \
METADATA_TSV="$METADATA_TSV" \
PYTHON_BIN="$PYTHON_BIN" \
RUN_CORE=1 \
RUN_LAMBDA_SWEEP=0 \
RUN_IDH_SANITY=0 \
CORE_VARIANTS="$CORE_VARIANTS" \
CORE_SEEDS="$CORE_SEEDS" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
ROI_SIZE="$ROI_SIZE" \
Z_SLICES="$Z_SLICES" \
ALIGN_MAX_CASES="$ALIGN_MAX_CASES" \
GRAPH_TOP_K="$GRAPH_TOP_K" \
BASE_LAMBDA_ANCHOR="$BASE_LAMBDA_ANCHOR" \
BASE_LAMBDA_CONS="$BASE_LAMBDA_CONS" \
BASE_LAMBDA_DIFF="$BASE_LAMBDA_DIFF" \
SEMANTIC_EXTRA_ARGS="$SEMANTIC_EXTRA_ARGS" \
./run_all_pro.sh

log "Core stress matrix done. Running bootstrap..."
RUN_ID="$RUN_ID" \
OUTPUT_ROOT="output" \
N_BOOTSTRAP="$N_BOOTSTRAP" \
BOOTSTRAP_SEED="$BOOTSTRAP_SEED" \
./run_bootstrap_5seed.sh

log "===== EXTERNAL STRESS TEST DONE ====="
log "Summary CSV: results/summary_${RUN_ID}.csv"
log "Summary TeX: results/table_${RUN_ID}.tex"
log "Bootstrap CSV: results/semantic_bootstrap_5seed_${RUN_ID}.csv"
log "Bootstrap JSON: results/semantic_bootstrap_5seed_${RUN_ID}.json"
