#!/usr/bin/env bash
set -euo pipefail

########################################
# Round-2 semantic alignment scheduler
#
# Default goal:
#   1) rerun the strongest/most informative variants with 5 seeds
#   2) keep lambda sweeps opt-in, because they are more expensive
#   3) keep IDH downstream sanity opt-in, because it is not the main claim
########################################

RUN_ID="${RUN_ID:-round2_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-}"
METADATA_TSV="${METADATA_TSV:-}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROI_SIZE="${ROI_SIZE:-96}"
Z_SLICES="${Z_SLICES:-7}"
ALIGN_MAX_CASES="${ALIGN_MAX_CASES:-50}"
GRAPH_TOP_K="${GRAPH_TOP_K:-3}"

MAX_RETRY="${MAX_RETRY:-2}"
RETRY_SLEEP="${RETRY_SLEEP:-8}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-10}"
SKIP_DONE="${SKIP_DONE:-1}"

RUN_CORE="${RUN_CORE:-1}"
RUN_LAMBDA_SWEEP="${RUN_LAMBDA_SWEEP:-0}"
RUN_IDH_SANITY="${RUN_IDH_SANITY:-0}"

CORE_VARIANTS="${CORE_VARIANTS:-full clip no_graph no_anchor}"
CORE_SEEDS="${CORE_SEEDS:-42 43 44 45 46}"

SWEEP_SEEDS="${SWEEP_SEEDS:-42}"
LAMBDA_CONS_VALUES="${LAMBDA_CONS_VALUES:-0 0.005 0.01 0.02 0.05 0.1}"
LAMBDA_DIFF_VALUES="${LAMBDA_DIFF_VALUES:-0 0.01 0.03 0.05 0.1}"
BASE_LAMBDA_CONS="${BASE_LAMBDA_CONS:-0.05}"
BASE_LAMBDA_DIFF="${BASE_LAMBDA_DIFF:-0.05}"
BASE_LAMBDA_ANCHOR="${BASE_LAMBDA_ANCHOR:-0.05}"

IDH_VARIANTS="${IDH_VARIANTS:-full graph shared_private}"
IDH_SEEDS="${IDH_SEEDS:-42}"

SEMANTIC_EXTRA_ARGS="${SEMANTIC_EXTRA_ARGS:-}"
IDH_EXTRA_ARGS="${IDH_EXTRA_ARGS:-}"

mkdir -p logs output results

MASTER_LOG="logs/master_${RUN_ID}.log"
GPU_LOG="logs/gpu_${RUN_ID}.csv"
RESULT_CSV="results/summary_${RUN_ID}.csv"
RESULT_TEX="results/table_${RUN_ID}.tex"
MANIFEST_CSV="results/manifest_${RUN_ID}.csv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
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

first_existing_file() {
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

slug() {
  echo "$1" | sed 's/\./p/g; s/-/m/g; s/+//g'
}

cleanup() {
  if [[ -n "${GPU_LOGGER_PID:-}" ]] && kill -0 "$GPU_LOGGER_PID" 2>/dev/null; then
    kill "$GPU_LOGGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_with_retry() {
  local name="$1"
  local log_path="$2"
  local done_file="$3"
  shift 3

  if [[ "$SKIP_DONE" == "1" && -f "$done_file" ]]; then
    log "[SKIP] ${name}; found ${done_file}"
    return 0
  fi

  local attempt=1
  while [[ "$attempt" -le "$MAX_RETRY" ]]; do
    log "[RUN] ${name} attempt ${attempt}/${MAX_RETRY}"
    log "[CMD] $*"
    if "$@" > "$log_path" 2>&1; then
      log "[SUCCESS] ${name}"
      return 0
    fi

    log "[FAIL] ${name}; tail follows:"
    tail -n 50 "$log_path" | tee -a "$MASTER_LOG" || true
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done

  log "[ERROR] ${name} failed after ${MAX_RETRY} attempts"
  return 1
}

append_manifest() {
  local group="$1"
  local name="$2"
  local variant="$3"
  local seed="$4"
  local lambda_anchor="$5"
  local lambda_cons="$6"
  local lambda_diff="$7"
  local out_dir="$8"
  echo "${group},${name},${variant},${seed},${lambda_anchor},${lambda_cons},${lambda_diff},${out_dir}" >> "$MANIFEST_CSV"
}

run_semantic_job() {
  local group="$1"
  local variant="$2"
  local seed="$3"
  local lambda_anchor="$4"
  local lambda_cons="$5"
  local lambda_diff="$6"
  local suffix="$7"

  local name="semantic_${variant}_${suffix}_s${seed}"
  local out_dir="output/${name}_${RUN_ID}"
  local log_path="logs/${name}_${RUN_ID}.log"

  append_manifest "$group" "$name" "$variant" "$seed" "$lambda_anchor" "$lambda_cons" "$lambda_diff" "$out_dir"

  local metadata_args=()
  if [[ -n "$METADATA_TSV" && -f "$METADATA_TSV" ]]; then
    metadata_args=(--metadata_tsv "$METADATA_TSV")
  fi

  local extra_args=()
  if [[ -n "$SEMANTIC_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=($SEMANTIC_EXTRA_ARGS)
  fi

  local cmd=(
    stdbuf -oL -eL "$PYTHON_BIN" -u train_semantic_alignment.py
    --data_root "$DATA_ROOT"
    "${metadata_args[@]}"
    --variant "$variant"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --roi_size "$ROI_SIZE"
    --z_slices "$Z_SLICES"
    --align_max_cases "$ALIGN_MAX_CASES"
    --graph_top_k "$GRAPH_TOP_K"
    --seed "$seed"
    --graph_type learnable
    --lambda_anchor "$lambda_anchor"
    --lambda_cons "$lambda_cons"
    --lambda_diff "$lambda_diff"
    --augment
    --out_dir "$out_dir"
    "${extra_args[@]}"
  )

  run_with_retry "$name" "$log_path" "${out_dir}/semantic_alignment_metrics.json" "${cmd[@]}"
}

run_idh_job() {
  local variant="$1"
  local seed="$2"

  local name="utsw_idh_${variant}_s${seed}"
  local out_dir="output/${name}_${RUN_ID}"
  local log_path="logs/${name}_${RUN_ID}.log"

  append_manifest "idh_sanity" "$name" "$variant" "$seed" "" "" "" "$out_dir"

  local metadata_args=()
  if [[ -n "$METADATA_TSV" && -f "$METADATA_TSV" ]]; then
    metadata_args=(--metadata_tsv "$METADATA_TSV")
  fi

  local extra_args=()
  if [[ -n "$IDH_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=($IDH_EXTRA_ARGS)
  fi

  local cmd=(
    stdbuf -oL -eL "$PYTHON_BIN" -u train_utsw.py
    --data_root "$DATA_ROOT"
    "${metadata_args[@]}"
    --task idh
    --variant "$variant"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --roi_size "$ROI_SIZE"
    --z_slices "$Z_SLICES"
    --seed "$seed"
    --graph_type learnable
    --class_weight
    --augment
    --out_dir "$out_dir"
    "${extra_args[@]}"
  )

  run_with_retry "$name" "$log_path" "${out_dir}/test_metrics.json" "${cmd[@]}"
}

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$(first_existing_dir \
    /root/autodl-tmp/dataset/UTSW-Glioma \
    /root/autodl-tmp/UTSW-Glioma \
    ../dataset/UTSW-Glioma \
    ./dataset/UTSW-Glioma \
  )" || {
    log "[ERROR] DATA_ROOT not set and no default UTSW-Glioma directory found"
    exit 1
  }
fi

if [[ -z "$METADATA_TSV" ]]; then
  METADATA_TSV="$(first_existing_file \
    /root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
    /root/autodl-tmp/UTSW_Glioma_Metadata-2-1.tsv \
    "$(dirname "$DATA_ROOT")/UTSW_Glioma_Metadata-2-1.tsv" \
  )" || true
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  log "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi

echo "group,name,variant,seed,lambda_anchor,lambda_cons,lambda_diff,out_dir" > "$MANIFEST_CSV"

log "===== START RUN_ID=${RUN_ID} ====="
log "DATA_ROOT=${DATA_ROOT}"
log "METADATA_TSV=${METADATA_TSV:-<auto-discovery inside Python>}"
log "RUN_CORE=${RUN_CORE} CORE_VARIANTS=${CORE_VARIANTS} CORE_SEEDS=${CORE_SEEDS}"
log "RUN_LAMBDA_SWEEP=${RUN_LAMBDA_SWEEP} SWEEP_SEEDS=${SWEEP_SEEDS}"
log "RUN_IDH_SANITY=${RUN_IDH_SANITY} IDH_VARIANTS=${IDH_VARIANTS}"
log "EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} ROI_SIZE=${ROI_SIZE} Z_SLICES=${Z_SLICES}"
log "BASE_LAMBDA_ANCHOR=${BASE_LAMBDA_ANCHOR} BASE_LAMBDA_CONS=${BASE_LAMBDA_CONS} BASE_LAMBDA_DIFF=${BASE_LAMBDA_DIFF}"

bash utils/gpu_logger.sh "$GPU_LOG" "$GPU_LOG_INTERVAL" &
GPU_LOGGER_PID=$!
log "GPU logger PID=${GPU_LOGGER_PID}, file=${GPU_LOG}"

if [[ "$RUN_CORE" == "1" ]]; then
  log "===== CORE VARIANTS ====="
  for seed in $CORE_SEEDS; do
    for variant in $CORE_VARIANTS; do
      run_semantic_job "core" "$variant" "$seed" "$BASE_LAMBDA_ANCHOR" "$BASE_LAMBDA_CONS" "$BASE_LAMBDA_DIFF" "core"
    done
  done
fi

if [[ "$RUN_LAMBDA_SWEEP" == "1" ]]; then
  log "===== LAMBDA CONS SWEEP ====="
  for seed in $SWEEP_SEEDS; do
    for lambda_cons in $LAMBDA_CONS_VALUES; do
      suffix="cons$(slug "$lambda_cons")_diff$(slug "$BASE_LAMBDA_DIFF")"
      run_semantic_job "lambda_cons" "full" "$seed" "$BASE_LAMBDA_ANCHOR" "$lambda_cons" "$BASE_LAMBDA_DIFF" "$suffix"
    done
  done

  log "===== LAMBDA DIFF SWEEP ====="
  for seed in $SWEEP_SEEDS; do
    for lambda_diff in $LAMBDA_DIFF_VALUES; do
      suffix="cons$(slug "$BASE_LAMBDA_CONS")_diff$(slug "$lambda_diff")"
      run_semantic_job "lambda_diff" "full" "$seed" "$BASE_LAMBDA_ANCHOR" "$BASE_LAMBDA_CONS" "$lambda_diff" "$suffix"
    done
  done
fi

if [[ "$RUN_IDH_SANITY" == "1" ]]; then
  log "===== IDH DOWNSTREAM SANITY ====="
  log "[NOTE] train_utsw.py has no true clip/no_graph semantic variant; this is only a downstream sanity panel."
  for seed in $IDH_SEEDS; do
    for variant in $IDH_VARIANTS; do
      run_idh_job "$variant" "$seed"
    done
  done
fi

log "Extracting metrics..."
"$PYTHON_BIN" utils/extract_metrics.py output "$RESULT_CSV" --run_id "$RUN_ID" | tee -a "$MASTER_LOG"

log "Generating LaTeX table..."
"$PYTHON_BIN" utils/make_table.py "$RESULT_CSV" "$RESULT_TEX" | tee -a "$MASTER_LOG"

log "===== ALL DONE ====="
log "Manifest: ${MANIFEST_CSV}"
log "CSV: ${RESULT_CSV}"
log "LaTeX: ${RESULT_TEX}"
log "GPU log: ${GPU_LOG}"
