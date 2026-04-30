#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/UTSW-Glioma}"
METADATA_TSV="${METADATA_TSV:-/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROI_SIZE="${ROI_SIZE:-96}"
Z_SLICES="${Z_SLICES:-7}"
MAX_RETRY="${MAX_RETRY:-2}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-10}"

VARIANTS="${VARIANTS:-full clip no_anchor graph_only modality_vector no_private no_graph}"
SEEDS="${SEEDS:-42}"
RUN_UTSW_SANITY="${RUN_UTSW_SANITY:-0}"

mkdir -p logs output results

MASTER_LOG="logs/master_${RUN_ID}.log"
GPU_LOG="logs/gpu_${RUN_ID}.csv"
RESULT_CSV="results/summary_${RUN_ID}.csv"
RESULT_TEX="results/table_${RUN_ID}.tex"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
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
  shift 2

  local attempt=1
  while [[ "$attempt" -le "$MAX_RETRY" ]]; do
    log "[RUN] ${name} attempt ${attempt}/${MAX_RETRY}"
    log "[CMD] $*"
    if "$@" > "$log_path" 2>&1; then
      log "[SUCCESS] ${name}"
      return 0
    fi
    log "[FAIL] ${name}; tail follows:"
    tail -n 40 "$log_path" | tee -a "$MASTER_LOG" || true
    attempt=$((attempt + 1))
    sleep 8
  done

  log "[ERROR] ${name} failed after ${MAX_RETRY} attempts"
  return 1
}

metadata_args=()
if [[ -f "$METADATA_TSV" ]]; then
  metadata_args=(--metadata_tsv "$METADATA_TSV")
else
  log "[WARN] METADATA_TSV not found: ${METADATA_TSV}; relying on auto-discovery"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  log "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi

log "===== START RUN_ID=${RUN_ID} ====="
log "DATA_ROOT=${DATA_ROOT}"
log "METADATA_TSV=${METADATA_TSV}"
log "VARIANTS=${VARIANTS}"
log "SEEDS=${SEEDS}"

bash utils/gpu_logger.sh "$GPU_LOG" "$GPU_LOG_INTERVAL" &
GPU_LOGGER_PID=$!
log "GPU logger PID=${GPU_LOGGER_PID}, file=${GPU_LOG}"

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    name="semantic_${variant}_s${seed}"
    out_dir="output/${name}_${RUN_ID}"
    log_path="logs/${name}_${RUN_ID}.log"

    cmd=(
      stdbuf -oL -eL "$PYTHON_BIN" -u train_semantic_alignment.py
      --data_root "$DATA_ROOT"
      "${metadata_args[@]}"
      --variant "$variant"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --roi_size "$ROI_SIZE"
      --z_slices "$Z_SLICES"
      --seed "$seed"
      --graph_type learnable
      --augment
      --out_dir "$out_dir"
    )

    run_with_retry "$name" "$log_path" "${cmd[@]}"
  done
done

if [[ "$RUN_UTSW_SANITY" == "1" ]]; then
  name="utsw_idh_full_s42"
  out_dir="output/${name}_${RUN_ID}"
  log_path="logs/${name}_${RUN_ID}.log"
  cmd=(
    stdbuf -oL -eL "$PYTHON_BIN" -u train_utsw.py
    --data_root "$DATA_ROOT"
    "${metadata_args[@]}"
    --task idh
    --variant full
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --roi_size "$ROI_SIZE"
    --z_slices "$Z_SLICES"
    --seed 42
    --graph_type learnable
    --class_weight
    --augment
    --out_dir "$out_dir"
  )
  run_with_retry "$name" "$log_path" "${cmd[@]}"
fi

log "Extracting metrics..."
"$PYTHON_BIN" utils/extract_metrics.py output "$RESULT_CSV" --run_id "$RUN_ID" | tee -a "$MASTER_LOG"

log "Generating LaTeX table..."
"$PYTHON_BIN" utils/make_table.py "$RESULT_CSV" "$RESULT_TEX" | tee -a "$MASTER_LOG"

log "===== ALL DONE ====="
log "CSV: ${RESULT_CSV}"
log "LaTeX: ${RESULT_TEX}"
log "GPU log: ${GPU_LOG}"
