#!/usr/bin/env bash
set -u -o pipefail

# Minimal smoke validation loop:
# - run tiny training/eval for all target variants
# - do not stop on single variant failure
# - collect per-variant pass/fail and artifact completeness

RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

DATA_ROOT="${DATA_ROOT:-}"
METADATA_TSV="${METADATA_TSV:-}"

CORE_VARIANTS="${CORE_VARIANTS:-full clip medclip_style dcca graph_shared_only no_graph no_anchor}"
CORE_SEEDS="${CORE_SEEDS:-42}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ROI_SIZE="${ROI_SIZE:-96}"
Z_SLICES="${Z_SLICES:-7}"
MAX_CASES="${MAX_CASES:-24}"
ALIGN_MAX_CASES="${ALIGN_MAX_CASES:-16}"
GRAPH_TOP_K="${GRAPH_TOP_K:-2}"

LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-0.05}"
LAMBDA_CONS="${LAMBDA_CONS:-0.05}"
LAMBDA_DIFF="${LAMBDA_DIFF:-0.05}"

N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2026}"

mkdir -p logs results "${OUTPUT_ROOT}"

MASTER_LOG="logs/smoke_${RUN_ID}.log"
STATUS_CSV="results/smoke_status_${RUN_ID}.csv"
SUMMARY_CSV="results/summary_${RUN_ID}.csv"
SUMMARY_TEX="results/table_${RUN_ID}.tex"
BOOT_CSV="results/semantic_bootstrap_5seed_${RUN_ID}.csv"
BOOT_JSON="results/semantic_bootstrap_5seed_${RUN_ID}.json"

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

echo "variant,seed,status,exit_code,out_dir,has_metrics,has_records,has_space_plot,has_graph_plot,nonfinite_dropped" > "$STATUS_CSV"

log "===== SMOKE START RUN_ID=${RUN_ID} ====="
log "DATA_ROOT=${DATA_ROOT}"
log "METADATA_TSV=${METADATA_TSV:-<auto-discovery inside Python>}"
log "CORE_VARIANTS=${CORE_VARIANTS} CORE_SEEDS=${CORE_SEEDS}"
log "EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} MAX_CASES=${MAX_CASES} ALIGN_MAX_CASES=${ALIGN_MAX_CASES}"

for seed in $CORE_SEEDS; do
  for variant in $CORE_VARIANTS; do
    name="smoke_${variant}_s${seed}"
    out_dir="${OUTPUT_ROOT}/${name}_${RUN_ID}"
    run_log="logs/${name}_${RUN_ID}.log"
    metadata_args=()
    if [[ -n "$METADATA_TSV" && -f "$METADATA_TSV" ]]; then
      metadata_args=(--metadata_tsv "$METADATA_TSV")
    fi

    cmd=(
      stdbuf -oL -eL "$PYTHON_BIN" -u train_semantic_alignment.py
      --data_root "$DATA_ROOT"
      "${metadata_args[@]}"
      --variant "$variant"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --roi_size "$ROI_SIZE"
      --z_slices "$Z_SLICES"
      --max_cases "$MAX_CASES"
      --align_max_cases "$ALIGN_MAX_CASES"
      --graph_top_k "$GRAPH_TOP_K"
      --seed "$seed"
      --graph_type learnable
      --lambda_anchor "$LAMBDA_ANCHOR"
      --lambda_cons "$LAMBDA_CONS"
      --lambda_diff "$LAMBDA_DIFF"
      --out_dir "$out_dir"
    )

    log "[RUN] ${name}"
    log "[CMD] ${cmd[*]}"
    "${cmd[@]}" > "$run_log" 2>&1
    code=$?
    if [[ "$code" -eq 0 ]]; then
      status="ok"
    else
      status="fail"
      log "[FAIL] ${name} (exit=${code}); tail:"
      tail -n 60 "$run_log" | tee -a "$MASTER_LOG" || true
    fi

    has_metrics=0
    has_records=0
    has_space_plot=0
    has_graph_plot=0
    nonfinite_dropped=""
    metrics_file="${out_dir}/semantic_alignment_metrics.json"
    records_file="${out_dir}/patient_level_records.json"
    if [[ -f "$metrics_file" ]]; then
      has_metrics=1
      nonfinite_dropped="$("$PYTHON_BIN" -c "import json;print(json.load(open(r'${metrics_file}','r',encoding='utf-8')).get('dropped_nonfinite_queries',''))")"
    fi
    if [[ -f "$records_file" ]]; then
      has_records=1
    fi
    if [[ -f "${out_dir}/semantic_unit_alignment_space.png" ]]; then
      has_space_plot=1
    fi
    if [[ -f "${out_dir}/semantic_unit_graph_50patients.png" ]]; then
      has_graph_plot=1
    fi

    echo "${variant},${seed},${status},${code},${out_dir},${has_metrics},${has_records},${has_space_plot},${has_graph_plot},${nonfinite_dropped}" >> "$STATUS_CSV"
  done
done

log "Extracting metrics to ${SUMMARY_CSV}"
"$PYTHON_BIN" utils/extract_metrics.py "${OUTPUT_ROOT}" "${SUMMARY_CSV}" --run_id "${RUN_ID}" | tee -a "$MASTER_LOG" || true

log "Generating LaTeX table to ${SUMMARY_TEX}"
"$PYTHON_BIN" utils/make_table.py "${SUMMARY_CSV}" "${SUMMARY_TEX}" | tee -a "$MASTER_LOG" || true

log "Running quick bootstrap (N=${N_BOOTSTRAP})"
"$PYTHON_BIN" utils/bootstrap_semantic_5seed.py \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --n_bootstrap "${N_BOOTSTRAP}" \
  --seed "${BOOTSTRAP_SEED}" \
  --out_csv "${BOOT_CSV}" \
  --out_json "${BOOT_JSON}" | tee -a "$MASTER_LOG" || true

log "===== SMOKE DONE ====="
log "STATUS_CSV=${STATUS_CSV}"
log "SUMMARY_CSV=${SUMMARY_CSV}"
log "SUMMARY_TEX=${SUMMARY_TEX}"
log "BOOT_CSV=${BOOT_CSV}"
log "BOOT_JSON=${BOOT_JSON}"

