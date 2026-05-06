#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
RUN_ID="${RUN_ID:-}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2026}"

if [[ -n "$RUN_ID" ]]; then
  OUT_CSV="results/semantic_bootstrap_5seed_${RUN_ID}.csv"
  OUT_JSON="results/semantic_bootstrap_5seed_${RUN_ID}.json"
  RUN_ARGS=(--run_id "$RUN_ID")
else
  OUT_CSV="results/semantic_bootstrap_5seed.csv"
  OUT_JSON="results/semantic_bootstrap_5seed.json"
  RUN_ARGS=()
fi

"$PYTHON_BIN" utils/bootstrap_semantic_5seed.py \
  --output_root "$OUTPUT_ROOT" \
  "${RUN_ARGS[@]}" \
  --n_bootstrap "$N_BOOTSTRAP" \
  --seed "$BOOTSTRAP_SEED" \
  --out_csv "$OUT_CSV" \
  --out_json "$OUT_JSON"

echo "Bootstrap done. CSV: $OUT_CSV"
echo "Bootstrap done. JSON: $OUT_JSON"
