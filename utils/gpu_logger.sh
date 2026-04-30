#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-logs/gpu.log}"
INTERVAL="${2:-10}"

mkdir -p "$(dirname "$OUT")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "timestamp,message" > "$OUT"
  echo "$(date '+%F %T'),nvidia-smi not found" >> "$OUT"
  exit 0
fi

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv \
  -l "$INTERVAL" >> "$OUT"
