#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p output
task_out="output/pasa_shape_probe_$(date +%Y%m%d_%H%M%S)_$$"
forward=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) [[ $# -ge 2 ]] || { printf 'Missing --out value\n' >&2; exit 2; }; task_out="$2"; shift 2 ;;
    --out=*) task_out="${1#--out=}"; shift ;;
    *) forward+=("$1"); shift ;;
  esac
done
task_log="${task_out}.log"
if [[ -e "$task_out" || -e "$task_log" ]]; then
  printf 'Output or log already exists: %s\n' "$task_out" >&2
  exit 2
fi
mkdir -p "$(dirname "$task_log")"
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -B tools/run_pasa_shape_probe.py \
  --metadata /root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
  --out "$task_out" "${forward[@]}" > "$task_log" 2>&1 < /dev/null &
printf 'Started CPU probe PID: %s\nResults: %s\nLog: %s\n' "$!" "$task_out" "$task_log"
printf 'View progress: tail -f %q\n' "$task_log"
