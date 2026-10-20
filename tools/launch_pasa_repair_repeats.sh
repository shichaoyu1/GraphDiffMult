#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p output
task_out=""
resume=false
forward=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) [[ $# -ge 2 ]] || { printf 'Missing --out value\n' >&2; exit 2; }; task_out="$2"; shift 2 ;;
    --out=*) task_out="${1#--out=}"; shift ;;
    --resume) resume=true; forward+=("$1"); shift ;;
    *) forward+=("$1"); shift ;;
  esac
done
if [[ "$resume" == true && -z "$task_out" ]]; then
  printf '%s\n' '--resume requires --out <existing directory>' >&2
  exit 2
fi
task_out="${task_out:-output/pasa_repair_repeats_$(date +%Y%m%d_%H%M%S)_$$}"
task_log="${task_out}.log"
if [[ "$resume" == true ]]; then task_log="${task_out}.resume_$(date +%Y%m%d_%H%M%S)_$$.log"; fi
nohup python -u -B tools/repeat_pasa_repair_pilot.py --out "$task_out" "${forward[@]}" > "$task_log" 2>&1 < /dev/null &
printf 'Started PID: %s\nResults: %s\nLog: %s\n' "$!" "$task_out" "$task_log"
printf 'View progress: tail -f %q\n' "$task_log"
