#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$stage" != smoke && "$stage" != pilot ]]; then
  printf 'Usage: bash tools/launch_pasa_repair.sh [smoke|pilot] [arguments]\n' >&2
  exit 2
fi
mkdir -p output
task_out="output/pasa_repair_${stage}_$(date +%Y%m%d_%H%M%S)_$$"
task_log="${task_out}.log"
nohup python -u -B tools/run_pasa_repair_pilot.py --stage "$stage" --out "$task_out" "$@" > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf 'Started PID: %s\nResults: %s\nLog: %s\n' "$task_pid" "$task_out" "$task_log"
printf 'View progress: tail -f %q\n' "$task_log"
