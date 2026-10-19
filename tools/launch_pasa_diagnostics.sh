#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p output
task_out="output/pasa_diagnostics_$(date +%Y%m%d_%H%M%S)_$$"
task_log="${task_out}.log"
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
nohup python -u -B tools/diagnose_pasa_server.py --out "$task_out" "$@" > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf 'Started PID: %s\nResults: %s\nLog: %s\n' "$task_pid" "$task_out" "$task_log"
printf 'View progress: tail -f %q\n' "$task_log"
