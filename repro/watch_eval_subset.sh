#!/usr/bin/env bash
# Every new snapshot: F.1 eval on a 1/40 BONES subset, deterministic ("clean")
# and with N(0,0.1^2) action noise ("noisy"). Results: <run>/eval_sub/*.json,
# one-liners in logs/stdout/eval_subset_summary.log.
# Usage: repro/watch_eval_subset.sh <gpu> <run_dir>
set -uo pipefail
GPU=${1:?gpu}; RUN=${2:?run}
cd "$(dirname "$0")/.."
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
mkdir -p "$RUN/eval_sub"
while true; do
  busy=0
  for snap in "$RUN"/snapshots/it*.pt; do
    [ -e "$snap" ] || continue
    name=$(basename "$snap" .pt)
    [ $(( $(date +%s) - $(stat -c %Y "$snap") )) -lt 30 ] && continue
    for mode in clean noisy; do
      out="$RUN/eval_sub/${name}_$mode.json"; [ -e "$out" ] && continue
      busy=1; extra=(); [ $mode = noisy ] && extra=(--action-noise 0.1)
      line=$(CUDA_VISIBLE_DEVICES=$GPU ~/.local/bin/uv run repro/evaluate.py --run-dir "$RUN" \
        --motion-file '../../data/bones_seed/npz_final/*.npz' --snapshot "$name.pt" \
        --subset-every 40 --sort-by-length "${extra[@]}" --output "$out" 2>&1 | grep -E '^it=|Traceback' | head -1)
      echo "[$(basename "$RUN")] $mode $line" >> logs/stdout/eval_subset_summary.log
    done
  done
  [ $busy -eq 0 ] && sleep 120
done
