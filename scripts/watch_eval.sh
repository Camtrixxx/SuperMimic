#!/usr/bin/env bash
# Evaluate new snapshots of the given run dirs as they appear (one GPU,
# sequential). Each snapshot's result lands in <run>/eval/<snap>.json.
#
# Usage: scripts/watch_eval.sh <gpu> <motion_glob> <run_dir>...
set -uo pipefail
GPU=${1:?gpu}; MOTION=${2:?motion glob}; shift 2
cd "$(dirname "$0")/.."
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
while true; do
  busy=0
  for run in "$@"; do
    mkdir -p "$run/eval"
    for snap in "$run"/snapshots/it*.pt; do
      [ -e "$snap" ] || continue
      name=$(basename "$snap" .pt)
      out="$run/eval/$name.json"
      [ -e "$out" ] && continue
      # Skip a snapshot that may still be being written.
      [ $(( $(date +%s) - $(stat -c %Y "$snap") )) -lt 20 ] && continue
      busy=1
      CUDA_VISIBLE_DEVICES=$GPU "$UV" run sm-eval \
        --run-dir "$run" --motion-file "$MOTION" --snapshot "$name.pt" \
        --output "$out" 2>&1 | grep -E '^it=|Error|Traceback' \
        | sed "s#^#[$(basename "$run")] #"
    done
  done
  [ $busy -eq 0 ] && sleep 60
done
