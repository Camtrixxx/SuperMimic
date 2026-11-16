#!/usr/bin/env bash
# Full-library (paper F.1) evaluation of snapshots as they appear.
# Every snapshot whose iteration is a multiple of EVERY is sharded over GPUS,
# merged into <run>/eval_full/merged_itNNNNNNN.json, and a one-line summary is
# appended to logs/stdout/eval_full_summary.log.
#
# Usage: EVERY=16000 GPUS=3,4,5,6,7 scripts/watch_eval_full.sh <motion_glob> <run_dir>...
set -uo pipefail
MOTION=${1:?motion glob}; shift
EVERY=${EVERY:-16000}
IFS=, read -ra GPU_LIST <<< "${GPUS:-3,4,5,6,7}"
N=${#GPU_LIST[@]}
cd "$(dirname "$0")/.."
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
SUMMARY=logs/stdout/eval_full_summary.log
while true; do
  busy=0
  for run in "$@"; do
    mkdir -p "$run/eval_full"
    for snap in "$run"/snapshots/it*.pt; do
      [ -e "$snap" ] || continue
      name=$(basename "$snap" .pt); it=$((10#${name#it}))
      (( it % EVERY == 0 )) || continue
      merged="$run/eval_full/merged_$name.json"
      [ -e "$merged" ] && continue
      [ $(( $(date +%s) - $(stat -c %Y "$snap") )) -lt 30 ] && continue
      busy=1
      pids=()
      for s in $(seq 0 $((N-1))); do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$s]} "$UV" run sm-eval \
          --run-dir "$run" --motion-file "$MOTION" --snapshot "$name.pt" \
          --sort-by-length --shard "$s" --num-shards "$N" \
          --output "$run/eval_full/${name}_shard$s.json" \
          > "$run/eval_full/${name}_shard$s.log" 2>&1 &
        pids+=($!)
      done
      ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
      if [ $ok -eq 1 ]; then
        "$UV" run sm-merge-eval --pattern "$run/eval_full/${name}_shard*.json" \
          --out "$merged" 2>/dev/null | head -1 | sed "s#^#[$(basename "$run")] #" >> "$SUMMARY"
      else
        echo "[$(basename "$run")] $name FAILED (see shard logs)" >> "$SUMMARY"
        touch "$merged.failed"; mv "$merged.failed" "$merged"
      fi
    done
  done
  [ $busy -eq 0 ] && sleep 120
done
