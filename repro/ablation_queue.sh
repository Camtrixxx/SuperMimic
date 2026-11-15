#!/usr/bin/env bash
# Run LAFAN1 ablations sequentially on one GPU, each followed by an F.1 eval of
# all its snapshots (one process, env built once) -> <run>/eval_all.json.
# Usage: repro/ablation_queue.sh <gpu> "<name> <extra train args>" ...
set -uo pipefail
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.."
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
for spec in "$@"; do
  name=${spec%% *}; args=${spec#* }; [ "$args" = "$spec" ] && args=""
  echo "[queue gpu$GPU] $(date +%H:%M) start $name ($args)"
  repro/run_train.sh "$GPU" "$name" --env.seed 0 --runner.max-iterations 40001 $args > /dev/null 2>&1
  run=$(ls -d logs/SuperMimic-GenCoord-G1/*_"$name" | tail -1)
  CUDA_VISIBLE_DEVICES=$GPU ~/.local/bin/uv run repro/evaluate.py --run-dir "$run" \
    --motion-file '../../data/lafan1_g1/npz/*.npz' --all-snapshots \
    --output "$run/eval_all.json" 2>&1 | grep -E '^it=|Traceback' > "$run/eval_all.log"
  echo "[queue gpu$GPU] $(date +%H:%M) done $name: $(tail -1 "$run/eval_all.log")"
done
echo "[queue gpu$GPU] $(date +%H:%M) queue finished"
