#!/usr/bin/env bash
# Data-parallel training (repro): one rank per GPU, gradients averaged, rank 0
# logs and saves. LR scaled by sqrt(world size), as the paper's 2-GPU run.
#
# Usage: [MOTION=...] repro/run_train_ddp.sh <gpu_list e.g. 0,1> <run_name> [extra args]
set -euo pipefail
GPUS=${1:?gpu list}; NAME=${2:?run name}; shift 2
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-offline} MUJOCO_GL=${MUJOCO_GL:-disable} UV_LINK_MODE=copy
N=$(awk -F, '{print NF}' <<<"$GPUS")
LR=$(python3 -c "import math; print(2e-4 * math.sqrt($N))")
mkdir -p logs/stdout
CUDA_VISIBLE_DEVICES=$GPUS ~/.local/bin/uv run torchrun --standalone --nproc-per-node "$N" \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file "${MOTION:-../../data/lafan1_g1/npz/*.npz}" \
  --num-envs 4096 \
  --runner.run-name "$NAME" \
  --runner.actor-obs proprio \
  --runner.history-length 10 \
  --runner.save-interval 2000 \
  --runner.wm.lr "$LR" --runner.policy.lr "$LR" \
  "$@" 2>&1 | tee "logs/stdout/$NAME.log"
