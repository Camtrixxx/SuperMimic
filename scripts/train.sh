#!/usr/bin/env bash
# Single-GPU training with the paper's actor config (proprio obs, 10-frame
# history). Default motion: LAFAN1 (40 clips); override with MOTION=<glob|packed dir>.
#
# Usage: [MOTION=<glob|packed dir>] scripts/train.sh <gpu_id> <run_name> [extra train.py args...]
set -euo pipefail
GPU=${1:?gpu id}; NAME=${2:?run name}; shift 2
cd "$(dirname "$0")/.."
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
export WANDB_MODE=${WANDB_MODE:-offline} MUJOCO_GL=${MUJOCO_GL:-disable} UV_LINK_MODE=copy
mkdir -p logs/stdout
CUDA_VISIBLE_DEVICES=$GPU "$UV" run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1 \
  --motion-file "${MOTION:-${SUPERMIMIC_DATA:-data}/lafan1_g1/npz/*.npz}" \
  --num-envs 4096 \
  --runner.run-name "$NAME" \
  --runner.actor-obs proprio \
  --runner.history-length 10 \
  --runner.save-interval 2000 \
  "$@" 2>&1 | tee "logs/stdout/$NAME.log"
