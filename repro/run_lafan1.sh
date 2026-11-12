#!/usr/bin/env bash
# Milestone 1: LAFAN1 40 clips, 1 GPU, paper config (proprio obs, 10-frame
# history). Target from paper Fig. 4(a): eval return 0.82 within ~1.2 h (H100).
#
# Usage: repro/run_lafan1.sh <gpu_id> <run_name> [extra train.py args...]
set -euo pipefail
GPU=${1:?gpu id}; NAME=${2:?run name}; shift 2
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-offline} MUJOCO_GL=${MUJOCO_GL:-disable} UV_LINK_MODE=copy
mkdir -p logs/stdout
CUDA_VISIBLE_DEVICES=$GPU ~/.local/bin/uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1 \
  --motion-file '../../data/lafan1_g1/npz/*.npz' \
  --num-envs 4096 \
  --runner.run-name "$NAME" \
  --runner.actor-obs proprio \
  --runner.history-length 10 \
  --runner.save-interval 2000 \
  "$@" 2>&1 | tee "logs/stdout/$NAME.log"
