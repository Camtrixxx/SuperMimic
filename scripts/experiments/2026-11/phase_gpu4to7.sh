#!/usr/bin/env bash
# 2026-11-15 plan, GPUs 4-7:
#  phase 0: full 129,543-clip F.1 eval of the four overnight 144k checkpoints
#  phase 1: GPUs 4-6 LAFAN1 critic x horizon ablation queues; GPU 7 subset evaluator
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
L=logs/SuperMimic-GenCoord-G1
NIGHT=$L/2026-11-14_14-07-56
for r in night_official_const night_clean20 night_clean50 night_noisedecay; do
  run=${NIGHT}_$r; mkdir -p $run/eval_full; pids=()
  for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$((s+4)) ~/.local/bin/uv run repro/evaluate.py --run-dir $run \
      --motion-file '../../data/bones_seed/npz_final/*.npz' --snapshot it0144000.pt \
      --sort-by-length --shard $s --num-shards 4 --output $run/eval_full/it0144000_shard$s.json \
      > $run/eval_full/it0144000_shard$s.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  ~/.local/bin/uv run repro/merge_eval.py --pattern "$run/eval_full/it0144000_shard*.json" \
    --out $run/eval_full/merged_it0144000.json 2>/dev/null | sed "s#^#[$r] #" >> logs/stdout/phase0_full_eval.log
done
echo "[phase0] done $(date +%H:%M)" >> logs/stdout/phase0_full_eval.log

setsid nohup repro/ablation_queue.sh 4 "abl_H64_critic --runner.policy.horizon 64" \
  "abl_H8_critic --runner.policy.horizon 8" > logs/stdout/abl_queue4.log 2>&1 < /dev/null &
setsid nohup repro/ablation_queue.sh 5 "abl_H64_nocritic --runner.policy.horizon 64 --runner.use-critic False" \
  "abl_H8_nocritic --runner.policy.horizon 8 --runner.use-critic False" > logs/stdout/abl_queue5.log 2>&1 < /dev/null &
setsid nohup repro/ablation_queue.sh 6 "abl_H16_critic --runner.policy.horizon 16" \
  "abl_H16_nocritic --runner.policy.horizon 16 --runner.use-critic False" \
  "abl_H32_nocritic --runner.use-critic False" > logs/stdout/abl_queue6.log 2>&1 < /dev/null &
MODES=clean setsid nohup repro/watch_eval_subset.sh 7 $(ls -d $L/2026-11-15_10-20-39_fix_*) > /dev/null 2>&1 < /dev/null &
echo "[phase1] launched $(date +%H:%M)" >> logs/stdout/phase0_full_eval.log
