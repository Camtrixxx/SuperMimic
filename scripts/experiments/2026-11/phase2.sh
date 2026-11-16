#!/usr/bin/env bash
# 2026-11-15 phase 2 (starts as each GPU 4-6 ablation queue finishes):
#  GPU4: velocity encoder, RL (paper's method)        100k iterations
#  GPU5: velocity encoder, distillation baseline       100k iterations
#  GPU6: push-recovery tests, then (after GPU4/5) velocity step tests
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
L=logs/SuperMimic-GenCoord-G1
BASE=$L/2026-11-14_14-07-56_night_clean50   # best fixed BONES tracker so far
LOG=logs/stdout/phase2.log
wait_queue() { until grep -q 'queue finished' "logs/stdout/abl_queue$1.log" 2>/dev/null; do sleep 120; done; }

vel() {  # gpu mode
  wait_queue "$1"
  echo "$(date +%H:%M) start vel_$2 on gpu$1" >> $LOG
  CUDA_VISIBLE_DEVICES=$1 ~/.local/bin/uv run src/supermimic/scripts/train_velocity.py \
    --init-run-dir $BASE --init-snapshot it0144000.pt --mode "$2" \
    --motion-file ../../data/loco/packed --run-name "vel_$2" \
    --max-iterations 100001 --save-interval 10000 > "logs/stdout/vel_$2.log" 2>&1
  echo "$(date +%H:%M) done vel_$2 (exit $?)" >> $LOG
}
vel 4 rl &
vel 5 distill &

(
  wait_queue 6
  echo "$(date +%H:%M) start push tests on gpu6" >> $LOG
  for spec in "$L/2026-11-12_19-16-17_lafan1_s0:it0040000.pt:lafan_H32_critic" \
              "$(ls -d $L/*_abl_H32_nocritic | tail -1):it0040000.pt:lafan_H32_nocritic" \
              "$BASE:it0144000.pt:bones_clean50_144k"; do
    IFS=: read -r run snap tag <<< "$spec"
    CUDA_VISIBLE_DEVICES=6 ~/.local/bin/uv run repro/push_test.py --run-dir "$run" --snapshot "$snap" \
      --output "$run/push_test_$snap.json" 2>&1 | grep -E '^ +[0-9.]+ +[0-9]|speed +fell' \
      | sed "s#^#[push $tag] #" >> $LOG
  done
  until [ "$(grep -c 'done vel_' $LOG)" -ge 2 ]; do sleep 120; done
  for m in rl distill; do
    run=$(ls -d $L/*_vel_$m | tail -1)
    for snap in it0010000.pt it0050000.pt it0100000.pt; do
      CUDA_VISIBLE_DEVICES=6 ~/.local/bin/uv run repro/velocity_step_test.py --run-dir "$run" \
        --snapshot $snap --output "$run/step_test_$snap.json" 2>&1 | grep '^{' \
        | sed "s#^#[step vel_$m $snap] #" >> $LOG
    done
  done
  echo "$(date +%H:%M) phase2 finished" >> $LOG
) &
wait
