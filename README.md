# SuperMimic — reproduction

An independent reproduction of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) on 8×A100-40GB, built on the official code ([HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic), Apache-2.0). Upstream history is preserved; our changes are the commits on top, prefixed `repro:`. The original README is in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

Not affiliated with the paper's authors.

## Results

Paper benchmark (Appendix F.1): each clip plays once from its first frame, with friction 1.0, no domain randomization or observation noise, and a deterministic policy. Results are reported as normalized return / failed clips.

**BONES-SEED, 129,543 clips** (paper: 129,514)

| Setting | Ours (A100) | Paper (H100) |
|---|---|---|
| 2 GPUs, 320k iterations | **0.967 / 0** (20.6 h) | **0 failed** (9.8 h) |
| 1 GPU, 320k iterations | 0.961 / 6, 0.964 / 5 (two fix variants, ~20 h) | — |
| 1 GPU, ~3 h (48k iterations) | 0.940 / 114 | >0.90 / 25 |
| 2 GPUs, ~3 h | 0.950 / 68 | — |
| SONIC released checkpoint | — | 0.90 / 179 |

An A100 runs about half as many iterations per hour as the paper's H100s, so equal wall time means fewer updates.

**Other results**

| Experiment | Ours | Paper |
|---|---|---|
| LAFAN1, 40 clips, 1 GPU: time to 0.82 return | 1.00–1.12 h (3 seeds) | 1.2 h |
| Critic × horizon (LAFAN1) | No critic: H ≤ 16 fails. H=8 never reaches 0.82. H=16 with critic peaks at 0.806. | No critic: H ≤ 16 fails. H=16 with critic ≈ critic-free H=32. |
| Push recovery, 0.5–4 m/s | With critic: no falls up to 3 m/s. Without: 3/8 falls at 3 m/s, 8/8 at 4 m/s. | Critic policy settles in 0.5–2.4 s; critic-free never settles. |
| Velocity interface, 672 step responses | RL 0.030, distillation 0.355 (median relative error; only distillation falls) | RL 0.16, distillation 0.49 |

Each ablation, push and velocity setting was run with one seed. Full logs are in [results/](results/) (`final_eval.txt` = full BONES-SEED evals, `phase2.txt` = push/velocity tests, `abl_queue*.txt` = ablations), and the experiment diary is in [docs/repro/PROGRESS.md](docs/repro/PROGRESS.md).

## What we changed

1. **Late deterministic collapse fix (required).**
   - With the official code, every long BONES run collapses under deterministic evaluation after ~70k iterations. The same checkpoints still work when the 0.1 collection noise is added at test time.
   - Cause: the world model only ever sees noisy actions.
   - Fixes, either one works:
     - `--runner.clean-action-frac 0.5`: half the sim envs collect without noise.
     - `--runner.noise-final 0.02`: decay the collection noise.
2. **Cosine LR schedule** from paper Table A.5 (`--runner.lr-schedule cosine`).
   - The official code uses Muon for hidden layers, and torch's Muon freezes its LR under CUDA-graph capture.
   - `ScheduledMuon` fixes this. Tests are in `repro/test_lr_schedule.py`.
3. **Data-parallel multi-GPU training** (`repro/run_train_ddp.sh`): torchrun, gradient all-reduce, LR ×√N, and a weight-checksum check across ranks.
4. **BONES-SEED pipeline**: name filter, batched CSV→npz conversion, and Appendix E feasibility filter. This gives 129,543 clips vs the paper's 129,514.
5. **Evaluation**: Appendix F.1 benchmark without W&B, sharded across GPUs, with Table A.8-style group breakdown.
6. **Paper experiments missing from the release**:
   - Critic-free ablation (`--runner.use-critic False`).
   - Push-recovery test.
   - RL velocity encoder and step test (Appendix G). The release has only the distillation variant.
7. **Local checkpoints**: `run_config.json` and `snapshots/` are written into the log dir, so W&B is optional. Log dirs are unique per run.

## Setup

Linux x86_64, NVIDIA driver ≥ R525, [uv](https://docs.astral.sh/uv/). Workspace layout expected by the scripts:

```
workspace/
├── data/                  # motion data (not included, see below)
└── ws/
    ├── supermimic/        # this repo
    └── mjlab_research/    # git clone https://github.com/HybridRobotics/mjlab_research
```

```bash
cd ws/supermimic
uv sync
export MUJOCO_GL=disable WANDB_MODE=offline   # headless; W&B optional
uv run pytest -q tests/ repro/test_lr_schedule.py
```

`pyproject.toml` pins NVIDIA's CUDA-12 build of warp-lang. PyPI builds ≥ 1.18 need driver R580+.

## Data

Datasets are **not** redistributed here; get them under their own licenses.

**LAFAN1** (G1 retarget, [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), CC BY-NC-ND 4.0). Place the 40 CSVs in `data/lafan1_g1/csv/`, then:

```bash
uv run repro/csv_to_npz.py --input-glob '../../data/lafan1_g1/csv/*.csv' \
  --output-dir ../../data/lafan1_g1/npz --input-fps 30
```

**BONES-SEED** ([bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed), gated, custom license). Download `g1.tar.gz` and `metadata/` into `data/bones_seed/` and extract the archive there, then:

```bash
D=../../data/bones_seed
uv run repro/bones_name_filter.py --metadata $D/metadata/seed_metadata_v004.parquet --out-keep $D/keep_names.txt
uv run repro/csv_to_npz.py --input-glob "$D/g1/csv/*/*.csv" --output-dir $D/npz_name_filtered \
  --format bones --input-fps 120 --keep-list $D/keep_names.txt
uv run repro/feasibility.py --npz-glob "$D/npz_name_filtered/*.npz" --out $D/feasibility.parquet
uv run repro/bones_finalize.py --data $D
uv run python -m mjlab.scripts.pack_motions --input "$D/npz_final/*.npz" --output $D/packed_final
```

## Training

```bash
COS="--runner.lr-schedule cosine --runner.lr-decay-iters 250000 --runner.lr-final-ratio 0.03"
# LAFAN1, 1 GPU (paper actor config: proprio obs, 10-frame history)
repro/run_train.sh 0 lafan1_s0 --env.seed 0 --runner.max-iterations 40001
# BONES-SEED, 1 GPU
MOTION=../../data/bones_seed/packed_final repro/run_train.sh 0 bones_1gpu \
  --runner.max-iterations 320001 --runner.save-interval 8000 $COS --runner.clean-action-frac 0.5
# BONES-SEED, 2 GPUs
MOTION=../../data/bones_seed/packed_final repro/run_train_ddp.sh 0,1 bones_2gpu \
  --runner.max-iterations 320001 --runner.save-interval 8000 $COS --runner.clean-action-frac 0.5
```

Peak memory is 20.9 GB on LAFAN1 and 33.7 GB on BONES-SEED; speed is about 0.22 s per iteration on an A100.

## Evaluation

```bash
RUN=logs/SuperMimic-GenCoord-G1/<run dir>
# LAFAN1, all snapshots
uv run repro/evaluate.py --run-dir $RUN --motion-file '../../data/lafan1_g1/npz/*.npz' --all-snapshots
# BONES-SEED, sharded over GPUs, then merged (~10 min on 8 A100s)
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k uv run repro/evaluate.py --run-dir $RUN \
  --motion-file '../../data/bones_seed/npz_final/*.npz' --snapshot it0320000.pt --sort-by-length \
  --shard $k --num-shards 4 --output $RUN/eval_full/shard$k.json & done; wait
uv run repro/merge_eval.py --pattern "$RUN/eval_full/shard*.json"
# Push recovery / velocity interface
uv run repro/push_test.py --run-dir $RUN --snapshot it0040000.pt
uv run src/supermimic/scripts/train_velocity.py --init-run-dir $RUN --init-snapshot it0144000.pt \
  --mode rl --motion-file <walk/run packed library> --run-name vel_rl
uv run repro/velocity_step_test.py --run-dir <velocity run> --snapshot it0100000.pt
```

`repro/phase*.sh` and `repro/watch_eval*.sh` are the schedulers used for our runs. Their run-directory names are hard-coded.

## License and attribution

Code: Apache-2.0, same as upstream (see [LICENSE](LICENSE)). Copyright of upstream code is held by The SuperMimic Authors.

Models trained with this repository on BONES-SEED: *Training data includes Motion Data by Bones Studio* ([bones.studio](https://bones.studio/)). *Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.*
