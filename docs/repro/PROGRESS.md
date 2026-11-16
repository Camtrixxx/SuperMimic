# SuperMimic Reproduction Progress Log

## Directory Layout

```
SuperMimic/
├── RESEARCH.md              Research report
├── PROGRESS.md              This file
├── refs/                    Paper PDF/text and official code (read-only reference)
├── wheels/                  warp-lang 1.19.0+cu12 (needed for driver R550)
├── data/lafan1_g1/{csv,npz} LAFAN1 G1 retarget: 40 clips, 441,140 frames @ 50Hz = 2.45 h
└── ws/
    ├── mjlab_research/      Official dependency (HybridRobotics, dd51a9f)
    └── supermimic/          Official code; our changes are on the `repro` branch
        ├── repro/csv_to_npz.py       Batched CSV→npz converter (verified against the official script)
        ├── repro/evaluate.py         Paper Appendix F.1 evaluation (no W&B needed)
        ├── repro/run_lafan1.sh       Milestone 1 launch script (paper's actor configuration)
        └── logs/                     Training output (gitignored)
```

## Environment (2026-11-12)

### Software Versions
- Python 3.13
- torch 2.14.1+cu126
- mujoco / mujoco-warp 3.16.0
- warp-lang 1.19.0+cu12
- rsl-rl-lib 5.5.2

### Verification Results
- Official test suite: **217 passed**
- `repro/test_csv_to_npz.py`: all 40 clips match the official converter
  - Positions/quaternions: atol 1e-6
  - Finite-difference velocities: atol ≤ 5e-5

### Required Environment Variables
```bash
export MUJOCO_GL=disable WANDB_MODE=offline UV_LINK_MODE=copy
```

## Milestone 1: LAFAN1, Single GPU, 0.82 Return

**Paper reference:** Fig. 4(a): eval return 0.82 in 1.2 h on one H100

### Configuration
- Task: `SuperMimic-GenCoord-G1`
- Environments: 4096
- Actor observation: proprio with 10-frame history (paper Appendix A; differs from code default)
- Everything else uses code defaults:
  - World model: 10×2048×8
  - Policy: 4096 × H=32
  - Learning rate: 2e-4 constant
  - Return-based reset

### Known Differences from Paper Protocol
- **No cosine learning rate decay**
  - The official code doesn't implement it
  - The paper's decay spans 250k iterations, so it barely matters within this milestone's 40k iterations
- **Wall time includes torch.compile time**
  - The first iteration includes about 10 minutes of compilation
  - The paper doesn't say whether its timing includes this
  - Reported times include it as a conservative measure

### Smoke Test (2k iterations, GPU 0)

| Metric | Value |
|---|---|
| Peak GPU memory | 20.7 GB / 40 GB |
| Speed | ~0.22 s/iteration (~16k iterations/hour), GPU utilization 99% |
| Compile time | ~10 min |
| F.1 return @1k | 0.26 (26/40 failed) |
| F.1 return @2k | 0.45 (17/40 failed), 0.28 h |

### Formal Runs
- Seeds 0, 1, 2 on GPUs 0, 1, 2 respectively
- Seeds 0 and 1 initially launched in the same second and shared a log dir, overwriting each other's checkpoints. Their logs are marked INVALID and kept; both were relaunched at 18:46 after the fix (commit 5878dcb). Seed 2 was unaffected and started at 18:13.
- `repro/watch_eval.sh` evaluates each snapshot on GPU 3 as it is written (every 2k iterations).
- At 18:56 seeds 0 and 1 were killed by accident: they had been launched from inside a background tool task, and stopping that task (a hung wait loop) killed its child processes. Logs kept as INVALID_killed_*. Relaunched at 19:16 with `setsid` (log dirs 2026-11-12_19-16-17_lafan1_s{0,1}). All long jobs are now started with setsid.
- One eval watcher per seed: GPU 3 → s0, GPU 4 → s1, GPU 5 → s2. A single watcher on GPU 3 could not keep up (5 min per eval vs 3 snapshots per 7 min).
- 40k iterations each (~2.5 h)

### Results (Milestone 1 ✅ reached)

Protocol: paper Appendix F.1 (first frame, friction 1.0, no DR/noise, deterministic actor); one checkpoint every 2k iterations (~7.4 min); wall time includes torch.compile.

| Seed | First ≥ 0.82 | First 0 failures | Best return | Final @40k (2.46 h) |
|---|---|---|---|---|
| 0 | 0.838 @ 1.12 h | 0.854 @ 1.49 h | 0.867 @ 2.47 h | 0.867, 0/40 failed |
| 1 | 0.828 @ 1.00 h | 0.847 @ 1.25 h | 0.862 @ 2.23 h | 0.862, 0/40 failed |
| 2 | 0.836 @ 1.00 h | 0.849 @ 1.12 h | 0.864 @ 2.34 h | 0.860, 1/40 failed |

- Paper Fig. 4(a): 0.82 within 1.2 h on one H100. Here: 1.00–1.12 h on one A100-40GB. The crossing time is only resolved to one checkpoint interval (~7 min).
- Peak GPU memory 20.9 GB; 0.221 s/iteration (~16.3k iterations/h); running 3 seeds on 3 GPUs at once does not slow any of them.
- Return plateaus around 0.86 after ~1.7 h.
- Hardest clips (number of checkpoints at ≥10k iterations where the clip failed, summed over 3 seeds): jumps1_subject2 12, fallAndGetUp1_subject5 7, fallAndGetUp2_subject3 5; the other fallAndGetUp clips 3–4 each. Every late failure is a jump or a get-up-from-the-ground motion.
- Full per-checkpoint curves: `lafan1_summary.txt`. Per-clip results: `<run>/eval/it*.json`.

## Milestone 2: BONES-SEED data pipeline (2026-11-13)

License: user is a current student → BONES-SEED "Academic User". Raw data stays on this machine (no redistribution); attribute "Motion Data by Bones Studio" if any model is released.

| Stage | Clips | Paper (Table A.7) |
|---|---|---|
| Raw G1 retarget (`g1.tar.gz`, sha256 verified) | 142,220 | 142,220 |
| Name filter (`repro/bones_name_filter.py`) | 129,679 | 129,667 |
| Feasibility filter (`repro/feasibility.py`) | **129,543** (263.2 h) | 129,514 (264.56 h) |

- **Name filter.** SONIC's default keyword list from `gear_sonic/data_process/filter_and_copy_bones_data.py`, plus `jump_off`, `jump_on_50cm` and `look_over_fence`, which the paper's own examples name. This is 12 clips (0.01%) off the paper. Its exact extra terms are not published.
- **CSV format.**
  - Units are cm and degrees. Root rotation is converted with scipy `from_euler("xyz")`, as in SONIC's converter.
  - I checked the convention on six clips using foot-floor distance. `xyz` puts the standing feet at about −5 mm. `XYZ` and `zyx` put them as far off as −30 cm or +17 cm.
  - The CSVs are 120 fps, resampled to 50 Hz.
- **Feasibility filter.** Implements the paper's three rules with MuJoCo FK and `mj_geomDistance`.
  - Stance calibration: −4.4 ± 4.5 mm, against the paper's −4 ± 4 mm.
  - Removed 136 clips: 50 elevated, 44 penetration, 76 floating (a clip can trigger more than one rule).
  - The paper doesn't quantify the floating rule's "upright / standing height / stationary" gates. I chose pelvis up-z > 0.9, height > 0.65 m and speed < 0.25 m/s.
  - Side effect: 10 of the 22 `flip_180_R` clips are removed, because their retarget ends standing with both feet ~12 cm off the floor. They are genuinely infeasible endings, but the paper still counts 44 flips in its benchmark, so its gate must spare some of these.
- **Output.**
  - Final clip set: `data/bones_seed/npz_final/` (symlinks) and `keep_final.txt`.
  - Removed clips with reasons: `removed_by_feasibility.csv`.
  - Packed motion library: `data/bones_seed/packed_final` (47.37M frames, 13 GB).
- **Full-library smoke run.** 1 A100, packed library, paper actor config.
  - Peak memory: 33.7 GB of 40 GB (LAFAN1 was 20.9 GB). The extra ~13 GB is the motion library held on the GPU.
  - Speed: 0.223 s/iteration, unchanged from LAFAN1.
  - Single-GPU full training is feasible as-is. Headroom is ~6 GB.

## Training-code gaps closed (2026-11-13)

- **Cosine LR.**
  - Paper Table A.5: decay to 0.03× over 250k iterations, then constant. Enabled with `--runner.lr-schedule cosine`; the default stays the official code's `constant`.
  - Found while implementing: the official code doesn't use plain Adam everywhere, as the paper says. Hidden-layer weights use **Muon** (`torch.optim.Muon`, `adjust_lr_fn=match_rms_adamw`, default weight decay 0.1).
  - torch's Muon reads its LR as a Python scalar. Under the code's CUDA-graph capture, that froze the Muon LR, so a schedule would have been a silent no-op for every hidden layer.
  - Fixed with `ScheduledMuon`: same math, but the LR is a device tensor. Tests (`repro/test_lr_schedule.py`): matches torch Muon to within fp32 roundoff, and an LR change reaches the captured graph. An end-to-end run showed `Perf/lr_scale` 1 → 0.03.
- **Data parallel**:
  - Launcher: `repro/run_train_ddp.sh <gpus> <name>` (torchrun).
  - Each rank runs its own 4096 envs with seed + rank.
  - Weights are broadcast from rank 0 at start. Gradients are averaged with one flat all-reduce before clipping, which is capture-safe.
  - Normalizers sync through rsl_rl's built-in `torch.distributed` path.
  - LR is scaled by √N (paper: √2 for 2 GPUs). Only rank 0 logs and saves.
  - A weight-checksum check across ranks prints `[DDP] ... OK/DIVERGED` every 1000 iterations.
- **Large-library eval**: `repro/evaluate.py --sort-by-length --shard i --num-shards n`, then `repro/merge_eval.py`, which reports Table A.8-style groups.

## Main run: BONES-SEED, 1 GPU (started 2026-11-13 08:55)

- `bones_1gpu_s0`, GPU 0. Paper settings: cosine LR, 320k iterations, snapshot every 4k.
- Speed 0.225 s/iteration, so ~20 h in total. The paper's "3 h on 1 H100" point is roughly 50k iterations.

## Main run: BONES-SEED, 2 GPUs (started 2026-11-13 09:22)

- `bones_2gpu_s0`, GPUs 1 and 2. DDP, LR ×√2, cosine, 320k iterations. 0.229 s/iteration, so ~20 h.
- Paper target: 0 failures after 9.8 h on 2×H100, which is the full 320k iterations at H100 speed.

## Monitoring

- `repro/watch_eval_full.sh` runs the full 129,543-clip F.1 benchmark on GPUs 3–7. It scores every 16k-iteration snapshot of both runs, taking about 12 min per checkpoint on 5 GPUs.
- One-line results go to `ws/supermimic/logs/stdout/eval_full_summary.log`. Per-group details are in `<run>/eval_full/merged_it*.json`.
- Sanity check: on the untrained smoke checkpoint, the motion groups the paper names come out the same size as in its Table A.8: side rolls 22, flips 44, fire-in-the-hole 20, broad jumps 24. My name patterns for the crossed-legs, lying and dance-routine groups don't match the paper's yet.

## Main-run results and late collapse (2026-11-14)

F.1 protocol, all 129,543 clips (`ws/supermimic/logs/stdout/eval_full_summary.log`):

| Run | Best checkpoint | Return | Failed | After that |
|---|---|---|---|---|
| 1 GPU (`bones_1gpu_s0`) | 64k iterations (4.05 h) | 0.933 | 71 | collapses: 80k 736 failed, 96k ~25k failed, 320k 0.49 / 96,906 failed |
| 2 GPU (`bones_2gpu_s0`) | 48k iterations (3.1 h) | 0.936 | 80 | 64k 218 failed, 80k ~52k failed, 320k 0.66 / 60,762 failed |
| Paper, 1×H100 | 3 h | >0.90 | 25 | 2×H100, 9.8 h: 0 failed |

The training logs show nothing at the collapse: imagined reward keeps rising, WM loss keeps falling, episode length stays at 500, and the DDP checksum is OK.

**Diagnosis.** Evaluated the 48k and 320k checkpoints of the 1-GPU run on a 1/40 subset (3,239 clips):

| Eval setting | 48k | 320k |
|---|---|---|
| Deterministic (F.1) | 0.940, 0 failed | 0.494, 2,443 failed |
| + action noise N(0, 0.1²), as in collection | 0.920, 1 failed | **0.947, 0 failed** |
| + observation noise only | 0.937, 1 failed | 0.508, 2,350 failed |
| both | 0.918, 0 failed | 0.945, 0 failed |

The late policy is good, but only when its actions are perturbed. Every simulator action and every world-model training sample includes the 0.1 collection noise, so the deterministic action path drifts off what the model was fit on. RaraAlu's reproduction reports the same late regression. This matters for deployment, because the real robot runs the deterministic policy.

## Overnight fix experiments (started 2026-11-14 14:08, 144k iterations each, ~9 h)

| GPU | Run | Change vs main 1-GPU run |
|---|---|---|
| 0 | `night_clean20` | 20% of envs collect without noise (`--runner.clean-action-frac 0.2`) |
| 1 | `night_clean50` | 50% of envs collect without noise |
| 2 | `night_noisedecay` | collection noise 0.1 → 0.02, linear over 100k iterations |
| 3 | `night_official_const` | control: official settings (constant LR, no fix). Tests whether the collapse comes from my cosine/Muon changes |

GPUs 4–7 run `repro/watch_eval_subset.sh`: every 8k-iteration snapshot is scored on the 1/40 subset, both deterministic and with action noise. One-liners go to `logs/stdout/eval_subset_summary.log`.

## Final results vs paper (2026-11-16)

Full F.1 benchmark, 129,543 clips (paper: 129,514), deterministic policy. A100 is roughly 2× slower than H100 per iteration here, so equal wall time ≈ half the iterations.

| Setting | Ours | Paper |
|---|---|---|
| 1 GPU, ~3 h (48k it) | 0.940–0.941 return, 114–121 failed | >0.90, 25 failed (1×H100, 3 h) |
| 2 GPU, ~3 h (48k it) | 0.950, 68 failed | — |
| 1 GPU, 320k it (~20 h) | 0.961 / 6 failed (clean50), 0.964 / 5 failed (noise decay) | — |
| 2 GPU, 320k it (~20.6 h) | **0.967, 0 failed** | **0 failed (2×H100, 9.8 h)** |
| SONIC released ckpt | — | 0.90, 179 failed |

Required fix (not in the official code): without it, every long run collapses under deterministic eval after ~70k iterations. 50% noise-free collection or collection-noise decay both fix it; their results are within noise of each other.

Other reproduced findings:
- LAFAN1: 0.82 return in 1.0–1.12 h on 1 A100 (paper 1.2 h, 1 H100), 3 seeds.
- Critic × H ablation: without a critic H ≤ 16 fails; H=8 never reaches 0.82. Differences from paper: H=16 with critic peaks at 0.806; H=64 reaches 0.82 fastest. One seed each.
- Push recovery: the critic clearly helps (no critic: 8/8 falls at 4 m/s, 3/8 at 3 m/s). The fixed BONES policy settles in 1.2–2.8 s (paper 0.5–2.4 s).
- Velocity interface: RL encoder median relative error 0.030, no falls; distillation 0.355 with falls (paper 0.16 vs 0.49).

Open: second seeds for the BONES runs; checking our push/step-test protocols against the paper; sim-to-real validation (latency, action limits, ONNX export parity).
