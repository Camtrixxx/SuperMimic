# Archived experiment schedulers (2026-11)

The one-off scripts that scheduled the runs reported in the top-level README.
They are kept for the record and **will not run unchanged**: they hard-code run
directory names from that session and the pre-refactor `repro/` script paths
(now `sm-*` console commands and `scripts/*.sh`).

| Script | What it ran |
|---|---|
| `run_lafan1.sh` | Milestone 1 launcher (LAFAN1, 1 GPU) |
| `phase_gpu4to7.sh` | Full evals of the overnight fix runs, then the critic × H ablation queues |
| `ablation_queue.sh` | Sequential LAFAN1 ablations + per-snapshot eval on one GPU |
| `phase2.sh` | Velocity interface (RL / distill), push-recovery and step tests |
