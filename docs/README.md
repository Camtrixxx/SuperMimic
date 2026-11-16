# 文档目录 / Documentation

| 文档 | Document | 内容 / Contents |
|---|---|---|
| [../README.zh-CN.md](../README.zh-CN.md) | [../README.md](../README.md) | 项目说明：结果、改动、配置、数据、训练与评估命令 / Overview: results, changes, setup, data, training and evaluation commands |
| [repro/RESEARCH.md](repro/RESEARCH.md) | — | 调研报告：论文要点、官方代码现状、第三方复现、数据、风险 / Research notes: paper summary, official code status, third-party reproductions, data, risks |
| [repro/PROGRESS.md](repro/PROGRESS.md) | — | 实验日志：每个阶段的设置、问题、诊断和结果 / Experiment diary: setup, issues, diagnoses and results of every stage |
| [UPSTREAM_README.md](UPSTREAM_README.md) | [UPSTREAM_README.md](UPSTREAM_README.md) | 官方原始 README / Original upstream README |
| [../results/](../results/) | [../results/](../results/) | 评估日志汇总 / Evaluation log summaries |

## 结果文件 / Result files

| 文件 / File | 内容 / Contents |
|---|---|
| `final_eval.txt` | BONES-SEED 全量评估（3 小时与 32 万步检查点，含分组失败数）/ Full BONES-SEED evals (3 h and 320k checkpoints, per-group failures) |
| `eval_full_summary.txt` | 未修复主实验的全量评估曲线（展示后期崩溃）/ Full-eval curves of the unfixed main runs (late collapse) |
| `phase0_full_eval.txt` | 三种修复方案 144k 检查点的全量评估 / Full evals of the three fixes at 144k |
| `eval_subset_summary.txt` | 1/40 子集评估（确定性与加噪声）/ 1/40 subset evals (deterministic and noisy) |
| `lafan1_summary.txt` | LAFAN1 三个种子的回报曲线 / LAFAN1 return curves, 3 seeds |
| `abl_queue4/5/6.txt` | critic × H 消融 / Critic × horizon ablations |
| `phase2.txt` | 推力恢复与速度阶跃测试 / Push recovery and velocity step tests |

## 代码索引 / Code map

| 路径 / Path | 作用 / Purpose |
|---|---|
| `src/supermimic/runners/supermimic_runner.py` | 训练主循环（含修复、余弦学习率、多卡）/ Training loop (fixes, cosine LR, DDP) |
| `src/supermimic/learners/learner_utils.py` | 优化器、`ScheduledMuon`、梯度 all-reduce / Optimizers, `ScheduledMuon`, gradient all-reduce |
| `src/supermimic/velocity.py`, `runners/velocity_runner.py`, `scripts/train_velocity.py` | 速度指令接口（RL 与蒸馏）/ Velocity-command interface (RL and distillation) |
| `repro/csv_to_npz.py`, `bones_name_filter.py`, `feasibility.py`, `bones_finalize.py` | 数据流程 / Data pipeline |
| `repro/evaluate.py`, `merge_eval.py` | 附录 F.1 评估 / Appendix F.1 benchmark |
| `repro/push_test.py`, `velocity_step_test.py` | 推力恢复、速度阶跃测试 / Push recovery, velocity step test |
| `repro/run_train.sh`, `run_train_ddp.sh` | 单卡 / 多卡训练启动脚本 / Single- and multi-GPU launchers |
