[English](README.md) | 中文

# SuperMimic 复现

本仓库在 8×A100-40GB 上独立复现了 [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)（Liao 等，CoRL 2026）。

- 代码基于官方实现 [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)（Apache-2.0）。
- 第一个提交导入了官方代码（上游提交 `abc124c`），之后的提交都是我们的修改。
- 官方原始 README 见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)，全部文档见 [docs/](docs/README.md)。

本项目与论文作者无关。

## 结果

评估采用论文附录 F.1 的协议：
- 每个片段从第一帧开始播放一次；
- 摩擦系数 1.0，不加域随机化，也不加观测噪声；
- 策略为确定性策略。

表中格式为"归一化回报 / 失败片段数"。

**BONES-SEED，129,543 个片段**（论文为 129,514 个）

| 设置 | 我们（A100） | 论文（H100） |
|---|---|---|
| 双卡，32 万步 | **0.967 / 0**（20.6 小时） | **0 失败**（9.8 小时） |
| 单卡，32 万步 | 0.961 / 6、0.964 / 5（两种修复方案，约 20 小时） | — |
| 单卡，约 3 小时（4.8 万步） | 0.940 / 114 | > 0.90 / 25 |
| 双卡，约 3 小时 | 0.950 / 68 | — |
| SONIC 公开检查点 | — | 0.90 / 179 |

同样的训练时长下，A100 跑的步数只有论文所用 H100 的一半左右。

**其他结果**

| 实验 | 我们 | 论文 |
|---|---|---|
| LAFAN1，40 个片段，单卡：回报达到 0.82 的用时 | 1.00–1.12 小时（3 个种子） | 1.2 小时 |
| critic × 想象步数 H（LAFAN1） | 无 critic 时 H ≤ 16 训练失败；H=8 始终达不到 0.82；有 critic 的 H=16 最高只到 0.806 | 无 critic 时 H ≤ 16 训练失败；有 critic 的 H=16 ≈ 无 critic 的 H=32 |
| 推力恢复，0.5–4 m/s | 有 critic：3 m/s 以内不摔倒；无 critic：3 m/s 摔 3/8 次，4 m/s 摔 8/8 次 | 有 critic 的策略 0.5–2.4 秒稳住；无 critic 的策略无法稳住 |
| 速度接口，672 个阶跃响应 | 稳态相对误差中位数：RL 0.030，蒸馏 0.355；只有蒸馏会摔倒 | RL 0.16，蒸馏 0.49 |

消融、推力恢复和速度接口每组只跑了一个种子。

- 完整日志在 [results/](results/)：
  - `final_eval.txt`：BONES-SEED 全量评估；
  - `phase2.txt`：推力恢复和速度接口；
  - `abl_queue*.txt`：消融实验。
- 实验日志在 [docs/repro/PROGRESS.md](docs/repro/PROGRESS.md)。

## 相对官方代码的改动

1. **修复训练后期确定性策略崩溃（必需）。**
   - 用官方代码在 BONES 上长时间训练，约 7 万步后，确定性评估会全面崩溃。
   - 同一批检查点如果在测试时加上 0.1 的采集噪声，表现仍然正常。
   - 原因是世界模型只见过带噪声的动作。
   - 两种修复任选其一：
     - `--runner.clean-action-frac 0.5`：一半仿真环境采集时不加噪声；
     - `--runner.noise-final 0.02`：采集噪声随训练逐渐衰减。
2. **余弦学习率**（论文表 A.5），参数为 `--runner.lr-schedule cosine`。
   - 官方代码的隐藏层用的是 Muon 优化器。torch 的 Muon 在 CUDA graph 捕获下学习率会被冻住，导致学习率调度不生效。
   - 我们用 `ScheduledMuon` 修复了这个问题，测试见 `repro/test_lr_schedule.py`。
3. **多卡数据并行**（`repro/run_train_ddp.sh`）：
   - 用 torchrun 启动，对梯度做 all-reduce；
   - 学习率乘以 √N；
   - 定期校验各卡权重一致。
4. **BONES-SEED 数据流程**：名字过滤、批量 CSV→npz 转换、附录 E 的可行性过滤。最终得到 129,543 个片段（论文为 129,514 个）。
5. **评估**：
   - 实现附录 F.1 的评估协议，不依赖 W&B；
   - 支持多卡分片评估；
   - 按论文表 A.8 的方式统计各动作组的失败数。
6. **补齐发布版缺少的论文实验**：
   - 无 critic 消融（`--runner.use-critic False`）；
   - 推力恢复测试；
   - RL 速度编码器和阶跃测试（附录 G）。发布版只有蒸馏版本。
7. **本地检查点**：
   - `run_config.json` 和 `snapshots/` 直接写入日志目录，所以 W&B 不再是必需的；
   - 每次运行使用独立的日志目录。

## 环境配置

环境要求：Linux x86_64，NVIDIA 驱动 ≥ R525，并安装 [uv](https://docs.astral.sh/uv/)。

脚本默认的目录结构如下：

```
workspace/
├── data/                  # 动作数据（不包含在仓库中，见下文）
└── ws/
    ├── supermimic/        # 本仓库
    └── mjlab_research/    # git clone https://github.com/HybridRobotics/mjlab_research
```

```bash
cd ws/supermimic
uv sync
export MUJOCO_GL=disable WANDB_MODE=offline   # 无显示环境；W&B 可选
uv run pytest -q tests/ repro/test_lr_schedule.py
```

`pyproject.toml` 固定使用 NVIDIA 的 CUDA 12 版 warp-lang。原因是 PyPI 上 1.18 及以后的版本需要 R580 以上的驱动。

## 数据

本仓库**不**附带任何数据集，请按各自的许可自行获取。

**LAFAN1**（G1 重定向版本，[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，许可为 CC BY-NC-ND 4.0）

把 40 个 CSV 放到 `data/lafan1_g1/csv/`，然后运行：

```bash
uv run repro/csv_to_npz.py --input-glob '../../data/lafan1_g1/csv/*.csv' \
  --output-dir ../../data/lafan1_g1/npz --input-fps 30
```

**BONES-SEED**（[bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)，需要申请访问，使用自定义许可）

把 `g1.tar.gz` 和 `metadata/` 下载到 `data/bones_seed/`，在该目录解压，然后运行：

```bash
D=../../data/bones_seed
uv run repro/bones_name_filter.py --metadata $D/metadata/seed_metadata_v004.parquet --out-keep $D/keep_names.txt
uv run repro/csv_to_npz.py --input-glob "$D/g1/csv/*/*.csv" --output-dir $D/npz_name_filtered \
  --format bones --input-fps 120 --keep-list $D/keep_names.txt
uv run repro/feasibility.py --npz-glob "$D/npz_name_filtered/*.npz" --out $D/feasibility.parquet
uv run repro/bones_finalize.py --data $D
uv run python -m mjlab.scripts.pack_motions --input "$D/npz_final/*.npz" --output $D/packed_final
```

## 训练

```bash
COS="--runner.lr-schedule cosine --runner.lr-decay-iters 250000 --runner.lr-final-ratio 0.03"
# LAFAN1，单卡（论文的 actor 配置：本体感知观测，10 帧历史）
repro/run_train.sh 0 lafan1_s0 --env.seed 0 --runner.max-iterations 40001
# BONES-SEED，单卡
MOTION=../../data/bones_seed/packed_final repro/run_train.sh 0 bones_1gpu \
  --runner.max-iterations 320001 --runner.save-interval 8000 $COS --runner.clean-action-frac 0.5
# BONES-SEED，双卡
MOTION=../../data/bones_seed/packed_final repro/run_train_ddp.sh 0,1 bones_2gpu \
  --runner.max-iterations 320001 --runner.save-interval 8000 $COS --runner.clean-action-frac 0.5
```

在 A100 上：
- 显存峰值：LAFAN1 为 20.9 GB，BONES-SEED 为 33.7 GB；
- 速度约 0.22 秒/步。

## 评估

```bash
RUN=logs/SuperMimic-GenCoord-G1/<运行目录>
# LAFAN1，评估所有检查点
uv run repro/evaluate.py --run-dir $RUN --motion-file '../../data/lafan1_g1/npz/*.npz' --all-snapshots
# BONES-SEED：多卡分片评估后合并（8 张 A100 约 10 分钟）
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k uv run repro/evaluate.py --run-dir $RUN \
  --motion-file '../../data/bones_seed/npz_final/*.npz' --snapshot it0320000.pt --sort-by-length \
  --shard $k --num-shards 4 --output $RUN/eval_full/shard$k.json & done; wait
uv run repro/merge_eval.py --pattern "$RUN/eval_full/shard*.json"
# 推力恢复 / 速度接口
uv run repro/push_test.py --run-dir $RUN --snapshot it0040000.pt
uv run src/supermimic/scripts/train_velocity.py --init-run-dir $RUN --init-snapshot it0144000.pt \
  --mode rl --motion-file <走跑动作库打包目录> --run-name vel_rl
uv run repro/velocity_step_test.py --run-dir <速度接口运行目录> --snapshot it0100000.pt
```

`repro/phase*.sh` 和 `repro/watch_eval*.sh` 是我们实验时用的调度脚本，里面写死了运行目录名。

## 许可与署名

代码采用 Apache-2.0 许可，与官方一致，见 [LICENSE](LICENSE)。官方代码的版权归 The SuperMimic Authors 所有。

用本仓库在 BONES-SEED 上训练的模型需注明：*Training data includes Motion Data by Bones Studio*（[bones.studio](https://bones.studio/)）。*Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.*
