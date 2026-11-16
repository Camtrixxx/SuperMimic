# SuperMimic 复现调研

调研日期 2026-11-12。硬件：8×A100-SXM4-40GB。
标注说明：**[论文]** 出自 arXiv 2610.28480v2；**[代码]** 是我读官方仓库 `HybridRobotics/supermimic@abc124c`（2026-11-11）源码确认过的；**[第三方]** 是第三方 README 的自述，我没有核实；**[未知]** 是待确认的内容。

本地材料：
- `refs/paper/supermimic_v2.pdf`：论文
- `refs/paper/supermimic_v2.txt`：全文文本
- `refs/official/supermimic/`：官方代码，只读

---

## 1. 论文要点

- **问题：** SONIC 用 PPO 跟踪 BONES-SEED 和其他数据集，在 128×L40 上训练了 7 天（约 21k GPU-h）。
- **方法：** 用一个可微分世界模型 f_ω 拟合所有动作共享的 G1 刚体动力学，训练数据是 MuJoCo Warp 仿真器的转移。策略在 f_ω 上做 H=32 步想象 rollout，用 BPTT 拿到解析梯度来优化。仿真器本身不求导。
- **训练循环：** 每个 iteration 依次做三件事：
  1. 带噪声地采集仿真数据；
  2. 用 k=8 步自回归损失拟合世界模型；
  3. 冻结世界模型，做 BPTT 更新 actor，并用 GAE(λ=0.95) 拟合 critic。critic 负责对 H 步之外的回报做 bootstrap。
- **关键设计：**
  - 世界模型是确定性的，输入里拼接 sim context ξ（摩擦、关节零偏、躯干质心偏移）。
  - 观测噪声在世界模型外部加。
  - 软终止概率 d̂ 让 survival weight 可微。
  - reward 和终止都由预测状态通过 FK 解析计算，不学。
- **结果 [论文]：**
  - 在 129,514 个 clip 的训练集 benchmark 上，1×H100 训练 3 小时，normalized return 超过 SONIC 的 0.90，失败 25 个（SONIC 失败 179 个）。
  - 2×H100 训练 9.8 小时，失败数降到 0。
  - LAFAN1（40 个 clip）上 1 小时就能到 0.82 return，PPO-Sim 在 8 卡上训练 18 小时也只到 0.72。
- **消融 [论文]：**
  - 不用 critic 时，H≤16 训练失败。
  - 用 critic 时，H=16 的效果就能追平无 critic 的 H=32。
  - critic 对抗推力恢复至关重要。
  - 在 world model 上，BPTT 达到同样 return 所需的想象步数比 PPO 少 64 倍。
- **局限 [论文]：** 只做了平地，没有地形和物体交互；世界模型只用了仿真数据；速度接口只是初步结果。

## 2. 官方代码现状 [代码]

- **依赖：**
  - Python 3.13。
  - 依赖同级目录下的 `../mjlab_research`。它是 mjlab 1.6.0 的 fork，要求 torch≥2.14、mujoco / mujoco-warp ~3.16、rsl-rl-lib 5.5.2。
  - 用 uv 管理环境，强依赖 W&B。
- **规模与入口：**
  - 约 4.5k 行核心代码，另有 20 多个测试文件。
  - 两个 task：`SuperMimic-GenCoord-G1`（默认）和 `SuperMimic-MaxCoord-G1`。
- **已实现：**
  - 完整的训练循环（`runners/supermimic_runner.py`）。
  - 世界模型：MLP(2048×3)，配 LayerNorm 和 GELU；输出加速度级别的增量，然后做半隐式积分。损失按 Bryson 规则加权。
  - Actor：多编码器 AE（motion / velocity 两个编码器 + 共享解码器，32 维 tanh latent）。
  - Critic：MLP(256,128)。
  - ReturnSampler（基于 return 的重置采样）。
  - PPO 基线（`train_ppo.py`）。
  - ONNX 导出。
  - 播放与可视化（viser）。
  - `torch.compile` 和 CUDA graph 捕获。
- **与论文不一致、需要注意的默认值：**
  1. **actor 观测：** 默认 `actor_obs="rep_state"`（特权状态）、`history_length=1`。论文用的是带噪声本体感知加 10 帧历史，复现论文需要传 `--runner.actor-obs proprio --runner.history-length 10`。
  2. **训练长度与学习率：** 默认 `max_iterations=1e6`，学习率恒定 2e-4，**代码里没有 cosine 衰减**。论文写的是"衰减到 0.03× / 250k iters，总共 320k iters"。
  3. **多卡：** **没有多 GPU 代码**，整个代码里搜不到 distributed、all_reduce 或 torchrun。论文 2 卡结果所用的数据并行没有随代码发布。
  4. **默认动作数据：** 来自私有的 W&B registry `supermimic/wandb-registry-motions/lafan_run1_subject5`，我们拉不到，需要改用 `--motion-file` 加载本地 npz。这个参数支持 glob 和逗号分隔列表。
  5. **Actor 的 goal：** 包含 10 帧未来参考关节位置和速度，帧间隔 stride=5（约 0.1s，所以前瞻约 0.9s），再加上参考 root 相对当前 root 的姿态（rot6d）。相对位置只给世界模型和 critic，不给 actor。
- **缺失的部分：**
  - 没有 BONES-SEED 的名字过滤和可行性过滤脚本（论文附录 E）。
  - 没有 129,514 clip 的评估脚本（论文附录 F.1）。`scripts/benchmark.py` 只测速度。
  - 数据转换可以用 mjlab_research 的 `src/mjlab/scripts/csv_to_npz.py`，它逐个文件跑仿真来补全 body 状态。对 14 万个文件来说太慢，需要改成批量并行。
- **Issue 状态：**
  - #1：README 里 clone 的地址写错了，正确地址是 `HybridRobotics/mjlab_research`。社区已给出 workaround，issue 仍未关闭。
  - #2：只用 velocity slot 时初始化报错。

## 3. 第三方复现 [第三方，均未核实]

| 仓库 | 框架 | 自述结果 | 价值 |
|---|---|---|---|
| CuboiLeo/SuperMimic-Reproduced（MIT，已停更） | mjlab | BONES-SEED 上 4×5090 训练 7.1 小时：0.930 return，145 个失败；翻转动作全部失败 | 中文设计文档写得详细；自己实现了多卡（√N 学习率缩放）；记录了 yaw 漂移问题 |
| RaraAlu/Supermimic（无 license） | mjlab 1.6.0 | 114k 个 clip 上 2×A100 训练 10.4 小时：35 个失败；**130k iter 之后出现退化** | 遇到过的坑在 A100 上同样会碰到：世界模型只见过带噪声的动作，确定性策略会钻模型误差的空子 |
| zita-ch/supermimic_vibed（MIT） | Isaac Lab 3 | LAFAN1 上 0.95 小时达到 0.842，0/40 失败 | 可以作为 LAFAN1 的参照 |

## 4. 数据

- **BONES-SEED**（HF `bones-studio/seed`）：
  - 需要登录并同意自定义许可，**能否用于你的场景需要你自己确认**。
  - 共 142,220 个 clip（含镜像），120fps。
  - G1 重定向数据是 CSV，路径为 `g1/csv/{date}/*.csv`。
  - 过滤后剩 129,514 个，共 264.56 小时。
- **LAFAN1：** G1 重定向版本在 HF `lvhaidong/LAFAN1_Retargeting_Dataset`，许可为 CC BY-NC-ND。适合做小规模验证。

## 5. A100-40GB 上的风险

- **显存：** 论文用的是 H100-80GB，最长能跑到 H=128。默认配置是 4096 条 rollout、H=32、2048 宽的 MLP。如果 BPTT 计算图放不进 40GB，就需要减 batch 或者用 activation checkpointing。**[未知，需实测]**
- **速度：** A100 的 TF32 峰值大约是 H100 的三分之一到一半，单卡 3 小时的结果在 A100 上预计需要 6 到 10 小时。**[估算]**
- **多卡：** 要用 8 卡就得自己写数据并行，这也是论文 2 卡结果复现的前提。
- **长训退化：** 见 RaraAlu 的现象。可以考虑在采集时混入一部分干净动作、保存检查点后挑最好的，或者按论文加上学习率衰减。

## 6. 建议路线

1. **跑通环境：** 用 uv、官方仓库和 mjlab_research 搭环境，先跑测试，再用 LAFAN1 单条动作做一次冒烟训练。
2. **LAFAN1 的 40 个 clip（单卡）：** 用论文配置（proprio、10 帧历史），目标是 1 到 2 小时达到 0.82 return。顺带测出显存和吞吐。
3. **补齐缺失部分：**
   - 并行版的 CSV 转 npz；
   - 附录 E 的过滤脚本；
   - 附录 F.1 的评估脚本（14 万个 clip 分片到 8 卡上跑）；
   - cosine 学习率；
   - DDP 多卡训练。
4. **BONES-SEED 全量训练：** 先单卡对标"3 小时 / 25 个失败"，再上多卡对标"0 个失败"。
5. **可选：** 消融实验（critic、H、采样方式、GenCoord 与 MaxCoord 对比）、速度接口、ONNX 部署到真机。

## 7. 待确认

- 你的最终目标是只做仿真，还是要上真机 G1？
- BONES-SEED 的许可是否适合你的用途？
- 用官方代码做底座加补丁，还是自己重写一遍、拿官方代码对照？
- 有没有 W&B 账号？官方代码强依赖它，否则需要去掉这层依赖。
