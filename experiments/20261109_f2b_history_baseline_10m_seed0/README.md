# F2b 历史观测 baseline：10m，seed 0

目的：恢复 SII-585-THU 的历史观测和 F2 dense reward，在两个 limiter 初始 shot、每次 reset 的 ±5% 参数扰动下训练，并用留出 shot 检查泛化能力。训练奖励已恢复为上游实现；最终成绩仍按正式 competition 评分规则计算。

上游提交：`ffda09ab4621ede477ff71247547f8cbc43d133a`。运行代码提交：`10264746549e45aa38b3be02b17e024c3b84cfb8`。方案见 `DESIGN.md`，执行记录见 `PLAN.md`。

| 项目 | 设置 |
|---|---|
| 输入 / 输出 | 812 维历史观测 / 7 维 delta 动作，映射为 12 维电压 |
| 历史 | 4 帧观测、5 帧实际动作，观测延迟 1 步 |
| 并行环境 | 32 个 HFM，端口 2223–2254，500 步/episode |
| 训练 shot | 15892_400、15892_500 |
| 泛化设置 | 每次 reset 的 signeo、bp、q0 独立缩放至 95%–105% |
| PPO | 2×256 Tanh，lr 1e-4，n_steps 256，batch 1024，6 epochs |
| 训练步数 | 请求 10,000,000；按完整 rollout 预计实际 10,002,432 |
| 随机种子 | 0；环境加 index，电源延迟 RNG 再加 30000 |
| 本地曲线 | training/logs/ppo/progress.csv、TensorBoard、reward_components.csv、monitor.csv |
| W&B | offline，本地保存，可之后同步 |

| 评估划分 | Shot | 起始 seed | episode 数 | 正式评分 /30 |
|---|---|---:|---:|---|
| seen | 15892_400 | 100 | 3 | 待完整训练结束 |
| validation | 14795_400 | 200 | 3 | 待完整训练结束 |
| test | 13705_500 | 300 | 3 | 待完整训练结束 |

评估每个 episode 最多 500 步，额外加入上游 README 所用的传感噪声：Ip 标准差 2300 A、R 标准差 0.008 m。因此评估同时检查未见初始位形与观测噪声鲁棒性。单 seed 的结果不代表统计上的泛化保证，尚未测量推理内核的延迟。

## 实际启动记录

- 2026-11-09 21:46:44（北京时间）启动，runner PID `2468796`，训练 PID `2469008`。
- 远端目录：`/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control`（2026-11-10 重命名；当前训练结束前保留旧名称软链接，以兼容运行中的绝对路径）。
- 使用 `competition_env`，CPU 训练；复用既有 HFM 服务，未停止或重启它们。
- 启动后已确认 32 个端口连接且仿真状态持续推进；首轮 PPO 汇总需收集满 8192 条环境转移后生成。
- `remote_snapshot/` 是有采集时间戳的本地快照，不能用快照中的 TRAINING 状态代替实时状态。完整模型、曲线、评估结果持续保存在远端实验目录。
- 训练完成后 runner 自动进行上述三个划分的正式评估，生成 JSON/CSV、结果表和 `picture/` 图片。

从仓库根目录运行以下命令可复现实验；已有同名实验时应更换 ID，避免覆盖记录：

```bash
python tools/run_baseline_experiment.py \
  --experiment-id 20261109_f2b_history_baseline_10m_seed0 \
  --total-timesteps 10000000 --num-envs 32 --base-port 2223 \
  --wandb-mode offline
```

查看当前远端训练日志：

```bash
tail -f experiments/20261109_f2b_history_baseline_10m_seed0/train.log
```

runner 不管理 HFM 的启停。配置、依赖清单、源文件哈希、电源延迟 seed 规则和运行状态均保存在实验目录；`planned_config.yaml` 表示预定设置，`remote_snapshot/training/.hydra/config.yaml` 表示实际生效设置。

## 预检查边界

Smoke 仅临时用于联调，结束后删除全部产物，不作为正式实验归档。本次正式训练及其验证记录继续保留。

## 2026-11-12 中断恢复

已从 2,211,840 步 checkpoint 恢复，累计目标仍为 10m。原日志和模型保留，新输出位于 `training/resume_20261112_01/`。恢复点、检查结果、实际命令及 TensorBoard 查看方法见 [本次续训记录](resumes/20261112_01/README.md)。
