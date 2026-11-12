# 四帧磁通 CNN critic：10m，seed 0

对照实验：[历史观测 baseline](../20261109_f2b_history_baseline_10m_seed0/README.md)。仅替换 critic，actor、训练奖励、shot、扰动、动作、电源延迟、PPO 超参数、训练步数和正式评估设置保持一致。

运行代码：`8a5b9db34a003291bdfa6d1f852eae3c89f3a353`。模型设计与实现计划分别见 [DESIGN.md](DESIGN.md) 和 [PLAN.md](PLAN.md)。`baseline_comparison.json` 对照归档 baseline 配置逐项比较，只允许 `critic` 字段不同。核心依赖版本与 baseline 一致，见 `runtime_comparison.json`；奖励、环境、参考轨迹和电源源码的相等哈希见 `source_parity.json`。

| 项目 | 设置 |
|---|---|
| Actor | 原始 812 维历史特征，2×256 Tanh，7 维 delta 动作；275,726 个参数，含 log_std |
| Critic | 四帧 66×65 磁通 Fx，共享三层 CNN，按时间顺序融合；同时输入原 812 维辅助特征和 4 维有效帧掩码；989,185 个参数 |
| 时间与归一化 | 与 actor 使用同一延迟观测；reset 后补零并掩码；所有有效磁通像素共享一组标量 RMS |
| 并行训练 | 32 环境，端口 2223–2254，CPU，seed 0 |
| 训练 shot / 扰动 | 15892_400、15892_500；每次 reset 的 signeo、bp、q0 独立 ±5% |
| PPO | lr 1e-4，warmup/cosine，n_steps 256，batch 1024，6 epochs，gamma 0.995，GAE lambda 0.95 |
| 训练步数 | 请求 10,000,000；完整 rollout 向上取整为 10,002,432 |
| 第 500 步 | 时间截断继续 bootstrap；真实失败不 bootstrap |
| 训练奖励 | SII-585-THU 上游 F2 dense reward，未修改 |
| 评估 | seen：15892_400/seed100；validation：14795_400/seed200；test：13705_500/seed300；各 3×500 步 |
| 评估噪声 | Ip 标准差 2300 A，R 标准差 0.008 m，保持 baseline 设置 |
| 曲线 | PPO progress.csv、TensorBoard、reward_components.csv、monitor.csv 和 W&B offline |

本实验同时改变 critic 的架构和可用信息，不能单独归因于 CNN。Actor 的初始化张量及初始化后的 Torch 随机状态通过与原 MLP 策略逐项相等的测试；actor-only 推理不会执行 critic CNN。

## 部署与复现

在启智 Notebook `plasma-control2` 的独立目录 `/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-cnn-critic` 运行，复用该 Notebook 已有的空闲 HFM 服务。原 `plasma-control` Notebook 的 baseline 进程和仿真服务不受改动。

Notebook2 的 CPU 配额为 110 核、内存 500 GiB；baseline Notebook 的配额为 54 核、内存 300 GiB。训练进程仍使用相同的 32 个环境及单线程数值库；壁钟耗时不能直接视为相同硬件下的速度对比。

2026-11-10 **22:03:31（北京时间）**已启动正式训练：runner PID `3690091`，训练 PID `3690338`。启动快照中已确认两进程存活、32 路 HFM 已连接；两次独立读取均观察到 `STEP` 请求，32 个仿真进程的 CPU 时间持续增长。最初快照处于首个 rollout 采集阶段，尚无正式 PPO 汇总行；首次汇总需收集满 8192 条转移，不能将空 CSV 解释为训练完成或失败。

```bash
python tools/run_baseline_experiment.py \
  --experiment-id 20261110_cnn_flux_critic_10m_seed0 \
  --critic-type flux_cnn \
  --baseline-config experiments/20261109_f2b_history_baseline_10m_seed0/remote_snapshot/requested_config.json \
  --total-timesteps 10000000 --num-envs 32 --base-port 2223 \
  --wandb-mode offline
```

已有正式运行记录时不要重复启动；复现请换新 ID。runner 在训练完成后自动执行三个划分的正式评估并生成 `picture/` 图片。比赛满分 30，训练奖励不等于比赛成绩；正式分数待完整训练和评估结束后报告。

远端实验目录保存完整权重、曲线、解析后配置和来源记录；`remote_snapshot/` 是带采集时间的本地快照。临时联调模型、轨迹、评分、图片及日志只使用 `/tmp`，检查结束后全部删除，不纳入实验记录。

## 验证

本地相关回归测试 148 passed、4 skipped；启智匹配 baseline 的 SB3 2.8.0 环境中，新增相关测试 37 passed。验证包括原 actor 特征及归一化一致性、磁通排列和掩码、实际 SB3 rollout 中的截断 bootstrap、PPO 更新与存取、完整 checkpoint 推理及 actor-only ONNX 导出。独立代码审查未发现阻断问题。真实 HFM 临时训练、权重保存/载入及三个 shot 的评估链路均通过；临时产物已全部删除。正式启动证据见 `remote_snapshot/launch.json`、`remote_snapshot/training/architecture.json` 和 `remote_snapshot/live_hfm_check.json`。

## 2026-11-12 中断恢复

已从 901,120 步 checkpoint 恢复，累计目标仍为 10m。原日志和模型保留，新输出位于 `training/resume_20261112_01/`。恢复点、检查结果、实际命令及 TensorBoard 查看方法见 [本次续训记录](resumes/20261112_01/README.md)。
