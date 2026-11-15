# baseline 随机种子对照：10m，seed 1

目的：保持 `20261109_f2b_history_baseline_10m_seed0` 的全部训练设置，仅将 `ppo.seed` 从 0 改为 1，观察随机种子对学习轨迹和最终控制成绩的影响。在启智 CPU 资源空间的 `plasma-control4` 独立运行。

2026-11-15 **19:54:43（北京时间）**已发起正式训练：runner PID `445494`，训练 PID `446156`，比较 PID `445498`。19:57 的现场检查已确认 32 路 HFM 正在执行 STEP。首次 PPO 汇总需采集满 8192 条转移，当前还不能报告 seed 对最终成绩的影响。

| 项目 | 设置 |
|---|---|
| 运行源码 | 原始提交 `10264746549e45aa38b3be02b17e024c3b84cfb8`，独立 checkout |
| 唯一训练配置差异 | `ppo.seed: 0 → 1` |
| 训练规模 | 从头训练 10,000,000 步，32 环境，每环境每 rollout 256 步 |
| 模型、观测、动作 | 原 MLP actor/critic，812 维历史观测，history4、delay1，7 维 delta 映射为 12 维电压 |
| shot、reset | 15892_400 / 15892_500；每次 reset 的 signeo、bp、q0 独立 ±5% |
| 奖励、PPO | 原始 F2 dense reward，原有学习率、warmup/cosine、batch、epoch、GAE、bootstrap 与归一化设置 |
| 正式评估 | 保持 seen / validation / test 的 shot、3 个 episode、起始 seed 100 / 200 / 300 不变 |
| 资源 | plasma-control4：55 核、500 GiB；原 baseline：54 核、300 GiB；seed=0 复现实例 plasma-control3 同为 55 核、500 GiB |

`ppo.seed` 同时影响网络初始化、动作采样、minibatch 打乱，以及派生的 reset 参数、电源延迟随机流。因此本实验测量整体训练种子的影响，不单独隔离某一种随机源。训练环境使用 seed 1–32，电源使用 seed 30001–30032。

## 比较与记录

训练和测评源码固定为原始提交；仅在外部增加初始模型及 RNG 记录，校验记录前后不消耗训练随机数。配置及依赖逐项核对，见 `config_parity.json`、`resolved_config_parity.json`、`dependency_parity.json`。`initial_state.json` 保存初始参数与采样 RNG 的哈希。

CSV、TensorBoard、W&B offline 和 checkpoint 保存在远端实验目录中。`comparison.json` 每分钟对齐原 baseline 中断前的公共步数，排除耗时/FPS，列出训练指标差异；不同 seed 出现差异是预期现象，不表示复现失败。

原 baseline 在 2,211,840 步后中断并续训。分析时还应对照 `20261115_f2b_history_baseline_10m_seed0_repro` 的连续训练；单独比较 seed 0 和 seed 1 两次运行，不能估计训练结果的完整分布或统计显著性。最终性能以相同评估随机条件下的正式分数为准。

## 使用位置

远端项目根目录：

```text
/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed1
```

在该目录查看日志：

```bash
tail -f experiments/20261115_f2b_history_baseline_10m_seed1/train.log
cat experiments/20261115_f2b_history_baseline_10m_seed1/comparison.json
```

启动与现场验证结果见 `launch.json`、`live_check.json`。本地提交的状态文件是带时间戳快照，不能代替实时状态。已有正式运行时不要重复启动编排脚本。

MATLAB 启动时出现过 MCI 辅助服务错误文本；随后 32 个 socket 均成功连接并执行 INIT、RESET 和 STEP。该现象原样留在远端日志中，未通过修改仿真代码或训练设置规避。
