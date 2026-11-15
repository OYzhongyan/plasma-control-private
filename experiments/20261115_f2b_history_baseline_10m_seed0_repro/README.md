# baseline 同种子复现：10m，seed 0

本实验用于检验固定随机种子后训练能否逐步复现，不改变原 baseline 的网络、奖励、PPO 或环境随机化设置。比较方案及边界见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

2026-11-15 **19:24:56（北京时间）**在启智 CPU 资源空间的 `plasma-control3` 发起正式训练。训练 runner PID 为 `1652292`，训练进程 PID 为 `1652682`，比较进程 PID 为 `1652295`。这些是启动快照，不代替实时状态。

| 项目 | 设置 |
|---|---|
| 原始实验 | `20261109_f2b_history_baseline_10m_seed0` |
| 运行源码 | 原始提交 `10264746549e45aa38b3be02b17e024c3b84cfb8`，独立 checkout |
| 训练 | 从头开始，10,000,000 步，seed=0，CPU，32 个环境 |
| 观测、动作 | 812 维，history4，delay1，7 维 delta 映射为 12 维电压 |
| shot、reset | 15892_400 / 15892_500，三个物理参数每次 reset 独立 ±5% |
| 软件 | PyTorch 2.10.0+cu128，SB3 2.8.0，NumPy 2.4.4 |
| 记录 | CSV、TensorBoard、W&B offline、初始模型和 RNG、定时曲线对比 |

远端项目根目录：

```text
/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed-repro
```

在该目录查看训练与比较结果：

```bash
tail -f experiments/20261115_f2b_history_baseline_10m_seed0_repro/train.log
cat experiments/20261115_f2b_history_baseline_10m_seed0_repro/comparison.json
```

`comparison.json` 每分钟更新，首先检查原 baseline 中断前的公共步数。尚无公共日志行时，不代表数值不一致；应先检查 `common_rows`。初始状态采集已验证未改变 Python、NumPy、PyTorch 随机流，见 `initial_state.json`。

原实验缺少逐步观测与动作记录，曲线一致也不足以证明全部内部状态一致；新实验与原实例的硬件配额也存在已记录的差别。当前还不能宣称整段训练完全复现。
