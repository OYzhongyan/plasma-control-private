# 2026-11-12 中断恢复：MLP baseline

已在重建的 `plasma-control` 实例启动续训，时间 **2026-11-12 14:39:29（北京时间）**。恢复代码为 `94c517eedff5ea36c0e6a7d689b0bc18c1154978`，原始运行代码与配置仍保留在实验根目录的 manifest 和原训练目录中。

| 项目 | 记录 |
|---|---|
| 中断前 PPO CSV 最大步数 | 2,211,840 |
| 实际恢复 checkpoint | 2,211,840 步 |
| 需要追加的转移 | 7,788,160 |
| 累计训练目标 | 10,000,000；完整 rollout 预计到 10,002,432 |
| runner / 训练 PID | 1086091 / 1090883 |
| 本段模型与曲线目录 | `training/resume_20261112_01/` |
| 旧 TensorBoard 有效指标 | 36 项 |
| 旧 reward 曲线点数 | 270 |
| 中断前平均 episode reward | 676.6097 |
| 中断前平均 episode 长度 | 496.62 |
| 中断前 Ip 相对误差 | 1.212% |
| 中断前 LCFS RMS | 2.592 cm |

这些是训练统计，未完成最终三个划分的正式评估。不同训练步数下的两组均值不能直接证明模型优劣。

## 恢复边界

已实际载入网络、优化器和同一步的 VecNormalize，并校验 checkpoint ZIP 完整性、参数量、归一化维度及有限值。32 个 HFM 均有持续 STEP 请求和增长的 CPU 时间。算法、观测编码、奖励、shot、reset 扰动、电源延迟、PPO 设置和第 500 步 bootstrap 保持原设置。物理环境、wrapper、奖励及正式评分源文件哈希与原 manifest 相等，见 `code_update.json`。

新增 `+resume.target_total_timesteps=10000000`、关闭步数重置，训练只补足累计目标，warmup/cosine 使用原 10m 总进度。原 checkpoint、CSV、事件及 W&B 文件均保留；新输出写入独立的续训目录。

HFM 内部状态、完整随机流和 rollout buffer 未保存，重建后重新 reset，因此这是从保存模型继续训练，不能逐步复现中断时的后续轨迹。现有 callback 在 rollout 结束、当轮 PPO 更新之前保存；未保存的当轮 buffer 无法补回。CNN 的旧日志领先 checkpoint 的部分也需重新采集。

## 查看可视化

在对应项目根目录的 `competition_env` 中运行：

```bash
tensorboard \
  --logdir experiments/20261109_f2b_history_baseline_10m_seed0/training \
  --host 0.0.0.0 --port 6006
```

该目录会显示原始训练及 `resume_20261112_01/logs/ppo` 两段曲线。在启智的 Notebook 端口预览中访问 6006。两段使用累计 global step；CNN 续训从较早 checkpoint 开始，两段之间会有少量步数重叠，应分别查看。

原 TensorBoard 已读出真实标量，不只是文件存在；样本和计数见 `curves_before_resume.json`。续训已完成第一次 8192 转移 rollout，并写入新的 PPO 标量；新数据核验见 `curves_after_resume.json`。W&B 当前为 **offline**，历史文件保存在共享存储，尚未同步至 W&B 网页。

续训初期 `rollout/ep_rew_mean` 会沿用 checkpoint 的 episode 缓存；`reward_components` 首个日志窗口可能不足 8192 步，不能据首个点判断恢复后的控制效果。需等待新 episode 和完整日志窗口再分析。

原始事件位于 `training/logs/ppo/`，续训事件位于 `training/resume_20261112_01/logs/ppo/`。原 W&B 位于 `wandb/`，续训 W&B 位于 `wandb/resume_20261112_01/`；CSV 同时保留，不依赖外部服务。

## 可复现记录

`resume_request.json` 记录恢复点和目标；`checkpoint_validation.json` 记录模型/统计 SHA256、优化器及形状校验；`commands.json` 保存实际训练及后续三组正式评估命令；`resolved_config.yaml` 是本段生效配置；`environment.json` 保存依赖和资源配额；`hfm_launch.json` 记录新 HFM 进程、核绑定和容器内隔离缓存。`continuation_driver.py` / `start_hfm.py` 保留实际编排代码。

`before_resume/` 是中断前 CSV 的本地副本，`status.json`、`live_check.json` 和 `hfm_live_check.json` 是带时间的启动快照，不能替代实时状态。实例当前已运行，不应重复启动这些记录脚本。训练结束后继续进行原 seen、validation、test 三组正式评估，每组 3×500 步，成绩写回实验的 `metrics.json` / `metrics.csv`。
