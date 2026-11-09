# 真实训练与评分流程预检查（1024 步）

此实验用于确认 HFM → 观测 wrapper → PPO 更新 → checkpoint → 测评与作图的完整链路，不用于汇报 baseline 成绩。

- 两个 HFM 环境完成了 1024 条训练转移，成功保存 checkpoint、归一化统计与元数据，并产生本地 CSV、TensorBoard 和离线 W&B 文件。
- 首次自动评估发现 Hydra 配置合并拒绝 `save_dir` 的问题；原始 `status.json` 保留 FAILED，具体错误见 `remote_snapshot/first_failure.json`。
- 修复提交为 `10264746549e45aa38b3be02b17e024c3b84cfb8`。修复后复用预检查模型，在 seen、validation、test 三个 shot 各进行 1 episode、16 步的真实仿真与评分；见 `remote_snapshot/short_verification_status.json`，状态 COMPLETED。
- 16 步结果只验证数据和评分接口，不能当作 500 步评估或 10m 模型成绩。
- `picture/` 保存训练曲线及三个 shot 的图。`remote_snapshot/` 保存指定时刻的实际配置、环境、模型哈希、曲线、短程轨迹和评分。

正式 10m 实验位于相邻目录 `20261109_f2b_history_baseline_10m_seed0/`；它使用随机初始化，未从此预检查模型续训。
