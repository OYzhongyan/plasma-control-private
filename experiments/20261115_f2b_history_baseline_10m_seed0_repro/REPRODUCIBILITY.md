# baseline 同种子复现实验

目的：在启智 CPU 资源空间的 `plasma-control3` 上，从头重复 `20261109_f2b_history_baseline_10m_seed0`，检验固定种子后的训练数值是否一致。

原始运行提交为 `10264746549e45aa38b3be02b17e024c3b84cfb8`，新实验使用该提交的独立 checkout；训练源码不做修改。请求 10,000,000 步，seed=0，32 环境，端口 2223–2254，历史观测 4 帧、一步观测延迟、delta 动作；reset 的 signeo、bp、q0 每次独立 ±5% 扰动。奖励、归一化、PPO、500 步截断 bootstrap、shot 分配和正式评估设置均复用原始配置。

原 baseline 在 2,211,840 步之后中断并续训；新实验从头运行。因此严格逐步比较只针对原始 `training/` 下的不中断前缀，不能把续训后的不同归因于种子缺陷。

如果本次实例也被中断，不能用续训后的轨迹声称通过连续训练的逐步复现检验；需要保留中断点，并另行从头重跑。

## 对齐与记录

- 原实验的源码、目标文件、HFM 启动脚本及初始化 MAT 文件哈希逐一核验。
- Python 环境从 baseline 实例复制到新实例独立前缀，保留原实例环境；重点对齐 PyTorch 2.10.0+cu128、SB3 2.8.0、NumPy 2.4.4。实际依赖记录见 `environment.json`。
- 新实例 CPU 配额 55 核，原实例 54 核；宿主机及资源差异会记录，不能宣称硬件逐位相同。
- 使用本实例新建的 32 个 HFM 进程及 `/tmp` 下独立缓存；不重用其他实验的 HFM 状态。
- 原训练入口外增加初始状态观察器，保存初始模型、归一化与 RNG 状态。记录前后校验 Python、NumPy、PyTorch 随机流未改变；不修改训练参数或算法。
- PPO CSV、reward CSV、TensorBoard 与 W&B offline 同时保留。初始模型等大文件仅保存在远端 `training/`，不推送 Git。

## 如何判断

`comparison.json` 定时按累计环境步数对齐原实验与复现实验的 PPO、reward CSV，排除 `time/*` 等时间指标，检查其余指标的数值是否完全相等，并记录首次差异。`initial_state.json` 保存初始 actor、完整模型及动作采样 RNG 的 SHA256。

曲线相同只是采样指标一致，不能单独证明全部内部状态逐位相同。原始实验缺少逐步观测和电压记录；若发现差异，可结合首个公共 checkpoint 进一步定位，不能据此直接判定 HFM 或 PPO 有 bug。

## 路径

远端项目：`/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-seed-repro`。

实验目录：该项目下 `experiments/20261115_f2b_history_baseline_10m_seed0_repro/`。当前启动状态以带时间戳的 `launch.json`、`live_check.json` 为准；配置和来源分别见 `config_parity.json`、`source_parity.json`、`runtime_parity.json`。

运行编排代码位于当前仓库 `scripts/seed_repro/`，实际执行副本也保存在远端实验 `scripts/` 中。核心训练代码固定在历史 checkout；临时环境传输包在安装成功后删除。
