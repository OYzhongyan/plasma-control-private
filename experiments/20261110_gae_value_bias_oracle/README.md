# GAE 后期 value 偏差：方法分析草案

状态：**CANCELLED，用户于 2026-11-10 取消，暂不考虑此实验**。未实施或运行，不再推进。以下内容仅保留为此前的方法分析；已复制的参考文档保留。本实验没有修改训练代码、启动 HFM 或使用 `plasma-control` 的训练计算资源，不影响原有 10m baseline。

## 1. 要验证的命题

需要分别验证三个环节：

1. episode 后期的 critic 是否存在更大的**系统偏差**，而不只是较大的单次预测误差或回报方差。
2. 偏差是否进入 GAE，使 actor 的更新方向失真。
3. 只改善 value 估计、保持其他条件不变，是否改善独立仿真中的控制效果。

这不是一个“换成 MC 后训练分数是否更高”的单一对照。后者同时改变偏差、方差、样本量和训练轨迹，无法单独归因。

用户已明确：**后续实验优先保留第 500 步 bootstrap**。本实验因此以持续控制的折扣回报为主目标；500 步是采样截断点，不是价值为零的吸收终点。真正物理失败后的 continuation 才置零。

**当前已知限制：用户确认无法获得第 500 步之后的 continuation。** 下述直接 MC value oracle 路线暂不可执行，其所需能力仅作为条件说明。当前可行方案改为第 8 节的“bootstrap 敏感性与 actor 干预诊断”，不能声称已经检验真实持续 value 的偏差。

## 2. 文档与当前代码的含义

参考文档：`docs/gae_reward_support/gae_reward_support_analysis.tex`，以及同目录 `build/gae_reward_support_analysis.pdf`。已从 SII-585-THU 原样复制到本地与远端项目，17 个文件通过 SHA256 校验。

文档证明的是 target 对预测误差的敏感性，而非实测 critic 误差必然随 episode 时间增加。当前配置与文档示例相同：episode 上限 500、rollout 每环境 256、gamma=0.995、lambda=0.95。固定长度 500、无提前失败且覆盖完整 rollout 相位周期时，episode 第一位置平均有 130 个 reward 支持 target，最后位置只有 1 个。

令片段包含 K 个 reward，q=gamma*lambda。只扰动片段终点的 value、固定 reward 和中间 value 时：

$$
\Delta Y_t=\gamma(\gamma\lambda)^{K-1}\Delta V_{t+K}.
$$

文档中第一位置的平均终点敏感系数约 0.0651，最后位置为 0.995，约相差 15.3 倍。**这个倍数不是实际 critic 误差之比**。若所有中间状态也有误差，总误差还包含中间项；文档对应的一致误差上界系数由约 0.9087 增至 0.995，不能套用 15.3 倍。实际训练还存在提前失败，必须记录真实 episode 起点和片段长度，不能直接套固定 500 步周期公式。

代码证据：

- `environment/hfm_simulator.py:393`：物理失败设置 terminated；达到 max_steps 设置 truncated。
- `train/dense_rewards.py`：超流也会设置 terminated；不会仅因第 500 步而终止。
- `train/train_ppo.py`：SubprocVecEnv/VecMonitor/VecNormalize，SB3 2.8.0。正常达到 500 步时，SB3 向末步奖励加入 gamma*V(terminal_observation)，并停止该 episode 的 GAE 递推；不拼接 reset 后新 episode 的 reward。
- `train/wrappers.py:261`：812 维历史观测没有显式输出剩余时间。history 不等于完整 Markov state，不能把可观测性问题直接归为 value 优化误差。
- `VecNormalize` 同时归一化观测和 reward；诊断必须冻结统计并使用相同 reward 单位。

Critic 不在推理时直接决定动作。它通过训练时的 advantage 影响 actor。一个仅依赖当前状态的 baseline 误差，在理想 on-policy score-function 梯度期望中可抵消；未来状态的 bootstrap 误差、有限样本及 PPO clipping 等仍可能影响更新。不能由 value MSE 大直接推出动作方向错误。

## 3. 三种方法及取舍

| 方法 | 能回答什么 | 主要限制 |
|---|---|---|
| 固定策略、独立长轨迹的 MC suffix 回报 | 按时间分组检查 calibration 与平均偏差 | 一个状态只有一次 suffix 时噪声大；不能给每个状态高精度真值 |
| 同一个 anchor 多次独立 continuation 取均值 | 估计单个状态的 value 参考值及其不确定度 | 需要完整状态恢复或可靠的确定性前缀重放，仿真成本更高 |
| 替换 value 的配对 actor 更新 | 检查 value 误差是否造成控制退化 | 需要独立的参考数据、足够精确的参考 value 及重复验证 |

建议依次进行：先低成本分组诊断，再对关键 anchor 做重复 continuation，最后做 actor 干预。lambda=1 或加大 n_steps 可以作为后续对照，但不充当 oracle：256 步片段末端及第 500 步仍然 bootstrap；更长 rollout 也不会消除 episode 截断。

## 4. MC 参考值如何构造

### 4.1 冻结被估计的对象

复制一个已经完整保存的 checkpoint 与同一步 VecNormalize 统计，记录哈希。固定 actor 参数、critic 参数、观测/回报归一化统计和目标参考轨迹；不同 checkpoint 分别分析。诊断执行训练所对应的**随机策略**，不能用 deterministic=True 的均值策略回报冒充随机策略的 value。

主参考量是当前固定策略的 value，不是最优控制器的 V*。采用训练的 dense reward，而非直接拿正式比赛总分回归 critic；比赛分数另外用于评价干预后的控制效果。

### 4.2 跨过 500 步观察真实 continuation

以 h_t 表示 actor/critic 实际看到的历史，以 x_t 表示包含仿真器、延迟队列、动作累计量等的完整状态。从同一个 anchor 恢复后，当前及后续动作均按冻结策略采样，未来电源随机延迟也独立采样。第 500 步不 reset，继续观察 reward；真正失败则停止并置零尾值。

必须定义 500 步之后的 reference。目前提议：原样保存前 500 步 reference，之后保持最后一个目标。这个延拓是新实验对持续任务的明确补充假设，不能悄悄把原 500 步参考曲线拉伸为更长曲线。需要先核验 HFM 后端能否持续推进；这一轮没有启动仿真核验。

无限持续回报不能只跑到 500 后置零。若只采样 L 步、不使用原 critic 补尾，尾项仍有截断偏差。对于 |r|<=B、gamma<1：

$$
\left|V^\pi(x)-\mathbb E[G^{(L)}\mid x]\right|
\leq \frac{B\gamma^L}{1-\gamma}.
$$

同时检查更长 L 的回报变化并报告此上界；不能将“MC”称为无条件无偏 oracle。若用原 critic 补尾，又会重新引入待验证的同源偏差。

### 4.3 用重复均值控制方差

对 anchor i，R 条相互独立的 continuation 给出回报 G_i^(1),...,G_i^(R)：

$$
\bar G_i=\frac1R\sum_{k=1}^{R}G_i^{(k)},\qquad
\mathrm{SE}_i=\frac{s_i}{\sqrt R}.
$$

比较 V_i 与均值及其置信区间，不以一次 G_i 作为真值。可以先用独立的小批量估方差，再为正式估计预先确定 R=16/32/64 等预算；方差过大时结论标为不确定。单状态的平方残差还含 MC 均值噪声，聚合 MSE 可同时报告减去 s_i^2/R 的噪声修正估计；不要把负的有限样本修正值解释成负误差。

不同方法之间使用配对的环境与策略随机数降低差值方差；同一 anchor 的重复样本之间仍需独立，不能不断复用同一 seed 假装增加样本量。若多次重放只是同一初始状态，数据集也不因此拥有更多独立 anchor。

目前 socket 客户端只暴露 INIT/RESET/STEP/EXIT，没有任意状态 SAVE/RESTORE。不能仅凭一个观测字典重建完整状态。若后端无快照接口，需要从 reset 重放相同前缀，并恢复动作累计量、5 帧动作历史、4 帧观测历史、观测延迟和电源队列/RNG，再从 anchor 分叉。必须先验证分叉前状态一致，否则不称为同状态 continuation。

### 4.4 统计与可观测性

先采集固定策略的独立 episode，例如每个 seen shot 32 次启动；包含所有提前失败，而不是只挑能跑满 500 步的 episode。按 1–100、101–200、201–300、301–400、401–450、451–500 分组；后期稀疏时给出实际样本数与 survival rate，必要时合并相邻分组。

报告 signed bias、MAE/RMSE、MC 标准误、value 与参考回报的 calibration、K 分布和 beta 分布。区间估计按 episode/anchor 聚类，不能把同一轨迹的 500 个时刻当成 500 个独立样本。seen/held-out shot、物理失败与正常到达采样终点分开报告。

完整状态 x 的 MC 均值不等于 critic 输入 h 的条件均值。若 V(h) 无法区分相同观测下的隐藏状态，逐 anchor 差值可能含不可约部分观测误差。聚合 calibration 与同输入的交叉验证拟合是必要补充；时间特征或特权状态仅作为单独的表示能力对照，不能混进“只修正估计”的主对照。

## 5. 从相关性走向因果验证

第一层，固定同一批轨迹和 rollout 切分，注入终点 value 的 +/-c 偏移，检查 target/advantage 的变化是否符合 beta_K*c。这只验证机制敏感性，不证明实际 critic 有该偏差，不需要额外 HFM 调用。

第二层，使用独立 continuation 的均值替换**仅片段终点**的 value，中间 value、reward、gamma、lambda 与切分全部保持相同。比较 advantage 偏移及 actor 梯度方向。随后才考虑用交叉拟合的 MC critic 替换全部 value；参考 critic 需在独立 anchor 上验证误差确实下降，不能只看训练拟合。

第三层，从同一 actor、优化器状态与数据批次复制两个分支：原 value 与替换 value。进行小幅、同等 KL 约束的 actor-only 更新，冻结 critic/value loss，防止额外的 critic 更新或全局梯度裁剪耦合造成混淆。数据、minibatch 顺序、归一化规则、entropy 系数等相同。MC 参考数据与检验数据分开。

在独立、配对的仿真 reset 与随机数上比较两个 actor：前 500 步正式分数、后期 Ip/LCFS/X 等偏差、失败率，以及持续 dense return。重复数据批次/seed 并给出差值置信区间。优势接近零且参考区间跨零时不判定“符号翻转”；比较方向时还需考虑 advantage normalization/PPO clipping。

若 value 更准、advantage/梯度更合理且独立控制效果也改善，才支持“value 误差损害了 actor”的因果链。若只出现第一项，则不能排除 reward 与比赛目标不一致、探索、控制饱和、隐藏状态或优化过程等原因。不建议直接启动另一个 10m 训练来做第一轮诊断。

## 6. 可能的代码改动（尚未实施）

| 位置 | 可能改动 | 隔离要求 |
|---|---|---|
| 新增 tools/diagnose_value_bias.py | 加载冻结 checkpoint，采集独立长轨迹与 value，导出分时段分析 | 不调用现有在训环境或服务 |
| 新增诊断专用 continuation/replay 模块 | 保留 reference 前缀、延拓目标、跨 500 步 continuation，前缀重放和完整 wrapper 状态管理 | 不改变现有 HFMSimulator 的默认 500 步行为 |
| 新增诊断专用 GAE 分析模块 | 保存原始/归一化 reward、原始/修正后 buffer reward、terminal value 与 masks；重算 Y/A 与敏感系数 | timeout bootstrap 只加一次，不能把 SB3 修正后 reward 再次补尾 |
| 新增诊断专用 actor probe | 值替换与 actor-only 的配对更新、梯度与 KL 记录 | 不修改正在运行的 PPO 实例，不编辑 site-packages |
| train/wrappers.py（可选独立对照） | critic-only 时间/隐藏状态特征或独立 critic 表示 | 主对照保持 actor 812 维输入与动作变换不变 |
| 独立 YAML 与验证测试 | 明确 continuation 目标、MC horizon/R、seed、归一化和真实 terminal 规则 | 验证 500 步 bootstrap、真实终止零尾值、reset 隔离、归一化尺度与重放一致性 |

实现前还要核验：HFM 长程推进与快照/重放能力、延拓 reference 规则及实际仿真预算。当前只建立分析草案，不将这些提议视为已验证能力。

## 7. 资源与记录边界

所有后续采样、HFM 实例、actor/critic 计算都放在 Notebook `plasma-control2`，使用它自己的 CPU 配额和 localhost socket 服务；不连接 `plasma-control` 的端口池，不停止或重启现有 baseline。共享存储上的文档/记录可共用，但未来试验代码应独立检出，避免修改在训工程。

该目录是正式研究实验的分析记录，不是 smoke 归档。没有生成仿真或训练结果；`picture/` 只在真正采集后保存研究图。后续 smoke 依旧使用临时目录并全部清理。

## 来源

- 本项目导入的 gae_reward_support_analysis.tex：数值表与敏感系数。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：value、advantage 与梯度偏差的区分。
- [Time Limits in Reinforcement Learning](https://proceedings.mlr.press/v80/pardo18a.html)：人工截断与真正终止的区别。
- [SB3 2.8.0 on-policy collector](https://stable-baselines3.readthedocs.io/en/v2.8.0/_modules/stable_baselines3/common/on_policy_algorithm.html)：timeout reward bootstrap。
- [SB3 2.8.0 RolloutBuffer](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.8.0/stable_baselines3/common/buffers.py)：GAE/returns 递推。
- [SB3 2.8.0 VecNormalize](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.8.0/stable_baselines3/common/vec_env/vec_normalize.py)：reward 尺度与统计冻结。

## 8. 无法获得 500 步后 continuation 时的可行实验

对于正常到达人工截断的轨迹，持续任务的 value 满足：

$$
V^\pi(s_t)=\mathbb E_\pi\left[
\sum_{k=t}^{499}\gamma^{k-t}r_k+
\gamma^{500-t}V^\pi(s_{500})\mid s_t\right].
$$

前 500 步数据只提供第一项的样本，不能直接确定第二项。更多重复轨迹可以减少已观察回报的方差，无法消除未知尾值。若末状态也没有在其他等价完整状态的轨迹中出现并提供 outgoing transitions，且没有额外可验证的模型假设，则不存在仅凭这些数据构造无偏持续 value oracle 的办法。前文所列 MC oracle 路线因此不能作为当前可执行的主实验。

不过仍可检验一个较弱且有用的命题：**actor 更新与控制质量是否对终点 bootstrap 的估计方式敏感？**

建议的对照如下，所有组始终保留 500 步 bootstrap：

1. 固定策略、数据、归一化与 GAE 切分，保存原始 endpoint value，计算每个位置实际 K 与 beta。用已知扰动重算 target/advantage，核对传播系数。
2. 保留原 value 作为对照，对 endpoint value 加入预先规定的正/负扰动。既包含常数扰动作为机制对照，也包含固定的状态相关扰动，例如基于标准化电流偏差的平滑函数；不以重训后的分数反向选择扰动函数。
3. 从相同 actor/优化器/批次复制分支，只改变 advantage 里的 value，进行相同小幅 KL 约束的 actor-only 更新。在独立、配对的正常 500 步仿真中比较早/中/后期控制误差、正式分数和失败率。结果按重复批次/seed 报告区间。

**常数扰动不能作为唯一证据。** 当新增 advantage 只是当前状态/时间的确定函数时，理想 on-policy score-function 梯度中的期望贡献可以抵消。advantage 数值明显变化不等于期望 actor 梯度被扭曲；状态相关的未来 endpoint 误差、提前失败掩码和 PPO 有限更新也需要分别研究。

可以补充 lambda/rollout 长度与独立 critic 的敏感性对照，但它们改变其他因素或仍共享未知尾值，均不作为真值。Bellman residual 小、ensemble 一致、critic 对自身 bootstrapped target 拟合良好也不能证明末状态 value 准确。

若扰动引起稳定的控制变化，只能支持“该机制能影响当前控制结果”；若某种校正提高分数，只能证明该校正在已评价控制任务上有效。**仍不能据此证明原 critic 的真实尾值偏高/偏低，也不能把改进归因于更接近真实 V。** 这个限制会随实验结果一起报告。
