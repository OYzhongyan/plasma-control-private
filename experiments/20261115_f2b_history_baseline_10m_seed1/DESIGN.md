# 不同训练种子的影响

训练自变量只有 `ppo.seed`：原 baseline 为 0，本实验为 1。使用同一历史提交、依赖版本、网络、奖励、归一化、学习率计划、PPO 超参数、训练 shot、reset 扰动分布、观测延迟、电源模型和 10m 步数目标。

全局训练种子会同时改变初始化、动作探索、minibatch 排列和各环境的随机扰动流。本实验不试图将这些影响分别归因。

正式测评使用与 seed=0 相同的 seen/validation/test shot、episode 数、评估种子和传感噪声，避免同时改变测试条件。训练奖励与正式 competition 得分分别报告。

原 baseline 存在中断续训；分析连续轨迹时，同时采用 plasma-control3 上的 `20261115_f2b_history_baseline_10m_seed0_repro` 作为对照。新 seed=0、seed=1 两实例同为 55 核/500 GiB，硬件和系统信息仍单独归档，耗时不纳入逐步数值比较。

记录初始模型/归一化/RNG、配置、软件版本、代码与目标哈希、每轮 PPO 和 reward CSV、TensorBoard、W&B offline 及正式测评结果。初始记录器必须不改变训练 RNG。发现不同种子的数值差异属于预期；只有两次运行不能估计训练结果的完整方差或统计显著性。
