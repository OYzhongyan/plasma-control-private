# F1 Direct Target Reward Design

本文档总结 F1 direct target 训练任务的当前设计、已观察到的问题和下一步 reward 改进方向。对应配置主要是 `training/configs/f1_direct_target.json` 和 `training/configs/f1_direct_target_progress.json`，代码实现以 `training/rewards.py` 为准。

## 1. 任务定义

F1 direct target 是不使用课程学习、直接从官方 target 训练 shutdown 变位形策略的实验。训练从 `13844_*` divertor 初态出发，目标是安全完成 `divertor -> limiter` 位形变化，并按官方 `Ip` 曲线降流。

官方 `evaluation/results/target.json` 的 F1 target 只提供：

- 300 步 `Ip` 轨迹
- 最终 `lcfs_points`

官方 target 不提供 `R/Z` 轨迹。HFM 环境在 trajectory reference 缺少 `R/Z` 时，会用 reset 初态填充 `reference_R/reference_Z`。因此 direct target 训练中关闭：

```json
"r_score": 0.0,
"z_score": 0.0
```

否则模型会被额外拉回初始 `R/Z`，和最终 LCFS target 发生冲突。

## 2. 当前配置

`f1_direct_target.json` 是第一版 direct target 配置：

- 从零训练，`resume.enabled = false`
- `alive_offset = 0.2`
- `failure_penalty = -100.0`
- `ip_score = 0.25`
- `lcfs_absolute_score = 0.45`
- `topology_score = 0.20`
- `limiter_transition_score = 0.10`

第一版结果显示：

- early failure 被明显压住
- `Ip` 控制明显改善
- `LCFS/topology/limiter_transition` 没有启动

主要原因是原始 LCFS score 只在 3 cm 内有信号：

```text
lcfs_score = max(0, 1 - lcfs_cm / 3.0)
```

我们用 HFM reset 测到 F1 初态与官方 target 的 LCFS RMS 差异约为：

| shot | 初始 LCFS RMS error |
| --- | ---: |
| `13844_500` | 4.56 cm |
| `13844_600` | 3.64 cm |

初态已经超过 3 cm 阈值，因此从零训练时模型一开始看不到 LCFS 的连续改进信号。

`f1_direct_target_progress.json` 是第二版配置：

```json
"ip_score": 0.2,
"lcfs_absolute_score": 0.6,
"topology_score": 0.2,
"limiter_transition_score": 0.0,
"lcfs_score_mode": "progress_precision",
"lcfs_precision_cm": 3.0,
"lcfs_progress_weight": 0.7,
"lcfs_precision_weight": 0.3
```

其中 `limiter_transition_score` 被关闭，因为它等价于：

```text
topology_score * lcfs_absolute_score * ip_safe_score
```

这个项和 LCFS/topology 高度重合，并且在 LCFS/topology 稀疏时长期为 0，增加调参复杂度。

## 3. LCFS Progress + Precision

第二版把 `lcfs_absolute_score` 的实际含义改成一个统一 LCFS score。字段名保留为 `lcfs_absolute_score` 是为了兼容原有 reward 加权逻辑。

当前实现：

```text
lcfs_cm = 当前 LCFS 到官方 target LCFS 的绝对坐标 RMS error，单位 cm
lcfs_initial_cm = 当前 episode reset 后记录的初始 LCFS RMS error

precision_score = max(0, 1 - lcfs_cm / lcfs_precision_cm)

progress_scale = max(lcfs_initial_cm - lcfs_precision_cm, lcfs_progress_min_scale_cm)
progress_score = clip((lcfs_initial_cm - lcfs_cm) / progress_scale, 0, 1)

lcfs_score =
    0.7 * progress_score
  + 0.3 * precision_score

lcfs_absolute_score = lcfs_score
```

因此日志里：

```text
lcfs_score == lcfs_absolute_score
```

是预期行为。真正用于诊断的拆分字段是：

- `lcfs_cm`
- `lcfs_initial_cm`
- `lcfs_progress_score`
- `lcfs_precision_score`

`lcfs_initial_cm` 会变化，因为它是每个 episode 自己的 reset 基准。并行训练中不同 shot、不同 env、不同 episode 会被聚合到同一行日志，因此 CSV 里的 `lcfs_initial_cm` 是一个 log interval 内的均值，不是全训练固定常数。

## 4. 当前 Progress 版观察

Progress 版训练早期已经证明新字段生效：

- `lcfs_cm/lcfs_initial_cm` 有非零记录
- `lcfs_progress_score` 有非零记录
- `lcfs_precision_score` 有非零记录

但前 50 万步显示了新的问题：策略仍然倾向于先把 LCFS 推远。

典型现象：

```text
lcfs_initial_cm ≈ 4 cm
lcfs_cm ≈ 15-20 cm
lcfs_progress_score ≈ 0
```

这说明 progress reward 只解决了“变近时有奖励”，但没有解决“变远时没有额外惩罚”的漏洞。策略仍可通过：

```text
alive_offset + Ip score
```

获得相对更好的回报，而不必保持 LCFS 接近 target。

## 5. 下一版建议：有界 LCFS Regression Penalty

下一版不建议简单增大 LCFS 权重，也不建议把 `alive_offset` 改回负数。当前训练已经证明：

- 正 `alive_offset`
- 大 `failure_penalty`

能有效压制 early failure，这是应保留的成果。

更合理的做法是新增一个 penalty 类项，惩罚 LCFS 比 episode 初态更远：

```text
lcfs_regression_cm = max(0, lcfs_cm - lcfs_initial_cm - margin_cm)
lcfs_regression_ratio = lcfs_regression_cm / scale_cm
lcfs_regression_penalty = capped_huber(lcfs_regression_ratio)
```

建议初始配置：

```json
"lcfs_regression_penalty": 0.3,
"lcfs_regression_margin_cm": 1.0,
"lcfs_regression_scale_cm": 5.0,
"lcfs_regression_penalty_shape": "huber",
"lcfs_regression_penalty_cap": 1.0
```

含义：

- 比初态远 `1 cm` 以内不罚，避免数值波动导致惩罚。
- 超过初态越远，惩罚逐渐增大。
- 惩罚封顶，避免单项过大重新诱发 early failure。
- 初始权重 `0.3` 比较保守，可观察后再调。

该项应该放在总 reward 的 penalty 部分，而不是混入 `normalized_score`：

```text
reward =
    alive_offset
  + score_scale * normalized_score
  - coil_penalty_weight * coil_penalty
  - ip_error_boundary_penalty_weight * ip_error_boundary_penalty
  - lcfs_regression_penalty_weight * lcfs_regression_penalty
  - action_l2_penalty
  - action_smooth_penalty
```

这样 score 和 penalty 语义清晰：

- score 项描述“越好越高”
- penalty 项描述“越危险或越不合理越低”

## 6. 调参判据

当前 progress 版训练建议继续跑完，至少得到最终权重和完整对照结果。训练过程中重点观察：

- `ep_len_mean`
- `ep_rew_mean`
- `failure_penalty`
- `ip_abs_error`
- `lcfs_cm`
- `lcfs_initial_cm`
- `lcfs_progress_score`
- `lcfs_precision_score`
- `topology_score`
- `coil_penalty`

如果下一版加入 regression penalty，判断规则如下：

| 现象 | 解释 | 建议 |
| --- | --- | --- |
| `ep_len_mean` 下降，`failure_penalty` 变大 | 惩罚过强，重新诱发 early failure | `lcfs_regression_penalty` 从 `0.3` 降到 `0.15` |
| `ep_len_mean` 上升，但 `lcfs_cm` 仍到 15-20 cm | 惩罚过弱 | `lcfs_regression_penalty` 从 `0.3` 升到 `0.5` |
| `lcfs_cm` 不再显著超过 `lcfs_initial_cm`，但 `progress_score` 不上升 | 模型学会不推远，但未学会靠近 | 继续提高 LCFS score 权重或降低 Ip 权重 |
| `topology_score` 长期为 0 | LCFS 约束仍不足以触发 limiter 语义 | 后续再考虑 topology soft proxy |

当前阶段建议只修 LCFS 变远漏洞，不同时引入 topology soft proxy。否则变量太多，难以判断哪一项导致改进或退化。

## 7. 当前结论

F1 direct target 的阶段性结论：

1. 正 `alive_offset` 和大 `failure_penalty` 是有效的，可以显著降低 early failure。
2. 仅靠 3 cm LCFS 精度奖励过于稀疏，direct target 从零训练看不到初始 shaping 信号。
3. Progress + precision 能提供“变近有奖励”的信号，但还不足以阻止策略把 LCFS 推远。
4. 下一版 reward 应增加有界 LCFS regression penalty，同时保留当前 survival/failure 设计。
5. 当前 progress 版训练仍值得跑完，用于形成完整对照和最终权重，即使结果不好也有分析价值。
