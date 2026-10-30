# F1 Stage0 Reward Design

本文档总结当前 F1 stage0a/stage0b 的奖励函数设计，便于查看训练目标、每一项奖惩含义和当前配置权重。代码实现以 `training/rewards.py` 为准，配置以 `training/configs/f1_stage0a_parallel_hold_no_perturb.json` 和 `training/configs/f1_stage0b_parallel_hold_perturb.json` 为准。

## 1. Stage0 训练目标

F1 stage0 不是训练完整 shutdown，也不是训练 `divertor -> limiter` 过渡，而是先训练多位形的平衡维持能力。

| 阶段 | 配置文件 | 目标 | 最大步数 | reset 扰动 |
| --- | --- | --- | --- | --- |
| stage0a | `f1_stage0a_parallel_hold_no_perturb.json` | 多 shot 并行维持初始平衡 | 100 | 关闭 |
| stage0b | `f1_stage0b_parallel_hold_perturb.json` | 从 stage0a 继续，扰动初始平衡后维持 | 200 | 开启 |

两个阶段都使用 `reference_mode = hold_initial`。环境 reset 后，当前初态会被构造成整段 reference，包括：

- `reference_Ip`
- `reference_R`
- `reference_Z`
- `reference_lX`
- `reference_lcfs_points`

因此 stage0 的 reward 语义是“维持当前 shot 自身的平衡”，不是强行把所有 shot 拉到 F1 最终 limiter target。

## 2. Shot 分布

F1 stage0 使用两类 shot：

| 组 | shots | 语义 |
| --- | --- | --- |
| `divertor_hold` | `13844_500`, `13844_600` | 偏滤器类初态维持 |
| `limiter_hold` | `15892_400`, `15892_500` | 限制器类平衡维持 |

并行环境会按 shot 列表轮转分配。64 个环境下，4 个 shot 会近似均匀分布。

## 3. 总奖励公式

当前配置显式提供了 `reward_weights`，所以使用加权归一化 score 版本：

```text
normalized_score = sum(weight_i * score_i) / sum(weight_i)

reward =
    alive_offset
  + score_scale * normalized_score
  - coil_penalty_weight * coil_penalty
  - ip_error_boundary_penalty_weight * ip_error_boundary_penalty
  - action_l2_penalty
  - action_smooth_penalty
```

终止或 failure 时直接返回：

```text
reward = failure_penalty
```

当前 `alive_offset = -1.0`，`score_scale = 2.0`。如果各项 score 都接近 1 且没有惩罚，单步 reward 约为 `+1.0`；如果 score 很差且没有 failure，单步 reward 会接近 `-1.0` 或更低。

## 4. Score 项

### 4.1 Ip score

```text
ip_abs_error = abs(Ip - reference_Ip)
ip_rel_error = ip_abs_error / max(abs(reference_Ip), 1)
ip_score = max(0, 1 - ip_rel_error / 0.05)
```

含义：鼓励电流跟踪 reset 初态电流。`5%` 相对误差对应 score 降到 0。

stage0a 权重较高，stage0b 略低：

| 阶段 | `ip_score` 权重 |
| --- | --- |
| stage0a | 0.45 |
| stage0b | 0.40 |

### 4.2 Ip boundary penalty

复赛评分中 `|Ip - Ip_ref| > 50 kA` 会触发电流熔断。stage0 reward 用 soft/hard 边界提前惩罚：

```text
soft_limit = 40 kA
hard_limit = 50 kA
ratio = max(0, (ip_abs_error - soft_limit) / (hard_limit - soft_limit))
```

当前采用 Huber + cap：

```text
if ratio <= delta:
    raw_penalty = ratio^2
else:
    raw_penalty = 2 * delta * ratio - delta^2

ip_error_boundary_penalty = min(raw_penalty, cap)
```

当前参数：

| 参数 | 值 |
| --- | --- |
| `ip_error_soft_limit_a` | 40000 |
| `ip_error_hard_limit_a` | 50000 |
| `ip_error_boundary_penalty_shape` | `huber` |
| `ip_error_boundary_huber_delta` | 1.0 |
| `ip_error_boundary_penalty_cap` | 4.0 |
| `ip_error_boundary_penalty` 权重 | 1.0 |

设计原因：避免无限平方惩罚在严重失控时压制所有 shaping 信号，同时在 40-50 kA 附近保留明显的边界感。

### 4.3 R/Z score

```text
r_score = max(0, 1 - abs(R - reference_R) / 0.03)
z_score = max(0, 1 - abs(Z - reference_Z) / 0.03)
```

含义：训练径向和垂直位置稳定能力。复赛评分没有单独 R/Z 项，但 stage0 是平衡维持预训练，保留低权重 R/Z 有助于学到基础控制。

| 阶段 | `r_score` | `z_score` |
| --- | --- | --- |
| stage0a | 0.10 | 0.10 |
| stage0b | 0.10 | 0.10 |

### 4.4 LCFS absolute score

```text
lcfs_error_cm = RMS(||lcfs_points - reference_lcfs_points||) * 100
lcfs_absolute_score = max(0, 1 - lcfs_error_cm / 3.0)
```

含义：使用装置绝对坐标下的 LCFS RMS 误差，不做中心平移对齐。`3 cm` RMS 误差对应 score 降到 0。

| 阶段 | `lcfs_absolute_score` 权重 |
| --- | --- |
| stage0a | 0.30 |
| stage0b | 0.35 |

stage0b 增大 LCFS 权重，是因为模型已经具备基础平衡能力后，需要更强调形状维持。

### 4.5 Topology score

```text
target_lX = reference_lX
topology_score = 1.0 if round(lX) == round(reference_lX) else 0.0
```

含义：stage0 使用 reset 初态给出的 `reference_lX`，而不是硬编码 F1 必须为 limiter。

这点很关键：

- `13844_*` hold 时，模型应维持偏滤器语义。
- `15892_*` hold 时，模型应维持限制器语义。
- stage0 不训练 `divertor -> limiter` 转换。

| 阶段 | `topology_score` 权重 |
| --- | --- |
| stage0a | 0.05 |
| stage0b | 0.05 |

### 4.6 Coil penalty

```text
coil_margin = max(abs(I_PF) / COIL_LIMITS_A)
coil_penalty = max(0, coil_margin - 1)
```

线圈限值：

| 线圈 | 限值 |
| --- | --- |
| CS | 45 kA |
| PF1-PF10 | 14 kA |
| VS | 4 kA |

当前权重：

| 阶段 | `coil_penalty` 权重 |
| --- | --- |
| stage0a | 0.50 |
| stage0b | 0.50 |

注意：复赛评分中线圈超限是强硬约束，第一次超限后该步及之后清零。reward 中的 coil penalty 是提前 shaping，但最终还应尽量通过动作/电源约束和训练分布降低超限概率。

### 4.7 Action penalties

动作惩罚基于传入环境的物理 12D 电压动作：

```text
action_l2 = mean((action / 500)^2)
action_smooth = mean(((action - prev_action) / 500)^2)

action_l2_penalty = action_l2_weight * action_l2
action_smooth_penalty = action_smooth_weight * action_smooth
```

| 阶段 | `action_l2_penalty` | `action_smooth_penalty` |
| --- | --- | --- |
| stage0a | 0.0 | 0.0 |
| stage0b | 0.0002 | 0.001 |

stage0a 暂时关闭动作惩罚，是为了先让模型学会基本平衡维持。stage0b 恢复较小动作惩罚，用于抑制扰动训练下的过激电压和高频抖动。

## 5. Stage0a 与 Stage0b 的差异

| 项 | stage0a | stage0b |
| --- | --- | --- |
| 目标 | 无扰动平衡维持 | 微扰动后平衡维持 |
| `max_steps` | 100 | 200 |
| resume | 不启用 | 从 stage0a latest 继续 |
| reset perturb | 关闭 | 开启 |
| `ip_score` | 0.45 | 0.40 |
| `lcfs_absolute_score` | 0.30 | 0.35 |
| action penalty | 关闭 | 小权重开启 |

stage0b 的 reset 扰动：

| 参数 | 范围 |
| --- | --- |
| `signeo_scale_range` | `[0.98, 1.02]` |
| `bp_delta_range` | `[-0.005, 0.005]` |
| `q0_delta_range` | `[-0.03, 0.03]` |

当前实现每次 reset 从 `signeo/bp/q0` 候选中随机扰动一个量，避免三个参数同时变化导致初态落入不合理或难求解区域。

## 6. Reward log 解读

训练时 `RewardComponentLoggerCallback` 会把各项写入：

```text
results/<stage>/logs/reward_components.csv
```

重点看以下列：

| 列 | 含义 | 诊断建议 |
| --- | --- | --- |
| `reward` | 最终单步 reward | 与 `monitor.csv` 的 episode reward 联合看 |
| `ip_abs_error` | Ip 绝对误差 | 是否长期接近或超过 50 kA |
| `ip_error_boundary_ratio` | 40-50 kA 边界区归一化误差 | 大于 1 表示已越过 hard limit |
| `ip_error_boundary_penalty` | Huber+cap 后的边界惩罚 | 是否过大压制其他项 |
| `lcfs_absolute_score` | LCFS 绝对误差 score | 是否随训练上升 |
| `topology_score` | `lX` 是否匹配初态 reference | 是否频繁丢拓扑 |
| `coil_penalty` | 线圈超限 margin | 是否出现硬约束风险 |
| `action_l2`, `action_smooth` | 动作幅值和平滑程度 | stage0b 后用于诊断动作抖动 |

## 7. 当前设计取舍

1. Stage0 保留 R/Z 项，是因为平衡维持阶段需要学习基础位置控制；进入 F1 stage1 后，R/Z 权重应谨慎下调或置零，避免与最终 LCFS 绝对目标冲突。
2. LCFS 使用绝对坐标 RMS，对齐复赛评分；centered shape 只能作为辅助特征，不能替代 reward。
3. Ip 边界惩罚使用 Huber + cap，而不是无限平方项，避免失控 episode 中 reward 被单项惩罚完全主导。
4. Stage0 topology 使用 `reference_lX`，不硬编码 F1 limiter，保证 divertor 和 limiter shot 都能进行合理 hold 预训练。
5. Stage0b 只做微扰动，目标是扩展初始分布而不是直接追求最终任务泛化。

## 8. 后续调参方向

- 如果 episode 长度上不去，优先检查 `failure`、`ip_abs_error` 和 `coil_penalty`，不要只看总 reward。
- 如果 `ip_error_boundary_penalty` 长期很大，说明 Ip 熔断风险主导训练，可考虑提高 Ip 控制能力或缩小扰动，而不是继续提高 LCFS 权重。
- 如果 `lcfs_absolute_score` 长期不上升但 Ip 已稳定，可以适度提高 LCFS 权重，或降低 R/Z 权重。
- 如果 stage0b 出现动作抖动，再逐步提高 `action_smooth_penalty`，不建议一次提高过多。
