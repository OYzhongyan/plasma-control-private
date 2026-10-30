# Limiter Shot Reward Issue

本文档记录 F1 stage0 rollout 中暴露出的 `15892_*` limiter shot 问题。该问题对当前 F1 stage1 的直接影响有限，因为 F1 正式任务是 `13844_*` 偏滤器起点到 limiter 目标；但它对 F2b 和 `policy2` 影响显著，因为 F2b 是 limiter 起点到 XPT 目标，推荐起点正是 `15892_*`。

## 1. 背景

当前复赛任务语义：

| 子任务 | 起点 | 推荐 shot | 目标 | 模型 |
| --- | --- | --- | --- | --- |
| F1 | divertor | `13844_500`, `13844_600` | limiter shutdown | `policy1` |
| F2a | divertor | `13844_500`, `13844_600` | XPT | `policy2` |
| F2b | limiter | `15892_400`, `15892_500` | XPT | `policy2` |

因此：

- F1 stage1 不应被 `15892` hold 问题牵着走，`f1_stage1.json` 当前只使用 `13844_500/13844_600` 是合理的。
- F2b 必须解决 `15892` 控制问题，否则 `policy2` 很可能在 limiter 起点快速失稳，无法进入 XPT 转换。

## 2. 现象

使用 stage0b policy rollout 对比：

| 模式 | shot | steps | failure | reward sum | mean Ip error | max Ip error | mean LCFS RMS | max LCFS RMS | max coil margin |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model | `13844_500` | 200 | false | 149.95 | 0.44 kA | 1.46 kA | 0.72 cm | 1.18 cm | 1.32 |
| model | `13844_600` | 200 | false | 149.17 | 1.54 kA | 3.06 kA | 0.63 cm | 1.21 cm | 1.29 |
| model | `15892_400` | 22 | true | -52.14 | 46.09 kA | 93.04 kA | 6.32 cm | 13.62 cm | 1.02 |
| model | `15892_500` | 25 | true | -72.73 | 58.52 kA | 115.77 kA | 8.58 cm | 19.22 cm | 1.01 |
| zero | `15892_400` | 200 | false | -890.99 | 234.35 kA | 385.68 kA | 31.36 cm | 57.76 cm | 0.82 |
| zero | `15892_500` | 197 | true | -889.33 | 238.30 kA | 393.62 kA | 34.44 cm | 62.61 cm | 0.81 |

关键观察：

1. `15892_*` 初态是合理的，Ip 约 500 kA，`lX=0`，`reference_lX=0`，线圈不超限。
2. zero 模式不是稳定维持，而是慢漂：Ip 持续下降，LCFS RMS 持续增大，但 topology 维持 limiter，线圈 margin 低于 1，因此不容易早 fail。
3. model 模式会主动把慢漂加速成 failure，20 多步内 Ip 误差超过 90-110 kA，LCFS RMS 达到 13-19 cm。
4. model 对 `15892_*` 的动作明显不同于 `13844_*`，并且更像错误方向的强控制，而不是温和纠偏。

## 3. 已排除的代码错配

目前未发现以下代码层面的错配：

- `configs/shots.yaml` 中 `15892_400/500` 分别指向 `ini_15892_400_v2.mat` 和 `ini_15892_500_v2.mat`。
- HFM predictor 会在 INIT 阶段根据 `shot_id` 生成对应 `L_addr/LX_addr/signeo/bp/q0`，不是只保留日志标签。
- stage0b 64 并发分配均衡，4 个 shot 各 16 个环境。
- `hold_initial` reference 从 reset raw observation 复制 `Ip/R/Z/lX/lcfs_points`。
- reward 和 feature 都读取 `reference_lX`，不会把 F1 固定硬编码成错误拓扑。
- rollout CSV 初始行显示 `15892_*` 的 `reference_Ip/R/Z/lX` 与 reset 初态一致。

因此当前更合理的判断是：`15892` 暴露的是训练和 reward 设计问题，而不是 shot/target/reference 显式错配。

## 4. 15892 的物理与控制特点

### 4.1 开环慢漂

zero action 下，`15892_*` 可以接近跑完整个 episode，但 Ip 与 LCFS 会长期漂移：

- `15892_400` 到 step 200：Ip error `-385.68 kA`，LCFS RMS `57.76 cm`。
- `15892_500` 到 step 190 附近：Ip error 接近 `-390 kA`，LCFS RMS 超过 `60 cm`。

这说明 `15892` 不是自然稳定 hold；它需要主动控制，但控制必须方向正确且足够温和。

### 4.2 过强动作会快速破坏平衡

model 模式下，策略对 `15892_*` 很快输出接近饱和的动作：

- `action7_2 ≈ -1`
- `action7_3 ≈ -1`
- `action7_4 ≈ +1`
- VS 通道被快速推向负方向

对比 zero 模式，model 下 VS 和若干 PF 通道动态明显不同。以 `15892_400` step20 为例：

- zero：VS norm 约 `+0.12`
- model：VS norm 约 `-0.92`

该行为会把 Ip 下降、R/Z 偏移和 LCFS 漂移显著加速。

### 4.3 topology 不是主要问题

在 model 和 zero rollout 中，`topology_match_rate` 都是 1.0。也就是说，`15892` 的短期 failure 不是因为 `lX` 立即错了，而是 Ip、LCFS、R/Z 和线圈动态持续恶化。

## 5. Reward 根因

当前 stage0 reward 在坏状态下有一个重要问题：早 fail 可能比长时间活着但状态很差更划算。

当前结构近似为：

```text
reward =
    alive_offset
  + score_scale * normalized_score
  - ip_error_boundary_penalty
  - coil_penalty
  - action_penalties
```

其中：

- `alive_offset = -1`
- `score_scale = 2`
- Ip boundary penalty 使用 Huber + cap，cap 当前为 4
- failure penalty 当前是固定 `-10`

当 `Ip/LCFS/R/Z` score 都接近 0，并且 Ip penalty 到 cap 时，单步 reward 可接近：

```text
-1 - 4 = -5
```

因此：

- zero 模式虽然活得久，但长期处于坏状态，总 return 约 `-890`。
- model 模式 20 多步 fail，总 return 仅 `-50 ~ -70`。

从 PPO 的角度，早 fail 没有被足够强地惩罚，甚至可能比长期坏状态更优。这会削弱“慢慢修回来、不要主动打坏”的学习动力。

## 6. 对 F1 的影响

对当前 F1 stage1，直接影响有限：

- F1 正式语义是 `divertor -> limiter shutdown`。
- 当前 `f1_stage1.json` 只使用 `13844_500/13844_600`。
- `13844_*` 在 stage0b model rollout 中能稳定跑满 200 steps，Ip 和 LCFS 误差都很小。

建议：

- F1 stage1 继续以 `13844_*` 为主，不要为了修 `15892` 牺牲 F1 主任务进度。
- 如果继续使用 stage0b 作为 F1 初始化，需要重点确认它没有因混训损害 `13844_*` 分支。目前 rollout 显示 `13844_*` 表现正常。

## 7. 对 F2 的影响

对 F2，尤其是 F2b，影响显著：

- F2b 起点是 limiter，推荐 shot 是 `15892_400/15892_500`。
- F2b 目标是 XPT，过程比 hold 更难，需要先在 `15892` 上安全维持，再逐步引导到 XPT。
- 如果 policy2 在 `15892` 上学到类似当前 F1 stage0b 的错误动作，F2b 会在进入 XPT 过渡前快速 failure。

因此，F2 不能简单复用当前 F1 stage0 reward。F2 stage0/stage1 需要专门处理 `15892` 的慢漂和过强动作问题。

## 8. Reward 修复方向

### 8.1 failure penalty 与剩余步数挂钩

固定 `failure_penalty = -10` 不足以惩罚 early fail。建议改为：

```text
failure_penalty = -failure_base - failure_remaining_scale * (max_steps - step)
```

目标是保证 step 20 failure 一定比活到 step 200 且状态较差更不划算。

### 8.2 stage0 hold 使用 survival bonus

stage0 的首要目标是维持平衡，应显式鼓励活着：

```text
reward =
    survival_bonus
  + score_scale * normalized_score
  + improvement_score
  - penalties
```

可以考虑把 `alive_offset` 从 `-1` 改成 `0` 或小正值，至少在 hold 预训练阶段避免“活得久反而累计更差”的结构。

### 8.3 加入误差改善项

对 `15892` 这种慢漂系统，绝对误差 score 不足以告诉模型“动作方向是否正确”。建议加入：

```text
error_now =
    w_ip * |Ip - reference_Ip| / 50k
  + w_lcfs * lcfs_rms / 3cm
  + w_rz * position_error

improvement = error_prev - error_now
improvement_score = clip(improvement, -1, 1)
```

这会直接奖励减小误差、惩罚加速漂移。

### 8.4 加入 Ip 纠偏方向项

hold_initial 中 reference Ip 固定。若当前 `Ip < reference_Ip`，但下一步 `delta_Ip < 0`，说明动作继续把电流推离目标，应额外惩罚。

可设计为：

```text
ip_direction_penalty =
    max(0, -sign(reference_Ip - Ip) * delta_Ip / scale)
```

该项对 `15892` 很重要，因为其核心失稳模式是 Ip 单调下降。

### 8.5 增强动作幅值与平滑惩罚

zero rollout 说明 `15892` 对强动作很敏感。F2 的 limiter 起点训练应更强地约束动作：

```text
action_l2_penalty:      0.001 - 0.003
action_smooth_penalty:  0.003 - 0.010
```

并可考虑加入“偏离 zero action”的保守先验，让模型先学会小修小补，而不是一开始就大幅打电压。

### 8.6 按 shot 或任务阶段区分 reward

不建议把 `13844` 和 `15892` 完全当成同一类 hold 任务处理。更合适的结构是：

- `13844`：偏滤器起点，服务 F1/F2a 转换。
- `15892`：limiter 起点，服务 F2b 转换。

F2 训练中可以考虑：

- F2a stage：`13844_* -> XPT`
- F2b stage：`15892_* -> XPT`
- policy2 共享权重，但 rollout/reward log 必须按子任务和 shot 分开统计。

## 9. 建议后续实验

### 9.1 15892-only hold 修复课程

目的：让策略先学会在 `15892_400/500` 上不主动打坏。

建议：

- shot：只用 `15892_400/15892_500`
- reference：`hold_initial`
- perturb：先关闭
- max_steps：200
- learning rate：低于当前 stage0b
- reward：使用 survival bonus、remaining failure penalty、improvement score、较强 action penalty

### 9.2 zero policy 作为安全基线

每次训练后都跑：

```text
policy-mode = zero
policy-mode = model
```

若 model 比 zero 更早 failure，说明策略主动破坏了平衡，不能进入下一阶段。

### 9.3 分 shot 训练监控

全局 `ep_len_mean` 会被 `13844_*` 掩盖。必须按 shot 输出：

- episode length
- failure rate
- Ip error
- LCFS RMS
- coil margin
- action saturation ratio

尤其关注 `15892_400/500` 的 failure step 和动作饱和情况。

## 10. 当前结论

`15892` 问题的核心不是配置错配，而是 reward 与动态共同导致的训练偏差：

- zero action 下 `15892` 慢漂但不易早 fail。
- 当前 model 对 `15892` 施加错误方向或过强动作，快速触发 failure。
- 当前 reward 结构下，早 fail 的 return 可能比长期坏状态更高，削弱了恢复学习。

对 F1：该问题不是当前主线阻塞，F1 stage1 应继续聚焦 `13844_*`。

对 F2：该问题是必须解决的核心风险，尤其是 F2b。F2 训练设计应把 `15892` limiter 起点作为独立课程处理，并从 reward 上保证“长期安全维持和逐步改善”优于“快速失稳”。
