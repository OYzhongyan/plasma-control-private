# 复赛补充说明

本文件说明复赛相对初赛的主要变化、环境更新和提交差异。基础安装、仿真器启动等通用内容仍可参考 `README.md`；复赛任务、训练入口和提交差异以本文和 `submission/README.md` 为准。

复赛完整评分细节见 `智能控制挑战赛——复赛评估标准v1`。

## 1. 复赛任务说明

复赛仍基于 FGE/HFM 仿真环境，但任务目标从初赛的基础控制与维持，扩展为不同位形之间的受约束转换。控制器需要在电源响应和执行器约束下，将等离子体从初始位形转换到目标位形，并在转换过程中尽量保持电流和位形稳定。

复赛包含 2 道赛题、3 个子任务：

| 任务 | 转换目标 | 起始位形 | 目标位形 | 步数 | 分值 | 服务 |
| --- | --- | --- | --- | --- | --- | --- |
| `F1` | 偏滤器 -> 限制器 | 偏滤器类初态 | 限制器目标 | 300 | 40 | policy1 |
| `F2a` | 偏滤器 -> XPT | 偏滤器类初态 | XPT 目标 | 500 | 30 | policy2 |
| `F2b` | 限制器 -> XPT | 限制器类初态 | XPT 目标 | 500 | 30 | policy2 |

`F2a` 和 `F2b` 的目标 XPT 位形一致，包含 LCFS、主 X 点、次级 X 点、打击点和相关磁通约束；两者仅初始位形不同，独立运行、独立计分。

赛题测试使用的控制目标会以参考文件形式提供，供选手训练和调试使用。`F1` 的限制器目标参考见 `configs/f1_reference_targets.json`，`F2a/F2b` 的 XPT 目标参考见 `configs/xpt_reference_targets.json`。正式测试环境的初态与训练环境类似，具体测试配置不公开；选手应基于公开训练环境和目标参考提升泛化能力，这一点与初赛一致。

## 2. 相比初赛的主要变化

- **任务目标变化**：复赛重点考核位形转换，不只是维持初始位形或跟踪简单 reference。
- **目标位形增加**：新增 XPT 位形，除 LCFS 外，还关注主 X 点、次级 X 点、打击点和 X 点磁通。
- **初始态变化**：正式评估的初始平衡与训练环境提供的平衡不完全相同，但属于同一位形类型。
- **评分逻辑变化**：复赛新增位形类型判断、电流偏差熔断、XPT 拓扑约束和线圈电流硬约束。
- **工程模型增强**：动作进入 HFM 前会经过电源模型，包含传输延迟、电压变化率限幅和 PSM 仿射标定。建议选手训练和本地测试时自行加入 observation 噪声、延迟等扰动，检查策略鲁棒性。
- **提交服务变化**：一个提交镜像内启动两个推理服务，分别处理 `F1` 和 `F2a/F2b`。

## 3. 环境与代码更新提醒

请选手参考初赛 `README.md` 的方式更新仿真器镜像或 runtime。复赛环境考虑了新的初始态、位形判断、XPT/reference 字段、电源响应和双服务推理流程，旧环境可能无法复现实测评估行为。

本次复赛相关代码主要有以下调整。选手第一次看代码时，可以先按这个表定位入口：

| 目录 / 文件 | 作用 |
| --- | --- |
| `environment/hfm_simulator.py` | Gymnasium 环境入口。复赛 observation 增加了 LCFS 与 XPT 相关 reference 字段，`reset(options=...)` 可接收 trajectory reference。 |
| `environment/hfm_predictor.py` | HFM socket 通信层。在 `step()` 中接入电源模型，所以策略输出电压不会直接进入 HFM。 |
| `environment/power_supply.py` | 电源响应模型：传输延迟 -> 变化率限幅 -> PSM 仿射标定。训练和推理都会经过这层。 |
| `environment/preprocessing.py` | 基础预处理工具，含 7 维对称动作到 12 维动作映射、字典 observation flatten 示例。不是强制接口。 |
| `environment/xpt_utils.py` | XPT 观测辅助工具。可从原始 observation 中整理 X 点、打击点、X 点磁通、磁通梯度等，供训练特征或 reward 使用。 |
| `configs/env_default.yaml` / `configs/shots.yaml` | 本地训练环境默认配置和公开 shot 预设。正式测试配置不完全公开。 |
| `configs/f1_reference_targets.json` | `F1` 限制器目标参考 |
| `configs/xpt_reference_targets.json` | `F2a/F2b` XPT 目标参考 |
| `examples/train_f1_ppo.py` | F1 最小 PPO 训练入口，读取 `f1_reference_targets.json`，演示偏滤器类初态到限制器目标的训练流程。 |
| `examples/train_f2_ppo.py` | F2a/F2b 最小 PPO 训练入口，读取 `xpt_reference_targets.json`，演示到 XPT 目标的训练流程。 |
| `examples/semifinal_training_common.py` | F1/F2 训练脚手架共用代码，不表示共用同一个模型。它会根据目标文件是否包含 XPT 字段自动分支：F1 只使用 Ip、LCFS、I_PF 等基础特征和 Ip/LCFS reward；F2 才追加 XPT reference、X 点和打击点相关特征与 reward。 |
| `examples/example_power_supply_step.py` | 电源模型阶跃响应示例，用于理解延迟、限幅和 PSM 仿射对动作的影响。 |
| `submission/service1.py` / `submission/inference1.py` | `F1` 推理 HTTP 服务与策略入口。 |
| `submission/service2.py` / `submission/inference2.py` | `F2a/F2b` 推理 HTTP 服务与策略入口。 |
| `submission/start_infer.sh` | 评测启动脚本，固定同时启动 `service1` 和 `service2`，选手不要修改。 |
| `test/check_submission.py` | 不启动 HFM，只检查两个 submission HTTP 服务是否能 `/health`、`/reset`、`/act`。 |
| `test/test_submission.py` | 与真实 HFM 环境做少步联调，检查 submission 服务和环境闭环是否能跑通。 |
| `evaluation/eval/` | 本地离线评分代码和示例结果格式，用于理解评分输出；正式评测以评测机为准。 |

电源模型的详细公式和参数说明见 `docs/power_supply_model.md`。简要地说，策略输出的 `U_set` 会先经过延迟和变化率限制，再映射为实际送入 HFM 的 `U_real`。公开训练环境和正式推理环境都会包含该电源模型；默认延迟会按步随机采样，电压变化率限幅按当前公开配置执行。

## 4. 复赛最短路径

如果只想先跑通复赛流程，可以按下面顺序做：

1. 按 `README.md` 完成 Python 环境安装，并启动 HFM 仿真器。
2. 运行 `examples/train_f1_ppo.py` 或 `examples/train_f2_ppo.py`，确认环境、reference 和 7 维到 12 维动作映射能跑通。
3. 将训练得到的策略导出为 ONNX：`F1` 默认读取 `submission/model/policy1.onnx`，`F2a/F2b` 默认读取 `submission/model/policy2.onnx`。如果三项任务共用同一套权重，可以额外放置同一个 `submission/model/policy.onnx` 作为回退模型。
4. 必须保留并实现两个推理入口：`submission/inference1.py` 负责 `F1`，`submission/inference2.py` 负责 `F2a/F2b`。评测会固定访问两个 HTTP 服务，不能只启动一个服务。
5. 运行 `python test/check_submission.py` 检查两个 HTTP 服务的 `/health`、`/reset`、`/act` 接口；需要 HFM 联调时再运行 `python test/test_submission.py`。

## 5. 动作对称控制要求

复赛更强调后续上机部署可行性。由于真实装置中上下不对称电压控制的工程风险较高，且不对称电压可能诱发垂直不稳定，进而需要更大的 VS 电压进行修正；若 VS 电流超过限幅，评估也会产生扣分。因此强烈建议控制策略遵循上下对称线圈控制逻辑。

当前接口仍以 HFM 的 12 路物理电压为最终动作格式；推荐选手在策略内部输出 7 维动作，再通过模板映射为 12 维对称电压：

```text
v[0]  -> u[0]
v[1]  -> u[1] = u[2]
v[2]  -> u[3] = u[4]
v[3]  -> u[5] = u[6]
v[4]  -> u[7] = u[8]
v[5]  -> u[9] = u[10]
v[6]  -> u[11]
```

也就是说，评测 HTTP 服务最终仍必须返回 12 维动作；如果策略内部使用 7 维动作，需要在推理代码里自行映射回 12 维。

## 6. 推理服务与上传调整

复赛提交模板会在一个镜像内固定启动两个 HTTP 服务：

```text
submission/service1.py -> inference1.Policy -> 默认读取 submission/model/policy1.onnx
submission/service2.py -> inference2.Policy -> 默认读取 submission/model/policy2.onnx
```

评测脚本按任务路由：

```text
F1        -> service1，端口 8000
F2a/F2b   -> service2，端口 8001
```

端口和服务文件名由 `start_infer.sh` 固定，选手不要修改；否则评测脚本无法访问推理服务。模型权重可以共用，但 `service1.py` / `service2.py` 和 `inference1.py` / `inference2.py` 两套入口必须都可用。

## 7. 训练思路建议

复赛是变位形任务，初始平衡和目标状态不一致。一个可行的训练路径是先让策略学会维持目标位形附近的平衡，例如参考初赛 `A1/B1` 的 hold 思路；随后再从初始平衡出发做课程学习，逐步增加转换距离和扰动强度。

另一种路径是先控稳初始平衡，再引入目标位形 reference，引导策略从初始态逐渐过渡到目标态。两种方式都只是参考，关键是避免一开始就把“稳态维持”和“长距离位形转换”混在一起训练，导致策略难以收敛。

示例训练入口保留为 `examples/train_f1_ppo.py` 和 `examples/train_f2_ppo.py`。两者都会读取对应目标 reference 文件，构造 trajectory reference，并默认使用 7 维对称动作经模板映射到 12 路物理电压。

这些 example 的定位是“跑通复赛接口和训练工程骨架”，不是官方最优策略。选手可以保留其中的环境创建、reference 构造和动作映射逻辑，替换 reward、模型结构、特征工程、训练算法和 ONNX 导出方式。

对于 `F2a/F2b`，建议至少关注：

- 电流轨迹，避免触发电流偏差熔断。
- LCFS 整体形状和位置。
- X 点数量、主 X 点、次级 X 点位置。
- 打击点和 X 点磁通，避免只学到近似边界形状但拓扑不正确。
- 线圈电流约束，避免一次超限后后续时间片全部清零。

`F2a/F2b` 的策略 observation 会额外收到目标 XPT 控制参考：

```text
reference_rX              # 4 个目标 X 点 R 坐标
reference_zX              # 4 个目标 X 点 Z 坐标
reference_x_valid         # 目标 X 点有效槽位
reference_strike_r        # 8 个目标打击点 R 坐标
reference_strike_z        # 8 个目标打击点 Z 坐标
reference_strike_valid    # 目标打击点有效槽位
reference_nX              # 目标 X 点个数，XPT 目标通常为 4
reference_n_strike        # 目标打击点个数
```

这些字段由评测脚本从目标位形动态提取，与 `configs/xpt_reference_targets.json` 中的调试参考含义一致；该文件同时提供目标 LCFS。为兼容已有 ONNX 模型，模板的 `flatten_dict_observation()` 默认不把这些字段拼进固定输入向量；新策略如需使用，需要在 `inference2.py` 中自定义 preprocessing keys 或自行处理 observation。

注意：环境原始观测中的 `nX` 可能不是 4，打击点候选数也可能不是 8。`environment/xpt_utils.py` 提供训练/预处理辅助函数，可把 X 点整理为固定 4 槽、打击点整理为固定 8 槽，并输出 `x_valid`、`strike_valid`、`nX`、`strike_n_actual` 等辅助量。选手如果自己连接 predictor 训练，可以直接使用这些工具；正式推理时评测只负责把 observation 发给 submission，`inference2.py` 不会自动调用 `xpt_utils`，需要选手按自己的模型输入自行处理。

## 8. 评测输出

复赛推理结果固定写入：

```text
/saisresult/infer_result.json
```

`infer_result.json` 是评测脚本内部用于算分的结果文件，选手提交服务不需要手动生成其中的 trajectory 字段。选手只需要保证 `/reset` 和 `/act` 接口可用，并在 `/act` 中根据 observation 返回 12 维动作。环境 observation 字段的详细说明见 `docs/reference.md`，XPT 控制辅助说明见 `docs/XPT_CONTROL.md`。

`run_test.py` 会在每个子任务结束后打印分项得分，最后打印总分：

```text
[subtask-score] F1 ...
[subtask-score] F2a ...
[subtask-score] F2b ...
[total-score] ...
```

## 9. 本地轻量检查

不启动 HFM 时，可以只检查 submission HTTP 层：

```bash
python submission/service1.py
python submission/service2.py
```

完整评估仍需要 HFM socket server 和评测机内部的 `inference/run_test.py`。
