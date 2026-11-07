# 自包含 ONNX 控制器提交包

## 目的与已选方案

选手可修改观测特征、归一化、网络和动作处理；提交一个 ZIP。主办方不运行上传的 Python、pickle 或训练代码，测评运行链路不得导入 `train`。用户已选择将处理逻辑和网络一起导出到 ONNX。

考虑过的方案：固定主办方的 90 维 reference wrapper 最简单，但不能覆盖选手自行改变处理逻辑的需求；上传 Python／Docker 控制器可以容纳任意代码，但不符合不执行选手 Python 的目标。采用自包含 ONNX，要求控制器计算可转换为 ONNX 支持的张量运算。

## 责任边界

- 选手本地：训练、读取 checkpoint／VecNormalize、构建并导出完整控制器。
- ONNX：观测特征提取、归一化、策略网络、动作裁剪／缩放／delta 累加、7→12 维转换及可选历史／循环状态。
- 主办方：从平台下载、安全解包、通用 ONNX CPU 推理、提供真实仿真观测、运行电源模型与 HFM、记录实际轨迹并正式评分。
- 仿真由主办方驱动；控制器不直接连接 socket 或发起网络请求。电源传输延迟、变化率限制和 PSM 不允许被提交包替代或重复施加。

## 提交格式与接口（实施提案）

```text
submission_model.zip
├── controller.onnx
└── controller.json
```

归一化统计量作为 ONNX 常量封装，不再单独上传 NPZ。旧的三文件 policy2 包不自动转换／兼容，必须重新导出。

用户在解释后批准改为按需字段，而非固定输入全部 4803 维：`controller.json` 按顺序声明需要的原始观测字段及尺寸，ONNX 直接以这些字段名作为输入。可用字段与 HFMSimulator 公开 observation space 一致，固化在不依赖 train 的 `submission/controller_contract.py`。服务器只增加 batch 轴并转换 float32；不做特征计算、归一化或非有限值替换。未声明字段不传递；必需字段缺失或尺寸不符时明确报错，不静默补零。

ONNX 按需观测输入和可选状态的含义如下：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| 各原始字段名 | `[1, *字段尺寸]` | 模型声明的原始观测，float32 |
| `state`（可选） | `[1, S]` | 前一步控制器状态，每回合全部清零 |
| `action` | `[1, 12]` | 电源模型之前的物理电压设定值，单位 V |
| `next_state`（可选） | `[1, S]` | 下一步状态，服务器仅保存并回传给下一步 |

action 必须为 float32。状态 dtype 可为 float32／float64，以保持训练算法的数值语义；baseline delta 使用 float64 保持与原 wrapper 累加一致。batch 可声明为动态但实际恒为 1。`S` 为 0 时不需要 state/next_state，否则必须为正整数且不超过 16384。只有推理及输出检查全部成功后才更新状态。沿用加载时零输入探测，模型须输出正确形状和有限值。

`controller.json` 记录 version=1、interface=hfm_controller_v1、observations=[{name, shape}]、state_dim、state_dtype、action_dim=12、action_units=voltage_setpoint、reset_state=zeros。服务端校验这些传输约束，不校验或重建选手内部 feature_mode、90 维特征、动作模式和网络架构。

## 导出与自定义

保留现有导出 CLI 的 `--model`、`--vecnormalize`、`--bundle` 用法，默认自动生成单 ZIP。默认导出器用 Torch 张量运算实现当前 baseline 的 reference 特征、冻结 VecNormalize、确定性 actor、absolute／delta 动作；保证与原 PPO 推理一致。

提供选手本地可修改的控制器导出模块，以及可选 `--controller-factory module:function`，返回 ControllerSpec(module, observations, state_dim=0, state_dtype="float32")。工厂仅在选手本地执行，接收训练模型、训练元数据及归一化数据。所有自定义处理均须体现在返回模块的 forward 中。任意 NumPy／Python wrapper 不会自动被 ONNX 捕获；选手改 wrapper 后需要同步其张量实现并重新导出。

自定义工厂可使用不同的内部输入维数与特征元数据；当前 90 维 reference 校验仅属于默认 baseline 工厂，不作为服务器或自定义工厂的限制。自定义 controller 的输入／输出仍须遵守原始字段及 12 维电压协议；非 SB3 算法也可自行导出满足协议的 ONNX，再用打包工具生成 ZIP。

导出器进行 ONNX 结构检查和 Torch／ONNX 多步数值一致性检查，覆盖状态重置。模型包只包括 ONNX 与 manifest，不包括工厂或训练源文件。

## 测评端与资源限制

通用控制器 loader 替换 reference adapter 推理链路，`submission` 运行时不导入 train。模型 ZIP、ONNX 外部张量、额外文件、CPU 线程、内存、加载／推理超时等检查继续保留。评分规则、shot、官方目标、扰动、电源延迟和固定种子保持主办方控制。

移除 ModelGrader 对 train 文件的运行依赖与指纹依赖。新格式使用新缓存目录 `server_evaluator/runtime-controller-v1`；旧结果保留，已回传失败的任务仍需重新提交／平台重置。

## 分支与文档

main 更新选手导出模块、独立推理 runtime、打包工具、测试及提交说明。server/offline-grading 使用同一通用 runtime／契约，更新 loader、审计配置、测试与服务器部署说明。沿用用户要求，在根目录切换分支，不创建 worktree；两分支的改动分别保存本地提交，不自动推送。

## 验收

1. 当前 baseline 在 absolute、delta 两种模式下，导出包与 PPO＋原训练 wrapper 多步动作一致，含非恒等归一化和重置。
2. 自定义特征、动作转换及状态可以封装，服务器只按通用接口运行。
3. 在禁止导入 train 的进程中，模型加载、推理、完整替身仿真评分链路成功；修改／移除 train 不改变该包动作与测评配置指纹。
4. 非法 manifest、尺寸、非有限输出、Python 文件、额外 ZIP 内容、外部 ONNX 张量均被拒绝；保留基础设施错误与选手文件错误区分。
5. 测试使用替身 HFM；真实服务器 HFM 联调与平台重新提交需另行在部署环境执行。
