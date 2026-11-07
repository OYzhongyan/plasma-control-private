# 完整控制器提交包

选手提交一个 `submission_model.zip`，根目录只有 `controller.onnx` 和 `controller.json`。ONNX 包含特征处理、归一化、策略网络和动作转换；测评端不导入 `train`，也不执行包内 Python。原始观测由主办方 HFM 提供，输出是电源模型之前的 **12 维电压设定值（V）**。

## 默认 baseline 导出

在项目根目录运行，checkpoint 也使用同一命令，并传入同一步数的归一化文件：

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl \
  --out submission/model/controller.onnx \
  --bundle results/submission_model.zip
```

默认控制器在 ONNX 内排列 90 维 reference、应用冻结 VecNormalize、运行确定性 actor，并执行 absolute／delta 动作处理和 7→12 维映射。delta 状态用 float64 保持与训练时累加一致；absolute 不需要状态。导出时检查 Torch／ONNX 的多步输出和状态重置一致性，成功后打印 `Model package ready`。归一化统计量成为 ONNX 常量，不单独上传 NPZ。

## 原始观测与可选状态接口

可用原始字段及尺寸见 [controller_contract.py](controller_contract.py) 中 `OBSERVATION_SHAPES`。控制器只声明自己需要的字段，不要求输入全部 4803 个观测数值。服务器只增加 batch 轴并转换 float32，不做特征筛选、计算误差、归一化或动作解码。

例如只使用电流和位置的无状态接口说明：

```json
{
  "version": 1,
  "interface": "hfm_controller_v1",
  "observations": [{"name": "Ip", "shape": [1]}, {"name": "R", "shape": [1]}],
  "state_dim": 0,
  "state_dtype": "float32",
  "action_dim": 12,
  "action_units": "voltage_setpoint",
  "reset_state": "zeros"
}
```

ONNX 输入名称必须与字段名称一致，形状为 `[1, *字段尺寸]`，dtype 为 float32；输出 `action` 形状 `[1, 12]`、dtype float32。如果 `state_dim>0`，增加输入 `state` 和输出 `next_state`，形状均为 `[1, state_dim]`，dtype 由 `state_dtype` 指定（float32 或 float64），维数最多 16384。服务器每回合清零状态，只有推理和所有输出检查成功后才更新状态。动作和状态必须有限。加载时用零输入探测接口，图须能返回形状正确的有限输出；除零等操作需在图内处理。

## 自定义特征、模型和动作转换

默认工厂为 [train/controller_export.py](../train/controller_export.py) 的 `build_controller()`。修改 wrapper 后，需要同步其可导出的张量实现；不能把任意 Python／NumPy 函数直接塞入 ONNX。

也可在本地创建 `train/my_controller.py`，工厂接口如下：

```python
import torch
from train.controller_export import ControllerSpec

class Controller(torch.nn.Module):
    def forward(self, Ip, R, state):
        # 此处实现自己的特征、网络和电压转换；示例仅说明接口。
        voltage = (Ip * 0.001 + R + state).expand(-1, 12)
        return voltage, state + R

def build(model, metadata, normalization):
    return ControllerSpec(Controller(), ("Ip", "R"), state_dim=1)
```

`model` 是本地加载的 SB3 PPO，`metadata` 是训练元数据，`normalization` 为冻结统计量字典或 `None`。工厂可以使用不同的内部特征维数，无需使用默认 90 维布局；所有控制计算都须在返回模块的 `forward()` 中。张量参数顺序是声明的观测字段顺序，最后才是可选状态。仅声明实际使用的字段，避免导出时被剪枝而与 manifest 不符。

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --metadata /path/to/training_metadata.json \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl \
  --controller-factory train.my_controller:build \
  --bundle results/submission_model.zip
```

此工厂仅在选手本地运行，不放进 ZIP。使用其他训练算法时，可自行导出符合接口的 ONNX 和 manifest，再执行：

```bash
python submission/model_bundle.py --model-dir /path/to/controller --output results/submission_model.zip
```

## 本地推理服务（开发用途）

```bash
pip install -r submission/requirements.txt
python submission/service2.py
```

默认读取 `submission/model/controller.onnx` 与 `controller.json`，可用 `SUBMISSION_MODEL_DIR` 覆盖目录。HTTP 仍提供 `/health`、`/reset`、`/act`，默认端口 8001；正式平台提交不需要启动该服务。每个 episode 前调用 `/reset`。

Docker 推理镜像不再复制 `train`；构建上下文仍包含主办方的环境和评测工具。物理电源延迟、变化率限制、PSM、官方目标和评分均由主办方控制，不应再封装进选手控制器。
