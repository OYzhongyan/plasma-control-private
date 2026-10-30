# Submission 提交说明

本目录是**推理提交模板**：评测只与这里的推理服务交互，不运行训练代码。提交物为**一个 Docker 镜像**，需推送到**阿里云 ACR 容器镜像服务**，地域请选**乌兰察布**。具体提交入口以赛事方最新通知为准。

---

## 选手可以改 / 不可以改

| 可改 | 不可改 |
|------|--------|
| `inference1.py` / `inference2.py`（你的策略逻辑） | `Dockerfile`（路径、工作目录等） |
| `model/policy1.onnx` / `model/policy2.onnx`（你的模型） | `run.sh`、`start_infer.sh`（评测启动脚本） |
| `requirements.txt`（仅增加依赖，勿删已有） | 容器内路径：`/saisdata/40`、`/saisresult`、`/app` 等 |
| `service1.py` / `service2.py` 仅在有把握时微调 | `run_hfm_socket_server.sh`、`run_test.py` 由评测机提供，路径不可改 |

---

## 目录结构

```text
submission/
├── Dockerfile              # 正式提交用（选手勿改）
├── start_infer.sh          # 推理评测启动脚本（选手勿改）
├── run.sh                  # 赛事方手动启动时执行 /app/run.sh（选手勿改）
├── entrypoint_platform.sh  # 旧版脚本，保留备份
├── entrypoint_platform.sh.bak
├── README.md
├── inference1.py           # F1 策略入口，默认读取 model/policy1.onnx
├── inference2.py           # F2a/F2b 策略入口，默认读取 model/policy2.onnx
├── requirements.txt        # 选手可追加依赖
├── service1.py             # F1 HTTP 服务层，一般不需改
├── service2.py             # F2a/F2b HTTP 服务层，一般不需改
└── model/
    ├── policy1.onnx        # F1 模型，选手提供
    ├── policy2.onnx        # F2a/F2b 模型，选手提供
    └── policy.onnx         # 可选通用模型，缺少专用模型时回退使用
```

---

## 评测时容器里会发生什么

赛事方挂载好数据后，容器内会依次：

1. 启动**仿真服务**（评测机提供的 `run_hfm_socket_server.sh`，路径固定）
2. 启动**你的 submission 服务**（`service1.py` 端口 8000，`service2.py` 端口 8001）
3. 运行**评测脚本**（评测机提供的 `run_test.py`，路径固定）

结果固定写入 `/saisresult/infer_result.json`，日志在 `/saisresult/*.log`。这些路径与脚本**不可修改**。

---

## 模型与接口

- **推荐**：将 `F1` 模型导出为 `submission/model/policy1.onnx`，将 `F2a/F2b` 模型导出为 `submission/model/policy2.onnx`。
- **可选**：如果三个子任务共用同一套权重，可以放置同一个 `submission/model/policy.onnx` 作为回退模型。
- **必须**：`inference1.py` 和 `inference2.py` 中实现 `Policy` 类，包含：
  - `__init__(self, model_dir=None)`
  - `reset(self)`：每个 episode 开始时调用一次，清空内部状态
  - `act(self, observation: dict) -> np.ndarray`：返回 12 维真实电压动作

`observation` 中已包含当前步的 reference 等字段，无需自己生成 reference。其余预处理、网络结构等可自由实现，只要接口与返回 shape 符合要求即可。
`environment/preprocessing.py` 仅为模板示例，不是必选依赖；你可以完全使用自己的预处理实现。

---

## HTTP 接口（由 service1.py / service2.py 提供）

- `GET /health`：返回 `{"ok": true}`
- `POST /reset`：开始新 episode，内部会调 `Policy.reset()`
- `POST /act`：传入当前 observation，返回 `{"ok": true, "action": [12 维]}`

## 复赛双服务

复赛在同一个提交镜像中启动两个策略服务：

- `service1.py` / `inference1.py`：用于 `F1`，默认读取 `model/policy1.onnx`
- `service2.py` / `inference2.py`：用于 `F2a`、`F2b`，默认读取 `model/policy2.onnx`

`start_infer.sh` 会同时启动两个服务，端口固定为 `8000` 和 `8001`。评测脚本会将 `F1` 路由到第一个服务，将 `F2a/F2b` 路由到第二个服务。选手不要修改服务文件名、端口或启动脚本；模型权重可以共用，但两个 HTTP 服务必须都能启动并响应。

---

## 构建与提交

**必须基于赛事提供的官方基础镜像构建，不得修改 Dockerfile 中的基础镜像、工作目录和路径。**

在 `fusion-control-comp` 目录下执行（替换镜像名为你的 ACR 地址）：

```bash
cd <fusion-control-comp 所在路径>
docker build \
  --build-arg BASE_IMAGE=crpi-q4qg69o2szruzmaa.cn-beijing.personal.cr.aliyuncs.com/enn_smart/competition_test:r2025a-py311-v1 \
  -f submission/Dockerfile \
  -t <你的 ACR 仓库>/<镜像名>:<标签> .
```

推送至阿里云 ACR（**地域选乌兰察布**）：

```bash
docker login --username=<你的用户名> <ACR 注册表地址>
docker push <你的 ACR 仓库>/<镜像名>:<标签>
```

提交入口与赛题链接以赛事方公布为准。天池 Docker 提交教程可作参考（非本赛专属链接）：<https://tianchi.aliyun.com/competition/entrance/531863/customize253>。

---

## 本地自测（仅测 submission 服务）

不跑完整评测、只测 HTTP 接口时，可以直接运行双服务检查脚本：

```bash
python test/check_submission.py
```

该脚本会检查 `service1.py` 和 `service2.py` 的 `/health`、`/reset`、`/act`。需要真实 HFM 环境联调时，再运行 `python test/test_submission.py`。

---

## 注意事项

- 评测环境为 CPU-only，勿依赖 GPU 或交互界面。
- `model/policy1.onnx` / `model/policy2.onnx` 需自行导出并放入镜像；若共用模型，可放 `model/policy.onnx`。
- 镜像内 `/health`、`/reset`、`/act` 必须可用，且行为符合上述约定。
