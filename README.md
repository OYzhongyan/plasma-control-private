# 1. 主办方远程推理测评（手册 4.5）

此分支 `server/offline-grading` 部署在测评服务器 **C**；选手使用 `main` 分支在自己的服务器 **A** 运行 checkpoint 模型，并在平台 **B** 上传一个 `endpoint.json`。C 主动拉取该文件，在 C 启动 HFM，通过 HTTP 逐步向 A 发送观测、接收 **12 维物理电压设定值**，最后由 C 计算七项分数并回传 B。队名、提交历史和排行榜由 B 管理。

C 不下载或执行选手的模型、Python 代码或 wrapper，不接收选手自报的轨迹／分数。模型及其特征编码、冻结归一化和动作处理均在 A 执行。C 保存正式 shot、初始化文件和目标文件；A 能看到控制所需观测及 reference，但不会收到 shot 文件或其路径。

**正式远程流程不需要 ONNX。** 各方使用的文件和职责如下，与 `main` 的提交说明一致：

| 位置 | 保存的文件 | 负责的操作 |
| --- | --- | --- |
| A：选手服务器 | PPO checkpoint `.zip`、匹配的 `VecNormalize.pkl`、训练代码和 wrapper | 加载模型，处理观测，返回电压 |
| B：比赛平台 | 选手上传的 `endpoint.json`、提交记录和成绩 | 管理队伍、提交与排行榜 |
| C：测评服务器 | 官方目标、shot、HFM、可信评分代码及评测记录 | 拉取连接文件，运行仿真并回传分数 |

A 的 baseline 网络输出 7 维动作，经 A 的动作 wrapper 转为 **12 维物理电压设定值**（CS、PF1–PF10、VS）。C 接收这 12 维电压，再施加统一电源模型；C 不依赖 A 内部的观测维数或动作编码。

以下命令在 C 的 **Linux / WSL Bash、项目根目录**执行。

# 2. 安装与准备

```bash
git switch server/offline-grading
pip install -r server_evaluator/requirements.txt
```

主办方准备完整的 `hfm-runtime-release-sealed-latest`、官方 `evaluation/results/target.json`、`configs/env_default.yaml`、`configs/shots.yaml`、`eval_files/references.yaml` 和 `eval_files/obs_perturb.yaml`。正式 shot 由 C 的 `eval_files/references.yaml` 中 `shot_id` 决定，选手提交不能覆盖它。

先只检查配置（不登录平台、不连接 A 或 HFM）：

```bash
python server_evaluator/run.py --target evaluation/results/target.json --check-config
```

# 3. 启动 C 的 HFM

单独开一个终端，假设运行包位于项目同级目录：

```bash
LISTEN_PORT=5558 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash ../hfm-runtime-release-sealed-latest/run_hfm_server.sh
```

等待 `Waiting for a client to connect...`，保持运行。该端口只能由本评分器使用，不要让训练占用。也可以在评分命令添加 `--hfm-runtime-dir ../hfm-runtime-release-sealed-latest`，由评分进程启动并在退出时关闭 C 的 HFM；这时不要再手动启动同端口服务。自动启动日志保存为数据目录中的 `hfm-runtime.log`。

# 4. A 与 C 联调

选手启动 A 并生成连接文件的方法见 [submission/README.md](submission/README.md)。管理员将测试用连接文件放在 C，例如 `results/endpoint.json`，执行：

```bash
python server_evaluator/run.py \
  --endpoint-file results/endpoint.json \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-remote-v1
```

该步骤在 C 完成最多 500 步仿真和评分，不回传平台。连接文件只允许 `version`、`protocol`、`service_url` 和可选 `access_token`；不能指定目标、shot、步数或 HFM 地址。文件上限 **64 KiB**。

选手通过生成器得到的文件形式如下；`service_url` 必须是 **C 能访问的 A 服务根地址**：

```json
{
  "version": 1,
  "protocol": "hfm-remote-policy-v1",
  "service_url": "http://10.0.0.12:8001",
  "access_token": "YOUR_AGENT_TOKEN"
}
```

`8001` 是 A 的推理 HTTP 端口，`5558` 是 C 的 HFM socket 端口，两者分别配置。C 调用 A 的 `GET /health`、`POST /reset` 和 `POST /act`；reset 使用 C 生成的回合编号，act 从第 0 步开始交换观测和电压。重复请求由 A 缓存去重，正式测评期间不要从其他程序手动 reset。

默认允许公网服务。若 A 使用私网地址，**由 C 管理员**添加精确允许项：

```bash
python server_evaluator/run.py \
  --endpoint-file results/endpoint.json \
  --target evaluation/results/target.json \
  --allow-agent-origin http://10.0.0.12:8001
```

`--allow-agent-origin` 可重复，必须与连接文件中的协议、主机、端口一致。它只是允许 C 连接该私网服务，不提供网络隧道；仍需开放防火墙并保证路由可达。`127.0.0.1` 在 C 上指 C 自己，不能指向另一台选手机器。链路本地、组播和未指定地址始终禁止。公网通信建议用 HTTPS 反向代理保护 A 的专用令牌。

# 5. 配置平台及评分

平台课题允许上传 **JSON**，选手上传 `endpoint.json`，而非模型 ZIP 或轨迹 JSON。设置七个 Metric，名称区分大小写，平台权重均为 **1**：

| Metric | 贡献分上限 |
| --- | ---: |
| Ip | 6 |
| LCFS | 6 |
| X | 4 |
| strike | 3 |
| psiX | 4 |
| X2 | 3 |
| psiX2 | 4 |

平台综合分公式：

```text
[0:Ip]+[0:LCFS]+[0:X]+[0:strike]+[0:psiX]+[0:X2]+[0:psiX2]
```

回传值已经包含各项权重、500 步时间平均及正式约束惩罚，直接相加，满分 **30**，按分数降序排名。提前终止的剩余步贡献为零。网络往返时间只作审计，不套用 0.5 ms 模型耗时扣分。

在 C 设置助教凭据，从平台 API / CLI 帮助核实课题 ID：

```bash
export PLATFORM_USERNAME='YOUR_TEACHER_ACCOUNT'
export PLATFORM_PROBLEM_ID='6ae84a300a686d1bbf619417'
read -rsp '平台助教密码: ' PLATFORM_PASSWORD
echo
export PLATFORM_PASSWORD
```

也可设置有效 `PLATFORM_TOKEN`。平台默认 `https://competition.opensii.ai`，通过 `PLATFORM_BASE_URL` 覆盖。A 的访问令牌与平台凭据完全独立；不要互相混用。

# 6. 拉取、测评、回传

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-remote-v1 \
  --once
```

确认后常驻运行：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-remote-v1 \
  --poll-interval 15
```

内网 A 的允许项也要加到正式常驻命令。程序串行处理 `offline,manual` 待评提交，每份结果先落盘再回传；回传失败不会重新仿真。A 应持续运行服务，一个服务只支持一个活动回合，不要在测评中手动 reset。提交后 A 连接失败、鉴权失败、非法电压或推理超时记该提交 `failed`；C 的 HFM 故障保留待评，修复后重试。

例如，由 C 自动管理 HFM，并允许同一内网中的 A，可使用：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-remote-v1 \
  --hfm-runtime-dir ../hfm-runtime-release-sealed-latest \
  --allow-agent-origin http://10.0.0.12:8001 \
  --agent-timeout 15 --agent-retries 2 \
  --poll-interval 15
```

该命令与手动启动 HFM 的方式二选一。将 A 的地址替换为实际值；若使用公网 A，可省略允许项。A 服务最多保存 4096 个已结束回合编号，达到容量后需在无活动测评时重启 A，之后重新提交或由管理员恢复待评任务。

评分电源固定使用 C 的 `environment/power_supply.py`：1 ms 步长，CS/PF 传输延迟每步采样 2–5 ms，VS 为 0–1 ms，再执行变化率限制、PSM 和 HFM。每份提交重建电源并使用相同 `--seed`（默认 0）；A 只返回电压设定值，不重复处理电源。

# 7. 日志与部署

`server_evaluator/runtime-remote-v1/` 保存任务数据库、下载的连接文件，以及 `evaluations/*/attempt-*/` 中的 `trajectory.json`、`result.json`、`evaluation.log`、`phase.json` 和可信配置。连接文件含 A 的令牌，限制数据目录访问权限，勿公开或提交 Git；临时连接副本在评测后删除。

更换接口、目标或可信评测配置时停止旧服务并使用新的数据目录；不要复用旧 `runtime-models-v1` 或 `runtime-reference-v1`。已回传 `failed` 的提交需选手重新提交，或由平台管理员重置为待评。

参数、重试与 systemd 配置见 [server_evaluator/README.md](server_evaluator/README.md)。本分支保留显式 `--submission-mode onnx` 兼容旧 ZIP 流程，正式默认使用远程连接文件。
