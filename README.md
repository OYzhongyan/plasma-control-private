# 1. 服务器模型评测

`server/offline-grading` 是主办方部署的测评分支，按助教操作手册 **4.5** 主动从比赛平台拉取任务。选手使用 `main` 分支训练、导出并上传模型，平台负责队伍、提交历史和排行榜，测评服务器负责运行模型、仿真和回传成绩。

```text
选手上传模型 ZIP → 平台保存提交 → 服务器下载并自动解包
→ CPU 推理控制 HFM → 计算七项贡献分 → 回传平台 → 更新排行榜
```

以下命令在测评服务器的 **Linux Bash、项目根目录**执行。部署需要完整的 HFM 运行包、CPU 和可访问比赛平台的网络；推理使用 ONNX Runtime CPU。

# 2. 安装依赖与准备数据

```bash
git switch server/offline-grading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r server_evaluator/requirements.txt
```

已有 Conda 环境时，可直接在该环境中安装依赖。另行准备完整的 `hfm-runtime-release-sealed-latest`，后续假设位于项目同级目录。

主办方统一提供下列文件：

| 文件 | 用途 |
| --- | --- |
| `evaluation/results/target.json` | 官方目标，构造参考轨迹并计算评分 |
| `configs/env_default.yaml` | HFM 环境配置 |
| `eval_files/references.yaml` | 初始 shot 与评测参考配置 |
| `eval_files/obs_perturb.yaml` | 观测扰动配置 |
| `configs/shots.yaml` | shot 初始化配置 |

官方目标顶层应包含 `F2b`，其目标几何包括 `lcfs_points`、`Xpt_main`、`Xpt_sec` 和 `strike`。所有选手使用服务器上的同一套目标、配置和随机种子。

平台接收选手导出的单个模型 ZIP，默认文件名为 `submission_model.zip`，包内根目录包含：

```text
submission_model.zip
├── policy2.onnx
├── policy2.metadata.json
└── policy2_vecnormalize_stats.npz
```

服务器自动校验、解包并加载这三份文件。上传上限默认 **128 MiB**，包内文件名固定。当前支持 `historical_observation_v1` 历史观测接口；ONNX 输出 7 维独立动作，服务器按训练时的映射转换为 HFM 接收的 12 维线圈电压。

# 3. 启动 HFM socket 服务

在一个终端启动独立 HFM 实例，端口为 `5558`：

```bash
LISTEN_PORT=5558 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash ../hfm-runtime-release-sealed-latest/run_hfm_server.sh
```

等待输出 `Waiting for a client to connect...`，保持此终端运行。测评期间，该实例应由评分程序独占。Python 评分器和后面的 systemd 模板只管理评分流程，HFM 服务需单独启动和维护。

# 4. 在服务器核验模型包

另开终端，激活 Python 环境并进入项目根目录。先检查目标文件和评测配置：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --check-config
```

此命令只检查本地配置。再将一份导出的模型包放在 `results/submission_model.zip`，运行完整仿真核验：

```bash
python server_evaluator/run.py \
  --model-file results/submission_model.zip \
  --target evaluation/results/target.json \
  --hfm-host 127.0.0.1 \
  --hfm-port 5558 \
  --seed 0 \
  --onnx-threads 1
```

`--model-file` 会自动解包、运行最多 500 步 HFM 仿真并打印七项得分。该步骤供服务器管理员联调，无需平台账号，也不回传成绩。等离子体提前终止时，按实际完成的轨迹评分，剩余步记零贡献。

# 5. 配置比赛平台

在平台课题中允许上传 ZIP，并将文件大小上限与服务器的 **128 MiB** 保持一致。创建以下七个 Metric，名称区分大小写，**平台权重均为 1**：

| Metric | 贡献分上限 |
| --- | ---: |
| `Ip` | 6 |
| `LCFS` | 6 |
| `X` | 4 |
| `strike` | 3 |
| `psiX` | 4 |
| `X2` | 3 |
| `psiX2` | 4 |

平台综合分公式设置为：

```text
[0:Ip]+[0:LCFS]+[0:X]+[0:strike]+[0:psiX]+[0:X2]+[0:psiX2]
```

七项回传值已包含各自的评分权重、500 步时间平均和约束惩罚，直接求和，**满分 30 分、降序排名**。平台只需这七个 Metric，避免额外创建 `得分` 指标或重复加权。

当前测评沿用正式轨迹评分规则，推理耗时仅记录审计，不启用每步 0.5 ms 的整段扣分规则；单次推理运行上限由 `--inference-timeout` 控制，默认 1 秒。

从课题的「API / CLI 帮助」获取字符串 `problem_id`，并由平台管理员提供有本题评分权限的助教身份。配置环境变量：

```bash
export PLATFORM_USERNAME='autumn26_t02'
export PLATFORM_PROBLEM_ID='6ae84a300a686d1bbf619417'
read -rsp '平台助教密码: ' PLATFORM_PASSWORD
echo
export PLATFORM_PASSWORD
```

也可设置有效的 `PLATFORM_TOKEN`。平台默认地址为 `https://competition.opensii.ai`，需要修改时设置 `PLATFORM_BASE_URL`。凭据只在服务器配置，不写入代码仓库。

# 6. 拉取任务、测评并回传

在平台使用测试队伍上传一份模型 ZIP，先执行一轮拉取、测评和回传：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-models-v1 \
  --once
```

确认平台的提交状态、七项分数和排名后，常驻运行：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-models-v1 \
  --poll-interval 15
```

程序串行处理本题 `offline`、`manual` 两类待评任务，每轮结束后等待 15 秒再查询。每份结果先持久化，再回传平台。`--once` 在仍有未完成或未确认任务时返回退出码 1。

| 测评情况 | 处理方式 |
| --- | --- |
| 模型正常运行，含合法提前终止或零分 | 回传 `completed` 和七项贡献分 |
| 模型包无效、接口不符或超过模型运行限制 | 回传 `failed` 和原因 |
| HFM 不可达等基础设施故障 | 保留待评，修复后重试 |
| 成绩回传失败 | 保存已有结果，下轮重传 |

服务器通过平台助教 API 主动拉取和回传任务；本流程无需部署面向平台的 `/submit` 或 `/retrieve` HTTP 服务。同一课题只运行一套评分服务，使用同一持久化数据目录。

# 7. 日志、数据与常驻部署

默认工作数据保存在 `server_evaluator/runtime-models-v1/`：

| 路径 | 内容 |
| --- | --- |
| `tasks.sqlite3` | 任务状态与待回传成绩 |
| `downloads/` | 原始模型 ZIP |
| `evaluations/*/attempt-*/trajectory.json` | 服务器生成的实际仿真轨迹 |
| `evaluations/*/attempt-*/result.json` | 本次评测结果或失败原因 |
| `evaluations/*/attempt-*/evaluation.log` | 模型评测日志 |
| `evaluations/*/attempt-*/phase.json` | 当前或最后运行阶段 |

解包后的模型文件在本次评测结束后自动清理，原始 ZIP 与审计结果保留。重启时程序复用已保存结果，尚未完成的任务重新评测。更改官方目标或评分配置前停止服务，并使用新的 `--data-dir`，保留旧目录以核对历史成绩。

Linux 常驻部署可使用 [plasma-grader.service](server_evaluator/plasma-grader.service)，按实际环境配置运行账号、代码路径、Python 环境、凭据文件和数据目录；完整安装步骤见 [服务器测评说明](server_evaluator/README.md#6-systemd-常驻部署)。

全部命令参数、超时与内存限制、故障重试、缓存升级和联调说明见 [server_evaluator/README.md](server_evaluator/README.md)。
