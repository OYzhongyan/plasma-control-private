# 服务器评分分支：手册 4.5

本目录只供出题组部署。`main` 是交给选手的 baseline；`server/offline-grading` 是服务器评分分支。学生仍在比赛平台提交，平台保存队伍、历史记录和排行榜，本程序主动拉任务并回传成绩。

```text
平台接收轨迹 JSON → 本机拉取待评记录 → 下载并评分 → 回传平台 → 平台更新排行榜
```

需要一台能访问比赛平台和文件下载地址、持续运行 Python 的 CPU 机器。无需公网 IP、入站端口、HF、Pyodide、Playwright、GPU 或 HFM 仿真器。只解析轨迹 JSON，不执行选手代码。

## 1.安装

将本分支推送到你的代码仓库后，在服务器克隆或切换到该分支。以下命令在项目根目录执行：

```bash
git switch server/offline-grading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r server_evaluator/requirements.txt
```

如已有 Conda 环境，也可使用其 Python 安装这份依赖，无需安装训练依赖。

把**官方 F2b 目标文件**放到 `evaluation/results/target.json`，或使用 `--target` 指定实际路径。选手仅提交完整轨迹 JSON：

```json
{"F2b": {"trajectory": {"Ip": [], "lX": [], "Icoil": [], "lcfs_per_step": []}, "timeout": false}}
```

上面仅示意结构，真实数组来自 rollout。主要可评分字段包括 `Ip`、`lX`、`nX`、`Icoil`、`lcfs_per_step`、`Xpt_main`、`Xpt_sec`、`strike`、`psia`、`psib`、`psiX_main`、`psiX_sec` 及可选 `inference_time_ms`。非核心字段缺失时沿用原评分器的规则，不补造观测；建议正式提交包含全部 rollout 字段。ZIP、JSONL、ONNX、Python 代码均不是此入口的输入。

目标顶层为 `F2b`，兼容原评分器的 `F2a` 几何目标回退，须包含 `lcfs_points`、`Xpt_main`、`Xpt_sec`、`strike`。A1/A2/B1/B2/B3 目标不适用。目标、评分规则、指标输出格式或平台/课题配置变化时使用新的 `--data-dir`，避免回传旧规则缓存。

## 2.配置平台

按操作手册先完成：

1. 在本题 Metric 中创建以下 7 个指标，大小写完全一致，每个指标的平台权重均为 `1`。不再回传单独的 `得分` 指标。
2. **综合分必须显式使用下面的求和公式**，排序为 `desc`。不要留空使用加权平均：权重均为 1 的平均值会变成总分的七分之一。仍是原 F2b 的 0～30 分。
3. 从本题「API / CLI 帮助」获取**平台课题的字符串 problem_id**，不要使用手册 4.4 中的整数外部题号，也不要使用示例占位 ID。
4. 由平台运维确认学生账号、队伍、课题绑定和提交开放时间。当前课题不划分小问，无需 `sub_question_index`。

| 平台 Metric 名 | 回传的贡献分上限 |
| --- | ---: |
| `Ip` | 6 |
| `LCFS` | 6 |
| `X` | 4 |
| `strike` | 3 |
| `psiX` | 4 |
| `X2` | 3 |
| `psiX2` | 4 |

首页综合分公式：

```text
[0:Ip]+[0:LCFS]+[0:X]+[0:strike]+[0:psiX]+[0:X2]+[0:psiX2]
```

回传的是**已经加权、按 500 步平均且施加约束惩罚的贡献分**，不是原始误差，也不是 0～1 的归一化分。平台直接相加，不能再乘 6/6/4/3/4/3/4。为保持求和精度，各贡献分保留原评分器浮点值；日志中显示的总分保留 6 位小数。

在服务器终端配置助教身份，密码交互输入，不写入仓库：

```bash
export PLATFORM_USERNAME='autumn26_t02'
export PLATFORM_PROBLEM_ID='6ae84a300a686d1bbf619417'
read -rsp '平台助教密码: ' PLATFORM_PASSWORD
echo
export PLATFORM_PASSWORD
```

也可只设置有效的 `PLATFORM_TOKEN`。如果同时提供账号密码，Token 收到 401 后会自动重新登录；仅提供 Token 时，失效后需要人工更新。账号密码和 Token 仅用于平台 API，**不会附加到文件下载请求**。两位负责人应协调使用同一个运行服务与数据目录。

## 3.先检查，再运行

本地检查（不登录、不调用平台、不回传成绩）：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --check-config
```

用平台提供的测试学生提交一份轨迹后，运行一轮实际拉取、评分和回传：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime \
  --once
```

常驻运行，每 15 秒查询一次：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime \
  --poll-interval 15
```

默认同时处理 `offline`、`manual` 两种来源；不处理 `auto` 或其他课题。只处理离线来源可加 `--sources offline`。评分完成即使为零分也回传 `completed`；文件无效回传 `failed` 与原因。

| 参数 / 变量 | 默认值与用途 |
| --- | --- |
| `--problem-id` / `PLATFORM_PROBLEM_ID` | 必填，平台课题字符串 ID |
| `--platform-url` / `PLATFORM_BASE_URL` | `https://competition.opensii.ai` |
| `--target` | 项目根目录下 `evaluation/results/target.json` |
| 回传指标 | 固定为 `Ip, LCFS, X, strike, psiX, X2, psiX2`，不再使用 `--metric-name` |
| `--data-dir` | 项目根目录下 `server_evaluator/runtime` |
| `--sources` | `offline,manual` |
| `--limit` | 每种来源每轮最多 100 条，可设 1～500 |
| `--http-timeout` | 单次 HTTP 读取超时 30 秒，连接超时 10 秒 |
| `--max-file-mb` | 下载文件最多 128 MiB |
| `--download-retries` | 下载失败跨轮重试 3 次，之后回传失败 |
| `--once` | 一轮后退出，有尚未处理/确认的任务时退出码为 1 |

程序直接复用 `evaluation/eval/score`，包括提前结束、拓扑、电流熔断、线圈电流和推理时间惩罚。电流参考仍由原规则的 `config.build_Ip_ref()` 定义。

## 4.持久化与恢复

`--data-dir` 必须位于可持久保存的磁盘，默认已加入 Git 忽略：

```text
runtime/
├── tasks.sqlite3       # 提交 ID、处理状态和已计算结果
├── worker.lock         # 单实例锁
└── downloads/          # 按提交 ID 哈希命名的轨迹文件
```

- 先保存评分结果，再回传。网络失败和重启后优先复用缓存，不重复下载和评分。
- 批量回传最多 200 条，检查 `updated`、`failed`、`results`、`errors`；部分失败只保留未确认项。
- 回传响应不明确时查询平台状态。平台已 `completed/failed` 则不覆盖，不再次评分；查不到不代表成功，继续保留缓存。
- 平台列表可能有分页，无法确认的历史任务保留在缓存中，需在面板核对；不会因为记录不在某一页就认定已完成。
- 同一数据目录只能运行一个实例；多台机器不会共享 OS 文件锁，请只启动一套服务。
- 不要在有未回传结果时删除数据库或改变目标。要更换规则，先处理完旧队列，再使用新数据目录。
- 平台回传仅接受本题的 `offline/manual + pending`，权限、指标名或任务状态错误会保留缓存并记录日志，修正配置后再次运行。

**从旧的单一“得分”输出升级：**旧缓存的输出格式与新版本不同，程序会因 fingerprint 不一致拒绝直接续用。先停止旧服务并保留其数据目录，使用新的目录启动，例如 `--data-dir server_evaluator/runtime-metrics-v2`；使用 systemd 时同步修改模板的 `--data-dir`。已有未回传任务会按新格式重新评分，旧缓存文件不会被删除。平台中已经完成、仅有旧“得分”字段的历史记录不会自动重评或改写；如需更新，先与平台管理员确认历史成绩迁移或重评安排。

## 5.systemd 常驻部署（可选）

提供了 `plasma-grader.service` 模板。按实际情况修改运行账号、仓库路径和 Python 路径。示例环境文件 `/etc/plasma-grader.env`：

```text
PLATFORM_USERNAME=YOUR_TEACHER_ACCOUNT
PLATFORM_PASSWORD=YOUR_CURRENT_PASSWORD
PLATFORM_PROBLEM_ID=YOUR_PLATFORM_STRING_PROBLEM_ID
```

环境文件不放入 Git，设置权限 `sudo chmod 600 /etc/plasma-grader.env`。模板默认仓库位于 `/opt/plasma-control-server`，虚拟环境为其 `.venv`；官方目标位于 `evaluation/results/target.json`，持久化目录为 `/var/lib/plasma-grader`。准备运行账号及写权限：

```bash
sudo useradd --system --home /var/lib/plasma-grader --create-home plasma-grader
sudo install -d -o plasma-grader -g plasma-grader /var/lib/plasma-grader
sudo cp server_evaluator/plasma-grader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plasma-grader
sudo journalctl -u plasma-grader -f
```

确保该账号能读取代码、虚拟环境与目标文件。修改模板或环境后重启服务。临时停止：`sudo systemctl stop plasma-grader`。

## 6.联调验收

先在平台提交正常与错误 JSON 各一份，确认评分、失败原因、状态、分数和排行榜排序。再模拟断网/重启，确认已计算结果不丢失且不会重复评分。正常结果默认回传：

```json
{"items": [{"submission_id": "PLATFORM_STRING_SUBMISSION_ID", "metrics_scores": {"Ip": 6.0, "LCFS": 6.0, "X": 4.0, "strike": 3.0, "psiX": 4.0, "X2": 3.0, "psiX2": 4.0}, "status": "completed", "evaluator_message": "F2b score=30.000000/30; ..."}]}
```

此处使用待评记录的字符串 `id`，不是手册 4.4 的整数评测 ID。本程序不提供 `/submit`、`/retrieve`，也不接入方式 A 的串行评测队列。

本地测试只使用假平台，不会使用手册中的账号密码连接真实平台：

```bash
python -m pip install pytest
python -m pytest server_evaluator/tests -q
```
