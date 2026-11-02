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

目标顶层为 `F2b`，兼容原评分器的 `F2a` 几何目标回退，须包含 `lcfs_points`、`Xpt_main`、`Xpt_sec`、`strike`。A1/A2/B1/B2/B3 目标不适用。目标、评分规则、指标名或平台/课题配置变化时使用新的 `--data-dir`，避免回传旧规则缓存。

## 2.配置平台

按操作手册先完成：

1. 在本题 Metric 中创建 `得分`（或约定的其他名字）。
2. 综合分公式设为 `[0:得分]`，排序为 `desc`。默认仍是原 F2b 的 0～30 分，不转换为百分制。
3. 从本题「API / CLI 帮助」获取**平台课题的字符串 problem_id**，不要使用手册 4.4 中的整数外部题号，也不要使用示例占位 ID。
4. 由平台运维确认学生账号、队伍、课题绑定和提交开放时间。当前课题不划分小问，无需 `sub_question_index`。

在服务器终端配置助教身份，密码交互输入，不写入仓库：

```bash
export PLATFORM_USERNAME='YOUR_TEACHER_ACCOUNT'
export PLATFORM_PROBLEM_ID='YOUR_PLATFORM_STRING_PROBLEM_ID'
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
| `--metric-name` | `得分`，必须与平台 Metric 完全一致 |
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
{"items": [{"submission_id": "PLATFORM_STRING_SUBMISSION_ID", "metrics_scores": {"得分": 25.8}, "status": "completed", "evaluator_message": "..."}]}
```

此处使用待评记录的字符串 `id`，不是手册 4.4 的整数评测 ID。本程序不提供 `/submit`、`/retrieve`，也不接入方式 A 的串行评测队列。

本地测试只使用假平台，不会使用手册中的账号密码连接真实平台：

```bash
python -m pip install pytest
python -m pytest server_evaluator/tests -q
```
