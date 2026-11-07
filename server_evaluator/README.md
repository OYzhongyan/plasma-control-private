# 服务器模型评分：手册 4.5

本目录供出题组部署。`main` 是选手 baseline，`server/offline-grading` 是服务器评分分支。选手在比赛平台上传模型 ZIP；平台保存队伍、提交历史和排行榜，本程序主动拉取任务，在出题组服务器运行 HFM 仿真并回传成绩。

```text
上传模型 ZIP → 服务器下载并校验 → ONNX 策略控制 HFM → 500 步轨迹评分 → 回传七项贡献分 → 平台更新排行榜
```

需要一台安装完整 HFM 运行包、能访问比赛平台与文件下载地址的 Linux CPU 机器。评分器使用 ONNX Runtime CPU 推理，不需要 GPU、Torch、Stable-Baselines3、Flask 或 Hugging Face。HFM socket 服务由出题组另行启动，评分器不会自动启动它。

## 1. 安装与准备文件

在服务器切换到本分支，以下命令均在项目根目录的 Bash 终端执行：

```bash
git switch server/offline-grading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r server_evaluator/requirements.txt
```

也可在已有 Conda 环境中安装上述依赖。Python 依赖不包含 HFM 的 MATLAB 运行包，请单独准备完整的 `hfm-runtime-release-sealed-latest`。

选手按 `main` 分支 README 运行 `train/export_onnx.py`，导出成功后会自动生成 `results/submission_model.zip`，上传这一个文件即可。服务器下载后由 `ModelGrader.grade_file()` 自动调用 `unpack_bundle()` 校验并解包，再启动模型测评，无需管理员手动解压。ZIP 根目录必须包含同一次导出的三个文件：

```text
submission_model.zip
├── policy2.onnx
├── policy2.metadata.json
└── policy2_vecnormalize_stats.npz
```

不要在 ZIP 中再套一层目录。服务器使用自己的推理、特征处理、仿真和评分代码；选手不提交 Python 脚本、PPO checkpoint、pickle、目标文件或自报成绩的轨迹 JSON。

本版本仅接受与 `main` 一致的 `reference_only_v1` 接口，元数据版本为 `3`。输入是当前目标的 **90 维 reference 特征**，没有实际状态、历史观测或动作历史；`history_steps`、`action_history_steps`、`action_history_dim` 均为 `0`。

| 输入顺序 | 字段 | 维数 |
| --- | --- | ---: |
| 1 | `reference_Ip` | 1 |
| 2 | `reference_lX` | 1 |
| 3 | `reference_lcfs_points`（32×2，按行展开） | 64 |
| 4 | `reference_rX` | 4 |
| 5 | `reference_zX` | 4 |
| 6 | `reference_strike_r` | 8 |
| 7 | `reference_strike_z` | 8 |

服务器与 main 使用相同的 `ReferenceObservationWrapper`、`reference_metadata()` 和 `ReferencePolicyAdapter`。严格校验字段顺序、形状、动作模式和归一化标志；需要归一化时使用同次训练导出的 90 维冻结统计量。旧的 `historical_observation_v1`／812 维模型不兼容，须重新训练和导出，不能只改元数据。

ONNX 输出 7 维归一化动作，先裁剪到 `[-1, 1]`，然后按元数据的 `action_mode` 转换：

- `absolute`：乘以 `[1499, 230, 172, 172, 348, 348, 270]` V。
- `delta`：乘以 `0.16 × [1499, 175, 175, 175, 175, 175, 80]` V，并累加到上一设定值；回合重置时清零。

两种模式都按索引 `[0,1,1,2,2,3,3,4,4,5,5,6]` 展开为 12 维电压设定值，再经过电源模型送入 HFM。测评端保留 ONNX Runtime CPU 单线程限制，模型包中的权重、元数据和统计量必须来自同次导出。

### 电源延迟与观测扰动

训练与测评统一采用 `environment/power_supply.py` 中的设置：

| 设置 | 值 |
| --- | --- |
| 仿真步长 | 1 ms |
| 通道 0–10（CS/PF）传输延迟 | 每步均匀采样 2–5 ms |
| 通道 11（VS）传输延迟 | 每步均匀采样 0–1 ms |
| 离散延迟步数 | `round(delay_s / 0.001)` |
| 电源处理顺序 | 传输延迟 → 电压变化率限制 → PSM 标定 → HFM |
| 正式评测随机种子 | `--seed 0`，每份模型重建电源 RNG |

这是物理电源传输延迟，与 `features.delay_step`／`eval.obs_perturb.delay_steps` 的观测延迟不同。关闭观测扰动不会关闭电源延迟。当前测评不支持非零观测延迟，遇到该配置会报告基础设施配置错误；当前目标 reference 特征不使用测量状态历史。启动日志、`--check-config` 输出和每次评测的 `config.json` 记录接口、电源延迟与种子。

出题组准备以下文件，所有选手使用同一套配置：

| 文件 | 用途 |
| --- | --- |
| `evaluation/results/target.json` | 官方目标几何，只由服务器管理员指定 |
| `configs/env_default.yaml` | HFM 环境基础配置 |
| `eval_files/references.yaml` | 初始 shot、目标轨迹和 500 步评测配置 |
| `eval_files/obs_perturb.yaml` | 观测扰动配置，是否启用以文件内容为准 |
| `configs/shots.yaml` | shot 对应的仿真初始化配置 |

官方目标顶层应包含 `F2b`，其中有 `lcfs_points`、`Xpt_main`、`Xpt_sec`、`strike`；评分器兼容原始 `F2a` 几何目标回退，`A1/A2/B1/B2/B3` 目标不适用。`--target` 指定的管理员目标用于构造参考轨迹并评分，不能由模型包覆盖。

## 2. 启动 HFM 并核验模型

以下命令假设完整运行包位于项目同级目录 `../hfm-runtime-release-sealed-latest`，实际位置不同时替换路径。在一个终端启动单个 HFM socket 实例：

```bash
LISTEN_PORT=5558 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash ../hfm-runtime-release-sealed-latest/run_hfm_server.sh
```

保持该终端运行，等待输出 `Waiting for a client to connect...`。评测期间不要让训练或其他评测进程占用该实例。默认地址为 `127.0.0.1:5558`；HFM 与评分器应在同一台机器上，部署 systemd 后也须保证服务进程能连接该地址。

另开终端，激活同一 Python 环境并进入项目根目录，先检查本地配置：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --check-config
```

`--check-config` 不登录平台、不连接 HFM，也不运行模型，不能代替实际仿真。再用模型包核验完整流程：

```bash
python server_evaluator/run.py \
  --model-file results/submission_model.zip \
  --target evaluation/results/target.json \
  --env-config configs/env_default.yaml \
  --reference-config eval_files/references.yaml \
  --obs-perturb-config eval_files/obs_perturb.yaml \
  --hfm-host 127.0.0.1 \
  --hfm-port 5558 \
  --socket-timeout 300 \
  --evaluation-timeout 1800 \
  --seed 0 \
  --onnx-threads 1
```

`--model-file` 在本机运行最多 500 步 HFM 仿真并输出七项得分，不需要平台账号或 `problem_id`，不会上传成绩。等离子体提前终止时按正式规则计算已有轨迹，剩余步为零贡献。

## 3. 配置平台

按手册 4.5 配置本题：

1. 请平台运维确认该题允许上传 **ZIP**，并使上传大小限制与服务器默认 **128 MiB** 下载上限一致。
2. 创建以下 7 个 Metric，大小写完全一致，平台权重均为 `1`，不创建单独的 `得分` 指标。
3. 综合分显式使用下面的求和公式，排序为 `desc`。不要使用加权平均，否则总分会变成七项之和的七分之一。
4. 从本题「API / CLI 帮助」获取平台课题的**字符串 `problem_id`**，不要使用手册 4.4 的整数外部题号。
5. 确认队伍与课题绑定、提交开放时间及评分助教权限。本题不划分小问，无需 `sub_question_index`。

| 平台 Metric 名 | 回传贡献分上限 |
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

各项已包含正式权重、500 步时间平均及约束惩罚，直接相加即为 0～30 分综合分，不再乘一次 `6/6/4/3/4/3/4`。回传保留原评分器的浮点贡献分，日志中的总分显示 6 位小数。

服务器复用 `evaluation/eval/score` 的电流、边界、主次级 X 点、打击点和磁通指标，并保留位形、拓扑、电流及线圈约束。电流参考由 `config.build_Ip_ref()` 定义。当前与原本地仿真评测一致，**不启用每步 0.5 ms 推理耗时的整段惩罚**；实际推理耗时仅供审计，不加入该评分字段。服务器连接/执行超时按后面的运行故障规则处理。

在终端配置具有本题权限的**助教账号**，密码交互输入：

```bash
export PLATFORM_USERNAME='YOUR_TEACHER_ACCOUNT'
export PLATFORM_PROBLEM_ID='YOUR_PLATFORM_STRING_PROBLEM_ID'
read -rsp '平台助教密码: ' PLATFORM_PASSWORD
echo
export PLATFORM_PASSWORD
```

由平台管理员提供可登录助教 API 的账号；学生测试账号不能替代助教身份。也可设置有效的 `PLATFORM_TOKEN`。同时提供账号密码时，Token 收到 401 后会重新登录；只提供 Token 时，失效后需人工更新。平台凭据不会附加到模型下载请求，也不写入结果数据库。

## 4. 拉取、评分和回传

完成本地核验后，在平台用测试队伍提交模型 ZIP，先运行一轮：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-reference-v1 \
  --once
```

确认评分与排行榜后，常驻运行：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-reference-v1 \
  --poll-interval 15
```

默认串行处理本题 `offline`、`manual` 两类待评任务，不处理 `auto` 或其他课题。每份成绩先持久化，再及时回传，不等待整轮所有仿真结束。每轮处理完成后等待 15 秒再查询。

| 参数 / 变量 | 默认值与用途 |
| --- | --- |
| `--problem-id` / `PLATFORM_PROBLEM_ID` | 平台运行必填；本地 `--model-file` 不需要 |
| `--platform-url` / `PLATFORM_BASE_URL` | `https://competition.opensii.ai` |
| `--target` | `evaluation/results/target.json`，官方目标 |
| `--env-config` | `configs/env_default.yaml` |
| `--reference-config` | `eval_files/references.yaml` |
| `--obs-perturb-config` | `eval_files/obs_perturb.yaml` |
| `--hfm-host` / `--hfm-port` | `127.0.0.1` / `5558` |
| `--socket-timeout` | HFM socket 超时 300 秒 |
| `--evaluation-timeout` | 每次模型评测子进程的硬超时 1800 秒 |
| `--inference-timeout` | `1` 秒，单次推理的运行上限，由父进程按阶段监测；不是 0.5 ms 评分惩罚 |
| `--seed` | `0`，固定评测随机种子 |
| `--onnx-threads` | `1`，CPU 推理线程数 |
| `--memory-limit-mb` | `4096`，Linux 评测子进程地址空间上限（MiB，至少 256）；不限制独立运行的 HFM 进程 |
| `--data-dir` | 项目根目录下 `server_evaluator/runtime-reference-v1` |
| `--sources` | `offline,manual`；可只设 `offline` |
| `--limit` | 每种来源每轮最多 100 条，可设 1～500 |
| `--http-timeout` | 平台/下载 HTTP 读取超时 30 秒，连接超时 10 秒 |
| `--max-file-mb` | 模型 ZIP 下载上限 128 MiB |
| `--download-retries` | 下载失败跨轮重试 3 次后回传失败 |
| `--once` | 一轮后退出；仍有未处理或未确认任务时退出码为 1 |
| `--check-config` | 仅检查本地配置，不连接平台或 HFM |
| `--model-file` | 本地完整仿真核验一个 ZIP，不回传平台 |

任务结果区分如下：

- 成功运行并得到合法轨迹：回传 `completed`，允许零分或提前终止。
- 模型包无效、文件超限或不符合模型接口：回传 `failed` 与原因。
- 模型加载超过总时限、单次推理超过 `--inference-timeout`，或在模型加载／推理阶段异常退出：回传 `failed`。其他总时限超出、HFM 等运行故障保留待重试，避免将慢仿真误判为选手错误。
- Linux 子进程使用 `RLIMIT_AS` 限制地址空间，模型超出内存预算则失败；主办方可按机器资源统一调整。Windows 本地核验不施加此 Linux 内存限制，正式部署请使用 Linux。
- 下载失败：按 `--download-retries` 跨轮重试，达到上限才回传 `failed`。
- HFM 不可达等基础设施错误（`GradingUnavailable`）：保持 `pending`，修复服务后重试，不消耗下载失败次数。
- 成绩回传失败：保留已计算结果，下轮重传，不重新仿真。

平台 API 保持手册 4.5：助教登录 `/api/auth/login/teacher`，拉取 `/api/submissions/offline-pending` 或 `/api/submissions`，向 `/api/submissions/offline-grading-batch` 回传。本程序不提供手册 4.4 的 `/submit`、`/retrieve` 接口。

## 5. 持久化、审计与升级

将 `--data-dir` 放在可持久保存的磁盘。默认目录结构为：

```text
server_evaluator/runtime-reference-v1/
├── tasks.sqlite3
├── worker.lock
├── downloads/                 # 按提交 ID 哈希命名的模型 ZIP
└── evaluations/
    └── <下载文件名去掉扩展名>/
        └── attempt-.../       # 每次评测独立目录
            ├── trajectory.json
            ├── result.json
            ├── evaluation.log
            ├── config.json
            └── phase.json     # 当前／最后运行阶段
```

成功完成仿真后，`trajectory.json` 保存服务器实际生成的轨迹，可用于复查七项分数；失败时查看已有的 `result.json`、`evaluation.log` 和 HFM 日志。`--model-file` 本地核验使用模型绝对路径的哈希作为评测目录名。不要将下载包、轨迹缓存或平台凭据提交到 Git。

终止或退化位形中的非有限几何值按正式评分器规则处理，在审计 JSON 中保存为 `null`，不填造零坐标。解包后的 `model/` 仅在本次评测期间存在，结束或失败后删除，原始 ZIP 留在 `downloads/`，避免服务器故障重试时反复累积模型副本。

- 已保存结果优先用于回传；重启不重复评分。中断时尚未得到结果的评测会重新运行。
- 回传响应不明确时查询平台状态。已 `completed/failed` 的记录不覆盖；查不到不代表成功，继续保留缓存。部分回传失败只保留未确认项。
- 平台列表可能分页，无法确认的旧任务保留在本地，需到面板核对。
- 同一数据目录有单实例锁，不同目录或机器不会共享锁；同一课题只运行一套评分服务。
- 目标、模型包契约、可信推理/仿真/评分逻辑或有效评测配置改变时，指纹会变化，程序拒绝复用旧缓存。先停止旧服务、保留旧目录，再选择新的 `--data-dir`；尚未回传的旧结果应先核对。

**从旧历史观测模型或轨迹 JSON 评分升级时，先停止旧评分器，再使用新的 `runtime-reference-v1` 目录，不要复用 `runtime-models-v1`。** 旧缓存不会自动迁移。平台已完成或失败的历史提交不会自动重评，旧轨迹 JSON 不能作为模型包继续评测。需要重评时，由管理员安排重新提交模型或将平台任务重置为待评；仅重启评分器不会重评已回传的失败记录。

## 6. systemd 常驻部署

先保证完整 HFM 已在同机 `127.0.0.1:5558` 启动并可用，再部署评分服务。模板只管理 Python 评分器，**不会管理或启动 HFM**；HFM 退出或服务器重启后，需要恢复 HFM 服务。

按实际情况修改 [plasma-grader.service](plasma-grader.service) 中的账号、仓库和 Python 路径。模板使用 `/opt/plasma-control-server`、其 `.venv`，以及新数据目录 `/var/lib/plasma-grader/reference-v1`。

创建 `/etc/plasma-grader.env`，填写管理员提供的助教身份与课题 ID：

```text
PLATFORM_USERNAME=YOUR_TEACHER_ACCOUNT
PLATFORM_PASSWORD=YOUR_CURRENT_PASSWORD
PLATFORM_PROBLEM_ID=YOUR_PLATFORM_STRING_PROBLEM_ID
```

该文件不放入仓库。准备账号、权限并安装服务：

```bash
sudo chmod 600 /etc/plasma-grader.env
sudo useradd --system --home /var/lib/plasma-grader --create-home plasma-grader
sudo install -d -o plasma-grader -g plasma-grader /var/lib/plasma-grader/reference-v1
sudo cp server_evaluator/plasma-grader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plasma-grader
sudo journalctl -u plasma-grader -f
```

账号已存在则跳过 `useradd`。确保该账号能读取代码、虚拟环境、目标和评测配置，能写入数据目录。更改 HFM 地址/端口时同步修改 `ExecStart` 参数。

停止服务使用 `sudo systemctl stop plasma-grader`。停止信号会传递给评测子进程，systemd 最多等待 40 秒后清理残留进程；下次启动重新处理尚未保存结果的任务。该操作不会停止独立运行的 HFM。

## 7. 联调与测试

先用 `--model-file` 验证完整 HFM 流程，再在平台提交正常模型包与无效包各一份，确认状态、七项分数及排名。检查平台网络短暂断开后的成绩重传，以及 HFM 暂不可达时任务保持待评的行为。

回传示例：

```json
{"items": [{"submission_id": "PLATFORM_STRING_SUBMISSION_ID", "metrics_scores": {"Ip": 6.0, "LCFS": 6.0, "X": 4.0, "strike": 3.0, "psiX": 4.0, "X2": 3.0, "psiX2": 4.0}, "status": "completed", "evaluator_message": "F2b score=30.000000/30; ..."}]}
```

`submission_id` 是待评记录的字符串 `id`，不是手册 4.4 的整数评测 ID。此处只说明回传结构，不代表某个模型的实际成绩。

自动测试使用本地假平台与仿真替身，不连接真实平台或使用真实账号：

```bash
python -m pip install pytest
python -m pytest server_evaluator/tests -q
```
