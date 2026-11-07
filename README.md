# 1. 主办方共享目录测评（手册 4.5）

此分支 `server/offline-grading` 部署在测评服务器 **C**。选手使用 `main` 分支，在服务器 **A** 加载自己的 checkpoint、冻结归一化和 wrapper，在比赛平台 **B** 上传一个自动生成的 `endpoint.json`。C 主动拉取提交，在 C 运行 HFM，通过共享文件向 A 发送观测，读取 A 返回的 **12 维物理电压**，计算七项分数并回传 B。平台管理队伍、历史提交和排行榜。

正式流程不需要 ONNX、HTTP 推理端口或选手选择 IP。C 不下载模型或执行选手代码，不接收选手自报轨迹或分数。官方 shot、初始化文件、目标文件、平台凭据和评分逻辑留在 C；公共目录只交换控制所需观测（含 reference）和动作。所有命令在 C 的 **Linux Bash、项目根目录**执行。

# 2. 准备共享目录和依赖

```bash
git switch server/offline-grading
pip install -r server_evaluator/requirements.txt
```

A、C 默认使用：

```text
/inspire/hdd/project/26-nuclear-fusion/public/plasma-agents/
└── <自动生成的服务编号>/
    ├── requests/       # C 发布观测请求
    └── responses/      # A 返回电压
```

管理员先确认两台机器看到同一个目录，且 A、C 的运行账户对每个邮箱有进入、读写权限。通常为共享项目组设置目录组继承（setgid）或 ACL；仅能列出 `/public` 不代表跨账户写入已可用。代码为新目录使用 2770、消息使用 0660，实际项目组和 ACL 由管理员核实。共享文件系统必须支持跨机器文件锁和原子重命名；时钟需同步。C 的私有 `--data-dir` 正式部署应放在可靠的本地持久磁盘，避免共享挂载故障同时阻塞监督日志。不要通过关闭签名验证来解决权限问题。

主办方准备 `evaluation/results/target.json`、`configs/env_default.yaml`、`configs/shots.yaml`、`eval_files/references.yaml`、`eval_files/obs_perturb.yaml` 及完整 HFM 运行包。正式 shot 由 C 的 `eval_files/references.yaml` 中 `shot_id` 决定，选手提交不能覆盖。

```bash
python server_evaluator/run.py --target evaluation/results/target.json --check-config
```

该命令只检查可信本地评分配置，不连接平台或启动 HFM，也不能代替跨机器共享目录权限测试。

# 3. 启动 C 的 HFM

若运行包位于项目同级目录，单独开一个终端：

```bash
LISTEN_PORT=5558 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash ../hfm-runtime-release-sealed-latest/run_hfm_server.sh
```

等待 `Waiting for a client to connect...`，保持运行。这个 **5558 端口仅用于 C 与 HFM**，A 不需要连接它。也可在评分命令添加 `--hfm-runtime-dir ../hfm-runtime-release-sealed-latest` 自动管理 HFM，退出时自动关闭；这时不要再手动启动同端口实例。

# 4. A 与 C 联调

选手在 A 执行：

```bash
python submission/shared_service.py --model results/YOUR_RUN/checkpoints
```

程序自动创建服务编号、邮箱和本地 `results/endpoint.json`，看到 `Shared agent ready` 后保持运行。完整选手说明见 [submission/README.md](submission/README.md)。管理员将一份测试连接文件**私下传到 C**，不能把含密钥的文件放在 public：

```bash
python server_evaluator/run.py \
  --endpoint-file results/endpoint.json \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-shared-v1
```

C 会执行最多 500 步仿真并打印七项分数，不登录或回传平台。若管理员指定其他共享根目录，A 和 C 均添加相同的 `--shared-root /path/to/shared/plasma-agents`。提交文件只允许版本、协议 `hfm-shared-policy-v1`、`transport=shared_directory`、服务编号和密钥，不允许指定任意目录、shot、目标或 HFM 地址，上限 64 KiB。

# 5. 配置平台与评分

平台课题允许上传 JSON。选手只上传自动生成的 `endpoint.json`，并保持 A 进程运行到测评完成。文件包含访问密钥，应限制下载权限，不公开或提交 Git。C 验证消息的 HMAC 签名、请求摘要、回合／步编号和有效期；公共邮箱中不保存密钥。观测在公共目录中不具备保密性，签名也不能阻止有目录权限的人读取或删除文件。

平台创建以下七个 Metric，名称区分大小写，**平台权重全部为 1**：

| Metric | 贡献分上限 |
| --- | ---: |
| Ip | 6 |
| LCFS | 6 |
| X | 4 |
| strike | 3 |
| psiX | 4 |
| X2 | 3 |
| psiX2 | 4 |

综合分公式：

```text
[0:Ip]+[0:LCFS]+[0:X]+[0:strike]+[0:psiX]+[0:X2]+[0:psiX2]
```

回传值已包含权重、500 步平均和正式约束惩罚，直接相加，满分 30，降序排名。提前终止后的剩余步记零。文件往返时间只用于审计，不作为 0.5 ms 推理惩罚。电源模型保持 1 ms 步长，CS/PF 每步传输延迟 2–5 ms、VS 为 0–1 ms，再执行变化率限制与 PSM；A 只返回设定电压，不重复电源处理。

在 C 设置助教凭据，并核实当前课题 ID：

```bash
export PLATFORM_USERNAME='YOUR_TEACHER_ACCOUNT'
export PLATFORM_PROBLEM_ID='6ae84a300a686d1bbf619417'
read -rsp '平台助教密码: ' PLATFORM_PASSWORD
echo
export PLATFORM_PASSWORD
```

也可设置有效 `PLATFORM_TOKEN`。平台默认 `https://competition.opensii.ai`，通过 `PLATFORM_BASE_URL` 覆盖；平台拉取／回传仍使用平台 API，只有 A/C 通信改为共享文件。

# 6. 拉取、测评、回传

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-shared-v1 \
  --once
```

确认后常驻运行，可同时托管 C 的 HFM：

```bash
python server_evaluator/run.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-shared-v1 \
  --hfm-runtime-dir ../hfm-runtime-release-sealed-latest \
  --agent-timeout 15 --agent-retries 2 \
  --poll-interval 15
```

如果 HFM 已单独启动，省略 `--hfm-runtime-dir`。程序串行处理 `offline,manual` 待评任务；每份结果先落盘再回传，回传失败不会重新仿真。同一课题只运行一套评分器，一个 A 邮箱同时只允许一个 C 客户端。

A 邮箱不存在、未运行、响应超时、模型中途重启或返回非法动作时，该提交失败；共享存储 I/O、缺失挂载或 C 的 HFM 故障保留待评，修复后重试。切勿在活动测评中手动 reset 或更换 A 模型。消息写完后原子发布，重复请求不会重复推理；完成消息删除，过期孤立消息由 A 清理。

# 7. 审计、迁移和部署

## 7.1 自动监控与重启 C（HFM 独立常驻）

如果 HFM 已由你独立启动并保持运行，可用守护脚本代替直接启动 `run.py`。先按 `Ctrl+C` 停止旧 C 评分器，保留 A 和 HFM 运行，然后执行：

```bash
python server_evaluator/watchdog.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-watchdog \
  --agent-timeout 15 --agent-retries 2 \
  --poll-interval 15
```

其余 `run.py` 平台、shot、共享目录、种子等参数可继续传入。**不要添加 `--hfm-runtime-dir`**：守护脚本只管理自己的 C 评分器和测评子进程，不启动、不停止独立 HFM；它不负责 HFM 保活或故障恢复。

检测到 `Server code/configuration changed` 或启动时缓存指纹不一致时，守护脚本会停止旧评分器，等待默认 5 秒，创建全新的评分数据目录并重启 C。待评任务重新从 B 拉取；选手的普通失败、HFM 暂时不可用或网络警告不会触发缓存轮换。其他致命退出会保留日志并返回错误，不无限重启。

`--data-dir` 在此是稳定的**守护根目录**：实际评分数据库位于其 `runs/run-*` 下，每次因配置变化重启会创建新目录，旧日志和成绩缓存不会删除。`active.json` 中的 `active_data_dir` 指向当前实际目录；以后查询 tasks.sqlite3、下载文件或测评日志时，使用这个实际目录。正常停止后再次运行相同命令，会复用最近一次目录；只有指纹变化才自动换新目录。

```bash
cat server_evaluator/runtime-watchdog/active.json
tail -f server_evaluator/runtime-watchdog/supervisor.log
```

每个 `runs/run-*` 下的 `worker.log` 保存该次 C 完整输出，原有 `evaluations/*/attempt-*/` 日志也保留。旧目录中算完但未确认回传的成绩仍保留以便核对；平台仍待评的任务可能在新配置下重新仿真。正式部署将守护根目录放在 C 的可靠本地持久磁盘，只运行一套评分器。修改守护脚本自身时，应停止并重新启动它。

可用 `--restart-delay 5` 设置重启间隔、`--shutdown-timeout 45` 设置优雅停止等待时间。按 `Ctrl+C` 停止守护脚本会同时清理它自己的评分子进程，不影响独立 HFM。

## 7.2 数据与部署说明

直接运行 `run.py` 时，默认数据目录为 `server_evaluator/runtime-shared-v1`，保存任务数据库、原始私有连接文件和 `evaluations/*/attempt-*/` 中的可信配置、阶段、日志、真实轨迹与结果。守护模式的数据目录结构见上一小节。临时 endpoint 副本在评测后删除。限制整个目录访问权限；轨迹审计不包含密钥。

升级时停止旧评分进程，使用新缓存目录，不复用 `runtime-remote-v1`。A 启动共享服务时会把合法旧 HTTP endpoint 自动换成共享协议文件；选手需重新上传新文件。平台已回传的失败记录不会自动重评，需重交或管理员重置。A 保存 4096 个旧回合编号，达到容量后应在空闲时重启；service_id 能检测中途重启，model_id 不能保证选手模型不可更改。

参数、文件锁、故障说明和 systemd 配置见 [server_evaluator/README.md](server_evaluator/README.md)。保留显式 `--submission-mode http` 与 `onnx` 供旧版开发联调，默认 `remote` 已改为共享目录。
