# 1. 共享目录评分服务 C

按手册 4.5 主动从 B 拉取私有 `endpoint.json`，在 C 运行 HFM，通过签名共享文件向 A 请求观测对应的 12 维物理电压。A 使用自己的模型和 wrapper，C 不导入选手代码、不下载权重、不采信自报分数。完整部署命令见[项目 README](../README.md)。

```bash
python server_evaluator/run.py --target evaluation/results/target.json --check-config
python server_evaluator/run.py --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-shared-v1 --poll-interval 15
```

# 2. 参数与权限

| 参数 | 默认值及用途 |
| --- | --- |
| `--submission-mode` | `remote`：共享目录；`http` 与 `onnx` 是旧版兼容模式 |
| `--shared-root` | `/inspire/hdd/project/26-nuclear-fusion/public/plasma-agents`；也可用 `SHARED_AGENT_ROOT` |
| `--endpoint-file` | 在 C 联调私有连接文件，不连接平台 |
| `--agent-timeout` | 15 秒，每次等待有效签名回复的时限 |
| `--agent-retries` | 2，重复发布同一请求；最大 5 |
| `--agent-poll-interval` | 0.05 秒，共享回复轮询间隔 |
| `--hfm-runtime-dir` | 可选，由 C 启动并托管完整 Linux HFM 运行包 |
| `--hfm-startup-timeout` | 180 秒，HFM 就绪日志等待时间 |
| `--hfm-host` / `--hfm-port` | `127.0.0.1` / `5558`；只用于 C 与 HFM |
| `--socket-timeout` | HFM socket 超时 300 秒 |
| `--evaluation-timeout` | 每份评测子进程硬超时 1800 秒 |
| `--memory-limit-mb` | 4096，Linux 评测子进程地址空间预算，不限制 A 或独立 HFM |
| `--seed` | 0，观测扰动和电源随机种子 |
| `--reference-config` | `eval_files/references.yaml`，其中 `shot_id` 决定正式初始化 |
| `--env-config` / `--obs-perturb-config` | `configs/env_default.yaml` / `eval_files/obs_perturb.yaml` |
| `--data-dir` | `server_evaluator/runtime-shared-v1` |
| `--poll-interval` | 15 秒，整轮平台处理后等待时间 |
| `--sources` / `--limit` | offline,manual / 每来源每轮 100 条，范围 1–500 |
| `--max-file-mb` | 默认 0.0625 MiB，解析器也强制连接文件不超过 64 KiB |
| `--http-timeout` | B 平台及文件下载超时 30 秒，非 A/C 通信 |
| `--download-retries` | 文件下载跨轮重试 3 次后失败 |
| `--once` | 一轮后退出，仍有未确认任务时返回 1 |

每次等待有效回复使用 `agent_timeout`，超时后可重复发布同一请求。父进程对单个文件 I/O 阶段施加 `agent_timeout × (agent_retries + 1) + 5` 秒的硬时限；存储操作卡住时保留 pending，单纯没有 A 回复则按选手响应超时处理。该阶段预算最多 3600 秒，整场另有 `evaluation-timeout`。旧 `--inference-timeout` 只适用于 ONNX；`--allow-agent-origin` 只适用于旧 HTTP，不需要在共享模式配置。

管理员需确认 A、C 共享同一个根目录，两账户对 A 创建的邮箱均可读写／进入，且支持跨机器锁和原子重命名。代码使用目录 2770、消息 0660，组权限应结合实际项目组／ACL 设置，已有目录权限不会被擅自修改。不能只在一台机器上测试后就假定共享挂载正确。

Linux 路径操作通过固定目录描述符、no-follow 和普通文件检查限制在根目录下；选手 descriptor 不含目录路径。Windows 仅作本地功能测试，正式服务使用 Linux。公共目录可被有权限的账户读取或删除，HMAC 只提供真实性和关联校验，不提供保密或防拒绝服务。

# 3. 消息与故障处理

连接文件协议为 `hfm-shared-policy-v1`，transport 为 `shared_directory`，包含随机 agent_id 和密钥。共享请求／响应包含方向、随机 request_id、有效期、负载及 HMAC；响应还绑定请求摘要。密钥只保存在 A 本地和 B/C 的私有记录，绝不写入邮箱或日志。平台凭据不传给 A，也不继承到 HFM／评测子进程。

C 创建每回合 episode_id 和从 0 递增的 step_id；同一请求重发保持字节与编号一致。A 对重复最后一步返回缓存，不重复累加动作。C 只接受有限的 12 维电压，核对回合、步编号、model_id 和每次启动不同的 service_id；正式轨迹来自 C 的真实 post-step 观测。文件往返耗时只作审计。

| 故障 | 平台状态及处理 |
| --- | --- |
| 合法完成、提前终止、零分 | completed 和七项贡献分 |
| 无效 descriptor、A 邮箱不存在、无有效签名回复直至超时、非法电压、中途重启 | 当前提交 failed，继续其他任务 |
| C 共享根目录未挂载、存储 I/O／权限问题、已有活动 C 客户端、HFM 不可用 | 保留 pending，修复后重试 |
| 下载暂时失败 | 跨轮重试，达到次数才失败 |
| 成绩回传失败 | 缓存结果，下轮重传，不重新仿真 |

每份提交独立受监督子进程，退出自动释放 OS 文件锁，不依靠过期锁目录。A 与 C 的锁文件 inode 固定，不要手动删除锁文件；同一课题只部署一套评分器。中断但未得到结果的回合重新开始，已保存结果优先回传。已完成消息及时删除；超时后的孤立请求／响应到期后由仍运行的 A 清理。

# 4. 审计、升级与 systemd

HFM 独立常驻时，建议使用配置变化自动恢复入口：

```bash
python server_evaluator/watchdog.py \
  --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-watchdog \
  --poll-interval 15
```

首次启用前停止旧的直接运行评分器。守护脚本接收其余常驻 `run.py` 参数，但拒绝 `--hfm-runtime-dir`、`--once` 和本地单文件联调模式，不启动或停止 HFM。只识别可信的代码／配置变化和缓存指纹不一致日志；检测后优雅停止自己的 C 进程，清理其测评子进程，等待 5 秒并换新缓存重启。未完成提交继续从平台拉取。普通模型失败、平台／HFM 暂时故障不会触发换缓存；无已知配置警告的致命退出直接返回错误。

守护模式的 `--data-dir` 是稳定根目录，内部结构为：

```text
runtime-watchdog/
├── active.json             # active_data_dir 指向当前真正的评分数据目录
├── supervisor.log          # 启动、停止和重启原因
├── supervisor-lock/        # 同一守护根目录的实例锁
└── runs/run-*/
    ├── worker.log          # 该次 C 的 stdout/stderr
    ├── tasks.sqlite3
    ├── downloads/
    └── evaluations/
```

旧目录保留，不修改或强行复用旧指纹。正常停止后重启守护脚本，恢复 `active.json` 中最近一次缓存；若新配置不匹配会再自动轮换。旧的未确认回传成绩不迁移到新规则缓存，平台仍待评的提交可能重新评分；旧记录仍可审计。维护守护脚本自身时需重新启动守护进程。

参数 `--restart-delay` 默认 5 秒，`--shutdown-timeout` 默认 45 秒。Ctrl+C / SIGTERM 停止守护进程时不自动拉起 C；Linux 只清理它拥有的进程组，不向独立 HFM 发进程信号。监督日志／状态应放可靠本地持久磁盘。不要同时以其他数据目录另起一套同课题 C 评分器。

直接运行评分器时，`runtime-shared-v1` 保存 `tasks.sqlite3`、含私有密钥的 `downloads/*.json`（0600）、`evaluations/*/attempt-*/trajectory.json`、`result.json`、`evaluation.log`、`config.json` 和 `phase.json`；守护模式将这些文件放入当前 `runs/run-*`。临时 endpoint 副本评测后删除，可信配置和轨迹审计不含 A 密钥。正式部署的数据应位于 C 的可靠本地持久磁盘，并只允许运行账户和管理员读取；避免共享目录卡住时连监督阶段文件也被阻塞。

先停止旧服务，再换新缓存和共享协议文件；不要复用 HTTP 的 runtime-remote-v1。A 自动迁移合法旧 HTTP endpoint，但必须重新上传 B。已回传失败的提交需重交或平台管理员重置。

配置 [plasma-grader.service](plasma-grader.service) 的账户、项目、Python 和共享目录路径。模板使用 watchdog、缓存根目录 `/var/lib/plasma-grader/watchdog`，连接独立运行的 HFM，不支持托管 HFM。在 `/etc/plasma-grader.env` 设置平台助教 `PLATFORM_USERNAME`、`PLATFORM_PASSWORD`、`PLATFORM_PROBLEM_ID` 或有效 `PLATFORM_TOKEN`：

```bash
sudo chmod 600 /etc/plasma-grader.env
sudo cp server_evaluator/plasma-grader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plasma-grader
sudo journalctl -u plasma-grader -f
```

模板 `UMask=0077` 保护 C 的私有结果；公共消息显式保留项目组访问权限。配置变化由 watchdog 自行重启 C，systemd 使用 `Restart=no`，其他致命配置错误不会被无限重试；修复后执行 `sudo systemctl restart plasma-grader`。停止使用 `sudo systemctl stop plasma-grader`，独立启动的 HFM 需单独管理。

# 5. 回归与部署核验

```bash
python -m pytest server_evaluator/tests test/test_shared_channel.py test/test_shared_service.py
```

本地测试使用替身 HFM、平台和临时共享目录；真实跨机器挂载、权限、锁、时延和 HFM 必须按项目 README 的本地 endpoint 联调在部署环境核验。原 ONNX 和 HTTP 工具保留作显式开发兼容，不是正式默认路线。
