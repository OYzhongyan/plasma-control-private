# 1. C 服务器部署说明

主流程见[项目 README](../README.md)。按手册 **4.5**，C 主动从 B 拉取待评 `endpoint.json`，在 C 运行 HFM，将当前观测发给 A；A 使用自己的 checkpoint、归一化和 wrapper 返回 12 维物理电压，C 保存真实轨迹并回传七项贡献分。无需 Hugging Face，也不在 C 导入选手 Python 或加载选手权重。

# 2. 运行参数

```bash
python server_evaluator/run.py --target evaluation/results/target.json --check-config
python server_evaluator/run.py --target evaluation/results/target.json \
  --data-dir server_evaluator/runtime-remote-v1 --poll-interval 15
```

平台凭据使用 `PLATFORM_USERNAME` / `PLATFORM_PASSWORD`，或有效 `PLATFORM_TOKEN`；课题使用 `PLATFORM_PROBLEM_ID`。默认平台 `https://competition.opensii.ai`，可用 `PLATFORM_BASE_URL` 覆盖。凭据只存在 C，不传给 A，也不继承到评测／HFM 子进程。

| 参数 | 默认值及作用 |
| --- | --- |
| `--submission-mode` | `remote`；`onnx` 为旧三文件 ZIP 兼容模式 |
| `--endpoint-file PATH` | 在 C 本地联调连接文件，不登录或回传平台 |
| `--agent-timeout` | 15 秒，每次连接／读取 A 的 socket 超时 |
| `--agent-retries` | 2，仅重试传输失败；最大 5 |
| `--allow-agent-origin URL` | 默认空；C 管理员精确允许私网 origin，可重复 |
| `--hfm-host` / `--hfm-port` | `127.0.0.1` / `5558`，由 C 决定 |
| `--hfm-runtime-dir PATH` | 可选，自动启动 C 本地 Linux HFM；未指定则连接现有实例 |
| `--hfm-startup-timeout` | 180 秒；等待 HFM 就绪日志，不通过探测连接占用服务 |
| `--socket-timeout` | HFM socket 超时 300 秒 |
| `--evaluation-timeout` | 每份评测子进程硬超时 1800 秒 |
| `--memory-limit-mb` | 4096，Linux 评测子进程地址空间上限；不限制 A 或独立 HFM |
| `--seed` | 0，观测扰动／电源随机种子 |
| `--reference-config` | `eval_files/references.yaml`，其 `shot_id` 决定正式初始化 shot |
| `--env-config` | `configs/env_default.yaml` |
| `--obs-perturb-config` | `eval_files/obs_perturb.yaml`；目前支持噪声，拒绝非零观测延迟 |
| `--data-dir` | remote 默认 `server_evaluator/runtime-remote-v1` |
| `--sources` | `offline,manual`，不处理 auto |
| `--limit` | 每来源每轮最多 100 条，范围 1–500 |
| `--poll-interval` | 15 秒，整轮完成后等待时间 |
| `--http-timeout` | 平台／文件下载超时 30 秒 |
| `--max-file-mb` | remote 默认 0.0625 MiB，即 64 KiB；解析器也强制 64 KiB |
| `--download-retries` | 下载跨轮失败 3 次才记失败 |
| `--once` | 一轮后退出；仍有未处理／未确认项返回 1 |
| `--check-config` | 只验证本地配置，不连接 B、A 或 HFM |

每个远程调用还有父进程硬预算 `agent_timeout × (agent_retries + 1) + 5` 秒，覆盖 DNS 卡住和持续缓慢响应；它与整个仿真的总预算独立。正式总预算应按实际网络与 HFM 耗时统一设置。旧 `--inference-timeout` 只用于 ONNX 模式，不约束远程调用。

# 3. 地址、协议与提交身份

连接文件格式由 [submission/export_endpoint.py](../submission/export_endpoint.py) 生成，内容为版本、协议、HTTP(S) 根地址和可选 A 专用 Bearer 令牌。不接受额外字段、脚本、路径前缀、URL 用户信息或重定向。队伍身份以 B 的提交记录为准，不以 A 返回内容为准。

C 每次连接前检查 DNS 返回的所有 IP，并将 socket 固定到已检查 IP；HTTPS 仍校验原域名证书。HTTP 环境代理不生效，A 的令牌不会跟随跳转。默认只访问公网；例如 A 在同一集群的 `http://10.0.0.12:8001`，由 C 管理员在启动参数加：

```bash
--allow-agent-origin http://10.0.0.12:8001
```

这是精确协议／主机／端口允许项，不能用通配符。允许项不解决路由、防火墙或 NAT；两台机器仍需网络互通。公网建议 HTTPS 或受控 VPN。禁止访问链路本地（含云元数据地址）、组播与未指定地址，即使添加允许项也不能连接。

C 为每次评测生成随机 episode_id，step_id 从 0 递增。重试保持同一请求内容和编号；A 对重复最后一步返回缓存动作，避免 delta 累加两次。明确 HTTP 错误、错误编号、非法动作、模型身份变化不重试。model_id 只能帮助发现意外切换，不能证明远端代码不可被选手修改。

观测中的非有限数经 JSON null 传递，A 转回 NaN；C 原始观测用于评分。C 仅发送观测（包括控制目标 reference），不发送正式 shot 文件、初始化路径、目标文件或评分配置。网络往返耗时记录在轨迹审计中，不作为模型内核耗时扣分。

# 4. 状态、队列与故障

| 情况 | 处理 |
| --- | --- |
| 合法完成、合法提前终止、零分 | `completed` 与七项分数 |
| 非法连接 JSON、A 不可达／鉴权失败、错误响应、非法电压、RPC 硬超时 | 当前提交 `failed`，继续后续任务 |
| C 的 HFM／目标／可信环境故障 | 保留 `pending`，暂停本轮仿真，修复后重试 |
| 文件下载暂时失败 | 按下载次数跨轮重试 |
| 成绩回传失败 | 保存现有结果，下轮重传，不重新仿真 |

每份结果先持久化并及时回传。重启复用已保存结果；中断但未保存的回合重新开始。部分回传或响应丢失会查询平台记录确认；不能确认的结果继续保留。`--limit` 是每轮每来源上限，后续轮次继续查询剩余待评提交。

同一课题只运行一套评分器。数据目录有实例锁，但不同目录／机器不共享锁；同一个 A 服务也只能承载一个活动回合。收到停止信号后结束子进程；可选托管 HFM 随评分进程关闭，单独启动的 HFM 需单独管理。

# 5. 审计与升级

默认目录 `server_evaluator/runtime-remote-v1/`：

| 文件 | 用途 |
| --- | --- |
| `tasks.sqlite3` | 提交、结果与回传确认 |
| `downloads/*.json` | 原始 endpoint，可能包含 A 访问令牌，权限 0600 |
| `evaluations/*/attempt-*/trajectory.json` | C 生成的真实轨迹、网络耗时和无令牌审计 |
| `result.json` / `evaluation.log` | 子进程结果／失败原因 |
| `config.json` / `phase.json` | C 配置及当前／最后阶段，不含 A 令牌 |
| `hfm-runtime.log` | 选择托管 HFM 时的启动／运行输出 |

临时 endpoint 在每次评测后删除。整个数据目录应仅允许运行账户和管理员读取。升级格式、目标或评测配置后使用新目录；指纹不一致会拒绝旧缓存。已回传的 failed 需要选手重交或平台管理员重置；新缓存不会自动更改平台已完成的记录。

原 ONNX 模式保留：`--submission-mode onnx`，或者本地 `--model-file PATH.zip` 自动选择 ONNX；使用独立 `runtime-reference-v1` 缓存。新正式提交不需要导出 ONNX。

# 6. systemd 常驻部署

先手动完成一个真实 A + HFM 联调，再配置 [plasma-grader.service](plasma-grader.service) 的运行账户、项目路径和 Python。默认模板连接单独启动的 HFM；需要自动管理时添加 `--hfm-runtime-dir`。私网 A 的精确允许项同样添加到 `ExecStart`。

在 `/etc/plasma-grader.env` 放置 `PLATFORM_USERNAME`、`PLATFORM_PASSWORD`、`PLATFORM_PROBLEM_ID` 或有效 `PLATFORM_TOKEN`，格式为 `KEY=value`，限制权限：

```bash
sudo chmod 600 /etc/plasma-grader.env
sudo cp server_evaluator/plasma-grader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plasma-grader
sudo journalctl -u plasma-grader -f
```

停止：`sudo systemctl stop plasma-grader`。模板使用 `UMask=0077`，缓存目录应由运行账户持有；不要把凭据写进仓库。

# 7. 本地回归测试

```bash
python -m pytest server_evaluator/tests test/test_remote_service.py test/test_checkpoint_policy.py
```

测试使用本地假平台、真实 HTTP A 服务、小型 PPO 和替身 HFM，不访问生产平台。正式部署仍须在 C 运行真实 HFM 联调。
