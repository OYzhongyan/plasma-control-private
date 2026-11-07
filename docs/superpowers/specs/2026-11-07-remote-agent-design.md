# A/B/C 远程 checkpoint 推理测评

## 已确认的需求

用户已在对话中批准：A 在 B 上传连接配置文件；C 从 B 下载该文件，连接 A 的推理服务，在 C 运行隐藏 shot 的 HFM，逐步交换观测与 12 维物理电压，最终由 C 评分并回传 B。A 使用自己的 checkpoint、冻结归一化和实际训练 wrapper，无需 ONNX 或上传训练代码。

## 组件和默认决定

- A：现有 Flask 服务增加 checkpoint 后端，保留 ONNX 为开发选项；默认 checkpoint。A 启动时只加载本地模型，不启动 HFM。提供 health/reset/act。
- B：接收 `endpoint.json`，不接收 URL 之外的可执行内容；平台课题应允许 JSON 文件。队伍和提交身份以 B 的记录为准。
- C：默认使用 RemoteGrader；保留显式 ONNX 旧模式。复用既有 HFM、目标构造、真实轨迹与七项评分，连接配置不控制 shot、target、步数、HFM 地址或评分规则。
- HFM 仍由 C 管理。提供可选 `--hfm-runtime-dir` 自动启动能力；不指定时连接已在 C 启动的 HFM。绝不在 A 启动正式测评环境。

## endpoint.json

最多 64 KiB、UTF-8、严格 JSON，拒绝重复键和非有限数。字段：version=1、protocol=hfm-remote-policy-v1、service_url、可选 access_token。URL 为 HTTP(S) 根地址，禁止用户信息、query、fragment 和路径前缀。access_token 只作为 A 的 Bearer 凭据，不得使用平台凭据。

默认仅允许公网目的地址；C 的管理员可用重复的 `--allow-agent-origin http://host:port` 显式允许特定私网地址。C 在每次连接前解析并校验 IP、固定连接该 IP，HTTPS 仍校验证书和原主机名；不跟随重定向、不使用环境代理。限制响应大小和请求次数，防止连接文件把 C 当成任意内部请求代理。

## 协议与状态

- health 返回 protocol、ok、model_loaded，以及用于诊断的 model_id；不泄露本地模型路径。
- reset 请求含 protocol、episode_id，回传相同标识。相同回合的重复 reset 不重复清空；冲突参数返回 409。
- act 请求含 protocol、episode_id、step_id（从 0 开始）、observation。返回相同标识、12 个有限电压和 model_id。
- A 只支持每服务一个活动回合，处理串行。相同最后一步且 payload 相同返回缓存结果，不再次执行 wrapper/模型；相同步不同 payload、过期或跳步返回 409。reset 新回合后旧回合请求被拒绝。
- 推理失败后该回合标记不可继续，避免 encode 已更新历史后再次推理。C 只重试传输失败和丢失响应，不重试明确的 HTTP/协议/模型错误；重试使用相同标识和请求字节。
- C 只发送公开 observation，不发送 shot_id、初始化文件、目标文件路径、配置文件或评分代码。非有限观测在 JSON 中编码为 null，A 转回 NaN；原始 C 轨迹用于评分，不被网络序列化替代。
- HTTP 往返时间只作审计，不当作模型内核推理耗时扣分；C 的请求和总评测超时可配置。

## A 的 checkpoint 适配

直接加载本地 SB3 PPO checkpoint 和同一步数的 VecNormalize.pkl，冻结统计。支持传入 run 目录选择最终模型或最新 checkpoint。默认适配器实际实例化 train/wrappers.py 的观测和动作 wrapper，通过只接收 C 观测的缓冲环境复用 reset/step；不构造本地 HFMSimulator。无固定 90 维校验，按模型实际观测／动作空间检查。

选手改变类名或接口时，可用本地 `--adapter-factory module:function`，接收 model、metadata、config，返回 reset/encode/decode 对象。该工厂只在 A 运行，C 不获取、不导入它。

## 可靠性与数据

C 对每份提交启动受监督子进程，保留取消、总超时、阶段超时和内存限制。连接 A 的失败或非法输出记该提交 failed；HFM／主办方配置问题保留 pending。平台成绩缓存、回传重试和实例锁沿用现有 worker。

新目录 `server_evaluator/runtime-remote-v1` 防止复用旧模型格式结果。endpoint.json 在 A 生成和 C 保存时按敏感文件处理，日志和成绩不输出 access_token；C 子进程不继承 PLATFORM_* 凭据。远程 model_id 仅用于检测意外重启，不证明选手模型不可更换。

## 交付和验证

main 更新 A 的服务、checkpoint 适配、连接文件生成器、启动脚本及 README。server/offline-grading 同步公共协议，并增加 C 的 HTTP 客户端、远程评分、CLI、HFM 启动和部署说明。根目录切换分支，分别保存本地提交，不推送。

测试包括真实小 PPO＋非恒等归一化的训练 wrapper 动作一致性、重复请求和响应丢失、鉴权／会话、非法 URL／重定向／私网限制、真实 HTTP 服务＋替身 HFM 七项评分、平台拉取和回传。C 的远程链路禁止导入 train／Torch／SB3。真实 HFM 和生产平台联调另行在部署环境执行。
