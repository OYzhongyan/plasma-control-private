# A：共享目录 checkpoint 推理服务

在选手服务器 A 的项目根目录执行：

```bash
pip install -r submission/requirements.txt
python submission/shared_service.py --model results/YOUR_RUN/checkpoints
```

或 `bash submission/run.sh --model results/YOUR_RUN/checkpoints`。程序自动加载最新 checkpoint 和匹配归一化，创建邮箱及本地 `results/endpoint.json`。将该文件上传平台 B，保持进程运行；无需 IP、端口、HTTP 或 ONNX。C 在自己的 HFM 上仿真并评分。

| 参数 | 用途 |
| --- | --- |
| `--model` | 必填，checkpoint 文件、checkpoints 目录或训练运行目录 |
| `--vecnormalize` | 可选，显式指定匹配的 VecNormalize.pkl |
| `--shared-root` | 默认 `/inspire/hdd/project/26-nuclear-fusion/public/plasma-agents`；可用 `SHARED_AGENT_ROOT` 设置 |
| `--output` | 默认本地 `results/endpoint.json`；已有文件复用服务编号和密钥，多模型需不同文件 |
| `--poll-interval` | 默认 0.05 秒，检查新请求间隔 |
| `--adapter-factory` | 本机 `module:function` 适配工厂 |
| `--device` | 模型运行设备，默认 cpu |

checkpoint 的配套文件和动作配置应保留在原运行目录。观测归一化统计冻结；缺少所需匹配文件时不会静默用原始观测。模型修改后重启服务，正式测评期间不要并行启动第二个服务或手动 reset。

## 提交文件与共享消息

`endpoint.json` 是小于等于 64 KiB 的严格 JSON，固定包含 `version=1`、`protocol=hfm-shared-policy-v1`、`transport=shared_directory`、随机 `agent_id` 和 `access_token`。生成器自动填写，无需手工修改；C 使用管理员配置的共享根目录，不接受选手指定任意文件路径。

**密钥只存在 A 的本地私有提交文件及 B/C 的受控记录，不能放到公共邮箱中。** 公共邮箱为 `<shared-root>/<agent_id>/requests` 与 `responses`，消息含 HMAC-SHA256 签名、方向、请求 ID、有效期和请求摘要，写完后原子替换。C 先验证签名／关联编号再接收动作。签名不能防止有目录权限的人读取观测、删除消息或造成服务不可用；部署需合理设置项目组／ACL，A、C 均有访问权限，且时钟同步。Linux 文件系统必须支持跨机器锁和原子重命名。

内部复用 health/reset/act 和原有 episode_id/step_id 协议：重复最后一步返回缓存结果，跳步、旧回合和冲突请求失败。返回动作必须恰好 12 个有限物理电压；非有限观测用 JSON null 传递，在 A 转回 NaN。每次服务启动有新 service_id，C 检测中途重启，reset/act 的签名请求必须绑定该编号，旧进程留下的请求不能推进新进程；model_id 仅用于诊断。一个邮箱同时只允许一个 C 客户端，4096 个旧回合达到容量后需空闲重启。完成请求删除，过期孤立消息由 A 清理。

## 复用自己的训练 wrapper

`history.py` 中的 `CheckpointPolicyAdapter` 在 A 实际实例化训练的 `ObservationWrapper` 和 `NormalizedAction7DTo12DWrapper`。缓冲环境把 C 的原始 observation 交给 reset/step，不在 A 创建 HFM；归一化统计冻结，输入空间按模型、metadata 和实际 wrapper 校验。

默认 baseline 恢复自 SII-585-THU commit `ffda09ab4621ede477ff71247547f8cbc43d133a`：4 帧 188 维观测加 5 帧 12 维物理动作历史，共 812 维。Actor 与 Critic 使用相同输入；Actor 输出 7 维归一化 delta，经累加、配对映射后返回 12 维物理电压。每次 reset 清空历史及增量累加状态，动作历史保存转换后的物理电压，避免重复累加。旧 90 维 `reference_only_v1` 仅用于加载 metadata 与 checkpoint 匹配的已有产物，不作为默认训练接口。

训练使用同一上游的 dense reward；其平滑映射、soft current mask 和部分权重与正式评分不同，训练回报总和不能作为比赛得分。C 始终使用正式七项满分 30 分的评分器。训练、离线评估与记录命令见根目录 [README.md](../README.md) 第 5、6、9 节；评分服务维护分支为 `grading`。

改变 wrapper 类名、构造参数或接口时，在 A 提供工厂：

```python
# train/my_adapter.py
def build_adapter(model, metadata, config):
    # 对象需提供 reset()、encode(observation)、decode(model_action)。
    return MyAdapter(model, metadata, config)
```

```bash
python submission/shared_service.py --model results/YOUR_RUN/checkpoints \
  --adapter-factory train.my_adapter:build_adapter
```

encode 后应用冻结 VecNormalize，模型预测后由 decode 返回 12 维设定电压。电源传输延迟、变化率限制和 PSM 由 C 执行，A 不重复施加。

## 旧工具

`service2.py`、`export_endpoint.py` 是旧 HTTP 开发工具，原 ONNX 导出工具也保留。它们生成的 HTTP 连接文件不适用于当前正式共享目录模式；请使用上面的单条启动命令重新生成提交文件。
