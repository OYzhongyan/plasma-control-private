# A：远程 checkpoint 推理服务

A 保存模型和实际训练代码；B 接收 `endpoint.json`；C 保存官方测评 shot、运行 HFM 并评分。A 只接收当前观测并返回 **12 维物理电压设定值**。无需导出 ONNX，也不把模型、wrapper 或仿真环境上传给 B/C。

## 启动与提交

在 A 的项目根目录，使用已有训练环境：

```bash
pip install -r submission/requirements.txt
read -rsp 'A 服务访问令牌: ' SUBMISSION_ACCESS_TOKEN
echo
export SUBMISSION_ACCESS_TOKEN

python submission/export_endpoint.py \
  --service-url http://YOUR_SERVER_IP:8001 \
  --output results/endpoint.json

python submission/service2.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl \
  --host 0.0.0.0 --port 8001
```

上传 `results/endpoint.json` 到 B，然后保持 A 服务运行。将地址替换为 **C 可访问的实际 IP／域名**；0.0.0.0 是监听地址，127.0.0.1 仅指调用方自己。公网建议使用 HTTPS 反向代理。内网地址由 C 管理员提前加入允许列表。

访问令牌是 A 的专用凭据，不能使用平台登录 token；连接文件含该令牌，不要公开或提交 Git。新终端若需要生成文件／访问服务，需设置相同令牌。

模型路径也可为运行目录或 `checkpoints` 目录：运行目录优先最终模型；`checkpoints` 按最大训练步数选模型，只搭配同一步数归一化。可用 `SUBMISSION_MODEL_PATH`、`SUBMISSION_VECNORMALIZE_PATH`、`SUBMISSION_ADAPTER_FACTORY` 配置，或者直接传 CLI 参数。`bash submission/run.sh` 同样只启动推理，不启动 HFM。

## 如何复用 wrapper

[history.py](history.py) 中的 `CheckpointPolicyAdapter` 直接实例化 A 本机 `train/wrappers.py` 的观测和动作 wrapper。缓冲环境把 C 的 observation 交给原 wrapper 的 reset/step，保留历史、动作累加等行为；它不构造或调用本地 HFM。按 checkpoint 的真实空间检查，不强制 90 维。VecNormalize 使用训练文件并冻结统计，不更新奖励归一化。

默认适配器使用 `ReferenceObservationWrapper`、`NormalizedAction7DTo12DWrapper` 的既有构造接口。仅修改类内部逻辑时会复用这些修改；若改变类名、构造参数或采用其他接口，可在 A 新建工厂：

```python
# train/my_adapter.py
def build_adapter(model, metadata, config):
    # 返回自己的 Python 适配器对象，直接复用自己的训练代码。
    # 必须提供 reset()、encode(observation)、decode(model_action)。
    return MyAdapter(model, metadata, config)
```

```bash
python submission/service2.py --model results/YOUR_RUN/checkpoints \
  --adapter-factory train.my_adapter:build_adapter
```

`encode` 结果经冻结 VecNormalize 后传入 PPO.predict；`decode` 返回 12 维电压。A 不重复执行电源延迟、变化率限制或 PSM，这些由 C 管理。如果没有使用观测归一化，训练元数据必须明确 `observation_normalization=false`。

## HTTP 协议

所有请求可携带 `Authorization: Bearer <A专用令牌>`。设置 `SUBMISSION_ACCESS_TOKEN` 后三个端点都必须鉴权。一个服务只支持一个活动测评回合；C 串行调用，正式评测中不要从另一个程序手动 reset。

`GET /health`：返回 `ok`、`protocol=hfm-remote-policy-v1`、模型加载状态和诊断用 model_id。加载失败返回 503。

`POST /reset`：

```json
{"protocol":"hfm-remote-policy-v1","episode_id":"C生成的唯一回合ID"}
```

`POST /act`：

```json
{"protocol":"hfm-remote-policy-v1","episode_id":"C生成的唯一回合ID","step_id":0,"observation":{"Ip":[500000.0]}}
```

这里的 observation 仅示意格式；正式请求包含 C 的原始观测，默认 wrapper 需要 reference 字段。响应包含 `ok`、相同 protocol/episode_id/step_id、`action`（长度 12）及 model_id。缺失或非有限观测用 JSON null 表示，A 转为 NaN；返回动作必须有限。

同一回合的相同 reset 不再次重置。重复最后一步且内容相同返回缓存动作，不再次推理或累加；跳步、旧回合或相同步不同内容返回 409。推理出错后该回合不可继续，需要新的 episode_id。C 不信任选手返回的分数，也不向 A 发送 shot 文件／路径。

服务记住最多 4096 个已结束回合，拒绝旧 reset 重新激活它们；达到容量后拒绝新回合，需在无活动测评时重启 A 服务。不会通过删除旧编号来允许过期请求重放。

model_id 用于发现意外重启或加载变化，不是对选手模型不可修改的证明。网络往返耗时仅作 C 的审计，和模型内核耗时不同。

## 原 ONNX 工具

原 ONNX 导出和三文件打包工具保留作开发用途。服务可显式使用 `--backend onnx --model-dir submission/model`；正式默认路线是 checkpoint＋连接文件，不需要这一步。旧的无回合／步编号 HTTP 调用不是正式远程协议。
