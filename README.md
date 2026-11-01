## 1.训练模型

```bash
python train/train_ppo.py \
  output_dir=f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action \
  reset_perturbation.fix_reset_params=true \
  num_envs=32 \
  features.delay_step=1 \
  actions.mode=delta
```

## 2.验证模型

将 `YOUR_RUN` 替换为训练生成的 `results/` 下的运行目录名。

```bash
python train/evaluate.py \
  eval.model_path=results/YOUR_RUN \
  target.default_reference_path=configs/xpt_reference_targets.json \
  'eval.shots="14795_400"' \
  eval.obs_perturb.enabled=true \
  actions.mode=delta
```

例如
```
python train/evaluate.py \
  eval.model_path=results/20261101_092602_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action \
  target.default_reference_path=configs/xpt_reference_targets.json \
  'eval.shots="14795_400"' \
  eval.obs_perturb.enabled=true \
  actions.mode=delta
```

验证指定 checkpoint 时，在上述命令后添加 `eval.cp_num=CHECKPOINT_STEPS`。

## 3.可视化结果

验证完成后执行：

```bash
python rollout/visualize_rollout.py --run_name YOUR_RUN
```

例如
```
python rollout/visualize_rollout.py --run_name 20261101_092602_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action
```

## 4.submission
submission 用于把训练好的模型部署成推理服务，供仿真环境或评测平台调用。`submission` 使用 ONNX 模型，不能直接加载 PPO 的 `.zip` 文件。以下命令在项目根目录执行。

### 4.1.导出模型

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
```

例如
```
python train/export_onnx.py \
  --model results/20261101_092602_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/20261101_092602_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action/vecnormalize.pkl
```

导出产物为：

```text
submission/model/
├── policy2.onnx
├── policy2.metadata.json
└── policy2_vecnormalize_stats.npz
```

### 4.2.启动服务

启动模型推理 HTTP 服务，供评测程序或外部平台调用训练好的模型。

```bash
pip install -r submission/requirements.txt
python submission/service2.py
```

默认端口为 `8001`，可通过环境变量 `SUBMISSION_PORT2` 修改。另开终端检查接口：

```bash
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/reset \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":"F2b"}'
```

每个 episode 先调用 `/reset`，再通过 `POST /act` 发送 `{"observation": {...}}`，返回 12 维物理电压。`/health` 会报告模型输入维度和归一化加载情况；模型或必要统计文件缺失、维度不匹配时返回 HTTP 503，推理异常返回 HTTP 500，不再静默返回零动作。可用环境变量 `SUBMISSION_MODEL_DIR` 指定模型目录。

### 4.3.本地联调与镜像

在项目根目录打开一个终端，启动本地评测默认连接的 HFM socket 服务（端口 `5558`）：

```bash
LISTEN_PORT=5558 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash hfm-runtime-release-sealed-latest/run_hfm_server.sh
```

保持该终端运行，等待输出 `Waiting for a client to connect...`。如果仿真器不在项目根目录下，将上述脚本路径替换为实际路径。

准备包含 F2b 目标数据的官方文件 `evaluation/results/target.json`，然后在同一计算节点的另一个终端进入项目根目录执行：

```bash
bash eval_files/local_eval.sh
```

该脚本启动端口 `10001` 的推理服务并执行 F2b rollout 和评分。使用已有服务时，在项目根目录执行以下命令；`PYTHONPATH` 用于确保 Python 能导入项目中的 `environment` 等模块：

```bash
PYTHONPATH="$PWD:$PWD/evaluation/eval:${PYTHONPATH:-}" \
SAISDATA_SHOTS_CONFIG="$PWD/configs/shots.yaml" \
python eval_files/run_test.py --no-build \
  --service-url http://127.0.0.1:8001 \
  --reference-config eval_files/references.yaml \
  --target-json evaluation/results/target.json \
  --output results/submission_infer_result.json
```


## 5.platformtohf评测

该工具提交边界系数 JSONL，不接收 F2b 的 PPO / ONNX 模型，也不接收 rollout 结果。实际脚本位于两层同名目录中；以下命令在 Linux / WSL 的 Bash 中执行：

```bash
cd platformtohf-main/platformtohf-main
bash submit.sh                         # 提交自带的 submission.jsonl
bash submit.sh /path/to/submission.jsonl
TAG=my-run bash submit.sh /path/to/submission.jsonl
```

JSONL 每行是一个 JSON 对象，包含 `tag`、`nfp`、`boundary.r_cos` 和 `boundary.z_sin`；两个系数矩阵的样例形状均为 `5 × 9`，完整样例见同目录的 `submission.jsonl`。

### 5.1.配置 Hugging Face

以下配置均可通过 Bash 环境变量覆盖，无需把自己的 token 写进脚本：

`submit.sh` 中的配置应保持为以下 Bash 语法；地址使用纯 URL，不要粘贴 `[https://...](https://...)` 这样的 Markdown 链接。每个赋值单独占一行，`SCRIPT_DIR` 和 `JSONL_PATH` 之间不要添加续行反斜杠：

```bash
HF_SPACE_URL="${HF_SPACE_URL:-https://sii-lwy-sii-ste-eva.hf.space}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
JSONL_PATH="${1:-$SCRIPT_DIR/submission.jsonl}"
HF_TOKEN="${HF_TOKEN:-}"
TAG="${TAG:-}"
```

上面是推荐的脚本配置写法，token 默认留空。仓库现有脚本含预填 token，若采用此写法，请同步替换脚本中的 `HF_TOKEN` 配置行。`${HF_SPACE_URL:-默认地址}` 表示：环境变量非空时使用环境变量，否则使用默认地址；`${HF_TOKEN:-}` 和 `${TAG:-}` 表示未配置时为空。`SCRIPT_DIR` 自动定位脚本目录，通常无需修改。

| 配置项 | 如何填写 |
| --- | --- |
| `HF_SPACE_URL` | 评测服务的根地址，默认是 `https://sii-lwy-sii-ste-eva.hf.space`，不要添加 `/evaluate` 或使用 Markdown 链接格式。脚本会自动追加 `/evaluate`。 |
| `HF_TOKEN` | 有权限访问目标 Space 的 Hugging Face User Access Token。脚本会将其放入 `Authorization: Bearer ...` 请求头。 |
| `TAG` | 本次提交的标签，例如 `team-a-run-01`；优先于 JSONL 内的 `tag`，不会修改原文件，也不会自动创建排行榜。 |
| 第一个命令行参数 | JSONL 文件路径；省略时使用脚本目录中的 `submission.jsonl`。`/path/to/submission.jsonl` 是占位符，需要替换。 |

1. **确认评测服务地址。** 使用现有服务时，向维护者确认 Space 地址及访问权限。Space 页面地址（例如 `https://huggingface.co/spaces/SII-LWY/sii-ste-eva`）不是 API 根地址；应使用运行应用的 `https://...hf.space` 地址。地址从 Space 的应用链接或嵌入信息中获取。
2. **创建访问 token。** 登录 Hugging Face，打开 [Settings → Access Tokens](https://huggingface.co/settings/tokens)，创建自己的 token。私有 Space 需要账号本身已有访问权限；token 无法赋予账号原本没有的权限。可使用仅允许读取目标 Space 的细粒度 token；按服务端要求配置额外的应用认证。无需为了调用评测接口授予仓库写权限。参见 [官方 Token 文档](https://huggingface.co/docs/hub/security-tokens)。
3. **如果使用自己的 Space。** 先在 Space 中部署能接收 `{"points": [...], "tag": "..."}` 的 `POST /evaluate` 评分服务，再填写其应用地址。仅创建空 Space 或修改地址不会自动部署评测程序、保存成绩或生成排行榜。参见 [官方 Spaces 文档](https://huggingface.co/docs/hub/spaces-overview)。

在 `platformtohf-main/platformtohf-main` 目录中执行，交互输入 token（输入不会显示）：

```bash
export HF_SPACE_URL="https://sii-lwy-sii-ste-eva.hf.space"
read -rsp 'Hugging Face token: ' HF_TOKEN
echo
export HF_TOKEN
TAG=team-a-run-01 bash submit.sh ./submission.jsonl
unset HF_TOKEN
```

`HF_SPACE_URL` 可换成已部署相同接口的自有 Space。也可以为单次命令指定变量（以下均为占位值）：

```bash
HF_SPACE_URL=https://YOUR-SPACE.hf.space \
HF_TOKEN=YOUR_TOKEN \
TAG=my-run \
bash submit.sh /path/to/submission.jsonl
```

依赖 `bash`、`python3` 和 `curl`。脚本中存在预填 token，使用时请用自己的 `HF_TOKEN` 覆盖，不要将 token 写进提交文件或日志。公开服务若不需要认证，应先将脚本的 token 配置改为 `HF_TOKEN="${HF_TOKEN:-}"`，再不设置 `HF_TOKEN`；当前含预填值的 `${HF_TOKEN:-...}` 写法在变量为空时仍会使用预填 token。

远程返回 `"success": true` 的评分 JSON 才算评测完成；`"得分": 0.0` 表示评测成功但得分为零。此前一次提交已获得成功响应，耗时约 342 秒，计算期间可能长时间没有输出。`401/403` 应检查认证与权限；`404` 应同时检查 Space 地址、接口路径和私有服务的访问权限，不能单凭 404 判定服务下线。
