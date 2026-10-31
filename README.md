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

验证指定 checkpoint 时，在上述命令后添加 `eval.cp_num=CHECKPOINT_STEPS`。

## 3.可视化结果

验证完成后执行：

```bash
python rollout/visualize_rollout.py --run_name YOUR_RUN
```

## 4.submission
submission 用于把训练好的模型部署成推理服务，供仿真环境或评测平台调用。`submission` 使用 ONNX 模型，不能直接加载 PPO 的 `.zip` 文件。以下命令在项目根目录执行。

### 3.1.导出模型

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
```

python train/export_onnx.py \
  --model results/20261030_164400_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/20261030_164400_f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action/vecnormalize.pkl

导出产物为：

```text
submission/model/
├── policy2.onnx
├── policy2.metadata.json
└── policy2_vecnormalize_stats.npz
```

### 3.2.启动服务

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

### 3.3.本地联调与镜像

先启动 HFM socket 服务，并准备官方目标文件 `evaluation/results/target.json`，再执行：

```bash
bash eval_files/local_eval.sh
```

该脚本启动端口 `10001` 的推理服务并执行 F2b rollout 和评分。使用已有服务时：

```bash
python eval_files/run_test.py --no-build \
  --service-url http://127.0.0.1:8001 \
  --reference-config eval_files/references.yaml \
  --target-json evaluation/results/target.json \
  --output results/submission_infer_result.json
```
如果报错`ModuleNotFoundError: No module named 'environment'`使用：
```
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

默认服务地址为 `https://sii-lwy-sii-ste-eva.hf.space`，脚本向其 `/evaluate` 发送请求。指定服务地址和访问 token：

```bash
HF_SPACE_URL=https://YOUR-SPACE.hf.space \
HF_TOKEN=YOUR_TOKEN \
TAG=my-run \
bash submit.sh /path/to/submission.jsonl
```

依赖 `bash`、`python3` 和 `curl`。脚本中存在预填 token，使用时请用自己的 `HF_TOKEN` 覆盖，不要将 token 写进提交文件或日志。远程返回成功的评分 JSON 才算提交完成；当前本机核验的请求尚未获得评分响应，出现超时。
