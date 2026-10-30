## 训练模型

```bash
python train/train_ppo.py \
  output_dir=f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action \
  reset_perturbation.fix_reset_params=true \
  num_envs=32 \
  features.delay_step=1 \
  actions.mode=delta
```

## 验证模型

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

## 可视化结果

验证完成后执行：

```bash
python rollout/visualize_rollout.py --run_name YOUR_RUN
```

## submission：模型导出与推理服务

以下命令在项目根目录执行。`submission` 使用 ONNX 模型，不能直接加载 PPO 的 `.zip` 文件。

### 导出模型

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
```

如果有与模型匹配的特征 metadata，导出时追加 `--metadata /path/to/features.metadata.json`。导出产物为：

```text
submission/model/
├── policy2.onnx
├── policy2.metadata.json
└── policy2_vecnormalize_stats.npz
```

**当前限制：**`train/train_ppo.py` 使用历史观测特征，而现有导出工具默认 metadata 和 `submission` 预处理使用旧版特征接口。导出成功不代表能够直接部署当前训练模型，仍需核对输入特征、归一化和动作解释；不要用默认 metadata 替代匹配的 metadata。

### 启动服务

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

`POST /act` 接收 `{"observation": {...}}`，返回 12 维动作。模型缺失或推理异常时，当前服务可能返回零动作，因此 `/health` 成功不能证明训练模型已正确运行。

### 本地联调与镜像

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

在 Linux / WSL 中构建镜像并单独启动推理服务：

```bash
tar -cf - --exclude='__pycache__' submission eval_files evaluation/eval/score \
  | docker build -f submission/Dockerfile -t plasma-f2b:local -
docker run --rm -p 8001:8001 --entrypoint python3 \
  plasma-f2b:local /app/submission/service2.py
```

镜像内的完整入口是 `/app/run.sh`，需要赛事环境的 `/saisdata/40` 数据挂载及 MATLAB Runtime。上面的命令只启动推理服务，不启动仿真器。正式上传镜像需要另行指定目标平台的镜像仓库和凭据。

## platformtohf：远程边界设计评测

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
