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

以下命令在服务器的 `plasma-control` 项目根目录、Bash 终端中执行。将 `YOUR_RUN` 替换为训练生成的运行目录名。

训练至少保存一次 checkpoint 后，可以按 `Ctrl+C` 中断训练，再执行以下命令验证。命令从 `checkpoints/` 中选择训练步数最大的模型，通过 `eval.cp_num` 加载，同时查找对应的 `*_vecnormalize.pkl` 归一化文件：

```bash
RUN_DIR=results/YOUR_RUN

LATEST_CP=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \
  -name 'ppo_f2_*_steps.zip' -printf '%f\n' | sort -V | tail -n 1)

if [[ -z "$LATEST_CP" ]]; then
  echo "没有找到 checkpoint，请检查 $RUN_DIR/checkpoints"
else
  CP_NUM="${LATEST_CP#ppo_f2_}"
  CP_NUM="${CP_NUM%_steps.zip}"
  echo "使用模型：$RUN_DIR/checkpoints/$LATEST_CP"

  python train/evaluate.py \
    eval.model_path="$RUN_DIR" \
    eval.cp_num="$CP_NUM" \
    target.default_reference_path=configs/xpt_reference_targets.json \
    'eval.shots="14795_400"' \
    eval.obs_perturb.enabled=true \
    actions.mode=delta
fi
```

`ppo_f2_latest.zip` 在训练正常结束后保存，中断训练时可能不存在。若要验证正常结束后保存的最终模型，使用：

```bash
python train/evaluate.py \
  eval.model_path=results/YOUR_RUN \
  eval.cp_num=null \
  target.default_reference_path=configs/xpt_reference_targets.json \
  'eval.shots="14795_400"' \
  eval.obs_perturb.enabled=true \
  actions.mode=delta
```

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

训练中断后，可从 `checkpoints/` 中选择训练步数最大的模型导出。将 `YOUR_RUN` 替换为运行目录名；必须使用同一步数对应的归一化文件：

```bash
RUN_DIR=results/YOUR_RUN

LATEST_CP=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \
  -name 'ppo_f2_*_steps.zip' -printf '%f\n' | sort -V | tail -n 1)

if [[ -z "$LATEST_CP" ]]; then
  echo "没有找到 checkpoint，请检查 $RUN_DIR/checkpoints"
else
  MODEL_PATH="$RUN_DIR/checkpoints/$LATEST_CP"
  VECNORMALIZE_PATH="${MODEL_PATH%.zip}_vecnormalize.pkl"
  if [[ ! -f "$VECNORMALIZE_PATH" ]]; then
    echo "缺少对应的归一化文件：$VECNORMALIZE_PATH，请检查 checkpoint.save_vecnormalize 配置"
  else
    echo "导出模型：$MODEL_PATH"
    python train/export_onnx.py \
      --model "$MODEL_PATH" \
      --out submission/model/policy2.onnx \
      --vecnormalize "$VECNORMALIZE_PATH"
  fi
fi
```

若训练已正常结束，也可以导出最终模型：

```bash
python train/export_onnx.py \
  --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
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

重新导出模型后，如果推理服务已启动，需停止并重新运行上述服务命令，才能加载新的模型及归一化统计。

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


### 4.4.平台提交

在 https://competition.opensii.ai/ 中提交json文件。
