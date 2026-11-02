> 本分支 `server/offline-grading` 用于服务器按操作手册 **4.5** 主动拉取任务、评分并回传。部署说明见 [server_evaluator/README.md](server_evaluator/README.md)。交给选手的 baseline 保留在 `main` 分支。

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


## 5.platformtohf：轨迹 JSON 评分（免费 Static Space + Pyodide）

该工具读取仿真得到的完整轨迹 JSON 和官方目标 JSON，复用 `evaluation/eval/score` 比较实际轨迹与目标并评分。不再提交边界系数 JSONL，也不再请求师兄的 `/evaluate` 接口；无需 HF Token、PRO 或 CPU 服务器。

### 5.1.命令行评分

在项目根目录安装一次依赖，使用当前 Conda 环境的 Python：

```bash
python -m pip install -r platformtohf-main/platformtohf-main/requirements.txt
python -m playwright install chromium
```

Linux 如果提示缺少 Chromium 系统库，执行 `python -m playwright install --with-deps chromium`（可能需要管理员权限）。

两个必填路径依次为**轨迹、目标**：

```bash
PYTHON_BIN=python bash platformtohf-main/platformtohf-main/submit.sh \
  results/submission_infer_result.json \
  evaluation/results/target.json \
  --tag team-a-run-01 \
  --output results/submission_score.json
```

脚本启动本机无界面 Chromium 和临时回环静态文件服务，在浏览器的 Pyodide + NumPy 中计算，结束后自动关闭。计算不依赖远程 CPU 或 HFM 服务。路径相对于当前工作目录，也可传绝对路径；有空格的路径加引号。`PYTHON_BIN` 默认是 `python3`。Windows 可直接运行 `python platformtohf-main/platformtohf-main/submit.py`，参数相同。

成功时输出 `success: true`、总分 `score`、各指标贡献分 `scoreJson` 和有效步数 `K_eff`，满分 30 分；`--output` 同时保存 JSON，`--tag` 仅标记本次计算。失败时退出码非零，不生成有效分数。首次需联网下载 Pyodide 和 NumPy，网络较慢可添加 `--timeout 600`。

轨迹顶层应为 `{"F2b": {"trajectory": {...}, "timeout": false}}`。目标须包含 `F2b`（兼容原评分器的 `F2a` 目标回退），且目标块包含 `lcfs_points`、`Xpt_main`、`Xpt_sec`、`strike`。A1/A2/B1/B2/B3、训练参考配置或边界系数 JSONL 不可替代该目标。

### 5.2.部署免费 Static Space

1. 创建或使用已有 HF Space，选择 **Static → Blank**；公开展示时选择 Public。
2. 将 `platformtohf-main/platformtohf-main` 内的 `README.md`、`index.html`、`app.js`、`worker.js`、`compute.py`、`scorer_bundle.py`、`LICENSE` 上传到 Space 仓库**根目录**。上传解压后的文件，不是 ZIP 本身；无需上传训练代码或仿真器。
3. 等待发布，打开 App，选择轨迹和目标两个 JSON，点击“开始评分”，可查看分数并下载 `f2b_score.json`。网页用户无需安装 Python、Playwright 或填写 token。

本地预览（项目根目录）：

```bash
python -m http.server 8000 --bind 127.0.0.1 \
  --directory platformtohf-main/platformtohf-main
```

打开 `http://127.0.0.1:8000`，不要通过 `file://` 双击 HTML。

命令行也可以加载已部署的新版 Static 网页：

```bash
PYTHON_BIN=python bash platformtohf-main/platformtohf-main/submit.sh \
  results/submission_infer_result.json \
  evaluation/results/target.json \
  --space-url https://YOUR-STATIC-SPACE.hf.space \
  --output results/submission_score.json
```

`--space-url` 或 `HF_SPACE_URL` 是部署本目录文件的公开 Static 应用根地址，不加 `/evaluate`。未设置时使用本地页面；如果之前导出过旧地址，先 `unset HF_SPACE_URL`。数据在本机浏览器中计算，不发送到 HF 评分后端；只加载你信任的页面代码。

### 5.3.规则与限制

沿用原评分器的电流、LCFS、主/次 X 点、打击点和磁通偏差规则，以及提前结束、拓扑、线圈电流、超时和推理时间惩罚。电流参考曲线仍由 `config.build_Ip_ref()` 定义，并非任意读取目标 JSON 中的 `Ip`。

单个输入上限为 128 MiB，可处理规模取决于设备内存。此版本计算并下载单次评分，不运行仿真、不验证轨迹来源，也不自动维护多人共享排行榜。浏览器代码和用户选择的目标均可查看。

项目评分规则更新后，重新生成快照并同步上传：

```bash
python platformtohf-main/platformtohf-main/build_scorer.py
python platformtohf-main/platformtohf-main/build_scorer.py --check
```

详见 [platformtohf README](platformtohf-main/platformtohf-main/README.md)。参考：[HF Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)、[Pyodide Web Worker](https://pyodide.org/en/stable/usage/webworker.html)。
