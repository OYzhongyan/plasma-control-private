# F2b inference service

正式比赛请按[项目 README 第 8 节](../README.md#8-模型导出与平台提交)导出模型并上传单个 `submission_model.zip`。评分服务器统一运行 HFM 与评分；下面的 HTTP 服务和本地联调说明仅用于开发调试，不是选手提交步骤。

`inference2.py` and `service2.py` provide the F2b policy. The service listens on port `8001` by default; override `SUBMISSION_PORT2` if needed. It retains `GET /health`, `POST /reset` and `POST /act`, returning 12-dimensional actions.

Place models in `submission/model/`, or set `SUBMISSION_MODEL_DIR`. The policy prefers `policy2b.onnx`, otherwise `policy2.onnx`. No F2b model is bundled. Metadata and VecNormalize statistics must match the model.

Export a current training run and automatically create `results/submission_model.zip` (checkpoint paths are also supported):

```bash
python train/export_onnx.py --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
```

The exporter reads `f2_features.metadata.json` or `configs.pkl` from the run directory. For a relocated model, supply `--config /path/to/configs.pkl`. The supported interface is `reference_only_v1`: Actor and Critic both receive the current 90 target values in this order: `reference_Ip` (1), `reference_lX` (1), `reference_lcfs_points` (64), `reference_rX` (4), `reference_zX` (4), `reference_strike_r` (8), `reference_strike_z` (8). Neither measured state nor observation/action history is included. VecNormalize uses matching frozen 90D statistics. Absolute/delta action conversion still reuses `train/wrappers.py`, including delta accumulation. Old historical models are not supported; retrain and export a new model with its corresponding statistics.

Call `/reset` before each episode. `/health` returns 200 only when the model and required preprocessing artifacts load successfully, and reports feature mode, dimension and normalization readiness. Missing/mismatched artifacts return 503. `/act` failures return 500 instead of zero-action fallback.

```bash
python submission/service2.py
```

For local integration, start HFM and provide `evaluation/results/target.json`, then run:

```bash
bash eval_files/local_eval.sh
```

This starts one F2b service on local port `10001`. Use `python test/check_submission.py` for Docker API checks or `python test/test_submission.py` for real-environment integration.

The Docker `start_infer.sh` starts HFM, one F2b service and this repository's F2b runner. It requires the existing MATLAB Runtime and evaluation-data mounts. The image includes the F2b-only `eval_files/references.yaml`. The image is now intended for a single experiment rather than the original three-task evaluator.

Build from the project root with the shared training wrappers and environment helpers included:

```bash
tar -cf - --exclude='__pycache__' submission eval_files evaluation/eval/score \
  environment configs/shots.yaml train/__init__.py train/wrappers.py \
  | docker buildx build --load -f submission/Dockerfile -t plasma-f2b:local -
docker run --rm -p 8001:8001 --entrypoint python3 \
  plasma-f2b:local /app/submission/service2.py
```
