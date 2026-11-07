# F2b inference service

`inference2.py` and `service2.py` provide the F2b policy. The service listens on port `8001` by default; override `SUBMISSION_PORT2` if needed. It retains `GET /health`, `POST /reset` and `POST /act`, returning 12-dimensional actions.

Place models in `submission/model/`, or set `SUBMISSION_MODEL_DIR`. The policy prefers `policy2b.onnx`, otherwise `policy2.onnx`. No F2b model is bundled. Metadata and VecNormalize statistics must match the model.

Export a current training run (checkpoint paths are also supported):

```bash
python train/export_onnx.py --model results/YOUR_RUN/ppo_f2_latest.zip \
  --out submission/model/policy2.onnx \
  --vecnormalize results/YOUR_RUN/vecnormalize.pkl
```

The exporter reads `f2_features.metadata.json` or `configs.pkl` from the run directory. For a relocated model, supply `--config /path/to/configs.pkl`. Only `reference_only_v1` metadata (version 3) and newly trained 90D models are supported. The adapter uses the same `ReferenceObservationWrapper` and absolute/delta action wrapper as main. Features contain only the current reference, with no state or action history; VecNormalize uses the matching frozen training statistics. The export also packages the three artifacts into one `submission_model.zip` for platform upload.

The server grader uses the same physical supply delays as training: channels 0?10 (CS/PF) sample 2?5 ms each step and VS samples 0?1 ms, with a 1 ms simulator step. Those delays are applied by the HFM predictor, independently of observation perturbations; the policy adapter does not apply them a second time.

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
