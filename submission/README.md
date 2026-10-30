# F2b inference service

`inference2.py` and `service2.py` provide the F2b policy. The service listens on port `8001` by default; override `SUBMISSION_PORT2` if needed. It retains `GET /health`, `POST /reset` and `POST /act`, returning 12-dimensional actions.

Place models in `submission/model/`. The policy prefers `policy2b.onnx`, otherwise `policy2.onnx`. Metadata and VecNormalize statistics must match the model. No F2b model is bundled. The base policy retains its zero-action fallback when a model cannot be loaded.

```bash
python submission/service2.py
```

For local integration, start HFM and provide `evaluation/results/target.json`, then run:

```bash
bash eval_files/local_eval.sh
```

This starts one F2b service on local port `10001`. Use `python test/check_submission.py` for Docker API checks or `python test/test_submission.py` for real-environment integration.

The Docker `start_infer.sh` starts HFM, one F2b service and this repository's F2b runner. It requires the existing MATLAB Runtime and evaluation-data mounts. The image includes the F2b-only `eval_files/references.yaml`. The image is now intended for a single experiment rather than the original three-task evaluator.
