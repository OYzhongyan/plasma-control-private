# F2b scoring

Only `F2b` is scored, over 500 steps. The original 30-point weights are retained: Ip 6, LCFS 6, primary X-point distance 4, strike distance 3, primary X-point flux 4, secondary X-point distance 3, secondary X-point flux 4. Current-fuse, coil-limit, topology-penalty and X-point masking rules remain active.

```bash
python evaluation/eval/evaluate.py evaluation/results/target.json infer_result.json
```

Inference JSON must contain the `F2b` trajectory and timeout. Other task entries are no longer required. Targets prefer `F2b`, with a fallback to the historical official shared key `F2a`. Target fields include `lcfs_points`, `Xpt_main`, `Xpt_sec` and `strike`.

Output `score` and `scoreJson.score1` report the F2b score. Logged task scores and metric scores contain only F2b.

The ONNX rollout utility also runs only F2b:

```bash
python evaluation/eval/rollout_onnx_policy.py \
  --policy2-path submission/model/policy2.onnx \
  --target-path evaluation/results/target.json --f2b-shot-id 15892_500
```

Real rollout requires HFM and a model. Add `--validate-only --allow-missing-model` to check configuration and target format without a simulator.
