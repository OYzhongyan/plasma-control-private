#!/usr/bin/env bash
# Requires an HFM socket service and the F2b ONNX model.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/evaluation/eval:${PYTHONPATH:-}"
export SAISDATA_SHOTS_CONFIG="$ROOT/configs/shots.yaml"
SUBMISSION_PORT2=10001 python submission/service2.py >/tmp/f2b_service.log 2>&1 &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true' EXIT INT TERM
python eval_files/run_test.py --no-build \
  --service-url http://127.0.0.1:10001 \
  --output infer_result.json \
  --reference-config eval_files/references.yaml \
  --obs-perturb eval_files/obs_perturb.yaml \
  --target-json plasma-control/evaluation/result/target.json
python evaluation/eval/evaluate.py evaluation/results/target.json infer_result.json
