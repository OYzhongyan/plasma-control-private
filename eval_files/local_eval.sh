#!/usr/bin/env bash
# Run local semifinal evaluation against the Python submission services.
#
# Usage:
#   bash eval_files/local_eval.sh
#
# Prerequisites:
# - HFM socket service is already running on the port configured by
#   configs/env_default.yaml.
# - submission/model contains the ONNX model files and VecNormalize stats.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/evaluation/eval:${PYTHONPATH:-}"
export SAISDATA_SHOTS_CONFIG="$ROOT/configs/shots.yaml"

pkill -f "submission/service1.py" 2>/dev/null || true
pkill -f "submission/service2.py" 2>/dev/null || true
sleep 1

SUBMISSION_PORT=10000 python submission/service1.py >/tmp/svc1.log 2>&1 &
SVC1_PID=$!
SUBMISSION_PORT2=10001 python submission/service2.py >/tmp/svc2.log 2>&1 &
SVC2_PID=$!

cleanup() {
  kill "$SVC1_PID" "$SVC2_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
curl --fail http://127.0.0.1:10000/health
curl --fail http://127.0.0.1:10001/health

python eval_files/run_test.py --no-build \
  --service-url http://127.0.0.1:10000 \
  --service-url2 http://127.0.0.1:10001 \
  --output ./infer_result.json \
  --task-config eval_files/task_references.yaml \
  --obs-perturb eval_files/obs_perturb.yaml

python evaluation/eval/evaluate.py evaluation/results/target.json infer_result.json
