#!/usr/bin/env bash
set -euo pipefail

# Train F1 PPO, then train a denoising student from that training output.
# Usage:
#   bash tools/train_then_distill_f1.sh TRAIN_CONFIG [DISTILL_OUTPUT_DIR] [extra distill args...]
#
# Optional env:
#   PYTHON=python        Python executable, e.g. python or /path/to/env/bin/python
#   SKIP_TRAIN=1        Reuse existing PPO output and only run distillation
#   FORCE_COLLECT=0     Reuse cached distillation dataset if it exists

PYTHON="${PYTHON:-python}"
TRAIN_CONFIG="${1:?usage: bash tools/train_then_distill_f1.sh TRAIN_CONFIG [DISTILL_OUTPUT_DIR] [extra distill args...]}"
DISTILL_OUTPUT="${2:-}"
shift || true
if [[ $# -gt 0 ]]; then
  shift || true
fi

OUTPUT_DIR="$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig'))['output_dir'])" "$TRAIN_CONFIG")"
TASK="$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('task','F1').lower())" "$TRAIN_CONFIG")"

if [[ "$TASK" != "f1" ]]; then
  echo "ERROR: distill_noise_student.py currently supports F1 only, got task=$TASK" >&2
  exit 1
fi

TEACHER_MODEL="$OUTPUT_DIR/ppo_f1_latest.zip"
VECNORMALIZE="$OUTPUT_DIR/vecnormalize_stats.npz"
if [[ ! -f "$VECNORMALIZE" ]]; then
  VECNORMALIZE="$OUTPUT_DIR/vecnormalize.pkl"
fi
METADATA="$OUTPUT_DIR/f1_features.metadata.json"
if [[ -z "$DISTILL_OUTPUT" ]]; then
  DISTILL_OUTPUT="${OUTPUT_DIR}_noise_distill"
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  $PYTHON training/train_ppo.py --train-config "$TRAIN_CONFIG"
fi

for f in "$TEACHER_MODEL" "$VECNORMALIZE" "$METADATA"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
done

COLLECT_ARGS=()
if [[ "${FORCE_COLLECT:-1}" == "1" ]]; then
  COLLECT_ARGS+=(--force-collect)
fi

$PYTHON training/distill_noise_student.py \
  --train-config "$TRAIN_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  --vecnormalize "$VECNORMALIZE" \
  --metadata "$METADATA" \
  --output-dir "$DISTILL_OUTPUT" \
  "${COLLECT_ARGS[@]}" \
  "$@"
