#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/start_infer.sh" ]]; then
  exec bash "$SCRIPT_DIR/start_infer.sh" "$@"
fi
exec bash "$SCRIPT_DIR/submission/start_infer.sh" "$@"
