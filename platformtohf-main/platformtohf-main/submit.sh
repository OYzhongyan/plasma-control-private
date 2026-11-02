#!/usr/bin/env bash
# Score two full JSON documents in local Chromium/Pyodide (no POST backend).
# Usage: bash submit.sh INFER_JSON TARGET_JSON [--output SCORE_JSON] [--tag NAME]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN (set PYTHON_BIN to your environment's Python)" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/submit.py" "$@"
