#!/usr/bin/env bash
# A hosts inference only. Official HFM/shot/scoring remain on C.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/shared_service.py" "$@"
