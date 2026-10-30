"""Submission inference for policy1 (F1: divertor -> limiter)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_common import BaseOnnxPolicy  # noqa: E402


class Policy(BaseOnnxPolicy):
    """Policy interface used by service1."""

    def __init__(self, model_dir: str | Path | None = None):
        super().__init__(
            task="F1",
            model_dir=Path(model_dir) if model_dir is not None else MODEL_DIR,
            preferred_name="policy1.onnx",
        )
