"""F2b submission policy with optional policy2b filename."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from inference_common import BaseOnnxPolicy


class Policy(BaseOnnxPolicy):

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir) if model_dir is not None else ROOT / "model"
        preferred = (
            "policy2b.onnx"
            if (model_dir / "policy2b.onnx").exists()
            else "policy2.onnx"
        )
        super().__init__(model_dir=model_dir, preferred_name=preferred)

    def reset(self, episode_id=None, options=None):
        return super().reset()
