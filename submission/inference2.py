"""Participant-local policy selection; checkpoint is the remote-service default."""
from pathlib import Path
import os
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Policy:
    def __init__(self, model_dir=None, *, backend=None, model_path=None, vecnormalize_path=None,
                 adapter_factory=None, device="cpu"):
        backend = backend or os.environ.get("SUBMISSION_BACKEND", "checkpoint")
        directory = Path(model_dir or os.environ.get("SUBMISSION_MODEL_DIR", CURRENT_DIR / "model"))
        if backend == "checkpoint":
            from submission.checkpoint_policy import CheckpointPolicy
            self._policy = CheckpointPolicy(
                model_path or os.environ.get("SUBMISSION_MODEL_PATH") or directory,
                vecnormalize_path=vecnormalize_path or os.environ.get("SUBMISSION_VECNORMALIZE_PATH"),
                adapter_factory=adapter_factory or os.environ.get("SUBMISSION_ADAPTER_FACTORY"),
                device=device)
        elif backend == "onnx":
            from submission.inference_common import BaseOnnxPolicy
            preferred = "policy2b.onnx" if (directory / "policy2b.onnx").is_file() else "policy2.onnx"
            self._policy = BaseOnnxPolicy(model_dir=directory, preferred_name=preferred)
        else:
            raise ValueError("SUBMISSION_BACKEND must be checkpoint or onnx")

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def reset(self, episode_id=None, options=None):
        return self._policy.reset()
