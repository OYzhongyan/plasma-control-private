"""Submission inference for policy2 (F2a/F2b: to XPT)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_common import BaseOnnxPolicy  # noqa: E402


class Policy(BaseOnnxPolicy):
    """Policy interface used by service2.

    F2a and F2b can use separate ONNX files. If they are absent, this class
    falls back to the legacy single ``policy2.onnx`` model.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        self.fallback = BaseOnnxPolicy(
            task="F2",
            model_dir=self.model_dir,
            preferred_name="policy2.onnx",
        )
        self.policies: dict[str, BaseOnnxPolicy] = {}
        for key, filename in (("F2a", "policy2a.onnx"), ("F2b", "policy2b.onnx")):
            if (self.model_dir / filename).exists():
                self.policies[key] = BaseOnnxPolicy(
                    task="F2",
                    model_dir=self.model_dir,
                    preferred_name=filename,
                )
        self.active_task: str | None = None

    def reset(self, episode_id: str | None = None, options=None) -> None:
        task = str(episode_id or "").strip()
        self.active_task = task if task in {"F2a", "F2b"} else None
        self.fallback.reset()
        for policy in self.policies.values():
            policy.reset()
        return None

    @staticmethod
    def _fallback_task_from_observation(observation) -> str:
        try:
            lx = float(observation.get("lX", [1.0]).reshape(-1)[0])
        except Exception:
            lx = 1.0
        return "F2b" if lx < 0.5 else "F2a"

    def _select_policy(self, observation) -> BaseOnnxPolicy:
        if self.active_task is None:
            self.active_task = self._fallback_task_from_observation(observation)
        task = self.active_task
        return self.policies.get(task, self.fallback)

    def act(self, observation):
        return self._select_policy(observation).act(observation)
