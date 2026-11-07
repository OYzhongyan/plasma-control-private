"""Contestant-local controller construction. This file is never run by grading.

Custom factories return ControllerSpec and express all feature and action
processing as Torch tensor operations in forward(*observations[, state]).
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from train.model_metadata import validate_reference_metadata

REFERENCE_FIELDS = ("reference_Ip", "reference_lX", "reference_lcfs_points",
                    "reference_rX", "reference_zX", "reference_strike_r", "reference_strike_z")


@dataclass
class ControllerSpec:
    module: torch.nn.Module
    observations: Sequence[str]
    state_dim: int = 0
    state_dtype: str = "float32"


class ReferenceController(torch.nn.Module):
    """Current baseline, including frozen normalization and voltage conversion."""

    def __init__(self, model, metadata, normalization):
        super().__init__()
        validate_reference_metadata(metadata)
        if model.observation_space.shape != (90,) or model.action_space.shape != (7,):
            raise ValueError("Default factory requires a 90D reference policy with 7 actions")
        self.policy = model.policy
        self.delta = metadata["action_mode"] == "delta"
        self.normalize = metadata["observation_normalization"]
        if self.normalize:
            if normalization is None:
                raise ValueError("Matching VecNormalize statistics are required")
            mean = np.asarray(normalization["obs_mean"], dtype=np.float64)
            var = np.asarray(normalization["obs_var"], dtype=np.float64)
            if mean.shape != (90,) or var.shape != (90,) or np.any(var < 0):
                raise ValueError("VecNormalize statistics must match the 90D input")
            self.register_buffer("mean", torch.as_tensor(mean))
            self.register_buffer("var", torch.as_tensor(var))
            self.epsilon = float(normalization.get("epsilon", 1e-8))
            self.clip_obs = float(normalization.get("clip_obs", 10.))
        self.register_buffer("scale", torch.tensor([1499, 230, 172, 172, 348, 348, 270], dtype=torch.float32))
        self.register_buffer("increment", torch.tensor([1499, 175, 175, 175, 175, 175, 80], dtype=torch.float64) * .16)
        self.register_buffer("mapping", torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=torch.int64))

    def forward(self, *values):
        observations = values[:-1] if self.delta else values
        features = torch.cat([value.flatten(start_dim=1) for value in observations], dim=1)
        if self.normalize:
            features = torch.clamp((features.to(torch.float64) - self.mean)
                                   / torch.sqrt(self.var + self.epsilon),
                                   -self.clip_obs, self.clip_obs).to(torch.float32)
        raw, _, _ = self.policy(features, deterministic=True)
        raw = torch.clamp(raw, -1., 1.)
        if self.delta:
            # Training accumulates in float64; retain it to avoid long-run drift.
            absolute = values[-1] + raw.to(torch.float64) * self.increment
            return absolute.to(torch.float32).index_select(1, self.mapping), absolute
        return (raw * self.scale).index_select(1, self.mapping)


def build_controller(model, metadata, normalization):
    module = ReferenceController(model, metadata, normalization)
    return ControllerSpec(module, REFERENCE_FIELDS, state_dim=7 if module.delta else 0,
                          state_dtype="float64" if module.delta else "float32")
