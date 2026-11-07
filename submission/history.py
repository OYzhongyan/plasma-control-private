"""Current-reference inference using the exact training observation/action wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from train.wrappers import (
    ReferenceObservationWrapper,
    NormalizedAction7DTo12DWrapper,
)

REFERENCE_FEATURE_MODE = "reference_only_v1"


def reference_metadata(action_mode: str, *, norm_obs: bool = True) -> dict[str, Any]:
    """The current 90 target values, with no observation or action history."""
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"Invalid action_mode: {action_mode!r}")
    return {
        "version": 3,
        "feature_mode": REFERENCE_FEATURE_MODE,
        "history_steps": 0,
        "include_xpt": True,
        "action_history_steps": 0,
        "action_history_dim": 0,
        "reference_keys": list(ReferenceObservationWrapper.REFERENCE_SHAPES),
        "reference_shapes": {key: list(shape) for key, shape in ReferenceObservationWrapper.REFERENCE_SHAPES.items()},
        "obs_dim": 90,
        "action_dim": 7,
        "action_mode": action_mode,
        "action_units": "normalized",
        "observation_normalization": bool(norm_obs),
        "requires_vecnormalize": bool(norm_obs),
    }


def validate_reference_metadata(metadata: dict[str, Any]) -> None:
    """Validate the complete input layout before training artifacts are consumed."""
    expected = reference_metadata(
        metadata["action_mode"], norm_obs=metadata.get("observation_normalization", True))
    for key, value in expected.items():
        actual = metadata.get(key)
        if isinstance(value, bool):
            valid = isinstance(actual, bool) and actual == value
        elif isinstance(value, int):
            valid = type(actual) is int and actual == value
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"Reference metadata {key} does not match training layout")


class _ActionSink(gym.Env):
    """Supply action-space/reset plumbing without simulating an environment."""
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return {}, {}


class ReferencePolicyAdapter:
    """Encode the current target only; retain delta-action accumulation."""

    def __init__(self, metadata: dict[str, Any]):
        validate_reference_metadata(metadata)
        sink = _ActionSink()
        self.actions = NormalizedAction7DTo12DWrapper(sink, action_mode=metadata["action_mode"])
        self.observations = ReferenceObservationWrapper(sink, action_mode=metadata["action_mode"])
        self.reset()

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        return self.observations.convert_obs_dict_to_vector(observation)

    def reset(self) -> None:
        self.actions.reset()

    def decode(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise ValueError("The policy must produce 7 finite actions")
        # SB3 predict() clips Box actions; ONNX emits the raw actor output.
        return self.actions.action(np.clip(action, -1., 1.)).astype(np.float32)
