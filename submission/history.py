"""Stateful inference using the exact observation/action wrappers used in training."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from train.wrappers import (
    HistoricalOBservationWrapper,
    NormalizedAction7DTo12DWrapper,
)

HISTORY_FEATURE_MODE = "historical_observation_v1"


def history_metadata(
    history_steps: int,
    action_mode: str,
    *,
    include_xpt: bool = True,
    norm_obs: bool = True,
) -> dict[str, Any]:
    """Describe the input layout without inferring it from the input dimension."""
    history_steps = int(history_steps)
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"Invalid action_mode: {action_mode!r}")
    wrapper = HistoricalOBservationWrapper
    per_step_dim = wrapper.PER_STEP_DIM + (
        wrapper.XPT_PER_STEP_DIM if include_xpt else 0
    )
    action_dim = wrapper.PHYSICAL_ACTION_DIM if action_mode == "delta" else 7
    return {
        "version": 2,
        "feature_mode": HISTORY_FEATURE_MODE,
        "history_steps": history_steps,
        "include_xpt": bool(include_xpt),
        "action_history_steps": wrapper.MAX_ACTION_DELAY_STEP,
        "action_history_dim": action_dim,
        "obs_dim": per_step_dim * history_steps
        + action_dim * wrapper.MAX_ACTION_DELAY_STEP,
        "action_dim": 7,
        "action_mode": action_mode,
        "action_units": "normalized",
        "observation_normalization": bool(norm_obs),
        "requires_vecnormalize": bool(norm_obs),
    }


class _ReceivedObservation(gym.Env):
    """A buffer for externally supplied observations; it never simulates a step."""

    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def __init__(self):
        self.observation: dict[str, Any] = {}
        self.physical_action = np.zeros(12, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation, {}

    def step(self, action):
        return (
            self.observation,
            0.0,
            False,
            False,
            {"physical_action": self.physical_action.copy()},
        )


class HistoricalPolicyAdapter:
    """Advance history on receipt of the next observation, after the prior action."""

    def __init__(self, metadata: dict[str, Any]):
        expected = history_metadata(
            metadata["history_steps"],
            metadata["action_mode"],
            include_xpt=metadata.get("include_xpt", True),
            norm_obs=metadata.get("observation_normalization", True),
        )
        for key in (
            "obs_dim",
            "action_dim",
            "action_history_dim",
            "action_history_steps",
        ):
            if int(metadata.get(key, expected[key])) != expected[key]:
                raise ValueError(
                    f"History metadata {key} does not match training layout"
                )
        self.buffer = _ReceivedObservation()
        self.actions = NormalizedAction7DTo12DWrapper(
            self.buffer, action_mode=metadata["action_mode"]
        )
        # The buffered step already carries the physical action computed on the
        # previous /act call. Wrapping actions again here would accumulate twice.
        self.observations = HistoricalOBservationWrapper(
            self.buffer,
            obs_history_step=metadata["history_steps"],
            include_xpt=metadata.get("include_xpt", True),
            action_mode=metadata["action_mode"],
        )
        self.reset()

    def reset(self) -> None:
        self.started = False
        self.pending_action: np.ndarray | None = None
        self.buffer.physical_action = np.zeros(12, dtype=np.float32)
        self.actions.reset()

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        self.buffer.observation = observation
        if not self.started:
            vector, _ = self.observations.reset()
            self.started = True
        else:
            if self.pending_action is None:
                raise RuntimeError(
                    "No previous action; reset the episode after inference failure"
                )
            vector, _, _, _, _ = self.observations.step(self.pending_action)
        self.pending_action = None
        if not np.all(np.isfinite(vector)):
            raise ValueError("Historical observation contains non-finite features")
        return vector

    def decode(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise ValueError("The historical policy must produce 7 finite actions")
        # SB3 predict() clips Box actions before env.step(). ONNX emits the raw
        # deterministic actor output, so apply that same clipping here.
        action = np.clip(action, -1.0, 1.0)
        physical = self.actions.action(action)
        self.buffer.physical_action = np.asarray(physical, dtype=np.float32).copy()
        self.pending_action = action.copy()
        return self.buffer.physical_action.copy()
