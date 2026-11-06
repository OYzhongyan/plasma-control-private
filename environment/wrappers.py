#
# Copyright @2025 ENN Energy(enn.cn)
#
# Wrappers and preprocessing helpers.
#

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from .preprocessing import (
    ACTION_7D_TO_12D_INDEX,
    DEFAULT_FLAT_OBSERVATION_KEYS,
    action_7d_to_12d,
    flatten_dict_observation,
)


class DictObsFlattenWrapper(gym.ObservationWrapper):
    """Flatten a subset of dict observation keys into a single vector."""

    def __init__(
        self,
        env: gym.Env,
        keys: Optional[Sequence[str]] = None,
        dtype=np.float32,
    ):
        super().__init__(env)
        self.keys = list(keys or DEFAULT_FLAT_OBSERVATION_KEYS)
        self.dtype = dtype
        self._obs_flat_dim = self._compute_flat_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_flat_dim,),
            dtype=self.dtype,
        )

    def _compute_flat_dim(self) -> int:
        n = 0
        for key in self.keys:
            space = self.env.observation_space.spaces.get(key)
            if space is not None:
                n += int(np.prod(space.shape))
            else:
                n += 1
        return n

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        return flatten_dict_observation(observation, keys=self.keys, dtype=self.dtype)


class Action7DTo12DWrapper(gym.ActionWrapper):
    """Map 7D action space to 12D before passing to base env."""

    def __init__(
        self,
        env: gym.Env,
        action_low_7d: Optional[np.ndarray] = None,
        action_high_7d: Optional[np.ndarray] = None,
    ):
        super().__init__(env)
        low_7 = np.array([-1499, -230, -172, -172, -348, -348, -270], dtype=np.float32)
        high_7 = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)
        if action_low_7d is not None:
            low_7 = np.asarray(action_low_7d, dtype=np.float32)
        if action_high_7d is not None:
            high_7 = np.asarray(action_high_7d, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_7, high=high_7, shape=(7,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        return action_7d_to_12d(action)


class NormalizedAction7DTo12DWrapper(gym.ActionWrapper):
    """Map normalized [-1, 1]^7 policy actions to physical 12D voltages."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.scale_7d = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        action_7d = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_7d.size != 7:
            raise ValueError(f"normalized action must have size 7, got {action_7d.size}")
        action_7d = np.clip(action_7d, -1.0, 1.0) * self.scale_7d
        return action_7d_to_12d(action_7d)
