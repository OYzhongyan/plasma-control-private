"""Delayed flux histories and actor-compatible observation normalization."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize

from train.wrappers import ObservationWrapper


class FluxObservationWrapper(ObservationWrapper):
    """Keep baseline actor history and append the same observations' flux frames.

    The inherited reset/step methods own simulator access and actor history.
    Capturing flux during their existing conversion preserves any upstream
    observation delay without a second environment reset or step.
    Invalid critic flux is masked; malformed/missing flux required by the XPT
    actor encoder still raises the baseline validation error.
    """

    FLUX_SHAPE = (66, 65)

    def __init__(self, env: gym.Env, obs_history_step: int, dtype=np.float32,
                 include_xpt: bool = False, action_mode: str = "absolute"):
        super().__init__(env, obs_history_step, dtype=dtype,
                         include_xpt=include_xpt, action_mode=action_mode)
        actor_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            "actor_obs": gym.spaces.Box(-np.inf, np.inf, actor_shape, dtype=np.float32),
            "flux_history": gym.spaces.Box(-np.inf, np.inf,
                                          (self.history_steps, *self.FLUX_SHAPE), dtype=np.float32),
            "flux_mask": gym.spaces.Box(0, 1, (self.history_steps,), dtype=np.float32),
        })
        self.flux_history = deque(maxlen=self.history_steps)
        self.flux_mask = deque(maxlen=self.history_steps)

    def convert_obs_dict_to_vector(self, obs_dict):
        actor = super().convert_obs_dict_to_vector(obs_dict)
        # Missing, malformed or nonfinite flux invalidates the entire frame.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                frame = np.asarray(obs_dict.get("Fx"), dtype=np.float32)
            valid = frame.size == int(np.prod(self.FLUX_SHAPE)) and np.all(np.isfinite(frame))
        except (TypeError, ValueError):
            valid = False
        if valid:
            self._current_flux = frame.reshape(self.FLUX_SHAPE, order="C").copy()
        else:
            self._current_flux = np.zeros(self.FLUX_SHAPE, dtype=np.float32)
        self._current_flux_valid = float(valid)
        return actor

    def _flux_observation(self, actor):
        self.flux_history.append(self._current_flux)
        self.flux_mask.append(self._current_flux_valid)
        return {
            # Preserve baseline float64 intermediates before VecNormalize's RMS
            # update; SubprocVecEnv stacks these without a float32 buffer cast.
            "actor_obs": actor,
            "flux_history": np.stack(self.flux_history).astype(np.float32),
            "flux_mask": np.asarray(self.flux_mask, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        actor, info = super().reset(seed=seed, options=options)
        self.flux_history.clear()
        self.flux_mask.clear()
        for _ in range(self.history_steps - 1):
            self.flux_history.append(np.zeros(self.FLUX_SHAPE, dtype=np.float32))
            self.flux_mask.append(0.0)
        return self._flux_observation(actor), info

    def step(self, action):
        actor, reward, terminated, truncated, info = super().step(action)
        return self._flux_observation(actor), reward, terminated, truncated, info


class FluxVecNormalize(VecNormalize):
    """Baseline actor/reward normalization plus scalar valid-pixel flux RMS.

    Only the main observation returned by reset/step updates ``flux_rms``.
    Explicit normalization and terminal observations reuse current statistics.
    Standard VecNormalize pickle save/load preserves this subclass and its RMS.
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True,
                 clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8,
                 norm_obs_keys=None):
        if norm_obs_keys is not None and norm_obs_keys != ["actor_obs"]:
            raise ValueError("FluxVecNormalize normalizes only actor_obs with vector statistics")
        super().__init__(venv, training=training, norm_obs=norm_obs, norm_reward=norm_reward,
                         clip_obs=clip_obs, clip_reward=clip_reward, gamma=gamma,
                         epsilon=epsilon, norm_obs_keys=["actor_obs"])
        self.flux_rms = RunningMeanStd(shape=())
        self._flux_update_pending = False

    @staticmethod
    def _valid_flux_frames(obs):
        flux = np.asarray(obs["flux_history"])
        return (np.asarray(obs["flux_mask"]) > 0) & np.all(np.isfinite(flux), axis=(-2, -1))

    def reset(self):
        self._flux_update_pending = True
        try:
            return super().reset()
        finally:
            self._flux_update_pending = False

    def step_wait(self):
        self._flux_update_pending = True
        try:
            return super().step_wait()
        finally:
            self._flux_update_pending = False

    def normalize_obs(self, obs):
        update = self._flux_update_pending
        self._flux_update_pending = False
        flux = np.asarray(obs["flux_history"])
        valid = self._valid_flux_frames(obs)
        if update and self.training and self.norm_obs and np.any(valid):
            self.flux_rms.update(np.asarray(flux[valid], dtype=np.float64).reshape(-1))
        result = super().normalize_obs(obs)
        if self.norm_obs:
            normalized = self._normalize_obs(flux, self.flux_rms)
            result["flux_history"] = np.where(valid[..., None, None], normalized, 0).astype(np.float32)
        return result

    def unnormalize_obs(self, obs):
        result = super().unnormalize_obs(obs)
        if self.norm_obs:
            flux = self._unnormalize_obs(np.asarray(obs["flux_history"]), self.flux_rms)
            valid = self._valid_flux_frames(obs)
            result["flux_history"] = np.where(valid[..., None, None], flux, 0).astype(np.float32)
        return result
