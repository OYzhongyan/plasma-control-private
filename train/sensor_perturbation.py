from __future__ import annotations
from typing import Any
import gymnasium as gym
import numpy as np


class ObservationDelayNoiseWrapper(gym.Wrapper):
    """对观测施加 k 步延迟 + 可选高斯噪声，模拟赛题推理侧的 obs 污染。

    放在 DictObservationCaptureWrapper 之后、ReferenceObservationWrapper 之前，
    使得真值观测先被 capture（用于评分），而模型看到的 obs 经延迟+加噪。
    逻辑与 saisdata/inference/obs_perturb.py 对齐。
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        delay_steps: int = 0,
        noise_std: dict[str, float] | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.delay_steps = max(0, int(delay_steps))
        self.noise_std = dict(noise_std or {})
        self.rng = np.random.default_rng(int(seed))
        self.enabled = self.delay_steps > 0 or any(
            float(v) > 0.0 for v in self.noise_std.values()
        )
        self._obs_hist: list[dict[str, np.ndarray]] = []
        self._step_count = 0

    def _snapshot(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}

    def _delayed_noised(self, step_index: int) -> dict[str, Any]:

        di = step_index - self.delay_steps
        src = self._obs_hist[max(0, di)]
        out: dict[str, Any] = {}
        for k, v in src.items():
            if k.startswith("reference_"):
                v = self._obs_hist[step_index][k]
            arr = np.asarray(v, dtype=np.float64)
            out[k] = arr.copy()
        for key, std in self.noise_std.items():
            if key.startswith("reference_") or key not in out:
                continue
            std_value = float(std)
            if std_value <= 0.0:
                continue
            out[key] = np.asarray(out[key], dtype=np.float64) + self.rng.normal(
                0.0, std_value, size=np.asarray(out[key]).shape
            )
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.enabled:
            self._obs_hist = [self._snapshot(observation)]
            self._step_count = 0
            return self._delayed_noised(0), info
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.enabled:
            self._obs_hist.append(self._snapshot(observation))
            self._step_count += 1
            delayed = self._delayed_noised(self._step_count)
            return delayed, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
