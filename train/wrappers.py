from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import get_shot_spec, action_7d_to_12d

COIL_LIMITS_A = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)
_MAX_CHANGE_7D = np.array([1499, 175, 175, 175, 175, 175, 80], dtype=np.float64) * 0.16


def _as_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    arr = list(value)
    if len(arr) != 2:
        raise ValueError(f"range must contain exactly two numbers, got {value!r}")
    lo, hi = float(arr[0]), float(arr[1])
    return (lo, hi) if lo <= hi else (hi, lo)


class ResetPerturbationSampler:
    """Sample reset perturbations from the enabled signeo/bp/q0 ranges."""

    def __init__(
        self,
        *,
        shot_id: str,
        reset_perturbation: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.shot_id = str(shot_id)
        self.reset_perturbation = dict(reset_perturbation or {})
        self.rng = np.random.default_rng(int(seed))

    def sample(self) -> dict[str, float]:
        if not bool(self.reset_perturbation.get("enabled", False)):
            return {}

        defaults = dict(get_shot_spec(self.shot_id).get("reset_defaults", {}))
        candidates = self._perturbation_candidates(defaults)
        if not candidates:
            return {}
        apply_mode = str(self.reset_perturbation.get("apply_mode", "one")).lower()
        if apply_mode == "all":
            params: dict[str, float] = {}
            for candidate in candidates:
                value = self._sample_value(candidate, defaults)
                if value is not None:
                    params[candidate] = float(value)
            return params

        name = str(self.rng.choice(candidates))
        value = self._sample_value(name, defaults)
        return {name: value} if value is not None else {}

    def _sample_value(self, name: str, defaults: dict[str, Any]) -> float | None:
        if name not in defaults:
            return None
        default = float(defaults[name])
        scale_key = f"{name}_scale_range"
        delta_key = f"{name}_delta_range"

        if self.reset_perturbation.get(scale_key) is not None:
            lo, hi = _as_range(self.reset_perturbation.get(scale_key), (1.0, 1.0))
            return default * float(self.rng.uniform(lo, hi))

        lo, hi = _as_range(self.reset_perturbation.get(delta_key), (0.0, 0.0))
        return default + float(self.rng.uniform(lo, hi))

    def _perturbation_candidates(self, defaults: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for name in ("signeo", "bp", "q0"):
            if name not in defaults:
                continue
            scale = self.reset_perturbation.get(f"{name}_scale_range")
            if scale is not None:
                lo, hi = _as_range(scale, (1.0, 1.0))
                if lo != 1.0 or hi != 1.0:
                    candidates.append(name)
                    continue

            delta = self.reset_perturbation.get(f"{name}_delta_range")
            lo, hi = _as_range(delta, (0.0, 0.0))
            if lo != 0.0 or hi != 0.0:
                candidates.append(name)
        return candidates


class EpisodeResetWrapper(gym.Wrapper):
    """Change reset params and the reference in every episode reset."""

    def __init__(
        self,
        env: gym.Env,
        references: list[dict[str, Any]],
        *,
        shot_id: str,
        reset_perturbation: dict[str, Any] | None = None,
        fix_reset_params: bool = False,
        seed: int = 0,
    ):
        super().__init__(env)
        self.references = references
        self.reference_rng = np.random.default_rng(int(seed) + 50_000)
        self.perturbation_sampler = ResetPerturbationSampler(
            shot_id=shot_id,
            reset_perturbation=reset_perturbation,
            seed=seed,
        )
        self.reset_params_rng = np.random.default_rng(int(seed) + 10_000)
        self.fix_reset_params = fix_reset_params
        self.current_reset_params = None

    def _sample_reset_params(self) -> dict[str, float]:
        if self.current_reset_params is None or not self.fix_reset_params:
            self.current_reset_params = self.perturbation_sampler.sample()
        return self.current_reset_params
        # if self.selected_reset_params is not None:
        #     idx = int(self.reset_params_rng.integers(0, len(self.selected_reset_params)))
        #     return self.selected_reset_params[idx]

    def _sample_reference(self) -> tuple[dict[str, Any], int]:
        if len(self.references) == 1:
            return self.references[0], 0
        idx = int(self.reference_rng.integers(0, len(self.references)))
        return self.references[idx], idx

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        merged = dict(options or {})
        explicit_reset_params = dict(merged.get("reset_params", {}))
        reset_params = self._sample_reset_params()
        reset_params.update(explicit_reset_params)
        reference, reference_idx = self._sample_reference()
        merged["reference_mode"] = "trajectory"
        merged["reference"] = reference
        if reset_params:
            merged["reset_params"] = reset_params
        obs, info = self.env.reset(seed=seed, options=merged)
        return obs, info


class HoldInitialResetWrapper(gym.Wrapper):
    """Force hold-initial reference and optional reset micro-perturbations."""

    def __init__(
        self,
        env: gym.Env,
        *,
        shot_id: str,
        reset_perturbation: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.shot_id = str(shot_id)
        self.perturbation_sampler = ResetPerturbationSampler(
            shot_id=shot_id,
            reset_perturbation=reset_perturbation,
            seed=seed,
        )

    def _sample_reset_params(self) -> dict[str, float]:
        return self.perturbation_sampler.sample()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        merged = dict(options or {})
        explicit_reset_params = dict(merged.get("reset_params", {}))
        reset_params = self._sample_reset_params()
        reset_params.update(explicit_reset_params)
        merged["reference_mode"] = "hold"
        if reset_params:
            merged["reset_params"] = reset_params
        return self.env.reset(seed=seed, options=merged)


class RewardInfoWrapper(gym.Wrapper):
    """Expose raw dense-reward components through ``info`` for SB3 callbacks."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward_fn = getattr(self.unwrapped, "reward_fn", None)
        components = getattr(reward_fn, "last_components", None)
        if isinstance(components, dict):
            info = dict(info or {})
            info["reward_components"] = dict(components)
        return observation, reward, terminated, truncated, info


class ReferenceObservationWrapper(gym.Wrapper):
    """Convert the current reference dictionary to the 90D policy input."""

    REFERENCE_SHAPES = {
        "reference_Ip": (1,),
        "reference_lX": (1,),
        "reference_lcfs_points": (32, 2),
        "reference_rX": (4,),
        "reference_zX": (4,),
        "reference_strike_r": (8,),
        "reference_strike_z": (8,),
    }
    PER_STEP_DIM = 66
    XPT_PER_STEP_DIM = 24

    def __init__(self, env: gym.Env, obs_history_step: int = 0,
                 dtype=np.float32, include_xpt: bool = True,
                 action_mode: str = "absolute"):
        super().__init__(env)
        self.dtype = np.dtype(dtype)
        # Accepted for existing callers, but neither argument changes the 90D contract.
        self.history_steps = 0
        self.include_xpt = True
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {"absolute", "delta"}:
            raise ValueError(f"invalid action_mode: {action_mode}")
        self.per_step_dim = self.PER_STEP_DIM + self.XPT_PER_STEP_DIM
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(90,), dtype=self.dtype)

    def _reference(self, observation, key):
        if key not in observation:
            raise ValueError(f"Missing reference feature: {key}")
        value = np.asarray(observation[key], dtype=self.dtype).reshape(-1)
        size = int(np.prod(self.REFERENCE_SHAPES[key]))
        if value.size != size or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must contain {size} finite reference values")
        return value

    def _convert_xpt_features(self, obs_dict) -> np.ndarray:
        return np.concatenate([
            self._reference(obs_dict, key)
            for key in ("reference_rX", "reference_zX", "reference_strike_r", "reference_strike_z")
        ])

    def convert_obs_dict_to_vector(self, obs_dict):
        return np.concatenate([
            self._reference(obs_dict, "reference_Ip"),
            self._reference(obs_dict, "reference_lX"),
            self._reference(obs_dict, "reference_lcfs_points"),
            self._convert_xpt_features(obs_dict),
        ])

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.convert_obs_dict_to_vector(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.convert_obs_dict_to_vector(observation), reward, terminated, truncated, info


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


class NormalizedAction7DTo12DWrapper(gym.ActionWrapper):
    """Map normalized [-1, 1]^7 policy actions to physical 12D voltages."""

    def __init__(self, env: gym.Env, action_mode: str = "absolute"):
        super().__init__(env)
        # self.scale_7d = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)
        self.scale_7d = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.current_abs_action = None
        self.action_mode = action_mode

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.current_abs_action = np.zeros((7,), dtype=np.float32)
        return self.env.reset(seed=seed, options=options)

    def action(self, action: np.ndarray) -> np.ndarray:
        action_7d = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_7d.size != 7:
            raise ValueError(f"normalized action must have size 7, got {action_7d.size}")
        if self.action_mode == "absolute":
            action_7d = np.clip(action_7d, -1.0, 1.0) * self.scale_7d
            return action_7d_to_12d(action_7d)
        elif self.action_mode == "delta":
            action_7d = np.clip(action_7d, -1.0, 1.0) * _MAX_CHANGE_7D
            self.current_abs_action = self.current_abs_action + action_7d
            return action_7d_to_12d(self.current_abs_action)
        else:
            raise ValueError(f"invalid action_mode: {self.action_mode}")

    def step(self, action):
        # 重写 ActionWrapper.step：转换后的 12D 物理电压除传给底层 env 外，
        # 还注入 info["physical_action"]，供外层 wrapper（如 action history v2）使用
        physical_action = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(physical_action)
        info = dict(info or {})
        info["physical_action"] = np.asarray(physical_action, dtype=np.float32).reshape(-1)
        return observation, reward, terminated, truncated, info
        
