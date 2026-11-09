from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from collections import deque

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import get_shot_spec, action_7d_to_12d
from environment.xpt_utils import extract_xpt_observation_pack

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


class ObservationWrapper(gym.Wrapper):
    """Convert dict observations to physical-v1 features with fixed history slots."""

    # 每步基础特征长度:ip(1) + delta_ip(1) + reference_ip(1) + lx(1) + delta_lx(1)
    # + lcfs_points(64) + delta_lcfs_points(64) + i_pf_norm(12) + coil_margin(1)
    PER_STEP_DIM = 1 + 1 + 1 + 1 + 1 + 32 * 2 + 32 * 2 + 12 + 1
    # F2 XPT 附加特征: 4 槽 × [x_valid, delta_x_r, delta_x_z, delta_x_fx] + [nX, n_strike]
    #                 + 8 槽 × [strike_valid, delta_strike_r, delta_strike_z]
    XPT_PER_STEP_DIM = 4 * 4 + 2 + 8 * 3
    PER_STEP_ACTION_DIM = 7
    # v2: action history 存 NormalizedAction7DTo12DWrapper 转换后的 12D 物理电压
    PHYSICAL_ACTION_DIM = 12
    MAX_ACTION_DELAY_STEP = 5

    def __init__(
        self,
        env: gym.Env,
        obs_history_step: int,
        dtype=np.float32,
        include_xpt: bool = False,
        action_mode: str = "absolute",
    ):
        super().__init__(env)
        self.dtype = dtype
        self.include_xpt = bool(include_xpt)
        # action history 内容由 action_mode 决定：
        #   absolute: 存策略输出的 7D 归一化动作（v1，与旧 checkpoint 兼容）；
        #   delta:    策略输出为增量，存 ActionWrapper 转换后 info["physical_action"] 的 12D 物理电压（v2）
        self.action_mode = str(action_mode).strip().lower()
        self.action_history_v2 = self.action_mode == "delta"
        self.action_dim = self.PHYSICAL_ACTION_DIM if self.action_history_v2 else self.PER_STEP_ACTION_DIM
        self.per_step_dim = self.PER_STEP_DIM + (self.XPT_PER_STEP_DIM if self.include_xpt else 0)
        self.history_steps = max(1, int(obs_history_step))
        self.observation_history = deque(maxlen=self.history_steps)
        self.action_history = deque(maxlen=self.MAX_ACTION_DELAY_STEP)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.per_step_dim * self.history_steps + self.action_dim * self.MAX_ACTION_DELAY_STEP,),
            dtype=dtype,
        )
        self.episode_step = 0

    def _convert_xpt_features(self, obs_dict) -> np.ndarray:
        """F2 XPT 误差特征（与 SemifinalDenseReward 的 XPT 分量同构）。

        - 当前 X 点/打击点经 extract_xpt_observation_pack 匹配到 reference 固定槽位；
        - delta_x_r/z: X 点槽位坐标 - reference 坐标，乘 x_valid 掩码（无效槽为 0）；
        - delta_x_fx: (x_fx - FB) / (FB - FA)，即 reward 中的 psiX 归一化偏差；
        - delta_strike_r/z: 打击点槽位坐标 - reference 坐标，乘 strike_valid 掩码；
        - nX / n_strike: 拓扑计数（评分对 nX 有硬掩码，需让策略感知）。
        """
        pack = extract_xpt_observation_pack(
            obs_dict,
            np.asarray(obs_dict["reference_rX"], dtype=np.float64),
            np.asarray(obs_dict["reference_zX"], dtype=np.float64),
            np.asarray(obs_dict["reference_strike_r"], dtype=np.float64),
            np.asarray(obs_dict["reference_strike_z"], dtype=np.float64),
            fx_order="C",
        )
        x_valid = np.asarray(pack["x_valid"], dtype=np.float64).reshape(-1)
        ref_r = np.asarray(obs_dict["reference_rX"], dtype=np.float64).reshape(-1)[: x_valid.size]
        ref_z = np.asarray(obs_dict["reference_zX"], dtype=np.float64).reshape(-1)[: x_valid.size]
        x_r = np.asarray(pack["x_r"], dtype=np.float64).reshape(-1)
        x_z = np.asarray(pack["x_z"], dtype=np.float64).reshape(-1)
        x_fx = np.asarray(pack["x_fx"], dtype=np.float64).reshape(-1)
        fb = float(np.asarray(pack["fb"], dtype=np.float64).reshape(-1)[0])
        fa = float(np.asarray(obs_dict.get("FA", 0.0), dtype=np.float64).reshape(-1)[0])
        den = fb - fa
        psi_norm = 1.0 / den if abs(den) > 1e-9 else 0.0

        strike_valid = np.asarray(pack["strike_valid"], dtype=np.float64).reshape(-1)
        ref_strike_r = np.asarray(obs_dict["reference_strike_r"], dtype=np.float64).reshape(-1)[: strike_valid.size]
        ref_strike_z = np.asarray(obs_dict["reference_strike_z"], dtype=np.float64).reshape(-1)[: strike_valid.size]
        strike_r = np.asarray(pack["strike_r"], dtype=np.float64).reshape(-1)
        strike_z = np.asarray(pack["strike_z"], dtype=np.float64).reshape(-1)

        return np.concatenate(
            [
                x_valid,
                (x_r - ref_r) * x_valid,
                (x_z - ref_z) * x_valid,
                (x_fx - fb) * psi_norm * x_valid,
                np.asarray([float(np.asarray(pack["nX"]).reshape(-1)[0])], dtype=np.float64),
                np.asarray([float(np.asarray(pack["strike_n_actual"]).reshape(-1)[0])], dtype=np.float64),
                strike_valid,
                (strike_r - ref_strike_r) * strike_valid,
                (strike_z - ref_strike_z) * strike_valid,
            ]
        ).astype(self.dtype)

    def convert_obs_dict_to_vector(self, obs_dict):
        ip = np.asarray(obs_dict["Ip"], dtype=self.dtype).reshape(-1)
        ref_ip = np.asarray(obs_dict["reference_Ip"], dtype=self.dtype).reshape(-1)
        lx = np.asarray(obs_dict["lX"], dtype=self.dtype).reshape(-1)
        ref_lx = np.asarray(obs_dict["reference_lX"], dtype=self.dtype).reshape(-1)
        lcfs = np.asarray(obs_dict["lcfs_points"], dtype=self.dtype).reshape(-1)
        ref_lcfs = np.asarray(obs_dict["reference_lcfs_points"], dtype=self.dtype).reshape(-1)
        ipf = np.asarray(obs_dict["I_PF"], dtype=self.dtype).reshape(-1)
        features: dict[str, np.ndarray] = {
            "ip": ip,
            "delta_ip": ip - ref_ip,
            "reference_ip": ref_ip,
            "lx": lx,
            "delta_lx": lx - ref_lx,
            "lcfs_points": lcfs,
            "delta_lcfs_points": lcfs - ref_lcfs,
            "i_pf_norm": ipf / COIL_LIMITS_A[: ipf.size],
            "coil_margin": np.asarray([np.max(np.abs(ipf) / COIL_LIMITS_A[: ipf.size])], dtype=self.dtype),
        }
        parts = list(features.values())
        if self.include_xpt:
            parts.append(self._convert_xpt_features(obs_dict))
        return np.concatenate(parts)

    def build_full_observation_vector(self):
        return np.concatenate([np.concatenate(self.observation_history), np.concatenate(self.action_history)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        observation_vector = self.convert_obs_dict_to_vector(observation)
        self.observation_history.clear()
        self.action_history.clear()
        for _ in range(self.history_steps):
            self.observation_history.append(np.zeros_like(observation_vector))
        self.observation_history.append(observation_vector)
        for _ in range(self.MAX_ACTION_DELAY_STEP):
            self.action_history.append(np.zeros(self.action_dim, dtype=self.dtype))
        self.episode_step = 0
        return self.build_full_observation_vector(), info

    def _action_history_vec(self, action: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """v1: 策略输出的 7D 归一化动作；v2: ActionWrapper 注入 info 的 12D 物理电压。"""
        if self.action_history_v2:
            physical = info["physical_action"]
            return np.asarray(physical, dtype=self.dtype).reshape(-1)
        return np.asarray(action, dtype=self.dtype).reshape(-1)

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation_vector = self.convert_obs_dict_to_vector(observation)
        self.observation_history.append(observation_vector)
        action_vec = self._action_history_vec(action, info)
        if self.episode_step == 0:
            for _ in range(self.MAX_ACTION_DELAY_STEP):
                self.action_history.append(action_vec)
        else:
            self.action_history.append(action_vec)
        self.episode_step += 1
        return self.build_full_observation_vector(), reward, terminated, truncated, info


class ObservationDelayNoiseWrapper(gym.Wrapper):
    """对观测施加 k 步延迟 + 可选高斯噪声，模拟赛题推理侧的 obs 污染。

    放在 DictObservationCaptureWrapper 之后、ObservationWrapper 之前，
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
            arr = np.asarray(v, dtype=np.float64)
            out[k] = arr.copy()
        for key, std in self.noise_std.items():
            if key not in out:
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

