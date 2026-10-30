"""Unified PPO training entrypoint for semifinal F1/F2 baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import (  # noqa: E402
    Action7DTo12DWrapper,
    HFMSimulator,
    NormalizedAction7DTo12DWrapper,
    get_shot_spec,
)
from training.features import (  # noqa: E402
    FeatureMetadata,
    FixedFeatureFlattenWrapper,
    PhysicalFeatureHistoryWrapper,
)
from training.callbacks import (  # noqa: E402
    EpisodeRewardComponentsCallback,
    PlasmaCheckpointCallback,
    RewardComponentsCallback,
    find_vecnormalize,
    save_vecnormalize_stats,
)
from training.references import (  # noqa: E402
    TargetReference,
    build_reference,
    default_reference_path,
    load_target_reference,
    normalize_task,
)
from training.reward_f2 import F2OfficialProxyReward, F2XPTTargetReward  # noqa: E402
from training.rewards import SemifinalDenseReward, SemifinalDenseRewardV2  # noqa: E402


def warmup_cosine_schedule(
    initial_lr: float,
    timesteps_per_update: int,
    total_timesteps: int,
    warmup_updates: int = 10,
    anneal: bool = True,
) -> Callable[[float], float]:
    """Return an SB3 learning-rate schedule with optional warmup and cosine decay."""

    warmup_updates = max(0, int(warmup_updates))
    initial_lr = float(initial_lr)
    if warmup_updates <= 0:
        if not anneal:
            return lambda _progress_remaining: initial_lr
        warmup_progress = 1.0
    else:
        warmup_timesteps = warmup_updates * max(1, int(timesteps_per_update))
        warmup_progress = max(0.0, 1.0 - warmup_timesteps / max(1, int(total_timesteps)))
    warmup_lr_start = initial_lr / 10.0

    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(progress_remaining)
        if warmup_updates > 0 and progress_remaining >= warmup_progress:
            warmup_fraction = (1.0 - progress_remaining) / max(1.0 - warmup_progress, 1e-6)
            return warmup_lr_start + (initial_lr - warmup_lr_start) * warmup_fraction
        if anneal:
            decay_progress = (warmup_progress - progress_remaining) / max(warmup_progress, 1e-6)
            decay_progress = min(max(decay_progress, 0.0), 1.0)
            return initial_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return initial_lr

    return schedule


def parse_ports(spec: str) -> list[int]:
    ports: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start), int(end) + 1))
        else:
            ports.append(int(part))
    if not ports:
        raise ValueError("at least one port is required")
    return ports


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def load_train_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"training config must be a JSON object: {path}")
    if "task" not in data:
        raise ValueError(f"training config must include task: {path}")
    if data.get("reference_mode") not in {None, "trajectory", "hold_initial"}:
        raise ValueError("reference_mode must be 'trajectory' or 'hold_initial'")
    return data


def _rooted_path(value: str | Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def flatten_shot_groups(shot_groups: Any) -> list[str]:
    if shot_groups is None:
        return []
    if isinstance(shot_groups, dict):
        shots: list[str] = []
        for group_shots in shot_groups.values():
            if isinstance(group_shots, str):
                shots.append(group_shots)
            else:
                shots.extend(str(shot_id) for shot_id in group_shots)
        return shots
    if isinstance(shot_groups, str):
        return [shot_groups]
    return [str(shot_id) for shot_id in shot_groups]


def assign_shots_to_envs(
    shot_groups: Any,
    num_envs: int,
    fallback_shot_id: str,
) -> list[str]:
    num_envs = max(1, int(num_envs))
    shot_ids = flatten_shot_groups(shot_groups) or [str(fallback_shot_id)]
    return [shot_ids[i % len(shot_ids)] for i in range(num_envs)]


def ports_for_envs(spec: str | None, num_envs: int) -> list[int]:
    num_envs = max(1, int(num_envs))
    ports = parse_ports(spec or "2223")
    if len(ports) == 1 and num_envs > 1:
        start = ports[0]
        return list(range(start, start + num_envs))
    if len(ports) < num_envs:
        raise ValueError(f"need {num_envs} ports, got {ports}")
    return ports[:num_envs]


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


class ReferenceResetWrapper(gym.Wrapper):
    """Inject one trajectory reference into every episode reset."""

    def __init__(
        self,
        env: gym.Env,
        reference: dict[str, Any] | list[dict[str, Any]],
        *,
        shot_id: str,
        reset_perturbation: dict[str, Any] | None = None,
        seed: int = 0,
        reference_variant_index: int | None = None,
    ):
        super().__init__(env)
        self.references = self._normalize_references(reference)
        self.reference_rng = np.random.default_rng(int(seed) + 50_000)
        self.fixed_reference_variant_index = self._normalize_fixed_index(reference_variant_index)
        self.perturbation_sampler = ResetPerturbationSampler(
            shot_id=shot_id,
            reset_perturbation=reset_perturbation,
            seed=seed,
        )

    @staticmethod
    def _normalize_references(reference: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(reference, list):
            out = [dict(item) for item in reference if isinstance(item, dict)]
        else:
            out = [dict(reference)]
        if not out:
            raise ValueError("at least one trajectory reference is required")
        return out

    def _normalize_fixed_index(self, index: int | None) -> int | None:
        if index is None:
            return None
        return int(index) % len(self.references)

    def _sample_reset_params(self) -> dict[str, float]:
        return self.perturbation_sampler.sample()

    def _sample_reference(self) -> tuple[dict[str, Any], int]:
        if self.fixed_reference_variant_index is not None:
            idx = self.fixed_reference_variant_index
            return self.references[idx], idx
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
        info = dict(info or {})
        info["reference_variant_index"] = reference_idx
        if "_variant_id" in reference:
            info["reference_variant_id"] = reference["_variant_id"]
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


class RawObservationNoiseWrapper(gym.Wrapper):
    """Add configured Gaussian noise to raw observations before feature flattening."""

    def __init__(
        self,
        env: gym.Env,
        *,
        observation_noise: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.observation_noise = dict(observation_noise or {})
        self.rng = np.random.default_rng(int(seed))
        noise_block = self.observation_noise.get("noise", self.observation_noise)
        self.enabled = bool(self.observation_noise.get("enabled", False)) and bool(
            noise_block.get("enabled", True)
        )
        self.std_map = dict(noise_block.get("std") or {})

    def _add_noise(self, observation: Any) -> Any:
        if not self.enabled or not isinstance(observation, dict) or not self.std_map:
            return observation
        out = dict(observation)
        for key, std in self.std_map.items():
            if key not in out:
                continue
            std_value = float(std)
            if std_value <= 0.0:
                continue
            arr = np.asarray(out[key])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            noisy = arr.astype(np.float64, copy=True) + self.rng.normal(
                0.0, std_value, size=arr.shape
            )
            if np.issubdtype(arr.dtype, np.floating):
                out[key] = noisy.astype(arr.dtype, copy=False)
            else:
                out[key] = noisy
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._add_noise(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(observation), reward, terminated, truncated, info


def make_env_config(
    base_config: dict[str, Any],
    *,
    port: int,
    shot_id: str,
    max_steps: int,
    reference_mode: str,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_config))
    cfg["max_steps"] = int(max_steps)
    cfg.setdefault("predictor", {})
    cfg["predictor"]["host"] = "127.0.0.1"
    cfg["predictor"]["port"] = int(port)
    cfg["predictor"]["shot_id"] = str(shot_id)
    cfg["reference"] = {"mode": "hold" if reference_mode == "hold_initial" else "trajectory"}
    return cfg


def make_env(
    base_config: dict[str, Any],
    *,
    task: str,
    port: int,
    shot_id: str,
    max_steps: int,
    reference_mode: str,
    reference: dict[str, Any] | None,
    metadata: FeatureMetadata,
    target,
    reward_weights: dict[str, float] | None = None,
    reward_type: str = "semifinal_dense",
    reset_perturbation: dict[str, Any] | None = None,
    observation_noise: dict[str, Any] | None = None,
    seed: int = 0,
    reference_variant_index: int | None = None,
):
    def _init() -> gym.Env:
        cfg = make_env_config(
            base_config,
            port=port,
            shot_id=shot_id,
            max_steps=max_steps,
            reference_mode=reference_mode,
        )
        if str(reward_type) == "f2_target_v1":
            if normalize_task(task) != "F2":
                raise ValueError("reward_type='f2_target_v1' is only valid for F2")
            reward = F2XPTTargetReward(target=target, reward_weights=reward_weights)
        elif str(reward_type) == "f2_official_proxy_v1":
            if normalize_task(task) != "F2":
                raise ValueError("reward_type='f2_official_proxy_v1' is only valid for F2")
            f2_reward_weights = dict(reward_weights or {})
            f2_reward_weights.setdefault("total_steps", float(max_steps))
            reward = F2OfficialProxyReward(target=target, reward_weights=f2_reward_weights)
        elif str(reward_type) == "semifinal_dense_v2":
            reward = SemifinalDenseRewardV2(task=task, target=target, reward_weights=reward_weights)
        else:
            reward = SemifinalDenseReward(task=task, target=target, reward_weights=reward_weights)
        env: gym.Env = HFMSimulator(cfg, reward_fn=reward)
        if reference_mode == "hold_initial":
            env = HoldInitialResetWrapper(
                env,
                shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                seed=seed,
            )
        else:
            if reference is None:
                raise ValueError("trajectory reference is required when reference_mode='trajectory'")
            env = ReferenceResetWrapper(
                env,
                reference=reference,
                shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                seed=seed,
                reference_variant_index=reference_variant_index,
            )
        if metadata.action_units == "normalized":
            env = NormalizedAction7DTo12DWrapper(env)
        else:
            env = Action7DTo12DWrapper(env)
        env = RewardInfoWrapper(env)
        env = RawObservationNoiseWrapper(
            env,
            observation_noise=observation_noise,
            seed=seed + 10_000,
        )
        if metadata.feature_mode in {
            "physical_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
            "physical_f1_no_ref_rz_v1",
        }:
            env = PhysicalFeatureHistoryWrapper(env, metadata=metadata)
        else:
            env = FixedFeatureFlattenWrapper(env, metadata=metadata)
        return env

    return _init


def _load_reference_variant_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("variants"), list):
            return list(data["variants"])
        if isinstance(data.get("target_lcfs_variants"), list):
            return list(data["target_lcfs_variants"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"reference variants must be a list or object with variants: {path}")


def _variant_array(
    record: dict[str, Any],
    names: tuple[str, ...],
    shape: tuple[int, ...],
    default: np.ndarray | None,
    *,
    required: bool = False,
    index: int = 0,
) -> np.ndarray | None:
    value = None
    for name in names:
        if name in record:
            value = record[name]
            break
    if value is None:
        if required:
            raise ValueError(f"reference variant #{index} is missing one of {names}")
        return None if default is None else np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"reference variant #{index} field {names[0]} must have shape {shape}, got {arr.shape}")
    return arr


def load_reference_variants(
    path: str | Path,
    *,
    task: str,
    base_target: TargetReference,
    max_steps: int,
) -> list[dict[str, Any]]:
    path = _rooted_path(path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"reference variants file not found: {path}")
    references: list[dict[str, Any]] = []
    task_norm = normalize_task(task)
    for idx, record in enumerate(_load_reference_variant_records(path)):
        if not isinstance(record, dict):
            raise ValueError(f"reference variant #{idx} must be an object")
        lcfs = _variant_array(
            record,
            ("lcfs_points", "lcfs", "reference_lcfs_points"),
            (32, 2),
            None,
            required=True,
            index=idx,
        )
        r_x = base_target.rX
        z_x = base_target.zX
        strike_r = base_target.strike_r
        strike_z = base_target.strike_z
        x_valid = base_target.x_valid
        strike_valid = base_target.strike_valid
        if task_norm == "F2":
            r_x = _variant_array(record, ("reference_rX", "rX"), (4,), base_target.rX, index=idx)
            z_x = _variant_array(record, ("reference_zX", "zX"), (4,), base_target.zX, index=idx)
            strike_r = _variant_array(record, ("reference_strike_r", "strike_r"), (8,), base_target.strike_r, index=idx)
            strike_z = _variant_array(record, ("reference_strike_z", "strike_z"), (8,), base_target.strike_z, index=idx)
            x_valid = _variant_array(record, ("reference_x_valid", "x_valid"), (4,), base_target.x_valid, index=idx)
            strike_valid = _variant_array(
                record,
                ("reference_strike_valid", "strike_valid"),
                (8,),
                base_target.strike_valid,
                index=idx,
            )
        variant_target = TargetReference(
            lcfs=np.asarray(lcfs, dtype=np.float64),
            ip_final=base_target.ip_final,
            rX=None if r_x is None else np.asarray(r_x, dtype=np.float64),
            zX=None if z_x is None else np.asarray(z_x, dtype=np.float64),
            strike_r=None if strike_r is None else np.asarray(strike_r, dtype=np.float64),
            strike_z=None if strike_z is None else np.asarray(strike_z, dtype=np.float64),
            x_valid=None if x_valid is None else np.asarray(x_valid, dtype=np.float64),
            strike_valid=None if strike_valid is None else np.asarray(strike_valid, dtype=np.float64),
        )
        reference = build_reference(task_norm, variant_target, max_steps)
        reference["_variant_id"] = str(record.get("variant_id", record.get("id", f"variant_{idx}")))
        references.append(reference)
    if not references:
        raise ValueError(f"reference variants file contains no variants: {path}")
    return references


def default_shot_id(task: str) -> str:
    return "13844_500" if normalize_task(task) == "F1" else "13844_500"


def default_max_steps(task: str) -> int:
    return 300 if normalize_task(task) == "F1" else 500


def parse_net_arch(value: Any) -> dict[str, list[int]]:
    if isinstance(value, dict):
        pi = value.get("pi", value.get("policy", [256, 256]))
        vf = value.get("vf", value.get("value", pi))
        return {"pi": parse_net_arch(pi)["pi"], "vf": parse_net_arch(vf)["pi"]}
    if isinstance(value, str):
        widths = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        widths = [int(width) for width in list(value or [256, 256])]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"net_arch must contain positive widths, got {value!r}")
    return {"pi": widths, "vf": list(widths)}


def parse_activation_fn(value: Any):
    import torch  # Imported lazily so smoke-rollout/offline config loading stays light.

    name = str(value or "tanh").strip().lower()
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "linear": torch.nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"activation_fn must be one of {sorted(activations)}, got {value!r}")
    return activations[name]


def build_policy_kwargs(ppo: dict[str, Any]) -> dict[str, Any]:
    raw = dict(ppo.get("policy_kwargs") or {})
    net_arch = raw.pop("net_arch", ppo.get("net_arch", [256, 256]))
    activation_fn = raw.pop("activation_fn", ppo.get("activation_fn", "tanh"))
    log_std_init = raw.pop("log_std_init", ppo.get("log_std_init", -1.0))
    policy_kwargs = dict(raw)
    policy_kwargs["net_arch"] = parse_net_arch(net_arch)
    policy_kwargs["activation_fn"] = parse_activation_fn(activation_fn)
    policy_kwargs["log_std_init"] = float(log_std_init)
    return policy_kwargs


def build_ppo_init_kwargs(ppo: dict[str, Any], max_steps: int) -> tuple[str, dict[str, Any]]:
    policy = str(ppo.get("policy", "MlpPolicy"))
    kwargs: dict[str, Any] = {
        "learning_rate": float(ppo["learning_rate"]),
        "n_steps": min(int(ppo["n_steps"]), int(max_steps)),
        "batch_size": int(ppo["batch_size"]),
        "n_epochs": int(ppo["n_epochs"]),
        "gamma": float(ppo["gamma"]),
        "gae_lambda": float(ppo.get("gae_lambda", 0.95)),
        "clip_range": float(ppo.get("clip_range", 0.2)),
        "normalize_advantage": bool(ppo.get("normalize_advantage", True)),
        "ent_coef": float(ppo.get("ent_coef", 0.0)),
        "vf_coef": float(ppo.get("vf_coef", 0.5)),
        "max_grad_norm": float(ppo.get("max_grad_norm", 0.5)),
        "use_sde": bool(ppo.get("use_sde", False)),
        "sde_sample_freq": int(ppo.get("sde_sample_freq", -1)),
        "policy_kwargs": build_policy_kwargs(ppo),
        "verbose": int(ppo.get("verbose", 1)),
        "seed": None if ppo.get("seed") is None else int(ppo["seed"]),
        "device": str(ppo.get("device", "auto")),
    }
    for key in ("clip_range_vf", "target_kl", "tensorboard_log"):
        if ppo.get(key) is not None:
            kwargs[key] = ppo[key]
    return policy, kwargs


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--task", choices=["F1", "F2"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--ports", type=str, default=None)
    parser.add_argument("--shot-id", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Alias for --save-dir.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--net-arch", type=str, default=None)
    parser.add_argument("--activation-fn", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reward-log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None, help="Enable WandB logging with this project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional WandB run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional WandB entity.")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated WandB tags.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB even if enabled in config.")
    parser.add_argument("--lr-anneal", dest="lr_anneal", action="store_true", default=None, help="Enable cosine annealing after optional warmup.")
    parser.add_argument("--no-lr-anneal", dest="lr_anneal", action="store_false", help="Disable cosine annealing.")
    parser.add_argument("--lr-warmup-updates", type=int, default=None, help="Number of PPO updates for LR warmup.")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress-bar", dest="progress_bar", action="store_true", default=None)
    progress_group.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument(
        "--smoke-rollout",
        action="store_true",
        help="Run a short zero-action rollout without importing PPO.",
    )


def _build_training_context(args: argparse.Namespace) -> dict[str, Any]:
    train_config = load_train_config(args.train_config) if args.train_config else {}
    task_value = args.task or train_config.get("task")
    if task_value is None:
        raise ValueError("--task is required when --train-config is not provided")
    task = normalize_task(task_value)

    if args.num_envs is not None:
        train_config["num_envs"] = int(args.num_envs)

    stage = str(train_config.get("stage", "baseline"))
    reference_mode = str(train_config.get("reference_mode", "trajectory"))
    max_steps = int(args.max_steps or train_config.get("max_steps") or default_max_steps(task))

    reference_path = _rooted_path(
        args.reference or train_config.get("reference"),
        default_reference_path(task),
    )
    if reference_path is None:
        raise ValueError("reference_path could not be resolved")
    target = load_target_reference(reference_path, task=task)
    reference_variants_path = _rooted_path(train_config.get("reference_variants"))
    reference_variants_enabled = bool(train_config.get("reference_variants_enabled", reference_variants_path is not None))
    if not reference_variants_enabled:
        reference_variants_path = None
    reference = None
    reference_variant_count = 0
    if reference_mode != "hold_initial":
        if reference_variants_path is not None:
            reference = load_reference_variants(
                reference_variants_path,
                task=task,
                base_target=target,
                max_steps=max_steps,
            )
            reference_variant_count = len(reference)
        else:
            reference = build_reference(task, target, max_steps)
            reference_variant_count = 1

    feature_cfg = dict(train_config.get("features", {}))
    feature_mode = str(feature_cfg.get("mode", "raw"))
    history_steps = int(feature_cfg.get("history_steps", 0))
    action_cfg = dict(train_config.get("actions", {}))
    action_mode = str(action_cfg.get("mode", "physical_7d"))
    if action_mode in {"normalized", "normalized_7d"}:
        action_units = "normalized"
    elif action_mode in {"physical", "physical_7d"}:
        action_units = "physical_voltage"
    else:
        raise ValueError(f"Unsupported actions.mode: {action_mode!r}")
    metadata = FeatureMetadata.for_task(
        task,
        mode=feature_mode,
        history_steps=history_steps,
        action_units=action_units,
    )

    env_config_path = _rooted_path(
        args.config or train_config.get("env_config"),
        ROOT / "configs" / "env_default.yaml",
    )
    if env_config_path is None:
        raise ValueError("env_config_path could not be resolved")
    base_config = load_yaml(env_config_path)

    ppo = {
        "total_timesteps": 10_000,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "use_sde": False,
        "sde_sample_freq": -1,
        "net_arch": [256, 256],
        "activation_fn": "tanh",
        "log_std_init": -1.0,
        "verbose": 1,
        "device": "auto",
        "progress_bar": True,
        "seed": 0,
        "lr_anneal": False,
        "lr_warmup_updates": 0,
    }
    ppo.update(dict(train_config.get("ppo", {})))
    cli_ppo_overrides = {
        "total_timesteps": (getattr(args, "total_timesteps", None), int),
        "learning_rate": (getattr(args, "learning_rate", None), float),
        "n_steps": (getattr(args, "n_steps", None), int),
        "batch_size": (getattr(args, "batch_size", None), int),
        "n_epochs": (getattr(args, "n_epochs", None), int),
        "gamma": (getattr(args, "gamma", None), float),
        "gae_lambda": (getattr(args, "gae_lambda", None), float),
        "ent_coef": (getattr(args, "ent_coef", None), float),
        "clip_range": (getattr(args, "clip_range", None), float),
        "target_kl": (getattr(args, "target_kl", None), float),
        "max_grad_norm": (getattr(args, "max_grad_norm", None), float),
        "vf_coef": (getattr(args, "vf_coef", None), float),
        "log_std_init": (getattr(args, "log_std_init", None), float),
        "net_arch": (getattr(args, "net_arch", None), str),
        "activation_fn": (getattr(args, "activation_fn", None), str),
        "device": (getattr(args, "device", None), str),
        "seed": (getattr(args, "seed", None), int),
        "lr_anneal": (getattr(args, "lr_anneal", None), bool),
        "lr_warmup_updates": (getattr(args, "lr_warmup_updates", None), int),
    }
    for key, (value, caster) in cli_ppo_overrides.items():
        if value is not None:
            ppo[key] = caster(value)
    if getattr(args, "progress_bar", None) is not None:
        ppo["progress_bar"] = bool(args.progress_bar)

    vecnormalize = {
        "enabled": False,
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
        "clip_reward": 10.0,
        "gamma": None,
        "epsilon": 1e-8,
        "load_path": None,
        "save": True,
    }
    vecnormalize.update(dict(train_config.get("vecnormalize", {})))

    checkpoint = {
        "enabled": False,
        "save_interval_timesteps": 50_000,
        "save_vecnormalize": True,
    }
    checkpoint.update(dict(train_config.get("checkpoint", {})))

    reward_log_interval = train_config.get("reward_log_interval", 1000)
    if getattr(args, "reward_log_interval", None) is not None:
        reward_log_interval = int(args.reward_log_interval)

    fallback_shot_id = args.shot_id or train_config.get("shot_id") or default_shot_id(task)
    ports_spec = args.ports or train_config.get("ports") or "2223"
    requested_num_envs = int(train_config.get("num_envs", 0) or 0)
    if requested_num_envs <= 0:
        requested_num_envs = len(parse_ports(ports_spec))
    num_envs = max(1, requested_num_envs)
    shot_ids = assign_shots_to_envs(train_config.get("shot_groups"), num_envs, fallback_shot_id)
    if args.shot_id is not None:
        shot_ids = [args.shot_id for _ in range(num_envs)]
    ports = ports_for_envs(ports_spec, num_envs)
    base_seed = 0 if ppo.get("seed") is None else int(ppo["seed"])
    reference_variant_assignment = str(
        train_config.get("reference_variant_assignment", "random_per_reset")
    ).lower()
    reference_variant_assignment_modes = {
        "random_per_reset",
        "per_reset_random",
        "random",
        "round_robin_by_env",
        "fixed_by_env",
        "random_by_env",
        "random_fixed_by_env",
        "random_env",
    }
    if reference_variant_assignment not in reference_variant_assignment_modes:
        raise ValueError(
            "unknown reference_variant_assignment="
            f"{reference_variant_assignment!r}; expected one of "
            f"{sorted(reference_variant_assignment_modes)}"
        )
    reference_variant_rng = np.random.default_rng(base_seed + 75_000)
    env_specs = []
    for i in range(num_envs):
        env_spec = {
            "index": i,
            "port": ports[i],
            "shot_id": shot_ids[i],
            "seed": base_seed + i,
        }
        if reference_variant_assignment in {"round_robin_by_env", "fixed_by_env"} and reference_variant_count > 0:
            env_spec["reference_variant_index"] = i % reference_variant_count
        elif reference_variant_assignment in {"random_by_env", "random_fixed_by_env", "random_env"} and reference_variant_count > 0:
            env_spec["reference_variant_index"] = int(
                reference_variant_rng.integers(0, reference_variant_count)
            )
        env_specs.append(env_spec)

    save_dir = _rooted_path(
        args.save_dir or args.output_dir or train_config.get("output_dir"),
        ROOT / "results" / f"{task.lower()}_ppo",
    )
    if save_dir is None:
        raise ValueError("save_dir could not be resolved")

    return {
        "task": task,
        "stage": stage,
        "max_steps": max_steps,
        "reference_mode": reference_mode,
        "reference_path": reference_path,
        "reference_variants_path": reference_variants_path,
        "reference_variants_enabled": reference_variants_enabled,
        "reference_variant_count": reference_variant_count,
        "reference_variant_assignment": reference_variant_assignment,
        "target": target,
        "reference": reference,
        "metadata": metadata,
        "env_config_path": env_config_path,
        "base_config": base_config,
        "save_dir": save_dir,
        "env_specs": env_specs,
        "ppo": ppo,
        "vecnormalize": vecnormalize,
        "checkpoint": checkpoint,
        "reward_log_interval": int(reward_log_interval),
        "reward_weights": train_config.get("reward_weights"),
        "reward_type": str(train_config.get("reward_type", "semifinal_dense")),
        "reset_perturbation": train_config.get("reset_perturbation"),
        "observation_noise": train_config.get("observation_noise"),
        "resume": train_config.get("resume", {"enabled": False}),
        "episode_reward_components": train_config.get("episode_reward_components", {"enabled": False}),
        "wandb": train_config.get("wandb", {"enabled": False}),
    }


def run_smoke_rollout(args: argparse.Namespace) -> int:
    ctx = _build_training_context(args)
    task = ctx["task"]
    max_steps = ctx["max_steps"]
    env_spec = ctx["env_specs"][0]
    env = make_env(
        ctx["base_config"],
        task=task,
        port=env_spec["port"],
        shot_id=env_spec["shot_id"],
        max_steps=max_steps,
        reference_mode=ctx["reference_mode"],
        reference=ctx["reference"],
        metadata=ctx["metadata"],
        target=ctx["target"],
        reward_weights=ctx["reward_weights"],
        reward_type=ctx["reward_type"],
        reset_perturbation=ctx["reset_perturbation"],
        observation_noise=ctx["observation_noise"],
        seed=env_spec["seed"],
        reference_variant_index=env_spec.get("reference_variant_index"),
    )()
    try:
        obs, info = env.reset(seed=int(ctx["ppo"]["seed"]))
        print(
            f"{task} reset ok: stage={ctx['stage']}, shot_id={info['shot_id']}, "
            f"port={env_spec['port']}, reference_mode={info.get('reference_mode')}, "
            f"obs_shape={obs.shape}"
        )
        for step in range(min(3, max_steps)):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"{task} step {step}: reward={reward:.4f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


def run_training(args: argparse.Namespace) -> int:
    if args.smoke_rollout:
        return run_smoke_rollout(args)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    except ImportError:
        print("Missing dependencies: pip install stable-baselines3 torch")
        return 1

    ctx = _build_training_context(args)
    wandb_run = None
    tensorboard_log_path = None
    wandb_cfg = dict(ctx.get("wandb") or {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False)) or any(
        getattr(args, name, None) for name in ("wandb_project", "wandb_name", "wandb_entity", "wandb_tags")
    )
    if args.no_wandb:
        wandb_enabled = False
    if wandb_enabled:
        try:
            import wandb

            wandb_kwargs = {"sync_tensorboard": True}
            project = args.wandb_project or wandb_cfg.get("project")
            name = args.wandb_name or wandb_cfg.get("name")
            entity = args.wandb_entity or wandb_cfg.get("entity")
            tags = args.wandb_tags or wandb_cfg.get("tags")
            if project:
                wandb_kwargs["project"] = project
            if name:
                wandb_kwargs["name"] = name
            if entity:
                wandb_kwargs["entity"] = entity
            if tags:
                wandb_kwargs["tags"] = [tag.strip() for tag in str(tags).split(",") if tag.strip()] if isinstance(tags, str) else list(tags)
            wandb_run = wandb.init(**wandb_kwargs)
            tensorboard_log_path = f"runs/{wandb_run.id}"
            print(f"WandB run initialized: {wandb_run.name}")
        except ImportError:
            print("WandB not installed, skipping wandb logging")
        except Exception as exc:
            print(f"Failed to initialize WandB: {exc}")
    task = ctx["task"]
    max_steps = ctx["max_steps"]
    metadata = ctx["metadata"]
    save_dir = ctx["save_dir"]
    ppo = ctx["ppo"]
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata.save(save_dir / f"{task.lower()}_features.metadata.json")

    env_fns = [
        make_env(
            ctx["base_config"],
            task=task,
            port=env_spec["port"],
            shot_id=env_spec["shot_id"],
            max_steps=max_steps,
            reference_mode=ctx["reference_mode"],
            reference=ctx["reference"],
            metadata=metadata,
            target=ctx["target"],
            reward_weights=ctx["reward_weights"],
            reward_type=ctx["reward_type"],
            reset_perturbation=ctx["reset_perturbation"],
            observation_noise=ctx["observation_noise"],
            seed=env_spec["seed"],
            reference_variant_index=env_spec.get("reference_variant_index"),
        )
        for env_spec in ctx["env_specs"]
    ]
    vec_env = DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env, filename=str(save_dir / "monitor.csv"))
    vecnormalize_cfg = dict(ctx["vecnormalize"])
    resume_cfg = dict(ctx["resume"] or {})
    if bool(vecnormalize_cfg.get("enabled", False)):
        vecnormalize_path = _rooted_path(
            vecnormalize_cfg.get("load_path") or resume_cfg.get("vecnormalize_path")
        )
        if vecnormalize_path is not None:
            if not vecnormalize_path.exists():
                raise FileNotFoundError(f"VecNormalize state not found: {vecnormalize_path}")
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            print(f"loaded VecNormalize: {vecnormalize_path}")
            vec_env.training = True
            vec_env.norm_reward = bool(vecnormalize_cfg.get("norm_reward", True))
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=bool(vecnormalize_cfg.get("norm_obs", True)),
                norm_reward=bool(vecnormalize_cfg.get("norm_reward", True)),
                clip_obs=float(vecnormalize_cfg.get("clip_obs", 10.0)),
                clip_reward=float(vecnormalize_cfg.get("clip_reward", 10.0)),
                gamma=float(vecnormalize_cfg.get("gamma") or ppo["gamma"]),
                epsilon=float(vecnormalize_cfg.get("epsilon", 1e-8)),
            )
        print(
            "VecNormalize: "
            f"norm_obs={vec_env.norm_obs}, norm_reward={vec_env.norm_reward}, "
            f"clip_obs={vec_env.clip_obs}, clip_reward={vec_env.clip_reward}"
        )

    def _assignment_label(spec: dict[str, Any]) -> str:
        label = f"env{spec['index']}:{spec['shot_id']}@{spec['port']}"
        if "reference_variant_index" in spec:
            label += f"#ref{spec['reference_variant_index']}"
        return label

    assignments = ", ".join(_assignment_label(spec) for spec in ctx["env_specs"])
    reference_variant_note = ""
    if ctx.get("reference_variants_path") is not None:
        reference_variant_note = (
            f", reference_variants={ctx['reference_variants_path']}"
            f" ({ctx.get('reference_variant_count', 0)} variants, "
            f"assignment={ctx.get('reference_variant_assignment', 'random_per_reset')})"
        )
    print(
        f"training {task}: stage={ctx['stage']}, reference_mode={ctx['reference_mode']}, "
        f"max_steps={max_steps}, obs_dim={metadata.obs_dim}, assignments=[{assignments}], "
        f"reference={ctx['reference_path']}{reference_variant_note}"
    )
    variant_count = int(ctx.get("reference_variant_count", 0) or 0)
    if (
        ctx.get("reference_variant_assignment") in {"round_robin_by_env", "fixed_by_env"}
        and variant_count > 0
        and len(ctx["env_specs"]) % variant_count != 0
    ):
        print(
            "warning: num_envs is not an integer multiple of reference variants; "
            "round-robin assignment will be imbalanced"
        )
    if bool(resume_cfg.get("enabled", False)):
        resume_path = _rooted_path(resume_cfg.get("model_path"))
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"resume model not found: {resume_path}")
        resume_custom_objects = {
            "learning_rate": float(ppo["learning_rate"]),
            "n_steps": min(int(ppo["n_steps"]), int(max_steps)),
            "batch_size": int(ppo["batch_size"]),
            "n_epochs": int(ppo["n_epochs"]),
            "gamma": float(ppo["gamma"]),
            "gae_lambda": float(ppo.get("gae_lambda", 0.95)),
            "clip_range": float(ppo.get("clip_range", 0.2)),
            "clip_range_vf": ppo.get("clip_range_vf"),
            "normalize_advantage": bool(ppo.get("normalize_advantage", True)),
            "ent_coef": float(ppo.get("ent_coef", 0.0)),
            "vf_coef": float(ppo.get("vf_coef", 0.5)),
            "max_grad_norm": float(ppo.get("max_grad_norm", 0.5)),
            "target_kl": ppo.get("target_kl"),
            "use_sde": bool(ppo.get("use_sde", False)),
            "sde_sample_freq": int(ppo.get("sde_sample_freq", -1)),
        }
        model = PPO.load(
            str(resume_path),
            env=vec_env,
            device=str(ppo.get("device", "auto")),
            custom_objects=resume_custom_objects,
        )
        print(f"resumed: {resume_path} with PPO overrides from train config")
    else:
        policy, ppo_kwargs = build_ppo_init_kwargs(ppo, max_steps)
        warmup_updates = int(ppo.get("lr_warmup_updates", 0) or 0)
        lr_anneal = bool(ppo.get("lr_anneal", False))
        if warmup_updates > 0 or lr_anneal:
            initial_lr = float(ppo_kwargs.get("learning_rate", ppo.get("learning_rate", 3e-4)))
            timesteps_per_update = int(ppo_kwargs.get("n_steps", ppo.get("n_steps", 128))) * max(1, len(env_fns))
            ppo_kwargs["learning_rate"] = warmup_cosine_schedule(
                initial_lr=initial_lr,
                timesteps_per_update=timesteps_per_update,
                total_timesteps=int(ppo["total_timesteps"]),
                warmup_updates=warmup_updates,
                anneal=lr_anneal,
            )
            print(
                f"Using LR schedule: initial_lr={initial_lr}, warmup_updates={warmup_updates}, "
                f"anneal={lr_anneal}, timesteps_per_update={timesteps_per_update}"
            )
        if tensorboard_log_path is not None:
            ppo_kwargs["tensorboard_log"] = tensorboard_log_path
        elif ppo_kwargs.get("tensorboard_log") is not None:
            ppo_kwargs["tensorboard_log"] = str(_rooted_path(ppo_kwargs["tensorboard_log"]))
        model = PPO(policy, vec_env, **ppo_kwargs)

    callbacks = []
    if int(ctx["reward_log_interval"]) > 0:
        callbacks.append(
            RewardComponentsCallback(
                log_dir=save_dir / "logs",
                log_interval=int(ctx["reward_log_interval"]),
            )
        )
    checkpoint_cfg = dict(ctx["checkpoint"])
    if bool(checkpoint_cfg.get("enabled", False)):
        save_interval = int(checkpoint_cfg.get("save_interval_timesteps", 50_000))
        save_freq = max(1, save_interval // max(1, len(env_fns)))
        callbacks.append(
            PlasmaCheckpointCallback(
                save_freq=save_freq,
                save_path=save_dir / "checkpoints",
                name_prefix=f"ppo_{task.lower()}",
                save_vecnormalize=bool(checkpoint_cfg.get("save_vecnormalize", True)),
            )
        )
    episode_components_cfg = dict(ctx.get("episode_reward_components") or {})
    if bool(episode_components_cfg.get("enabled", False)):
        callbacks.append(
            EpisodeRewardComponentsCallback(
                log_dir=save_dir / "logs",
                reward_weights=ctx.get("reward_weights") or {},
            )
        )
    callback = CallbackList(callbacks) if callbacks else None
    reset_num_timesteps = bool(resume_cfg.get("reset_num_timesteps", not bool(resume_cfg.get("enabled", False))))

    try:
        model.learn(
            total_timesteps=int(ppo["total_timesteps"]),
            callback=callback,
            progress_bar=bool(ppo.get("progress_bar", True)),
            reset_num_timesteps=reset_num_timesteps,
        )
        out = save_dir / f"ppo_{task.lower()}_latest"
        model.save(str(out))

        vecnormalize = find_vecnormalize(vec_env)
        if vecnormalize is not None and bool(vecnormalize_cfg.get("save", True)):
            vecnormalize.training = False
            vecnormalize.norm_reward = False
            vecnormalize.save(str(save_dir / "vecnormalize.pkl"))
            save_vecnormalize_stats(vecnormalize, save_dir / "vecnormalize_stats.npz")
            print(f"VecNormalize saved: {save_dir / 'vecnormalize.pkl'}")

        print(f"saved: {out}.zip")
        print(f"metadata: {save_dir / f'{task.lower()}_features.metadata.json'}")
    finally:
        vec_env.close()
        if wandb_run is not None:
            try:
                wandb_run.finish()
                print("WandB run finished")
            except Exception as exc:
                print(f"Failed to finish WandB run: {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semifinal F1/F2 PPO training baseline")
    add_args(parser)
    return run_training(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
