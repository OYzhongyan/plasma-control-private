"""Supervised noise distillation for F1/F2 PPO policies.

The teacher sees clean normalized features. The student prepends a trainable
SB3 feature extractor that sees noisy normalized features and is trained to
denoise them while matching the frozen teacher actor/critic outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import pickle
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import HFMSimulator, NormalizedAction7DTo12DWrapper, Action7DTo12DWrapper  # noqa: E402
from training.denoising import ResidualDenoisingFeatureExtractor  # noqa: E402
from training.features import (  # noqa: E402
    ACTION_SCALE_7D_V,
    COIL_LIMITS_A,
    FeatureMetadata,
    _array,
    _clip_finite,
    _scalar,
    physical_f2_xpt_feature_vector,
    physical_feature_vector,
)
from training.references import build_reference, load_target_reference, normalize_task  # noqa: E402
from training.rewards import SemifinalDenseReward  # noqa: E402
from training.train_ppo import (  # noqa: E402
    HoldInitialResetWrapper,
    ReferenceResetWrapper,
    RewardInfoWrapper,
    _rooted_path,
    load_train_config,
    load_yaml,
    make_env_config,
    parse_ports,
    load_reference_variants,
)


DEFAULT_TEACHER_MODEL = ROOT / "results/f1_direct_target_progress_regression_continue_lcfs_targets/ppo_f1_latest.zip"
DEFAULT_VECNORMALIZE = ROOT / "results/f1_direct_target_progress_regression_continue_lcfs_targets/vecnormalize_stats.npz"
DEFAULT_METADATA = ROOT / "results/f1_direct_target_progress_regression_continue_lcfs_targets/f1_features.metadata.json"
DEFAULT_TRAIN_CONFIG = ROOT / "training/configs/f1_direct_target_progress_regression_continue_lcfs_targets.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/f1_direct_target_noise_distill_lcfs_targets"


@dataclass
class DistillDataset:
    clean_obs: np.ndarray
    episode_ids: np.ndarray
    noisy_obs: np.ndarray | None = None


def parse_float_list(spec: str | float | int | None) -> list[float]:
    if spec is None:
        return []
    if isinstance(spec, (float, int)):
        return [float(spec)]
    values: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError(f"empty float list: {spec!r}")
    return values


def parse_noise_schedule(stds_spec: str, probs_spec: str | None) -> tuple[np.ndarray, np.ndarray]:
    stds = np.asarray(parse_float_list(stds_spec), dtype=np.float32)
    if np.any(stds < 0.0):
        raise ValueError(f"noise stds must be non-negative, got {stds}")
    if probs_spec:
        probs = np.asarray(parse_float_list(probs_spec), dtype=np.float64)
        if probs.shape != stds.shape:
            raise ValueError("--normalized-noise-std-probs must have the same length as --normalized-noise-stds")
        if np.any(probs < 0.0) or float(probs.sum()) <= 0.0:
            raise ValueError(f"invalid noise probabilities: {probs}")
        probs = probs / probs.sum()
    else:
        probs = np.full(stds.shape, 1.0 / len(stds), dtype=np.float64)
    return stds, probs



NOISE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "ip": ("ip", "ip_error_50ka"),
    "rz": ("r", "r_error_3cm", "z", "z_error_3cm"),
    "lcfs": ("lcfs_points_absolute", "lcfs_error_3cm"),
    "xpt": (
        "xpt_nx_norm",
        "xpt_nx_error",
        "xpt_x_valid",
        "xpt_r",
        "xpt_z",
        "xpt_position_error_3cm",
        "xpt_flux_error_0p003",
    ),
    "reference_xpt": ("reference_xpt_r", "reference_xpt_z", "reference_xpt_valid", "reference_nx"),
    "history_ip": ("history_delta_ip",),
    "history_rz": ("history_delta_rz",),
    "history": ("history_delta_ip", "history_delta_rz"),
    "all_current": (
        "ip",
        "ip_error_50ka",
        "r",
        "r_error_3cm",
        "z",
        "z_error_3cm",
        "lcfs_points_absolute",
        "lcfs_error_3cm",
        "xpt_nx_norm",
        "xpt_nx_error",
        "xpt_x_valid",
        "xpt_r",
        "xpt_z",
        "xpt_position_error_3cm",
        "xpt_flux_error_0p003",
    ),
    "all": ("*",),
}


def parse_name_list(spec: str | None) -> list[str]:
    return [part.strip() for part in str(spec or "").split(",") if part.strip()]


def noise_feature_mask(metadata: FeatureMetadata, groups_spec: str | None) -> np.ndarray:
    """Return a boolean mask over normalized features that should receive noise."""

    groups = parse_name_list(groups_spec) or ["ip", "rz", "lcfs", "history_ip", "history_rz"]
    key_ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for key in metadata.keys:
        size = int(np.prod(metadata.shapes[key]))
        key_ranges[key] = (start, start + size)
        start += size
    if start != metadata.obs_dim:
        raise ValueError(f"metadata obs_dim mismatch: ranges end at {start}, obs_dim={metadata.obs_dim}")

    mask = np.zeros(metadata.obs_dim, dtype=bool)
    for group in groups:
        names = NOISE_FEATURE_GROUPS.get(group, (group,))
        if "*" in names:
            mask[:] = True
            continue
        for name in names:
            if name not in key_ranges:
                if group in NOISE_FEATURE_GROUPS:
                    continue
                raise ValueError(
                    f"unknown noise feature group/key {name!r}; "
                    f"valid groups={sorted(NOISE_FEATURE_GROUPS)}, valid keys={metadata.keys}"
                )
            lo, hi = key_ranges[name]
            mask[lo:hi] = True
    if not np.any(mask):
        raise ValueError(f"noise feature mask is empty for groups={groups}")
    return mask



def _feature_key_ranges(metadata: FeatureMetadata) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for key in metadata.keys:
        size = int(np.prod(metadata.shapes[key]))
        ranges[key] = (start, start + size)
        start += size
    if start != metadata.obs_dim:
        raise ValueError(f"metadata obs_dim mismatch: ranges end at {start}, obs_dim={metadata.obs_dim}")
    return ranges


def _selected_noise_names(groups_spec: str | None) -> set[str]:
    groups = parse_name_list(groups_spec) or ["ip", "rz", "lcfs", "history_ip", "history_rz"]
    names: set[str] = set()
    for group in groups:
        group_names = NOISE_FEATURE_GROUPS.get(group, (group,))
        if "*" in group_names:
            return {"*"}
        names.update(group_names)
    return names


def _feature_scale(
    key_ranges: dict[str, tuple[int, int]],
    obs_var: np.ndarray,
    eps: float,
    key: str,
    shape: tuple[int, ...],
    factor: float = 1.0,
) -> np.ndarray:
    if key not in key_ranges:
        return np.zeros(shape, dtype=np.float32)
    lo, hi = key_ranges[key]
    scale = np.sqrt(np.maximum(obs_var[lo:hi], 0.0) + float(eps)).astype(np.float32) * float(factor)
    return scale.reshape(shape)


def _raw_observation_noise_scales(
    metadata: FeatureMetadata,
    obs_var: np.ndarray,
    eps: float,
    groups_spec: str | None,
) -> dict[str, np.ndarray]:
    selected = _selected_noise_names(groups_spec)
    all_selected = "*" in selected
    key_ranges = _feature_key_ranges(metadata)
    scales: dict[str, np.ndarray] = {}

    def enabled(*names: str) -> bool:
        return all_selected or any(name in selected for name in names)

    if enabled("ip", "ip_error_50ka", "history_delta_ip"):
        scales["Ip"] = _feature_scale(key_ranges, obs_var, eps, "ip", (1,), 500_000.0)
    if enabled("r", "r_error_3cm", "history_delta_rz"):
        scales["R"] = _feature_scale(key_ranges, obs_var, eps, "r", (1,), 1.0)
    if enabled("z", "z_error_3cm", "history_delta_rz"):
        scales["Z"] = _feature_scale(key_ranges, obs_var, eps, "z", (1,), 1.0)
    if enabled("lcfs_points_absolute", "lcfs_error_3cm"):
        scales["lcfs_points"] = _feature_scale(key_ranges, obs_var, eps, "lcfs_points_absolute", (32, 2), 1.0)
    if enabled("xpt_r", "xpt_position_error_3cm"):
        r_scale = _feature_scale(key_ranges, obs_var, eps, "xpt_r", (4,), 1.0)
        scales["rX"] = np.concatenate([r_scale, np.full(2, float(np.mean(r_scale)), dtype=np.float32)])
    if enabled("xpt_z", "xpt_position_error_3cm"):
        z_scale = _feature_scale(key_ranges, obs_var, eps, "xpt_z", (4,), 1.0)
        scales["zX"] = np.concatenate([z_scale, np.full(2, float(np.mean(z_scale)), dtype=np.float32)])
    if enabled("xpt_flux_error_0p003"):
        fx_scale = _feature_scale(key_ranges, obs_var, eps, "xpt_flux_error_0p003", (4,), 0.003)
        scales["FX"] = np.concatenate([fx_scale, np.full(2, float(np.mean(fx_scale)), dtype=np.float32)])
    return scales


def _add_raw_observation_noise(
    observation: dict[str, Any],
    *,
    scales: dict[str, np.ndarray],
    std: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if float(std) <= 0.0 or not scales:
        return dict(observation)
    noisy = dict(observation)
    for key, scale in scales.items():
        if key not in observation:
            continue
        value = np.asarray(observation[key], dtype=np.float32)
        scale_arr = np.asarray(scale, dtype=np.float32)
        try:
            scale_arr = np.broadcast_to(scale_arr, value.shape)
        except ValueError:
            scale_arr = np.resize(scale_arr.reshape(-1), value.size).reshape(value.shape).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=value.shape).astype(np.float32) * scale_arr * float(std)
        noisy[key] = (value + noise).astype(np.float32)
    return noisy
def load_dataset(path: Path, metadata: FeatureMetadata) -> DistillDataset:
    data = np.load(path)
    clean = np.asarray(data["clean_obs"], dtype=np.float32)
    episodes = np.asarray(data["episode_ids"], dtype=np.int64)
    noisy = np.asarray(data["noisy_obs"], dtype=np.float32) if "noisy_obs" in data.files else None
    if clean.ndim != 2 or clean.shape[1] != metadata.obs_dim:
        raise ValueError(f"cached dataset shape mismatch: clean={clean.shape}, obs_dim={metadata.obs_dim}")
    if episodes.shape[0] != clean.shape[0]:
        raise ValueError("cached dataset episode_ids length mismatch")
    if noisy is not None and noisy.shape != clean.shape:
        raise ValueError(f"cached noisy dataset shape mismatch: noisy={noisy.shape}, clean={clean.shape}")
    if not np.isfinite(clean).all():
        raise ValueError("cached dataset contains non-finite clean features")
    if noisy is not None and not np.isfinite(noisy).all():
        raise ValueError("cached dataset contains non-finite noisy features")
    return DistillDataset(clean_obs=clean, episode_ids=episodes, noisy_obs=noisy)


def save_dataset(path: Path, dataset: DistillDataset, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"clean_obs": dataset.clean_obs, "episode_ids": dataset.episode_ids}
    if dataset.noisy_obs is not None:
        payload["noisy_obs"] = dataset.noisy_obs
    np.savez_compressed(path, **payload)
    with open(path.with_suffix(".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

class FeatureSequenceBuilder:
    """Feature builder matching PhysicalFeatureHistoryWrapper for offline dict observations."""

    def __init__(self, metadata: FeatureMetadata):
        supported_modes = {
            "physical_v1",
            "physical_f1_no_ref_rz_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
        }
        if metadata.feature_mode not in supported_modes:
            raise ValueError(
                "noise distillation expects one of "
                f"{sorted(supported_modes)}, got {metadata.feature_mode!r}"
            )
        self.metadata = metadata
        self.history_steps = max(0, int(metadata.history_steps))
        self._last_observation: dict[str, Any] | None = None
        self._reset_history()

    def clone_empty(self) -> "FeatureSequenceBuilder":
        return FeatureSequenceBuilder(self.metadata)

    def _reset_history(self) -> None:
        self._history_action = np.zeros((self.history_steps, 7), dtype=np.float32)
        self._history_delta_ip = np.zeros((self.history_steps, 1), dtype=np.float32)
        self._history_delta_rz = np.zeros((self.history_steps, 2), dtype=np.float32)
        self._history_delta_i_pf = np.zeros((self.history_steps, 12), dtype=np.float32)

    @staticmethod
    def _push(buffer: np.ndarray, value: np.ndarray) -> np.ndarray:
        if buffer.shape[0] == 0:
            return buffer
        out = np.roll(buffer, shift=-1, axis=0)
        out[-1] = value
        return out

    def _history_vector(self) -> np.ndarray:
        if self.history_steps <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [
                self._history_action.reshape(-1),
                self._history_delta_ip.reshape(-1),
                self._history_delta_rz.reshape(-1),
                self._history_delta_i_pf.reshape(-1),
            ]
        ).astype(np.float32)

    def _base_feature(self, observation: dict[str, Any]) -> np.ndarray:
        if self.metadata.feature_mode in {"physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"}:
            return physical_f2_xpt_feature_vector(
                observation,
                no_ref_rz=self.metadata.feature_mode == "physical_f2_xpt_no_ref_rz_v1",
                dtype=np.float32,
            )
        return physical_feature_vector(
            observation,
            self.metadata.task,
            mode=self.metadata.feature_mode,
            dtype=np.float32,
        )

    def _build_observation(self, observation: dict[str, Any]) -> np.ndarray:
        base = self._base_feature(observation)
        out = np.concatenate([base, self._history_vector()]).astype(np.float32)
        if out.shape != (self.metadata.obs_dim,):
            raise ValueError(f"feature dim mismatch: got {out.shape}, expected {(self.metadata.obs_dim,)}")
        return out

    def reset(self, observation: dict[str, Any]) -> np.ndarray:
        self._last_observation = observation
        self._reset_history()
        return self._build_observation(observation)

    def step(self, action: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        prev = self._last_observation
        if self.history_steps > 0:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            action7 = np.zeros(7, dtype=np.float32)
            n_action = min(7, action_arr.size)
            if n_action > 0:
                action7[:n_action] = action_arr[:n_action]
            if self.metadata.action_units == "normalized":
                action_norm = np.clip(action7, -1.0, 1.0)
            else:
                action_norm = action7 / ACTION_SCALE_7D_V

            if prev is None:
                delta_ip = np.zeros(1, dtype=np.float32)
                delta_rz = np.zeros(2, dtype=np.float32)
                delta_i_pf = np.zeros(12, dtype=np.float32)
            else:
                delta_ip = np.asarray(
                    [(_scalar(observation, "Ip") - _scalar(prev, "Ip")) / 50_000.0],
                    dtype=np.float32,
                )
                delta_rz = np.asarray(
                    [
                        (_scalar(observation, "R") - _scalar(prev, "R")) / 0.03,
                        (_scalar(observation, "Z") - _scalar(prev, "Z")) / 0.03,
                    ],
                    dtype=np.float32,
                )
                delta_i_pf = (
                    _array(observation, "I_PF", (12,)) - _array(prev, "I_PF", (12,))
                ) / COIL_LIMITS_A

            self._history_action = self._push(self._history_action, _clip_finite(action_norm))
            self._history_delta_ip = self._push(self._history_delta_ip, _clip_finite(delta_ip))
            self._history_delta_rz = self._push(self._history_delta_rz, _clip_finite(delta_rz))
            self._history_delta_i_pf = self._push(self._history_delta_i_pf, _clip_finite(delta_i_pf))

        self._last_observation = observation
        return self._build_observation(observation)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill an F1/F2 PPO policy for observation-noise robustness.")
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-variants",
        type=Path,
        default=None,
        help="Optional LCFS target variants JSON. Defaults to train config reference_variants when enabled.",
    )
    parser.add_argument(
        "--reference-variants-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable collecting rollouts for reference variants. Default follows train config.",
    )
    parser.add_argument(
        "--reference-variant-indices",
        type=str,
        default="",
        help="Optional comma/range subset of variant indices, e.g. 0,2-4. Empty means all variants.",
    )
    parser.add_argument(
        "--include-base-reference",
        action="store_true",
        help="Also collect rollouts for the base --reference target.json reference in addition to variants.",
    )
    parser.add_argument("--ports", type=str, default=None, help="Comma/range port spec for rollout workers.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel rollout workers; each worker uses one HFM port.")
    parser.add_argument("--shots", type=str, default="13844_500,13844_600")
    parser.add_argument("--seeds", type=str, default="0-16")
    parser.add_argument("--noise-augmentations", type=int, default=24)
    parser.add_argument("--noise-source", choices=("observation", "feature"), default="observation", help="Use noisy raw observations to build student features/history, or legacy feature-space noise.")
    parser.add_argument("--rollout-noise-augmentations", type=int, default=1, help="Independent noisy observation sequences per clean rollout when --noise-source=observation.")
    parser.add_argument("--normalized-noise-stds", type=str, default="0.0,0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--normalized-noise-std-probs", type=str, default="0.10,0.15,0.20,0.25,0.15,0.15")
    parser.add_argument("--normalized-noise-std", type=float, default=None, help="Deprecated single-std alias.")
    parser.add_argument("--noise-feature-groups", type=str, default="ip,rz,lcfs,history_ip,history_rz")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-kl-weight", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--feature-reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--clean-consistency-weight", type=float, default=0.25)
    parser.add_argument("--denoiser-hidden-dim", type=int, default=256)
    parser.add_argument("--denoiser-depth", type=int, default=2)
    parser.add_argument("--denoiser-residual-scale", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dataset", action="store_true", help="Deprecated; dataset caching is enabled by default.")
    parser.add_argument(
        "--strict-rollout",
        action="store_true",
        help="Fail immediately on HFM rollout errors instead of skipping the failed episode.",
    )
    return parser.parse_args(argv)


def parse_int_spec(spec: str) -> list[int]:
    values: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError(f"empty integer spec: {spec!r}")
    return values


def load_vecnormalize_stats(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        mean = np.asarray(data["obs_mean"], dtype=np.float32)
        var = np.asarray(data["obs_var"], dtype=np.float32)
        clip_obs = float(np.asarray(data.get("clip_obs", 10.0)).reshape(-1)[0])
        epsilon = float(np.asarray(data.get("epsilon", 1e-8)).reshape(-1)[0])
        return mean, var, clip_obs, epsilon

    with open(path, "rb") as f:
        vecnormalize = pickle.load(f)
    obs_rms = getattr(vecnormalize, "obs_rms", None)
    if obs_rms is None:
        raise ValueError(f"VecNormalize object has no obs_rms: {path}")
    mean = np.asarray(obs_rms.mean, dtype=np.float32)
    var = np.asarray(obs_rms.var, dtype=np.float32)
    clip_obs = float(getattr(vecnormalize, "clip_obs", 10.0))
    epsilon = float(getattr(vecnormalize, "epsilon", 1e-8))
    return mean, var, clip_obs, epsilon


def normalize_obs(obs: np.ndarray, mean: np.ndarray, var: np.ndarray, clip_obs: float, epsilon: float) -> np.ndarray:
    norm = (np.asarray(obs, dtype=np.float32) - mean) / np.sqrt(var + epsilon)
    return np.clip(norm, -clip_obs, clip_obs).astype(np.float32)


def build_raw_env(
    *,
    train_config: dict[str, Any],
    base_config: dict[str, Any],
    reference: dict[str, Any] | None,
    target,
    shot_id: str,
    port: int,
    max_steps: int,
    seed: int,
    action_units: str,
    reset_perturbation: dict[str, Any] | None = None,
) -> gym.Env:
    cfg = make_env_config(
        base_config,
        port=port,
        shot_id=shot_id,
        max_steps=max_steps,
        reference_mode=str(train_config.get("reference_mode", "trajectory")),
    )
    reward = SemifinalDenseReward(
        task=str(train_config.get("task", "F1")),
        target=target,
        reward_weights=train_config.get("reward_weights"),
    )
    env: gym.Env = HFMSimulator(cfg, reward_fn=reward)
    reset_perturbation = reset_perturbation if reset_perturbation is not None else {"enabled": False}
    if str(train_config.get("reference_mode", "trajectory")) == "hold_initial":
        env = HoldInitialResetWrapper(env, shot_id=shot_id, reset_perturbation=reset_perturbation, seed=seed)
    else:
        if reference is None:
            raise ValueError("trajectory reference is required")
        env = ReferenceResetWrapper(
            env,
            reference=reference,
            shot_id=shot_id,
            reset_perturbation=reset_perturbation,
            seed=seed,
        )
    env = NormalizedAction7DTo12DWrapper(env) if action_units == "normalized" else Action7DTo12DWrapper(env)
    return RewardInfoWrapper(env)


class SpaceOnlyEnv(gym.Env):
    """Tiny env used only to instantiate an SB3 PPO policy with known spaces."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return obs, 0.0, False, False, {}


def build_denoising_student(args: argparse.Namespace, teacher):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: SpaceOnlyEnv(teacher.observation_space, teacher.action_space)])
    policy_kwargs = dict(getattr(teacher, "policy_kwargs", {}) or {})
    policy_kwargs["features_extractor_class"] = ResidualDenoisingFeatureExtractor
    policy_kwargs["features_extractor_kwargs"] = {
        "hidden_dim": int(args.denoiser_hidden_dim),
        "depth": int(args.denoiser_depth),
        "residual_scale": float(args.denoiser_residual_scale),
    }
    student = PPO(
        teacher.policy_class,
        env,
        learning_rate=float(args.learning_rate),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=float(getattr(teacher, "gamma", 0.995)),
        gae_lambda=float(getattr(teacher, "gae_lambda", 0.95)),
        ent_coef=float(getattr(teacher, "ent_coef", 0.0)),
        vf_coef=float(getattr(teacher, "vf_coef", 0.5)),
        max_grad_norm=float(args.max_grad_norm),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=int(args.seed),
        device=str(args.device),
    )
    incompatible = student.policy.load_state_dict(teacher.policy.state_dict(), strict=False)
    if hasattr(student.policy.features_extractor, "reset_identity"):
        student.policy.features_extractor.reset_identity()
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("features_extractor.")]
    if unexpected:
        raise RuntimeError(f"unexpected teacher keys while copying policy: {unexpected}")
    for param in student.policy.parameters():
        param.requires_grad_(False)
    for param in student.policy.features_extractor.parameters():
        param.requires_grad_(True)
    return student


def policy_outputs(policy, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    values = policy.value_net(latent_vf)
    dist = policy._get_action_dist_from_latent(latent_pi)
    normal = getattr(dist, "distribution", None)
    if normal is None:
        raise TypeError("expected a diagonal Gaussian action distribution")
    return normal.mean, normal.stddev, values


def _variant_target_from_reference(reference: dict[str, Any], base_target) -> Any:
    lcfs_seq = np.asarray(reference["lcfs_points"], dtype=np.float64)
    lcfs = lcfs_seq[0] if lcfs_seq.ndim == 3 else lcfs_seq
    return type(base_target)(
        lcfs=np.asarray(lcfs, dtype=np.float64),
        ip_final=float(base_target.ip_final),
        rX=base_target.rX,
        zX=base_target.zX,
        strike_r=base_target.strike_r,
        strike_z=base_target.strike_z,
        x_valid=base_target.x_valid,
        strike_valid=base_target.strike_valid,
    )


def _select_reference_cases(
    *,
    args: argparse.Namespace,
    train_config: dict[str, Any],
    task: str,
    base_target,
    max_steps: int,
) -> list[dict[str, Any]]:
    reference_mode = str(train_config.get("reference_mode", "trajectory"))
    if reference_mode == "hold_initial":
        return [{"variant_index": None, "variant_id": "hold_initial", "target": base_target, "reference": None}]

    variants_path = _rooted_path(args.reference_variants or train_config.get("reference_variants"))
    config_enabled = bool(train_config.get("reference_variants_enabled", variants_path is not None))
    variants_enabled = config_enabled if args.reference_variants_enabled is None else bool(args.reference_variants_enabled)
    cases: list[dict[str, Any]] = []
    if bool(getattr(args, "include_base_reference", False)):
        reference = build_reference(task, base_target, max_steps)
        cases.append({"variant_index": None, "variant_id": "base_reference", "target": base_target, "reference": reference})

    if variants_enabled and variants_path is not None:
        references = load_reference_variants(
            variants_path,
            task=task,
            base_target=base_target,
            max_steps=max_steps,
        )
        selected = parse_int_spec(args.reference_variant_indices) if str(args.reference_variant_indices).strip() else list(range(len(references)))
        for idx in selected:
            if idx < 0 or idx >= len(references):
                raise IndexError(f"reference variant index {idx} out of range 0..{len(references) - 1}")
            reference = references[idx]
            cases.append(
                {
                    "variant_index": int(idx),
                    "variant_id": str(reference.get("_variant_id", f"variant_{idx}")),
                    "target": _variant_target_from_reference(reference, base_target),
                    "reference": reference,
                }
            )
    else:
        reference = build_reference(task, base_target, max_steps)
        cases.append({"variant_index": None, "variant_id": "base_reference", "target": base_target, "reference": reference})
    if not cases:
        raise ValueError("no reference cases selected for distillation dataset collection")
    return cases


def collect_dataset(args: argparse.Namespace, teacher, metadata: FeatureMetadata) -> DistillDataset:
    train_config = load_train_config(_rooted_path(args.train_config))
    task = normalize_task(train_config.get("task", "F1"))
    reference_path = _rooted_path(args.reference or train_config.get("reference"))
    if reference_path is None:
        raise ValueError("reference path could not be resolved")
    base_target = load_target_reference(reference_path, task=task)
    max_steps = int(args.max_steps or train_config.get("max_steps") or 300)
    reference_cases = _select_reference_cases(
        args=args,
        train_config=train_config,
        task=task,
        base_target=base_target,
        max_steps=max_steps,
    )

    env_config_path = _rooted_path(args.env_config or train_config.get("env_config"), ROOT / "configs/env_default.yaml")
    base_config = load_yaml(env_config_path)
    ports = parse_ports(args.ports or train_config.get("ports") or "5558")
    requested_workers = max(1, int(getattr(args, "num_workers", 1)))
    if requested_workers > len(ports):
        raise ValueError(f"--num-workers={requested_workers} requires at least that many --ports, got {ports}")
    shots = [part.strip() for part in str(args.shots).split(",") if part.strip()]
    seeds = parse_int_spec(args.seeds)
    obs_mean, obs_var, clip_obs, eps = load_vecnormalize_stats(_rooted_path(args.vecnormalize))
    noise_source = str(getattr(args, "noise_source", "observation"))
    rollout_noise_augmentations = max(1, int(getattr(args, "rollout_noise_augmentations", 1)))
    if args.normalized_noise_std is not None:
        collect_noise_stds = np.asarray([max(0.0, float(args.normalized_noise_std))], dtype=np.float32)
        collect_noise_probs = np.asarray([1.0], dtype=np.float64)
    else:
        collect_noise_stds, collect_noise_probs = parse_noise_schedule(
            args.normalized_noise_stds,
            args.normalized_noise_std_probs,
        )
    raw_noise_scales = _raw_observation_noise_scales(metadata, obs_var, eps, args.noise_feature_groups)

    episode_specs: list[dict[str, Any]] = []
    episode_index = 0
    for ref_case in reference_cases:
        for shot_id in shots:
            for seed in seeds:
                episode_specs.append(
                    {
                        "episode": episode_index,
                        "reference_variant_index": ref_case["variant_index"],
                        "reference_variant_id": str(ref_case["variant_id"]),
                        "target": ref_case["target"],
                        "reference": ref_case["reference"],
                        "shot": shot_id,
                        "seed": int(seed),
                    }
                )
                episode_index += 1

    worker_count = min(requested_workers, max(1, len(episode_specs)))
    worker_ports = ports[:worker_count]
    print(
        f"collecting dataset: reference_cases={len(reference_cases)} shots={len(shots)} seeds={len(seeds)} "
        f"episodes={len(episode_specs)} max_steps={max_steps} workers={worker_count} ports={worker_ports} "
        f"noise_source={noise_source} rollout_noise_augmentations={rollout_noise_augmentations}",
        flush=True,
    )

    predict_lock = threading.Lock()

    def run_episode(spec: dict[str, Any], port: int) -> dict[str, Any]:
        episode = int(spec["episode"])
        variant_id = str(spec["reference_variant_id"])
        variant_index = spec["reference_variant_index"]
        shot_id = str(spec["shot"])
        seed = int(spec["seed"])
        env: gym.Env | None = None
        episode_clean_rows: list[np.ndarray] = []
        episode_noisy_rows: list[np.ndarray] = []
        episode_ids_local: list[int] = []
        rng = np.random.default_rng(int(args.seed) * 1_000_003 + episode)
        try:
            env = build_raw_env(
                train_config=train_config,
                base_config=base_config,
                reference=spec["reference"],
                target=spec["target"],
                shot_id=shot_id,
                port=port,
                max_steps=max_steps,
                seed=seed,
                action_units=metadata.action_units,
            )
            clean_builder = FeatureSequenceBuilder(metadata)
            noisy_builders = [FeatureSequenceBuilder(metadata) for _ in range(rollout_noise_augmentations)]
            raw_obs, _ = env.reset(seed=seed)
            raw_obs_dict = dict(raw_obs)
            clean_feature = clean_builder.reset(raw_obs_dict)
            noisy_features: list[np.ndarray] = []
            if noise_source == "observation":
                for noisy_builder in noisy_builders:
                    sampled_std = float(rng.choice(collect_noise_stds, p=collect_noise_probs))
                    noisy_obs = _add_raw_observation_noise(
                        raw_obs_dict,
                        scales=raw_noise_scales,
                        std=sampled_std,
                        rng=rng,
                    )
                    noisy_features.append(noisy_builder.reset(noisy_obs))
            for _step in range(max_steps + 1):
                clean_norm = normalize_obs(clean_feature, obs_mean, obs_var, clip_obs, eps)
                if noise_source == "observation":
                    for noisy_feature in noisy_features:
                        episode_clean_rows.append(clean_norm)
                        episode_noisy_rows.append(normalize_obs(noisy_feature, obs_mean, obs_var, clip_obs, eps))
                        episode_ids_local.append(episode)
                else:
                    episode_clean_rows.append(clean_norm)
                    episode_ids_local.append(episode)

                with predict_lock:
                    action, _ = teacher.predict(clean_norm.reshape(1, -1), deterministic=True)
                action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
                raw_next, _reward, terminated, truncated, _info = env.step(action7)
                raw_next_dict = dict(raw_next)
                clean_feature = clean_builder.step(action7, raw_next_dict)
                if noise_source == "observation":
                    next_noisy_features: list[np.ndarray] = []
                    for noisy_builder in noisy_builders:
                        sampled_std = float(rng.choice(collect_noise_stds, p=collect_noise_probs))
                        noisy_next = _add_raw_observation_noise(
                            raw_next_dict,
                            scales=raw_noise_scales,
                            std=sampled_std,
                            rng=rng,
                        )
                        next_noisy_features.append(noisy_builder.step(action7, noisy_next))
                    noisy_features = next_noisy_features
                if terminated or truncated:
                    break
            return {
                "ok": True,
                "episode": episode,
                "rows": episode_clean_rows,
                "noisy_rows": episode_noisy_rows if noise_source == "observation" else None,
                "episode_ids": episode_ids_local,
                "failed": None,
                "message": (
                    f"collected episode={episode} variant={variant_id} "
                    f"shot={shot_id} seed={seed} port={port} samples={len(episode_ids_local)}"
                ),
            }
        except Exception as exc:
            if args.strict_rollout:
                raise
            failed = {
                "episode": episode,
                "reference_variant_index": variant_index,
                "reference_variant_id": variant_id,
                "shot": shot_id,
                "seed": seed,
                "port": port,
                "error": repr(exc),
            }
            return {
                "ok": False,
                "episode": episode,
                "rows": [],
                "noisy_rows": None,
                "episode_ids": [],
                "failed": failed,
                "message": (
                    f"warning: skipped failed episode={episode} variant={variant_id} "
                    f"shot={shot_id} seed={seed} port={port}: {type(exc).__name__}: {exc}"
                ),
            }
        finally:
            if env is not None:
                env.close()

    def run_worker(worker_id: int, port: int, shard: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for spec in shard:
            results.append(run_episode(spec, port))
        return results

    shards: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for idx, spec in enumerate(episode_specs):
        shards[idx % worker_count].append(spec)

    results: list[dict[str, Any]] = []
    if worker_count == 1:
        results = run_worker(0, worker_ports[0], shards[0])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_worker = {
                executor.submit(run_worker, worker_id, worker_ports[worker_id], shards[worker_id]): worker_id
                for worker_id in range(worker_count)
            }
            for future in concurrent.futures.as_completed(future_to_worker):
                results.extend(future.result())

    clean_rows: list[np.ndarray] = []
    noisy_rows: list[np.ndarray] = []
    episode_ids: list[int] = []
    failed_episodes: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["episode"])):
        print(str(result["message"]), flush=True)
        if result["ok"]:
            clean_rows.extend(result["rows"])
            if result.get("noisy_rows") is not None:
                noisy_rows.extend(result["noisy_rows"])
            episode_ids.extend(result["episode_ids"])
        elif result["failed"] is not None:
            failed_episodes.append(result["failed"])

    clean = np.asarray(clean_rows, dtype=np.float32)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    noisy = np.asarray(noisy_rows, dtype=np.float32) if noise_source == "observation" else None
    if failed_episodes:
        print(f"skipped_failed_episodes={len(failed_episodes)}", flush=True)
    if clean.ndim != 2 or clean.shape[1] != metadata.obs_dim:
        raise ValueError(f"dataset shape mismatch: clean={clean.shape}, obs_dim={metadata.obs_dim}")
    if episodes.shape[0] != clean.shape[0]:
        raise ValueError("dataset episode_ids length mismatch")
    if noisy is not None and noisy.shape != clean.shape:
        raise ValueError(f"dataset noisy shape mismatch: noisy={noisy.shape}, clean={clean.shape}")
    if not np.isfinite(clean).all():
        raise ValueError("dataset contains non-finite clean features")
    if noisy is not None and not np.isfinite(noisy).all():
        raise ValueError("dataset contains non-finite noisy features")
    return DistillDataset(clean_obs=clean, episode_ids=episodes, noisy_obs=noisy)

def split_by_episode(dataset: DistillDataset, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    episodes = np.unique(dataset.episode_ids)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * float(val_fraction)))) if len(episodes) > 1 else 0
    val_eps = set(int(x) for x in episodes[:n_val])
    val_mask = np.asarray([int(ep) in val_eps for ep in dataset.episode_ids], dtype=bool)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train/val split is empty; collect more than one episode")
    return train_idx, val_idx


def train_student(args: argparse.Namespace, dataset: DistillDataset, teacher, student) -> dict[str, float]:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    teacher.policy.to(device)
    student.policy.to(device)
    teacher.policy.eval()
    student.policy.train()
    for param in teacher.policy.parameters():
        param.requires_grad_(False)

    denoiser_params = [param for param in student.policy.features_extractor.parameters() if param.requires_grad]
    if not denoiser_params:
        raise RuntimeError("student policy has no trainable denoiser parameters")
    optimizer = torch.optim.AdamW(
        denoiser_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_idx, val_idx = split_by_episode(dataset, args.val_fraction, args.seed)
    rng = np.random.default_rng(args.seed)
    if args.normalized_noise_std is not None:
        noise_stds = np.asarray([max(0.0, float(args.normalized_noise_std))], dtype=np.float32)
        noise_probs = np.asarray([1.0], dtype=np.float64)
    else:
        noise_stds, noise_probs = parse_noise_schedule(
            args.normalized_noise_stds,
            args.normalized_noise_std_probs,
        )
    feature_noise_mask = noise_feature_mask(
        FeatureMetadata.load(_rooted_path(args.metadata)),
        args.noise_feature_groups,
    ).astype(np.float32)
    print(
        f"noise feature groups={args.noise_feature_groups} "
        f"dims={int(feature_noise_mask.sum())}/{feature_noise_mask.size}",
        flush=True,
    )

    def make_noisy(clean_np: np.ndarray) -> np.ndarray:
        repeats = max(1, int(args.noise_augmentations))
        clean_rep = np.repeat(np.asarray(clean_np, dtype=np.float32), repeats, axis=0)
        sampled_stds = rng.choice(noise_stds, size=clean_rep.shape[0], p=noise_probs).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=clean_rep.shape).astype(np.float32)
        return clean_rep + noise * sampled_stds[:, None] * feature_noise_mask[None, :]

    def batch_pair(batch_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clean_np = dataset.clean_obs[batch_idx].astype(np.float32, copy=False)
        if dataset.noisy_obs is not None:
            return clean_np, dataset.noisy_obs[batch_idx].astype(np.float32, copy=False)
        noisy_np = make_noisy(clean_np)
        clean_rep = np.repeat(clean_np, max(1, int(args.noise_augmentations)), axis=0)
        return clean_rep.astype(np.float32, copy=False), noisy_np.astype(np.float32, copy=False)
    output_dir = _rooted_path(args.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "distill_metrics.csv"
    best_val = float("inf")
    best_state = copy.deepcopy(student.policy.state_dict())
    patience = 0

    def eval_indices(indices: np.ndarray) -> dict[str, float]:
        student.policy.eval()
        sums = {
            "actor_mse": 0.0,
            "actor_kl": 0.0,
            "value_mse": 0.0,
            "clean_actor_mse": 0.0,
            "clean_value_mse": 0.0,
            "feature_mse": 0.0,
            "clean_feature_mse": 0.0,
        }
        count = 0
        with torch.no_grad():
            for start in range(0, len(indices), int(args.batch_size)):
                batch_idx = indices[start : start + int(args.batch_size)]
                clean_np, noisy_np = batch_pair(batch_idx)
                clean = torch.as_tensor(clean_np, device=device)
                noisy = torch.as_tensor(noisy_np, device=device)
                t_mean, t_std, t_value = policy_outputs(teacher.policy, clean)
                s_mean, s_std, s_value = policy_outputs(student.policy, noisy)
                s_clean_mean, _s_clean_std, s_clean_value = policy_outputs(student.policy, clean)
                denoised_noisy = student.policy.extract_features(noisy)
                denoised_clean = student.policy.extract_features(clean)
                actor_mse = torch.mean((s_mean - t_mean) ** 2)
                clean_actor_mse = torch.mean((s_clean_mean - t_mean) ** 2)
                value_mse = torch.mean((s_value - t_value) ** 2)
                clean_value_mse = torch.mean((s_clean_value - t_value) ** 2)
                feature_mse = torch.mean((denoised_noisy - clean) ** 2)
                clean_feature_mse = torch.mean((denoised_clean - clean) ** 2)
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(s_mean, s_std),
                    torch.distributions.Normal(t_mean, t_std),
                ).sum(dim=1).mean()
                n = int(clean_np.shape[0])
                sums["actor_mse"] += float(actor_mse.item()) * n
                sums["actor_kl"] += float(kl.item()) * n
                sums["value_mse"] += float(value_mse.item()) * n
                sums["clean_actor_mse"] += float(clean_actor_mse.item()) * n
                sums["clean_value_mse"] += float(clean_value_mse.item()) * n
                sums["feature_mse"] += float(feature_mse.item()) * n
                sums["clean_feature_mse"] += float(clean_feature_mse.item()) * n
                count += n
        student.policy.train()
        return {k: v / max(1, count) for k, v in sums.items()}

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_actor_mse",
                "train_actor_kl",
                "train_value_mse",
                "train_clean_actor_mse",
                "train_clean_value_mse",
                "train_feature_mse",
                "train_clean_feature_mse",
                "val_actor_mse",
                "val_actor_kl",
                "val_value_mse",
                "val_clean_actor_mse",
                "val_clean_value_mse",
                "val_feature_mse",
                "val_clean_feature_mse",
                "val_total",
            ],
        )
        writer.writeheader()
        for epoch in range(1, int(args.epochs) + 1):
            rng.shuffle(train_idx)
            for start in range(0, len(train_idx), int(args.batch_size)):
                batch_idx = train_idx[start : start + int(args.batch_size)]
                clean_np, noisy_np = batch_pair(batch_idx)
                clean = torch.as_tensor(clean_np, device=device)
                noisy = torch.as_tensor(noisy_np, device=device)
                with torch.no_grad():
                    t_mean, t_std, t_value = policy_outputs(teacher.policy, clean)
                s_mean, s_std, s_value = policy_outputs(student.policy, noisy)
                s_clean_mean, _s_clean_std, s_clean_value = policy_outputs(student.policy, clean)
                denoised_noisy = student.policy.extract_features(noisy)
                denoised_clean = student.policy.extract_features(clean)
                actor_mse = torch.mean((s_mean - t_mean) ** 2)
                value_mse = torch.mean((s_value - t_value) ** 2)
                clean_actor_mse = torch.mean((s_clean_mean - t_mean) ** 2)
                clean_value_mse = torch.mean((s_clean_value - t_value) ** 2)
                feature_mse = torch.mean((denoised_noisy - clean) ** 2)
                clean_feature_mse = torch.mean((denoised_clean - clean) ** 2)
                actor_kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(s_mean, s_std),
                    torch.distributions.Normal(t_mean, t_std),
                ).sum(dim=1).mean()
                behavior_loss = actor_mse + float(args.actor_kl_weight) * actor_kl + float(args.value_weight) * value_mse
                clean_loss = clean_actor_mse + float(args.value_weight) * clean_value_mse
                feature_loss = feature_mse + clean_feature_mse
                loss = (
                    behavior_loss
                    + float(args.clean_consistency_weight) * clean_loss
                    + float(args.feature_reconstruction_weight) * feature_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser_params, float(args.max_grad_norm))
                optimizer.step()

            train_metrics = eval_indices(train_idx)
            val_metrics = eval_indices(val_idx)
            val_total = (
                val_metrics["actor_mse"]
                + float(args.actor_kl_weight) * val_metrics["actor_kl"]
                + float(args.value_weight) * val_metrics["value_mse"]
                + float(args.clean_consistency_weight)
                * (val_metrics["clean_actor_mse"] + float(args.value_weight) * val_metrics["clean_value_mse"])
                + float(args.feature_reconstruction_weight)
                * (val_metrics["feature_mse"] + val_metrics["clean_feature_mse"])
            )
            row = {
                "epoch": epoch,
                "train_actor_mse": train_metrics["actor_mse"],
                "train_actor_kl": train_metrics["actor_kl"],
                "train_value_mse": train_metrics["value_mse"],
                "train_clean_actor_mse": train_metrics["clean_actor_mse"],
                "train_clean_value_mse": train_metrics["clean_value_mse"],
                "train_feature_mse": train_metrics["feature_mse"],
                "train_clean_feature_mse": train_metrics["clean_feature_mse"],
                "val_actor_mse": val_metrics["actor_mse"],
                "val_actor_kl": val_metrics["actor_kl"],
                "val_value_mse": val_metrics["value_mse"],
                "val_clean_actor_mse": val_metrics["clean_actor_mse"],
                "val_clean_value_mse": val_metrics["clean_value_mse"],
                "val_feature_mse": val_metrics["feature_mse"],
                "val_clean_feature_mse": val_metrics["clean_feature_mse"],
                "val_total": val_total,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch} val_total={val_total:.6g} "
                f"val_actor_mse={val_metrics['actor_mse']:.6g} "
                f"val_value_mse={val_metrics['value_mse']:.6g}",
                flush=True,
            )
            if val_total < best_val:
                best_val = val_total
                best_state = copy.deepcopy(student.policy.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(args.early_stop_patience):
                    print(f"early stop at epoch={epoch}", flush=True)
                    break

    student.policy.load_state_dict(best_state)
    return {"best_val_total": best_val, "train_samples": float(len(train_idx)), "val_samples": float(len(val_idx))}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _rooted_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    metadata = FeatureMetadata.load(_rooted_path(args.metadata))
    teacher = PPO.load(str(_rooted_path(args.teacher_model)), device=args.device)
    student = build_denoising_student(args, teacher)
    teacher.policy.eval()
    student.policy.train()

    dataset_path = _rooted_path(args.dataset_path, output_dir / "distill_clean_dataset.npz")
    if dataset_path.exists() and not args.force_collect:
        dataset = load_dataset(dataset_path, metadata)
        print(f"loaded dataset: {dataset_path} samples={dataset.clean_obs.shape[0]}", flush=True)
    else:
        dataset = collect_dataset(args, teacher=teacher, metadata=metadata)
        save_dataset(
            dataset_path,
            dataset,
            {
                "teacher_model": str(_rooted_path(args.teacher_model)),
                "vecnormalize": str(_rooted_path(args.vecnormalize)),
                "metadata": str(_rooted_path(args.metadata)),
                "train_config": str(_rooted_path(args.train_config)),
                "shots": str(args.shots),
                "seeds": str(args.seeds),
                "ports": str(args.ports),
                "num_workers": int(getattr(args, "num_workers", 1)),
                "noise_source": str(getattr(args, "noise_source", "observation")),
                "rollout_noise_augmentations": int(getattr(args, "rollout_noise_augmentations", 1)),
                "max_steps": args.max_steps,
                "reference_variants": str(_rooted_path(args.reference_variants)) if args.reference_variants else str(load_train_config(_rooted_path(args.train_config)).get("reference_variants", "")),
                "reference_variants_enabled": args.reference_variants_enabled,
                "reference_variant_indices": str(args.reference_variant_indices),
                "include_base_reference": bool(args.include_base_reference),
                "samples": int(dataset.clean_obs.shape[0]),
                "has_noisy_obs": dataset.noisy_obs is not None,
                "obs_dim": int(dataset.clean_obs.shape[1]),
                "kind": "clean_and_noisy_normalized_features" if dataset.noisy_obs is not None else "clean_normalized_features",
            },
        )
        print(f"saved dataset: {dataset_path} samples={dataset.clean_obs.shape[0]}", flush=True)
    if args.collect_only:
        return 0

    metrics = train_student(args, dataset=dataset, teacher=teacher, student=student)

    task_lower = str(metadata.task).lower()
    model_path = output_dir / f"ppo_{task_lower}_student_latest"
    student.save(str(model_path))
    vec_src = _rooted_path(args.vecnormalize)
    vec_name = "vecnormalize_stats.npz" if vec_src.suffix.lower() == ".npz" else "vecnormalize.pkl"
    vec_out = output_dir / vec_name
    shutil.copyfile(vec_src, vec_out)
    metadata_path = output_dir / f"{task_lower}_features.metadata.json"
    metadata.save(metadata_path)
    with open(output_dir / "distill_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher_model": str(_rooted_path(args.teacher_model)),
                "vecnormalize": str(_rooted_path(args.vecnormalize)),
                "metadata": str(_rooted_path(args.metadata)),
                "train_config": str(_rooted_path(args.train_config)),
                "reference_variants": str(_rooted_path(args.reference_variants)) if args.reference_variants else str(load_train_config(_rooted_path(args.train_config)).get("reference_variants", "")),
                "reference_variants_enabled": args.reference_variants_enabled,
                "reference_variant_indices": str(args.reference_variant_indices),
                "include_base_reference": bool(args.include_base_reference),
                "dataset_path": str(dataset_path),
                "samples": int(dataset.clean_obs.shape[0]),
                "has_noisy_obs": dataset.noisy_obs is not None,
                "obs_dim": int(dataset.clean_obs.shape[1]),
                "noise_augmentations": int(args.noise_augmentations),
                "noise_source": str(getattr(args, "noise_source", "observation")),
                "rollout_noise_augmentations": int(getattr(args, "rollout_noise_augmentations", 1)),
                "noise_feature_groups": str(args.noise_feature_groups),
                "normalized_noise_stds": (
                    [float(args.normalized_noise_std)]
                    if args.normalized_noise_std is not None
                    else [float(x) for x in parse_float_list(args.normalized_noise_stds)]
                ),
                "normalized_noise_std_probs": (
                    [1.0]
                    if args.normalized_noise_std is not None
                    else [float(x) for x in parse_float_list(args.normalized_noise_std_probs)]
                ),
                "denoiser": {
                    "class": "training.denoising.ResidualDenoisingFeatureExtractor",
                    "hidden_dim": int(args.denoiser_hidden_dim),
                    "depth": int(args.denoiser_depth),
                    "residual_scale": float(args.denoiser_residual_scale),
                },
                "trainable": "features_extractor only",
                **metrics,
            },
            f,
            indent=2,
        )
    print(f"saved: {model_path}.zip")
    print(f"vecnormalize: {vec_out}")
    print(f"metadata: {metadata_path}")
    print(f"metrics: {output_dir / 'logs/distill_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
