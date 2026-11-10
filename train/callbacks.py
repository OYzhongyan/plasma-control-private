"""Training callbacks for semifinal PPO runs."""

from __future__ import annotations

import csv
import sys
from collections import deque
import json
from pathlib import Path
from typing import Any

import numpy as np


try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:  # pragma: no cover - training dependency is optional for offline imports.
    BaseCallback = object  # type: ignore[assignment]
    VecNormalize = None  # type: ignore[assignment]


REWARD_COMPONENT_KEYS = [
    "reward",
    "failure_penalty",
    "coil_current_exceeded",
    "ip_abs_error",
    "ip_relative_error",
    "ip_score",
    "lcfs_error_cm",
    "lcfs_score",
    "topology_score",
    "coil_margin",
    "ip_error_mask",
    "soft_ip_error_mask",
    "competition_score",
    "main_x_dist_score",
    "main_x_psi_score",
    "main_x_valid_ratio",
    "second_x_psi_score",
    "second_x_dist_score",
    "second_x_valid_ratio",
    "strike_dist_score",
    "strike_valid_ratio",
]

REWARD_COMPONENT_KEYS += [
    "competition_step_score",
    "competition_total_score",
    "competition_terminal_correction",
    "gamma",
    "eta",
    "rho",
    "Ip",
    "LCFS",
    "X",
    "strike",
    "psiX",
    "X2",
    "psiX2",
]

INFO_COMPONENT_KEYS = [
    "hfm_timeout_rate",
    "hfm_timeout_count",
]


def find_vecnormalize(env: Any):
    """Return the VecNormalize wrapper from a vector-env wrapper chain, if present."""

    if VecNormalize is None:
        return None
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, VecNormalize):
            return current
        if hasattr(current, "venv"):
            current = current.venv
        elif hasattr(current, "env"):
            current = current.env
        else:
            current = None
    return None


def save_vecnormalize_stats(vecnormalize: Any, path: str | Path) -> None:
    """Save lightweight VecNormalize statistics for inspection/export tooling."""

    obs_rms = vecnormalize.obs_rms
    if isinstance(obs_rms, dict):
        obs_rms = obs_rms["actor_obs"]
    values: dict[str, Any] = {
        "obs_mean": np.asarray(obs_rms.mean, dtype=np.float64),
        "obs_var": np.asarray(obs_rms.var, dtype=np.float64),
        "clip_obs": np.asarray(vecnormalize.clip_obs, dtype=np.float64),
        "epsilon": np.asarray(vecnormalize.epsilon, dtype=np.float64),
    }
    flux_rms = vars(vecnormalize).get("flux_rms")
    if flux_rms is not None:
        values.update(flux_mean=np.asarray(flux_rms.mean, dtype=np.float64),
                      flux_var=np.asarray(flux_rms.var, dtype=np.float64),
                      flux_count=np.asarray(flux_rms.count, dtype=np.float64))
    if getattr(vecnormalize, "ret_rms", None) is not None:
        values["ret_mean"] = np.asarray(vecnormalize.ret_rms.mean, dtype=np.float64)
        values["ret_var"] = np.asarray(vecnormalize.ret_rms.var, dtype=np.float64)
        values["clip_reward"] = np.asarray(vecnormalize.clip_reward, dtype=np.float64)
    np.savez(path, **values)


class RewardComponentsCallback(BaseCallback):
    """Aggregate raw reward components from env infos and log them periodically."""

    def __init__(self, log_dir: Path, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.log_interval = int(log_interval)
        self.csv_path = self.log_dir / "reward_components.csv"
        self._buffer: list[dict[str, float]] = []
        self._info_count = 0
        self._hfm_timeout_count = 0
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        header = ["timesteps", *REWARD_COMPONENT_KEYS, *INFO_COMPONENT_KEYS]
        wandb = sys.modules.get("wandb")
        if wandb is not None and wandb.run is not None:
            wandb.define_metric("reward_components/*", step_metric="timesteps")
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)
            return

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            self._info_count += 1
            if bool(info.get("hfm_timeout", False)):
                self._hfm_timeout_count += 1

            components = info.get("reward_components")
            if not isinstance(components, dict):
                continue
            row: dict[str, float] = {}
            for key in REWARD_COMPONENT_KEYS:
                value = components.get(key)
                if value is None:
                    continue
                value = float(value)
                if np.isfinite(value):
                    row[key] = value
            if row:
                self._buffer.append(row)

        if self.log_interval > 0 and self.num_timesteps - self._last_log_step >= self.log_interval:
            self._flush()
            self._last_log_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buffer and self._info_count == 0:
            return

        means: dict[str, float] = {}
        for key in REWARD_COMPONENT_KEYS:
            values = [row[key] for row in self._buffer if key in row]
            if values:
                means[key] = float(np.mean(values))
        info_values = {
            "hfm_timeout_rate": (
                float(self._hfm_timeout_count) / float(self._info_count)
                if self._info_count > 0
                else 0.0
            ),
            "hfm_timeout_count": float(self._hfm_timeout_count),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                self.num_timesteps,
                *[means.get(key, "") for key in REWARD_COMPONENT_KEYS],
                *[info_values[key] for key in INFO_COMPONENT_KEYS],
            ])

        for key, value in means.items():
            self.logger.record(f"reward_components/{key}", value)
        for key, value in info_values.items():
            self.logger.record(f"reward_components/{key}", value)
        self._buffer.clear()
        self._info_count = 0
        self._hfm_timeout_count = 0

        wandb = sys.modules.get("wandb")
        if wandb is not None and wandb.run is not None:
            payload: dict[str, Any] = {"timesteps": self.num_timesteps}
            for key, value in means.items():
                payload[f"reward_components/{key}"] = value
            for key, value in info_values.items():
                payload[f"reward_components/{key}"] = value
            wandb.log(payload)


class PlasmaCheckpointCallback(BaseCallback):
    """Save PPO checkpoints together with matching VecNormalize statistics."""

    def __init__(
        self,
        *,
        save_freq: int,
        save_path: Path,
        name_prefix: str,
        save_vecnormalize: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = Path(save_path)
        self.name_prefix = str(name_prefix)
        self.save_vecnormalize = bool(save_vecnormalize)
        self.completed_rollouts = 0

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.completed_rollouts += 1
        if self.completed_rollouts % self.save_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        step = int(self.num_timesteps)
        base = self.save_path / f"{self.name_prefix}_{step}_steps"
        self.model.save(str(base))

        if self.save_vecnormalize:
            vecnormalize = find_vecnormalize(self.training_env)
            if vecnormalize is not None:
                vecnormalize_path = base.with_name(f"{base.name}_vecnormalize.pkl")
                vecnormalize.save(str(vecnormalize_path))
                save_vecnormalize_stats(
                    vecnormalize,
                    base.with_name(f"{base.name}_vecnormalize_stats.npz"),
                )

        if self.verbose > 0:
            print(f"saved checkpoint: {base}.zip")
