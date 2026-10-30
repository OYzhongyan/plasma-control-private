"""Training callbacks for semifinal PPO runs."""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional.
    wandb = None  # type: ignore[assignment]

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:  # pragma: no cover - training dependency is optional for offline imports.
    BaseCallback = object  # type: ignore[assignment]
    VecNormalize = None  # type: ignore[assignment]


REWARD_COMPONENT_KEYS = [
    "reward",
    "failure",
    "failure_penalty",
    "ip_abs_error",
    "ip_relative_error",
    "ip_safe_score",
    "ip_score",
    "ip_error_boundary_ratio",
    "ip_error_boundary_penalty_raw",
    "ip_error_boundary_penalty",
    "r_score",
    "z_score",
    "lcfs_cm",
    "lcfs_initial_cm",
    "lcfs_progress_score",
    "lcfs_precision_score",
    "lcfs_regression_cm",
    "lcfs_regression_ratio",
    "lcfs_regression_penalty_raw",
    "lcfs_regression_penalty",
    "lcfs_relative_progress",
    "lcfs_quality",
    "lcfs_component",
    "lcfs_score",
    "lcfs_absolute_score",
    "x_score",
    "strike_score",
    "psix_score",
    "nx_score",
    "x_count_score",
    "x_position_cm",
    "x_position_initial_cm",
    "x_position_progress_score",
    "x_position_precision_score",
    "x_position_regression_cm",
    "x_position_regression_ratio",
    "x_position_regression_penalty_raw",
    "x_position_regression_penalty",
    "x_position_relative_progress",
    "x_position_quality",
    "x_position_component",
    "x_valid_count",
    "x_flux_error",
    "x_flux_initial_error",
    "x_flux_relative_progress",
    "x_flux_quality",
    "x_flux_component",
    "topology_score",
    "limiter_transition_score",
    "coil_penalty",
    "action_l2",
    "action_smooth",
    "action_l2_penalty",
    "action_smooth_penalty",
    "lcfs_error_cm",
    "competition_score",
    "coil_margin",
    "topology_mask",
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

    values: dict[str, Any] = {
        "obs_mean": np.asarray(vecnormalize.obs_rms.mean, dtype=np.float64),
        "obs_var": np.asarray(vecnormalize.obs_rms.var, dtype=np.float64),
        "clip_obs": np.asarray(vecnormalize.clip_obs, dtype=np.float64),
        "epsilon": np.asarray(vecnormalize.epsilon, dtype=np.float64),
    }
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
            means[key] = float(np.mean(values)) if values else 0.0
        info_values = {
            "hfm_timeout_rate": (
                float(self._hfm_timeout_count) / float(self._info_count)
                if self._info_count > 0
                else 0.0
            ),
            "hfm_timeout_count": float(self._hfm_timeout_count),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    *[means[key] for key in REWARD_COMPONENT_KEYS],
                    *[info_values[key] for key in INFO_COMPONENT_KEYS],
                ]
            )

        for key, value in means.items():
            self.logger.record(f"reward_components/{key}", value)
        for key, value in info_values.items():
            self.logger.record(f"reward_components/{key}", value)
        self._buffer.clear()
        self._info_count = 0
        self._hfm_timeout_count = 0

        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log({f"reward_components/{key}": value for key, value in means.items()}, step=self.num_timesteps)


class EpisodeRewardComponentsCallback(BaseCallback):
    """Track per-episode reward component sums and append rollout summaries as JSONL."""

    def __init__(self, log_dir: Path, reward_weights: dict[str, float] | None = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.reward_weights = dict(reward_weights or {})
        self.rollout_idx = 0
        self._finished_episodes: dict[str, deque[float]] = {}
        self._current_episode: dict[int, dict[str, float]] = {}

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_idx = 0
        self._finished_episodes.clear()
        self._current_episode.clear()
        output_path = self.log_dir / "episode_reward_components.jsonl"
        output_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            episode = self._current_episode.setdefault(env_idx, {"episode_length": 0.0})
            components = info.get("reward_components", {})
            if isinstance(components, dict):
                for key in REWARD_COMPONENT_KEYS:
                    value = components.get(key)
                    if value is None:
                        continue
                    value = float(value)
                    if np.isfinite(value):
                        episode[key] = episode.get(key, 0.0) + value
            episode["episode_length"] = episode.get("episode_length", 0.0) + 1.0
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if done:
                self._finalize_episode(env_idx)
        return True

    def _finalize_episode(self, env_idx: int) -> None:
        episode = self._current_episode.pop(env_idx, None)
        if not episode:
            return
        for key, accumulated in episode.items():
            if key not in self._finished_episodes:
                self._finished_episodes[key] = deque(maxlen=64)
            self._finished_episodes[key].append(float(accumulated))

    def _on_rollout_end(self) -> None:
        if not self._finished_episodes:
            self.rollout_idx += 1
            return
        components = {key: list(values) for key, values in self._finished_episodes.items()}
        output_path = self.log_dir / "episode_reward_components.jsonl"
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"rollout_idx": self.rollout_idx, "components": components}, ensure_ascii=False) + "\n")
        if wandb is not None and getattr(wandb, "run", None) is not None:
            mean_scores = {
                f"rollout/mean_{key}": float(np.mean(values))
                for key, values in components.items()
                if values
            }
            if mean_scores:
                wandb.log(mean_scores, step=self.rollout_idx)
        self.rollout_idx += 1
        self._finished_episodes.clear()


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

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
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
