#!/usr/bin/env python3
"""Roll out an SB3 semifinal policy per shot and plot physical trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
EVAL_FILES = PROJECT_ROOT / "eval_files"
if str(EVAL_FILES) not in sys.path:
    sys.path.insert(0, str(EVAL_FILES))

from environment import (  # noqa: E402
    Action7DTo12DWrapper,
    HFMSimulator,
    NormalizedAction7DTo12DWrapper,
)
from environment.preprocessing import action_7d_to_12d  # noqa: E402
from score.evaluate_core import evaluate  # noqa: E402
from trajectory_utils import extract_semifinal_trajectory_from_obs_list, to_jsonable  # noqa: E402
from training.features import (  # noqa: E402
    FixedFeatureFlattenWrapper,
    PhysicalFeatureHistoryWrapper,
)
from training.references import normalize_task  # noqa: E402
from training.reward_f2 import F2XPTTargetReward  # noqa: E402
from training.rewards import COIL_LIMITS_A, SemifinalDenseReward  # noqa: E402
from training.train_ppo import (  # noqa: E402
    HoldInitialResetWrapper,
    ReferenceResetWrapper,
    RewardInfoWrapper,
    _build_training_context,
    flatten_shot_groups,
    load_train_config,
    make_env_config,
    parse_ports,
)


ACTION_SCALE_7D_V = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float64)


class DictObservationCaptureWrapper(gym.Wrapper):
    """Keep the latest dict observation before feature flattening."""

    def __init__(self, env: gym.Env, sink: dict[str, Any]):
        super().__init__(env)
        self.sink = sink

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.sink["last_observation"] = observation
        self.sink["last_info"] = dict(info or {})
        self.sink["last_reward_components"] = {}
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info["captured_observation"] = observation
        self.sink["last_observation"] = observation
        self.sink["last_info"] = info
        self.sink["last_reward_components"] = dict(info.get("reward_components", {}))
        return observation, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=PROJECT_ROOT / "training" / "configs" / "f1_stage0a_parallel_hold_no_perturb.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "results" / "f1_stage0a_parallel_hold" / "ppo_f1_latest.zip",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=PROJECT_ROOT / "results" / "f1_stage0a_parallel_hold" / "vecnormalize.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "f1_stage0a_rollout_analysis",
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        default=None,
        help="Shot ids to roll out. Defaults to all unique shots from train config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HFM port. Defaults to the first port in train config.",
    )
    parser.add_argument("--episodes-per-shot", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--policy-mode",
        choices=["model", "zero"],
        default="model",
        help="Use the SB3 model, or send zero normalized/physical 7D actions.",
    )
    parser.add_argument(
        "--perturb-mode",
        choices=["none", "config"],
        default="none",
        help="Use no reset perturbation, or reuse reset perturbation from the train config.",
    )
    parser.add_argument(
        "--reference-variant-index",
        type=int,
        default=None,
        help="For configs with reference_variants, force one variant index for this rollout.",
    )
    parser.add_argument("--target-path", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "target.json")
    parser.add_argument("--infer-result", type=Path, default=None, help="Write scorer-compatible rollout JSON.")
    parser.add_argument("--score-output", type=Path, default=None, help="Write local score summary JSON.")
    parser.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write infer_result.json and score_summary.json for local evaluation. Use --no-score to disable.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files/config and exit without connecting to HFM.",
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scalar(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = observation.get(key)
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _array(observation: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = observation.get(key)
    out = np.zeros(int(np.prod(shape)), dtype=np.float64)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        out[:n] = flat[:n]
    return out.reshape(shape)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def context_from_config(args: argparse.Namespace) -> dict[str, Any]:
    namespace = argparse.Namespace(
        train_config=_rooted(args.train_config),
        task=None,
        config=None,
        reference=None,
        ports=None,
        shot_id=None,
        max_steps=args.max_steps,
        total_timesteps=None,
        save_dir=None,
        learning_rate=None,
        n_steps=None,
        batch_size=None,
        n_epochs=None,
        gamma=None,
        gae_lambda=None,
        ent_coef=None,
        clip_range=None,
        target_kl=None,
        max_grad_norm=None,
        vf_coef=None,
        log_std_init=None,
        net_arch=None,
        activation_fn=None,
        device=args.device,
        reward_log_interval=None,
        seed=args.seed,
        progress_bar=None,
        smoke_rollout=False,
        output_dir=args.output_dir,
    )
    return _build_training_context(namespace)


def unique_shots(train_config: dict[str, Any], explicit: list[str] | None) -> list[str]:
    if explicit:
        return [str(shot) for shot in explicit]
    out: list[str] = []
    for shot in flatten_shot_groups(train_config.get("shot_groups")):
        if shot not in out:
            out.append(shot)
    if not out:
        fallback = train_config.get("shot_id") or "13844_500"
        out.append(str(fallback))
    return out


def first_port(train_config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)
    ports = parse_ports(str(train_config.get("ports") or "5558"))
    return int(ports[0])


def reset_perturbation_for_mode(ctx: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode == "config":
        return ctx.get("reset_perturbation")
    return {"enabled": False}


def make_rollout_env(
    ctx: dict[str, Any],
    *,
    shot_id: str,
    port: int,
    sink: dict[str, Any],
    seed: int,
    reset_perturbation: dict[str, Any] | None,
):
    metadata = ctx["metadata"]

    def _init() -> gym.Env:
        cfg = make_env_config(
            ctx["base_config"],
            port=port,
            shot_id=shot_id,
            max_steps=int(ctx["max_steps"]),
            reference_mode=str(ctx["reference_mode"]),
        )
        if str(ctx.get("reward_type", "semifinal_dense")) == "f2_target_v1":
            reward = F2XPTTargetReward(
                target=ctx["target"],
                reward_weights=ctx["reward_weights"],
            )
        else:
            reward = SemifinalDenseReward(
                task=str(ctx["task"]),
                target=ctx["target"],
                reward_weights=ctx["reward_weights"],
            )
        env: gym.Env = HFMSimulator(cfg, reward_fn=reward)
        if ctx["reference_mode"] == "hold_initial":
            env = HoldInitialResetWrapper(
                env,
                shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                seed=seed,
            )
        else:
            env = ReferenceResetWrapper(
                env,
                reference=ctx["reference"],
                shot_id=shot_id,
                reset_perturbation=reset_perturbation,
                seed=seed,
                reference_variant_index=ctx.get("rollout_reference_variant_index"),
            )

        if metadata.action_units == "normalized":
            env = NormalizedAction7DTo12DWrapper(env)
        else:
            env = Action7DTo12DWrapper(env)
        env = RewardInfoWrapper(env)
        env = DictObservationCaptureWrapper(env, sink)
        if metadata.feature_mode in {
            "physical_v1",
            "physical_f1_no_ref_rz_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
        }:
            env = PhysicalFeatureHistoryWrapper(env, metadata=metadata)
        else:
            env = FixedFeatureFlattenWrapper(env, metadata=metadata)
        return env

    return _init


def normalized_action_to_physical_12d(action: np.ndarray, *, action_units: str) -> np.ndarray:
    action7 = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
    if action7.size < 7:
        action7 = np.pad(action7, (0, 7 - action7.size))
    if action_units == "normalized":
        action7 = np.clip(action7, -1.0, 1.0) * ACTION_SCALE_7D_V
    return action_7d_to_12d(action7.astype(np.float32)).astype(np.float64)


def lcfs_rms_cm(observation: dict[str, Any]) -> float:
    lcfs = _array(observation, "lcfs_points", (32, 2))
    ref = _array(observation, "reference_lcfs_points", (32, 2))
    return float(np.sqrt(np.mean(np.sum((lcfs - ref) ** 2, axis=1))) * 100.0)


def record_row(
    *,
    step: int,
    shot_id: str,
    episode: int,
    observation: dict[str, Any],
    action7: np.ndarray | None,
    action12: np.ndarray | None,
    reward: float,
    done: bool,
    info: dict[str, Any],
    reward_components: dict[str, Any],
) -> dict[str, Any]:
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    r = _scalar(observation, "R")
    ref_r = _scalar(observation, "reference_R", r)
    z = _scalar(observation, "Z")
    ref_z = _scalar(observation, "reference_Z", z)
    i_pf = _array(observation, "I_PF", (12,))
    coil_margin = float(np.max(np.abs(i_pf) / COIL_LIMITS_A))
    row: dict[str, Any] = {
        "task": str(info.get("task", "")),
        "reference_variant_index": info.get("reference_variant_index", ""),
        "reference_variant_id": info.get("reference_variant_id", ""),
        "shot_id": shot_id,
        "episode": episode,
        "step": step,
        "reward": float(reward),
        "done": int(bool(done)),
        "failure": int(bool(info.get("failure", False)) or _scalar(observation, "failure", 0.0) > 0.5),
        "Ip": ip,
        "reference_Ip": ref_ip,
        "ip_error": ip - ref_ip,
        "ip_abs_error": abs(ip - ref_ip),
        "R": r,
        "reference_R": ref_r,
        "r_error": r - ref_r,
        "Z": z,
        "reference_Z": ref_z,
        "z_error": z - ref_z,
        "lX": _scalar(observation, "lX"),
        "reference_lX": _scalar(observation, "reference_lX"),
        "nX": _scalar(observation, "nX"),
        "lcfs_rms_cm": lcfs_rms_cm(observation),
        "Rmax": _scalar(observation, "Rmax"),
        "Rmin": _scalar(observation, "Rmin"),
        "aminor": _scalar(observation, "aminor"),
        "kappa": _scalar(observation, "kappa"),
        "deltal": _scalar(observation, "deltal"),
        "deltau": _scalar(observation, "deltau"),
        "rc": _scalar(observation, "rc"),
        "zc": _scalar(observation, "zc"),
        "FA": _scalar(observation, "FA"),
        "FB": _scalar(observation, "FB"),
        "coil_margin": coil_margin,
        "coil_penalty": max(0.0, coil_margin - 1.0),
    }
    for idx, value in enumerate(i_pf):
        row[f"I_PF_{idx}"] = float(value)
    r_x = _array(observation, "rX", (6,))
    z_x = _array(observation, "zX", (6,))
    ref_r_x = _array(observation, "reference_rX", (4,))
    ref_z_x = _array(observation, "reference_zX", (4,))
    ref_x_valid = _array(observation, "reference_x_valid", (4,))
    for idx, value in enumerate(r_x):
        row[f"rX_{idx}"] = float(value)
    for idx, value in enumerate(z_x):
        row[f"zX_{idx}"] = float(value)
    for idx in range(4):
        row[f"reference_rX_{idx}"] = float(ref_r_x[idx])
        row[f"reference_zX_{idx}"] = float(ref_z_x[idx])
        row[f"reference_x_valid_{idx}"] = float(ref_x_valid[idx])
        if idx < r_x.size and ref_x_valid[idx] > 0.5:
            row[f"x_slot_dist_cm_{idx}"] = float(np.hypot(r_x[idx] - ref_r_x[idx], z_x[idx] - ref_z_x[idx]) * 100.0)
        else:
            row[f"x_slot_dist_cm_{idx}"] = float("nan")
    a7 = np.full(7, np.nan, dtype=np.float64) if action7 is None else np.asarray(action7, dtype=np.float64)
    a12 = np.full(12, np.nan, dtype=np.float64) if action12 is None else np.asarray(action12, dtype=np.float64)
    for idx, value in enumerate(a7[:7]):
        row[f"action7_{idx}"] = float(value)
    for idx, value in enumerate(a12[:12]):
        row[f"action12_{idx}"] = float(value)
    for key, value in reward_components.items():
        if isinstance(value, (int, float, np.generic)):
            row[f"reward_{key}"] = float(value)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], *, max_steps: int) -> dict[str, Any]:
    if not rows:
        return {}
    step_rows = [row for row in rows if int(row["step"]) > 0]
    final = rows[-1]

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(row.get(key, np.nan)) for row in step_rows], dtype=np.float64)

    lengths = int(final["step"])
    reward_sum = float(np.nansum(arr("reward"))) if step_rows else 0.0
    failure = bool(final.get("failure", 0))
    return {
        "shot_id": final["shot_id"],
        "episode": int(final["episode"]),
        "steps": lengths,
        "max_steps": max_steps,
        "survived_full": bool(lengths >= max_steps and not failure),
        "failure": failure,
        "reward_sum": reward_sum,
        "mean_ip_abs_error_ka": float(np.nanmean(arr("ip_abs_error")) / 1000.0) if step_rows else None,
        "max_ip_abs_error_ka": float(np.nanmax(arr("ip_abs_error")) / 1000.0) if step_rows else None,
        "final_ip_error_ka": float(final["ip_error"] / 1000.0),
        "mean_lcfs_rms_cm": float(np.nanmean(arr("lcfs_rms_cm"))) if step_rows else None,
        "max_lcfs_rms_cm": float(np.nanmax(arr("lcfs_rms_cm"))) if step_rows else None,
        "final_lcfs_rms_cm": float(final["lcfs_rms_cm"]),
        "mean_abs_r_error_cm": float(np.nanmean(np.abs(arr("r_error"))) * 100.0) if step_rows else None,
        "mean_abs_z_error_cm": float(np.nanmean(np.abs(arr("z_error"))) * 100.0) if step_rows else None,
        "topology_match_rate": float(np.nanmean(np.round(arr("lX")) == np.round(arr("reference_lX"))))
        if step_rows
        else None,
        "max_coil_margin": float(np.nanmax(arr("coil_margin"))) if step_rows else None,
        "max_abs_action12_v": float(
            np.nanmax(
                np.abs(
                    np.asarray(
                        [
                            [float(row.get(f"action12_{idx}", np.nan)) for idx in range(12)]
                            for row in step_rows
                        ],
                        dtype=np.float64,
                    )
                )
            )
        )
        if step_rows
        else None,
    }


def plot_episode(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    shot = str(rows[-1]["shot_id"])
    episode = int(rows[-1]["episode"])
    safe_shot = shot.replace("/", "_")
    steps = np.asarray([row["step"] for row in rows], dtype=float)

    def y(key: str, scale: float = 1.0) -> np.ndarray:
        return np.asarray([float(row.get(key, np.nan)) / scale for row in rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    fig.suptitle(f"Policy rollout: {shot} episode {episode}", fontsize=14)

    ax = axes[0]
    ax.plot(steps, y("Ip", 1000.0), label="Ip")
    ax.plot(steps, y("reference_Ip", 1000.0), linestyle="--", label="reference Ip")
    ax2 = ax.twinx()
    ax2.plot(steps, y("ip_error", 1000.0), color="tab:red", alpha=0.7, label="Ip error")
    ax2.axhline(50, color="tab:red", linestyle=":", linewidth=1)
    ax2.axhline(-50, color="tab:red", linestyle=":", linewidth=1)
    ax.set_ylabel("Ip kA")
    ax2.set_ylabel("Ip error kA")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(steps, y("R"), label="R")
    ax.plot(steps, y("reference_R"), linestyle="--", label="reference R")
    ax.plot(steps, y("Z"), label="Z")
    ax.plot(steps, y("reference_Z"), linestyle="--", label="reference Z")
    ax.set_ylabel("R/Z m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=4)

    ax = axes[2]
    ax.plot(steps, y("lcfs_rms_cm"), label="LCFS RMS cm")
    ax.plot(steps, y("coil_margin"), label="coil margin")
    ax.plot(steps, y("lX"), label="lX")
    ax.plot(steps, y("reference_lX"), linestyle="--", label="reference lX")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("shape / constraint")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=4)

    ax = axes[3]
    action_keys = [f"action12_{idx}" for idx in range(12)]
    actions = np.asarray([[float(row.get(key, np.nan)) for key in action_keys] for row in rows], dtype=float)
    ax.plot(steps, np.nanmax(np.abs(actions), axis=1), label="max |action12| V")
    ax.plot(steps, y("reward"), label="reward")
    ax.set_xlabel("step")
    ax.set_ylabel("action / reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    timeseries_path = out_dir / f"{safe_shot}_ep{episode:02d}_timeseries.png"
    fig.savefig(timeseries_path, dpi=160)
    plt.close(fig)

    lcfs_path = out_dir / f"{safe_shot}_ep{episode:02d}_lcfs.png"
    plot_lcfs(rows, lcfs_path, title=f"LCFS: {shot} episode {episode}")
    return timeseries_path, lcfs_path


def plot_lcfs(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    # Re-read LCFS arrays from sidecar npz if available; otherwise skip silently.
    sidecar = path.with_suffix(".lcfs.npz")
    if not sidecar.exists():
        return
    data = np.load(sidecar)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    for key, label, style in [
        ("reference", "reference", "--"),
        ("initial", "initial", "-"),
        ("final", "final", "-"),
    ]:
        arr = np.asarray(data[key], dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], style, label=label)
    ax.set_title(title)
    ax.set_xlabel("R m")
    ax.set_ylabel("Z m")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_lcfs_sidecar(path: Path, initial_obs: dict[str, Any], final_obs: dict[str, Any]) -> None:
    np.savez(
        path.with_suffix(".lcfs.npz"),
        reference=_array(final_obs, "reference_lcfs_points", (32, 2)),
        initial=_array(initial_obs, "lcfs_points", (32, 2)),
        final=_array(final_obs, "lcfs_points", (32, 2)),
    )


def rollout_one(
    *,
    model: Any,
    ctx: dict[str, Any],
    shot_id: str,
    episode: int,
    port: int,
    seed: int,
    vecnormalize_path: Path | None,
    output_dir: Path,
    reset_perturbation: dict[str, Any] | None,
    deterministic: bool,
    policy_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sink: dict[str, Any] = {}
    env_fn = make_rollout_env(
        ctx,
        shot_id=shot_id,
        port=port,
        sink=sink,
        seed=seed,
        reset_perturbation=reset_perturbation,
    )
    vec_env = DummyVecEnv([env_fn])
    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    rows: list[dict[str, Any]] = []
    obs_list: list[dict[str, Any]] = []
    initial_obs: dict[str, Any] | None = None
    final_obs: dict[str, Any] | None = None
    try:
        obs = vec_env.reset()
        initial_obs = dict(sink["last_observation"])
        reset_info = dict(sink.get("last_info", {}))
        reset_info["task"] = str(ctx["task"])
        rows.append(
            record_row(
                step=0,
                shot_id=shot_id,
                episode=episode,
                observation=sink["last_observation"],
                action7=None,
                action12=None,
                reward=0.0,
                done=False,
                info=reset_info,
                reward_components={},
            )
        )
        max_steps = int(ctx["max_steps"])
        for step in range(1, max_steps + 1):
            if policy_mode == "zero":
                action = np.zeros((1, 7), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=np.float32)
            action7 = action_arr.reshape(-1)[:7]
            action12 = normalized_action_to_physical_12d(
                action7,
                action_units=ctx["metadata"].action_units,
            )
            obs, rewards, dones, infos = vec_env.step(action_arr)
            done = bool(dones[0])
            info = dict(infos[0] or {})
            info["task"] = str(ctx["task"])
            raw_observation = info.get("captured_observation", sink["last_observation"])
            final_obs = dict(raw_observation)
            obs_list.append(final_obs)
            rows.append(
                record_row(
                    step=step,
                    shot_id=shot_id,
                    episode=episode,
                    observation=raw_observation,
                    action7=action7,
                    action12=action12,
                    reward=float(rewards[0]),
                    done=done,
                    info=info,
                    reward_components=info.get(
                        "reward_components",
                        sink.get("last_reward_components", {}),
                    ),
                )
            )
            if done:
                break
    finally:
        vec_env.close()

    if initial_obs is not None and final_obs is not None:
        lcfs_png = output_dir / f"{shot_id.replace('/', '_')}_ep{episode:02d}_lcfs.png"
        save_lcfs_sidecar(lcfs_png, initial_obs, final_obs)
    summary = summarize_rows(rows, max_steps=int(ctx["max_steps"]))
    return rows, summary, obs_list


def plot_summary(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "summary_by_shot.png"
    if not summaries:
        return path
    labels = [f"{item['shot_id']}#{item['episode']}" for item in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].bar(x, [item["steps"] for item in summaries])
    axes[0].set_ylabel("steps")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x - 0.2, [item["mean_ip_abs_error_ka"] for item in summaries], width=0.4, label="Ip abs err kA")
    axes[1].bar(x + 0.2, [item["mean_lcfs_rms_cm"] for item in summaries], width=0.4, label="LCFS RMS cm")
    axes[1].set_ylabel("tracking error")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].bar(x - 0.2, [item["max_coil_margin"] for item in summaries], width=0.4, label="max coil margin")
    axes[2].bar(x + 0.2, [item["topology_match_rate"] for item in summaries], width=0.4, label="topology match")
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[2].set_ylabel("constraint / topology")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="best")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def infer_task_id(stage: str, shots: list[str]) -> str:
    stage_lower = str(stage).lower()
    if "f2b" in stage_lower:
        return "F2b"
    if "f2a" in stage_lower:
        return "F2a"
    if "f1" in stage_lower:
        return "F1"
    if shots and all(str(shot).startswith("15892") for shot in shots):
        return "F2b"
    return "F2a"


def build_score_outputs(
    *,
    infer_tasks: dict[str, Any],
    target_all: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for rollout_id, entry in infer_tasks.items():
        tid = str(entry.get("task_id", task_id))
        result = evaluate(
            {tid: {"trajectory": entry["trajectory"], "timeout": bool(entry.get("timeout", False))}},
            target_all,
            task_ids=[tid],
        )
        scored.append(
            {
                "rollout_id": rollout_id,
                "task_id": tid,
                "shot_id": entry.get("shot_id"),
                "episode": entry.get("episode"),
                "n_steps": entry.get("n_steps"),
                "task_score": float(result["task_scores"].get(tid, 0.0)),
                "metric_scores": result["task_metric_scores"].get(tid, {}),
                "total_score": float(result.get("total_score", 0.0)),
            }
        )

    if len(infer_tasks) == 1:
        entry = next(iter(infer_tasks.values()))
        tid = str(entry.get("task_id", task_id))
        infer_result: dict[str, Any] = {
            tid: {
                "trajectory": entry["trajectory"],
                "timeout": bool(entry.get("timeout", False)),
            }
        }
    else:
        infer_result = {
            rollout_id: {
                "task_id": entry.get("task_id", task_id),
                "shot_id": entry.get("shot_id"),
                "episode": entry.get("episode"),
                "trajectory": entry["trajectory"],
                "timeout": bool(entry.get("timeout", False)),
            }
            for rollout_id, entry in infer_tasks.items()
        }

    scores = np.asarray([item["task_score"] for item in scored], dtype=np.float64)
    summary = {
        "task_id": task_id,
        "rollouts": scored,
        "mean_task_score": float(np.nanmean(scores)) if scores.size else 0.0,
        "min_task_score": float(np.nanmin(scores)) if scores.size else 0.0,
        "max_task_score": float(np.nanmax(scores)) if scores.size else 0.0,
    }
    return {"infer_result": infer_result, "score_summary": summary}


def main() -> int:
    args = parse_args()
    args.train_config = _rooted(args.train_config)
    args.model = _rooted(args.model)
    vec_spec = str(args.vecnormalize).strip().lower() if args.vecnormalize else ""
    args.vecnormalize = None if vec_spec in {"", "none", "null", "false"} else _rooted(args.vecnormalize)
    args.output_dir = _rooted(args.output_dir)
    args.target_path = _rooted(args.target_path)
    args.infer_result = _rooted(args.infer_result) if args.infer_result else None
    args.score_output = _rooted(args.score_output) if args.score_output else None

    train_config = load_train_config(args.train_config)
    ctx = context_from_config(args)
    task = normalize_task(ctx["task"])
    ctx["rollout_reference_variant_index"] = args.reference_variant_index

    shots = unique_shots(train_config, args.shots)
    port = first_port(train_config, args.port)
    vecnormalize_path = args.vecnormalize if args.vecnormalize and args.vecnormalize.exists() else None

    if args.policy_mode == "model" and not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    if args.vecnormalize and not args.vecnormalize.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {args.vecnormalize}")

    print(
        "rollout setup: "
        f"task={task}, stage={ctx['stage']}, model={args.model}, vecnormalize={vecnormalize_path}, "
        f"port={port}, shots={shots}, max_steps={ctx['max_steps']}, "
        f"policy_mode={args.policy_mode}, perturb_mode={args.perturb_mode}"
    )
    if args.validate_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_all = None
    if args.score or args.infer_result or args.score_output:
        if args.target_path is None or not args.target_path.exists():
            raise FileNotFoundError(f"target file not found: {args.target_path}")
        target_all = json.loads(args.target_path.read_text(encoding="utf-8"))
    model = None
    if args.policy_mode == "model":
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), device=args.device)
    reset_perturbation = reset_perturbation_for_mode(ctx, args.perturb_mode)

    all_summaries: list[dict[str, Any]] = []
    task_id = infer_task_id(ctx["stage"], shots)
    infer_tasks: dict[str, Any] = {}
    for shot_id in shots:
        for episode in range(int(args.episodes_per_shot)):
            seed = int(args.seed) + 1000 * shots.index(shot_id) + episode
            print(f"rolling out shot={shot_id}, episode={episode}, seed={seed}")
            rows, summary, obs_list = rollout_one(
                model=model,
                ctx=ctx,
                shot_id=shot_id,
                episode=episode,
                port=port,
                seed=seed,
                vecnormalize_path=vecnormalize_path,
                output_dir=args.output_dir,
                reset_perturbation=reset_perturbation,
                deterministic=bool(args.deterministic),
                policy_mode=str(args.policy_mode),
            )
            csv_path = args.output_dir / f"{shot_id.replace('/', '_')}_ep{episode:02d}_trajectory.csv"
            write_csv(csv_path, rows)
            timeseries_path, lcfs_path = plot_episode(rows, args.output_dir)
            summary.update(
                {
                    "trajectory_csv": str(csv_path),
                    "timeseries_plot": str(timeseries_path),
                    "lcfs_plot": str(lcfs_path),
                }
            )
            all_summaries.append(summary)
            if target_all is not None:
                target_obs = target_all.get(task_id, target_all.get("F2a"))
                trajectory = extract_semifinal_trajectory_from_obs_list(
                    obs_list,
                    task_id=task_id,
                    target_obs=target_obs,
                )
                infer_tasks[f"{task_id}_{shot_id}_ep{episode:02d}"] = {
                    "task_id": task_id,
                    "shot_id": shot_id,
                    "episode": episode,
                    "trajectory": trajectory,
                    "timeout": False,
                    "n_steps": len(obs_list),
                }
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))

    if target_all is not None:
        score_payload = build_score_outputs(
            infer_tasks=infer_tasks,
            target_all=target_all,
            task_id=task_id,
        )
        infer_path = args.infer_result or (args.output_dir / "infer_result.json")
        infer_path.write_text(
            json.dumps(to_jsonable(score_payload["infer_result"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        score_path = args.score_output or (args.output_dir / "score_summary.json")
        score_path.write_text(
            json.dumps(to_jsonable(score_payload["score_summary"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"infer result: {infer_path}")
        print(f"score summary: {score_path}")
    write_csv(args.output_dir / "rollout_summary.csv", all_summaries)
    (args.output_dir / "rollout_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary_plot = plot_summary(all_summaries, args.output_dir)
    print(f"summary: {args.output_dir / 'rollout_summary.json'}")
    print(f"summary plot: {summary_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

