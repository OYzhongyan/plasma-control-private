#!/usr/bin/env python3
"""Roll out an SB3 policy and render LCFS evolution as a GIF."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rollout_stage0_policy import (  # noqa: E402
    _array,
    _rooted,
    _scalar,
    first_port,
    lcfs_rms_cm,
    load_train_config,
    make_rollout_env,
    normalized_action_to_physical_12d,
)
from training.train_ppo import _build_training_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def build_context(train_config: Path, *, shot_id: str | None, port: int | None, seed: int):
    namespace = argparse.Namespace(
        train_config=train_config,
        task=None,
        ports=str(port) if port is not None else None,
        shot_id=shot_id,
        max_steps=None,
        total_timesteps=None,
        save_dir=None,
        reference=None,
        config=None,
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
        device=None,
        reward_log_interval=None,
        progress_bar=None,
        seed=seed,
        smoke_rollout=False,
    )
    return _build_training_context(namespace)


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


def observation_row(
    *,
    step: int,
    observation: dict[str, Any],
    reward: float,
    done: bool,
    action7: np.ndarray | None,
    action12: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": int(step),
        "reward": float(reward),
        "done": int(bool(done)),
        "failure": int(_scalar(observation, "failure", 0.0) > 0.5),
        "Ip": _scalar(observation, "Ip"),
        "reference_Ip": _scalar(observation, "reference_Ip"),
        "R": _scalar(observation, "R"),
        "Z": _scalar(observation, "Z"),
        "lX": _scalar(observation, "lX"),
        "reference_lX": _scalar(observation, "reference_lX"),
        "nX": _scalar(observation, "nX"),
        "lcfs_rms_cm": lcfs_rms_cm(observation),
    }
    a7 = np.full(7, np.nan) if action7 is None else np.asarray(action7, dtype=float).reshape(-1)
    a12 = np.full(12, np.nan) if action12 is None else np.asarray(action12, dtype=float).reshape(-1)
    for idx, value in enumerate(a7[:7]):
        row[f"action7_{idx}"] = float(value)
    for idx, value in enumerate(a12[:12]):
        row[f"action12_{idx}"] = float(value)
    return row


def frame_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= max_frames:
        return np.arange(num_frames, dtype=int)
    return np.unique(np.linspace(0, num_frames - 1, max_frames).round().astype(int))


def save_lcfs_gif(
    *,
    path: Path,
    lcfs: np.ndarray,
    reference: np.ndarray,
    rows: list[dict[str, Any]],
    fps: int,
    max_frames: int,
) -> None:
    indices = frame_indices(len(rows), max_frames)
    initial = lcfs[0]
    ref = reference[-1]

    all_points = np.concatenate([lcfs.reshape(-1, 2), reference.reshape(-1, 2)], axis=0)
    r_min, z_min = np.nanmin(all_points, axis=0)
    r_max, z_max = np.nanmax(all_points, axis=0)
    pad_r = max(0.03, 0.08 * float(r_max - r_min))
    pad_z = max(0.03, 0.08 * float(z_max - z_min))

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ax.plot(ref[:, 0], ref[:, 1], "--", color="black", linewidth=1.8, label="target LCFS")
    ax.plot(initial[:, 0], initial[:, 1], ":", color="tab:gray", linewidth=1.6, label="initial LCFS")
    (line,) = ax.plot([], [], "-", color="tab:blue", linewidth=2.2, label="policy LCFS")
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.set_xlim(r_min - pad_r, r_max + pad_r)
    ax.set_ylim(z_min - pad_z, z_max + pad_z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R / m")
    ax.set_ylabel("Z / m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    def update(frame_no: int):
        idx = int(indices[frame_no])
        pts = lcfs[idx]
        row = rows[idx]
        line.set_data(pts[:, 0], pts[:, 1])
        text.set_text(
            f"step {int(row['step'])}\n"
            f"LCFS RMS {float(row['lcfs_rms_cm']):.2f} cm\n"
            f"Ip {float(row['Ip']) / 1000.0:.1f} kA\n"
            f"lX {float(row['lX']):.0f}, nX {float(row['nX']):.0f}"
        )
        return line, text

    ani = animation.FuncAnimation(fig, update, frames=len(indices), interval=1000 / max(1, fps), blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    train_config_path = _rooted(args.train_config)
    model_path = _rooted(args.model)
    vecnormalize_path = _rooted(args.vecnormalize)
    output_dir = _rooted(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = load_train_config(train_config_path)
    shot_id = str(args.shot_id or train_config.get("shot_id") or "13844_500")
    port = first_port(train_config, args.port)
    ctx = build_context(train_config_path, shot_id=shot_id, port=port, seed=args.seed)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sink: dict[str, Any] = {}
    env_fn = make_rollout_env(
        ctx,
        shot_id=shot_id,
        port=port,
        sink=sink,
        seed=args.seed,
        reset_perturbation={"enabled": False},
    )
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(str(model_path), env=vec_env, device=args.device)

    rows: list[dict[str, Any]] = []
    lcfs_points: list[np.ndarray] = []
    reference_points: list[np.ndarray] = []
    try:
        obs = vec_env.reset()
        raw = dict(sink["last_observation"])
        rows.append(observation_row(step=0, observation=raw, reward=0.0, done=False, action7=None, action12=None))
        lcfs_points.append(_array(raw, "lcfs_points", (32, 2)))
        reference_points.append(_array(raw, "reference_lcfs_points", (32, 2)))
        for step in range(1, int(ctx["max_steps"]) + 1):
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            action12 = normalized_action_to_physical_12d(action7, action_units=ctx["metadata"].action_units)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            info = dict(infos[0] or {})
            raw = dict(info.get("captured_observation", sink["last_observation"]))
            rows.append(
                observation_row(
                    step=step,
                    observation=raw,
                    reward=float(rewards[0]),
                    done=done,
                    action7=action7,
                    action12=action12,
                )
            )
            lcfs_points.append(_array(raw, "lcfs_points", (32, 2)))
            reference_points.append(_array(raw, "reference_lcfs_points", (32, 2)))
            if done:
                break
    finally:
        vec_env.close()

    lcfs = np.asarray(lcfs_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    csv_path = output_dir / f"{shot_id}_lcfs_rollout.csv"
    npz_path = output_dir / f"{shot_id}_lcfs_points.npz"
    gif_path = output_dir / f"{shot_id}_lcfs_evolution.gif"
    summary_path = output_dir / f"{shot_id}_summary.json"
    write_csv(csv_path, rows)
    np.savez(npz_path, lcfs=lcfs, reference=reference)
    save_lcfs_gif(path=gif_path, lcfs=lcfs, reference=reference, rows=rows, fps=args.fps, max_frames=args.max_frames)

    summary = {
        "train_config": str(train_config_path),
        "model": str(model_path),
        "vecnormalize": str(vecnormalize_path),
        "shot_id": shot_id,
        "port": port,
        "steps": int(rows[-1]["step"]) if rows else 0,
        "done": bool(rows[-1]["done"]) if rows else False,
        "failure": bool(rows[-1]["failure"]) if rows else False,
        "initial_lcfs_rms_cm": float(rows[0]["lcfs_rms_cm"]) if rows else None,
        "final_lcfs_rms_cm": float(rows[-1]["lcfs_rms_cm"]) if rows else None,
        "min_lcfs_rms_cm": float(np.nanmin([row["lcfs_rms_cm"] for row in rows])) if rows else None,
        "gif": str(gif_path),
        "csv": str(csv_path),
        "npz": str(npz_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
