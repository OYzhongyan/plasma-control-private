"""Shared PPO training utilities for semifinal F1/F2 examples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import Action7DTo12DWrapper, DictObsFlattenWrapper, HFMSimulator  # noqa: E402
from environment.xpt_utils import extract_xpt_observation_pack  # noqa: E402

PF_CURRENT_LIMITS = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)


@dataclass
class TargetReference:
    lcfs: np.ndarray
    ip_final: float
    rX: np.ndarray | None = None
    zX: np.ndarray | None = None
    strike_r: np.ndarray | None = None
    strike_z: np.ndarray | None = None
    x_valid: np.ndarray | None = None
    strike_valid: np.ndarray | None = None

    @property
    def has_xpt(self) -> bool:
        return self.rX is not None and self.zX is not None


def scalar(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(np.asarray(obs.get(key, default), dtype=np.float64).reshape(-1)[0])


def parse_ports(spec: str) -> list[int]:
    ports: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start), int(end) + 1))
        else:
            ports.append(int(part))
    return ports


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_target_reference(path: Path) -> TargetReference:
    data = json.loads(path.read_text(encoding="utf-8"))
    lcfs = np.asarray(data["reference_lcfs_points"], dtype=np.float64)
    ip_final = float(data.get("reference_Ip_final", 400_000.0))

    if "reference_rX" not in data:
        return TargetReference(lcfs=lcfs, ip_final=ip_final)

    return TargetReference(
        lcfs=lcfs,
        ip_final=ip_final,
        rX=np.asarray(data["reference_rX"], dtype=np.float64),
        zX=np.asarray(data["reference_zX"], dtype=np.float64),
        strike_r=np.asarray(data["reference_strike_r"], dtype=np.float64),
        strike_z=np.asarray(data["reference_strike_z"], dtype=np.float64),
        x_valid=np.asarray(data.get("reference_x_valid", np.ones(4)), dtype=np.float64),
        strike_valid=np.asarray(data.get("reference_strike_valid", np.ones(8)), dtype=np.float64),
    )


def build_reference(task: str, target: TargetReference, max_steps: int) -> dict[str, Any]:
    if task == "F2":
        t = np.arange(max_steps, dtype=np.float64)
        ip = np.where(t <= 100.0, 500_000.0 - 1_000.0 * t, target.ip_final)
    else:
        ip = np.linspace(500_000.0, target.ip_final, max_steps, dtype=np.float64)

    reference: dict[str, Any] = {
        "Ip": ip.tolist(),
        "lcfs_points": np.repeat(target.lcfs[None, :, :], max_steps, axis=0).tolist(),
    }
    if target.has_xpt:
        reference.update(
            {
                "rX": target.rX.tolist(),
                "zX": target.zX.tolist(),
                "x_valid": target.x_valid.tolist(),
                "strike_r": target.strike_r.tolist(),
                "strike_z": target.strike_z.tolist(),
                "strike_valid": target.strike_valid.tolist(),
                "nX": [float(np.sum(target.x_valid > 0.5))],
                "n_strike": [float(np.sum(target.strike_valid > 0.5))],
            }
        )
    return reference


def config_for_port(base_config: dict[str, Any], *, port: int, shot_id: str, max_steps: int, target: TargetReference, task: str) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_config))
    cfg["max_steps"] = int(max_steps)
    cfg.setdefault("predictor", {})
    cfg["predictor"]["port"] = int(port)
    cfg["predictor"]["shot_id"] = shot_id
    cfg["reference"] = {
        "mode": "trajectory",
        "reference_keys": ["Ip", "R", "Z", "lcfs_points"],
        "reference": build_reference(task, target, max_steps),
    }
    return cfg


class SemifinalReward:
    def __init__(self, target: TargetReference):
        self.target = target

    def __call__(self, observation: dict[str, Any], action: np.ndarray, terminated: bool = False, truncated: bool = False, info: dict | None = None) -> float:
        if terminated:
            return -5.0

        ref_ip = scalar(observation, "reference_Ip", self.target.ip_final)
        ip_score = 1.0 - min(abs(scalar(observation, "Ip") - ref_ip) / 50_000.0, 1.0)

        lcfs = np.asarray(observation.get("lcfs_points", np.zeros((32, 2))), dtype=np.float64)
        ref_lcfs = np.asarray(observation.get("reference_lcfs_points", self.target.lcfs), dtype=np.float64)
        n = min(len(lcfs), len(ref_lcfs))
        lcfs_cm = float(np.sqrt(np.mean(np.sum((lcfs[:n] - ref_lcfs[:n]) ** 2, axis=1))) * 100.0)
        lcfs_score = 1.0 - min(lcfs_cm / 5.0, 1.0)

        score = 0.45 * ip_score + 0.55 * lcfs_score
        if self.target.has_xpt:
            pack = extract_xpt_observation_pack(
                observation,
                self.target.rX,
                self.target.zX,
                self.target.strike_r,
                self.target.strike_z,
                fx_order="C",
            )
            target_x_valid = np.asarray(self.target.x_valid, dtype=np.float64) > 0.5
            valid = (pack["x_valid"] > 0.5) & target_x_valid
            if np.any(valid):
                x_dist_cm = (
                    np.sqrt(
                        (pack["x_r"][valid] - self.target.rX[valid]) ** 2
                        + (pack["x_z"][valid] - self.target.zX[valid]) ** 2
                    )
                    * 100.0
                )
                x_score = 1.0 - min(float(np.mean(x_dist_cm)) / 5.0, 1.0)
            else:
                x_score = 0.0
            if int(pack["nX"]) != int(np.sum(target_x_valid)):
                x_score *= 0.25

            target_strike_valid = np.asarray(self.target.strike_valid, dtype=np.float64) > 0.5
            strike_valid = (pack["strike_valid"] > 0.5) & target_strike_valid
            if np.any(strike_valid):
                strike_dist_cm = (
                    np.sqrt(
                        (pack["strike_r"][strike_valid] - self.target.strike_r[strike_valid]) ** 2
                        + (pack["strike_z"][strike_valid] - self.target.strike_z[strike_valid]) ** 2
                    )
                    * 100.0
                )
                strike_score = 1.0 - min(float(np.mean(strike_dist_cm)) / 10.0, 1.0)
            else:
                strike_score = 0.0
            if int(pack["strike_n_actual"]) != int(np.sum(target_strike_valid)):
                strike_score *= 0.5

            xpt_score = 0.75 * x_score + 0.25 * strike_score
            score = 0.35 * score + 0.65 * xpt_score

        ipf = np.asarray(observation.get("I_PF", np.zeros(12)), dtype=np.float64).reshape(-1)
        if ipf.size == 12 and np.any(np.abs(ipf) > PF_CURRENT_LIMITS):
            score -= 0.5

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        score -= 0.001 * float(np.mean((action / 500.0) ** 2))
        return float(score)


def flatten_keys(target: TargetReference) -> list[str]:
    keys = ["Ip", "reference_Ip", "lcfs_points", "reference_lcfs_points", "I_PF"]
    if target.has_xpt:
        keys += [
            "rX",
            "zX",
            "nX",
            "FX",
            "reference_rX",
            "reference_zX",
            "reference_x_valid",
            "reference_strike_r",
            "reference_strike_z",
            "reference_strike_valid",
            "reference_nX",
            "reference_n_strike",
        ]
    return keys


def make_env(base_config: dict[str, Any], *, port: int, shot_id: str, max_steps: int, target: TargetReference, task: str):
    def _init():
        cfg = config_for_port(base_config, port=port, shot_id=shot_id, max_steps=max_steps, target=target, task=task)
        env: gym.Env = HFMSimulator(cfg, reward_fn=SemifinalReward(target))
        env = Action7DTo12DWrapper(env)
        env = DictObsFlattenWrapper(env, keys=flatten_keys(target))
        return env

    return _init


def add_common_args(parser: argparse.ArgumentParser, *, default_save_dir: Path, default_reference: Path, default_start_shot: str, default_max_steps: int) -> None:
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "env_default.yaml")
    parser.add_argument("--reference", type=Path, default=default_reference)
    parser.add_argument("--start-shot-id", type=str, default=default_start_shot)
    parser.add_argument("--ports", type=str, default="2223")
    parser.add_argument("--max-steps", type=int, default=default_max_steps)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--save-dir", type=Path, default=default_save_dir)
    parser.add_argument(
        "--smoke-rollout",
        action="store_true",
        help="Run a short zero-action rollout without importing PPO dependencies.",
    )


def run_smoke_rollout(args: argparse.Namespace, *, task: str) -> int:
    base_config = load_yaml(args.config)
    target = load_target_reference(args.reference)
    port = parse_ports(args.ports)[0]
    env = make_env(
        base_config,
        port=port,
        shot_id=args.start_shot_id,
        max_steps=args.max_steps,
        target=target,
        task=task,
    )()
    try:
        obs, info = env.reset(seed=0)
        print(f"{task} reset ok: shot_id={info['shot_id']}, obs_shape={obs.shape}")
        for step in range(min(3, args.max_steps)):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"{task} step {step}: reward={float(reward):.4f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


def run_training(args: argparse.Namespace, *, task: str) -> int:
    if args.smoke_rollout:
        return run_smoke_rollout(args, task=task)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError:
        print("缺少依赖：pip install stable-baselines3 torch")
        return 1

    base_config = load_yaml(args.config)
    target = load_target_reference(args.reference)
    ports = parse_ports(args.ports)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            base_config,
            port=port,
            shot_id=args.start_shot_id,
            max_steps=args.max_steps,
            target=target,
            task=task,
        )
        for port in ports
    ]
    vec_env = DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env, filename=str(args.save_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=min(128, args.max_steps),
        batch_size=64,
        n_epochs=5,
        gamma=0.995,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(str(args.save_dir / f"ppo_{task.lower()}_latest"))
    vec_env.close()
    print(f"saved: {args.save_dir / f'ppo_{task.lower()}_latest.zip'}")
    return 0
