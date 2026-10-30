"""Perturb official F1 LCFS target and rollout with a submitted policy.

Supports two teacher formats:
- onnx:  submission/model/policy1.onnx + metadata.json + vecnormalize_stats.npz
- sb3:   PPO .zip + f1_features.metadata.json + vecnormalize_stats.npz

Aligned with tools/rollout_f1_target_lcfs_ablation.py:
- extracts semifinal trajectory from raw observations
- scores the rollout against the (perturbed) target using evaluation/eval/score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from environment import HFMSimulator
from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list
from score.evaluate_core import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perturb F1 LCFS in target.json and rollout a policy"
    )
    parser.add_argument(
        "--target",
        default=str(ROOT / "evaluation" / "results" / "target.json"),
        help="Path to official target.json",
    )
    parser.add_argument(
        "--teacher-format",
        choices=["onnx", "sb3"],
        default="onnx",
        help="Policy format to load",
    )

    # ONNX defaults
    parser.add_argument(
        "--model-dir",
        default=str(
            ROOT
            / "submission_best_f1_late_ppo_f1_7925760_steps_20260622"
            / "submission"
            / "model"
        ),
        help="Directory containing policy1.onnx, metadata.json and vecnormalize stats",
    )

    # SB3 defaults
    parser.add_argument(
        "--model",
        default=str(ROOT / "results" / "f1_direct_target_progress_regression_continue" / "ppo_f1_latest.zip"),
        help="Path to SB3 PPO .zip model",
    )
    parser.add_argument(
        "--metadata",
        default=str(ROOT / "results" / "f1_direct_target_progress_regression_continue" / "f1_features.metadata.json"),
        help="Path to SB3 feature metadata JSON",
    )
    parser.add_argument(
        "--vecnormalize",
        default=str(ROOT / "results" / "f1_direct_target_progress_regression_continue" / "vecnormalize_stats.npz"),
        help="Path to SB3 VecNormalize stats (.npz or .pkl)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for SB3 model (cpu/cuda/auto)",
    )

    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "evaluation" / "results" / "perturb_test"),
        help="Directory to save perturbed target and rollout results",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Gaussian noise std for LCFS (R, Z) coordinates in meters",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shot-id", default="13844_500", help="HFM shot id")
    parser.add_argument("--max-steps", type=int, default=300, help="Episode length")
    parser.add_argument("--port", type=int, default=5558, help="HFM socket port")
    parser.add_argument("--host", default="127.0.0.1", help="HFM socket host")
    return parser.parse_args()


def load_target(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def perturb_f1_lcfs(target_data: dict, sigma: float, seed: int) -> dict:
    """Return a copy of target_data with F1 lcfs_points perturbed by Gaussian noise."""
    rng = np.random.default_rng(seed)
    perturbed = {k: v for k, v in target_data.items()}
    if "F1" not in perturbed:
        raise ValueError("target.json does not contain F1 entry")

    f1 = perturbed["F1"]
    lcfs = np.asarray(f1["lcfs_points"], dtype=np.float64)
    if lcfs.shape != (32, 2):
        raise ValueError(f"F1 lcfs_points has unexpected shape {lcfs.shape}")

    noise = rng.normal(loc=0.0, scale=sigma, size=lcfs.shape)
    f1["lcfs_points"] = (lcfs + noise).tolist()
    return perturbed


def build_f1_reference(entry: dict, max_steps: int) -> dict:
    """Build reference spec for HFMSimulator trajectory mode.

    target.json F1 only provides Ip and lcfs_points. Missing scalar references
    (R, Z) will be filled by the environment from the hold reference.
    """
    return {
        "Ip": entry["Ip"],
        "lX": 0.0,  # F1 target is limiter
        "lcfs_points": entry["lcfs_points"],
    }


def make_env_config(args: argparse.Namespace) -> dict:
    return {
        "max_steps": args.max_steps,
        "predictor": {
            "host": args.host,
            "port": args.port,
            "shot_id": args.shot_id,
            "timeout": 300.0,
        },
        "reference": {"mode": "trajectory"},
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def build_policy(args: argparse.Namespace):
    """Return a policy object with reset()/act(obs) interface."""
    if args.teacher_format == "onnx":
        from submission.inference_common import BaseOnnxPolicy

        policy = BaseOnnxPolicy(
            task="F1",
            model_dir=args.model_dir,
            preferred_name="policy1.onnx",
        )
        print(f"Loaded ONNX policy from {Path(args.model_dir) / 'policy1.onnx'}")
        print(
            f"feature_mode={policy.metadata.get('feature_mode')}, "
            f"action_units={policy.metadata.get('action_units')}, "
            f"history_steps={policy.metadata.get('history_steps')}"
        )
        return policy

    if args.teacher_format == "sb3":
        from stable_baselines3 import PPO

        from submission.inference_common import normalized_action_7d_to_12d
        from training.distill_noise_student import (  # noqa: PLC0415
            FeatureSequenceBuilder,
            load_vecnormalize_stats,
            normalize_obs,
        )
        from training.features import FeatureMetadata  # noqa: PLC0415

        metadata = FeatureMetadata.load(Path(args.metadata))
        if metadata.feature_mode != "physical_v1":
            raise ValueError(f"SB3 rollout currently supports physical_v1 only, got {metadata.feature_mode}")
        if metadata.action_units != "normalized":
            raise ValueError(f"SB3 rollout currently supports normalized 7D action only, got {metadata.action_units}")

        model = PPO.load(str(Path(args.model)), device=args.device)
        obs_mean, obs_var, clip_obs, eps = load_vecnormalize_stats(Path(args.vecnormalize))

        class Sb3Policy:
            def __init__(self, model, metadata, obs_mean, obs_var, clip_obs, eps):
                self.model = model
                self.metadata = metadata
                self.obs_mean = obs_mean
                self.obs_var = obs_var
                self.clip_obs = clip_obs
                self.eps = eps
                self.builder: FeatureSequenceBuilder | None = None
                self._last_action7_norm = np.zeros(7, dtype=np.float32)

            def reset(self) -> None:
                self.builder = FeatureSequenceBuilder(self.metadata)
                self._last_action7_norm = np.zeros(7, dtype=np.float32)

            def act(self, observation: dict[str, Any]) -> np.ndarray:
                if self.builder is None:
                    self.builder = FeatureSequenceBuilder(self.metadata)
                    feature = self.builder.reset(observation)
                else:
                    feature = self.builder.step(self._last_action7_norm, observation)

                obs_norm = normalize_obs(feature, self.obs_mean, self.obs_var, self.clip_obs, self.eps)
                action, _ = self.model.predict(obs_norm.reshape(1, -1), deterministic=True)
                action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
                action12, action7_norm = normalized_action_7d_to_12d(action7)
                self._last_action7_norm = action7_norm
                return action12.astype(np.float32)

        policy = Sb3Policy(model, metadata, obs_mean, obs_var, clip_obs, eps)
        print(f"Loaded SB3 policy from {args.model}")
        print(
            f"feature_mode={metadata.feature_mode}, "
            f"action_units={metadata.action_units}, "
            f"history_steps={metadata.history_steps}"
        )
        return policy

    raise ValueError(f"Unsupported teacher_format: {args.teacher_format}")


def rollout(
    env: HFMSimulator,
    policy: Any,
    reference: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one episode and return raw observations plus reset info."""
    obs, info = env.reset(
        seed=0,
        options={"reference_mode": "trajectory", "reference": reference},
    )
    policy.reset()

    obs_list: list[dict[str, Any]] = []
    rewards: list[float] = []
    actions: list[list[float]] = []

    start = time.time()
    for _step in range(env.max_steps):
        action = policy.act(obs)
        actions.append(action.tolist())

        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list.append(dict(obs))
        rewards.append(float(reward))

        if terminated or truncated:
            break

    elapsed = time.time() - start
    return obs_list, {
        "steps": len(obs_list),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "elapsed_seconds": elapsed,
        "actions": actions,
        "rewards": rewards,
        "shot_id": info.get("shot_id"),
        "reference_mode": info.get("reference_mode"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and perturb target
    target_data = load_target(args.target)
    perturbed = perturb_f1_lcfs(target_data, args.sigma, args.seed)
    perturbed_path = output_dir / "target_perturb.json"
    with open(perturbed_path, "w", encoding="utf-8") as f:
        json.dump(perturbed, f, ensure_ascii=False, indent=2)
    print(f"Saved perturbed target to {perturbed_path}")

    # 2. Build reference
    f1_ref = build_f1_reference(perturbed["F1"], args.max_steps)

    # 3. Setup environment and policy
    config = make_env_config(args)
    env = HFMSimulator(config)
    policy = build_policy(args)

    # 4. Rollout
    try:
        obs_list, summary = rollout(env, policy, f1_ref)
        print(
            f"Rollout finished: {summary['steps']} steps, "
            f"total_reward={summary['total_reward']:.4f}, "
            f"elapsed={summary['elapsed_seconds']:.2f}s"
        )
    finally:
        env.close()

    # 5. Score against perturbed target
    trajectory = extract_semifinal_trajectory_from_obs_list(
        obs_list, task_id="F1", target_obs=perturbed.get("F1")
    )
    score_result = evaluate(
        {"F1": {"trajectory": trajectory, "timeout": False}},
        perturbed,
        task_ids=["F1"],
    )
    metric_scores = score_result["task_metric_scores"].get("F1", {})
    summary["score"] = {
        "total_score": float(score_result["total_score"]),
        "task_score": float(score_result["task_scores"].get("F1", 0.0)),
        "Ip": float(metric_scores.get("Ip", 0.0)),
        "LCFS": float(metric_scores.get("LCFS", 0.0)),
    }
    summary["trajectory"] = trajectory

    print(
        f"Score: total={summary['score']['total_score']:.4f}, "
        f"task={summary['score']['task_score']:.4f}, "
        f"Ip={summary['score']['Ip']:.4f}, "
        f"LCFS={summary['score']['LCFS']:.4f}"
    )

    # 6. Save results
    result_path = output_dir / "rollout_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Saved rollout result to {result_path}")


if __name__ == "__main__":
    main()
