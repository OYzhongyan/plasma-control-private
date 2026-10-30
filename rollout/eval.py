"""Evaluate a trained PPO policy and save per-step reward components."""

from __future__ import annotations
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STEP_RE = re.compile("_(\\d+)_steps(?:\\.zip)?$")
DEFAULT_REWARD_COMPONENT_KEYS = [
    "reward",
    "failure",
    "failure_penalty",
    "ip_abs_error",
    "ip_score",
    "ip_error_boundary_ratio",
    "ip_error_boundary_penalty_raw",
    "ip_error_boundary_penalty",
    "r_score",
    "z_score",
    "lcfs_score",
    "lcfs_error_cm",
    "x_score",
    "strike_score",
    "psix_score",
    "nx_score",
    "topology_score",
    "coil_penalty",
    "action_l2",
    "action_smooth",
    "action_l2_penalty",
    "action_smooth_penalty",
    "competition_score",
    "coil_margin",
    "topology_mask",
]


def _reward_component_keys() -> list[str]:
    try:
        from train.legacy_callbacks import REWARD_COMPONENT_KEYS
    except ImportError:
        return list(DEFAULT_REWARD_COMPONENT_KEYS)
    return list(REWARD_COMPONENT_KEYS)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _checkpoint_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(run_dir: Path) -> Path:
    direct = sorted(run_dir.glob("ppo_*_latest.zip"))
    if direct:
        return max(direct, key=lambda path: path.stat().st_mtime)
    checkpoint_dir = run_dir / "checkpoints"
    candidates = (
        list(checkpoint_dir.glob("ppo_*_steps.zip")) if checkpoint_dir.exists() else []
    )
    if not candidates:
        candidates = list(run_dir.rglob("ppo_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError(f"no PPO checkpoint found under {run_dir}")
    return max(
        candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime)
    )


def _find_checkpoint(run_dir: Path, checkpoint_step: int) -> Path:
    if int(checkpoint_step) < 0:
        return _find_latest_checkpoint(run_dir)
    pattern = f"ppo_*_{int(checkpoint_step)}_steps.zip"
    candidates = []
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        candidates.extend(checkpoint_dir.glob(pattern))
    candidates.extend(run_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"checkpoint step {checkpoint_step} not found under {run_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_metadata(run_dir: Path, ctx: dict[str, Any]):
    from train.features import FeatureMetadata

    metadata = ctx.get("metadata")
    if metadata is not None:
        return metadata
    candidates = [
        run_dir / "f2_features.metadata.json",
        *sorted(run_dir.glob("*features.metadata.json")),
    ]
    for path in candidates:
        if path.exists():
            return FeatureMetadata.load(path)
    return FeatureMetadata.create()


def _find_vecnormalize(model_path: Path, run_dir: Path) -> Path | None:
    stem = model_path.with_suffix("")
    candidates = [
        stem.with_name(f"{stem.name}_vecnormalize.pkl"),
        model_path.parent / "vecnormalize.pkl",
        run_dir / "vecnormalize.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_saved_context(configs_path: Path) -> dict[str, Any]:
    with open(configs_path, "rb") as f:
        ctx = pickle.load(f)
    if not isinstance(ctx, dict):
        raise ValueError(f"configs.pkl must contain a dict, got {type(ctx)!r}")
    return ctx


def _load_eval_env_config(name: str | None) -> dict[str, Any]:
    if not name:
        raise ValueError("--eval_env_config is required")
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".json")
    if not path.is_absolute():
        path = ROOT / "rollout" / "configs" / path
    if not path.exists():
        raise FileNotFoundError(f"eval env config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"eval env config must be a JSON object: {path}")
    return data


def _build_eval_env_specs(
    eval_cfg: dict[str, Any], eval_episode_num: int, seed: int
) -> list[dict[str, Any]]:
    from train.legacy_train_ppo import assign_shots_to_envs, ports_for_envs

    num_envs = int(eval_episode_num)
    if num_envs <= 0:
        raise ValueError("--eval_episode_num must be positive")
    eval_cfg["num_envs"] = num_envs
    fallback_shot_id = str(eval_cfg.get("shot_id") or "13844_500")
    shot_ids = assign_shots_to_envs(
        eval_cfg.get("shot_groups"), num_envs, fallback_shot_id
    )
    ports = ports_for_envs(str(eval_cfg.get("ports") or "2223"), num_envs)
    return [
        {"index": i, "port": ports[i], "shot_id": shot_ids[i], "seed": int(seed) + i}
        for i in range(num_envs)
    ]


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_name:
        raise ValueError("--run_name is required")
    run_dir = ROOT / "results" / args.run_name
    configs_path = run_dir / "configs.pkl"
    output_dir = run_dir / "eval"
    if not run_dir.exists():
        raise FileNotFoundError(f"result run directory not found: {run_dir}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.pkl not found: {configs_path}")
    model_path = _find_checkpoint(run_dir, int(args.checkpoint_step))
    vecnormalize_path = _find_vecnormalize(model_path, run_dir)
    ctx = _load_saved_context(configs_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    eval_cfg = _load_eval_env_config(args.eval_env_config)
    eval_specs = _build_eval_env_specs(
        eval_cfg, int(args.eval_episode_num), int(args.seed)
    )
    reference_mode = str(eval_cfg.get("reference_mode", "trajectory"))
    max_steps = int(eval_cfg.get("max_steps") or ctx.get("max_steps") or 300)
    from train.references import (
        build_reference,
        default_reference_path,
        load_target_reference,
    )
    from train.legacy_train_ppo import load_yaml

    reference_path = (
        eval_cfg.get("reference")
        or ctx.get("reference_path")
        or default_reference_path()
    )
    reference_path = Path(reference_path)
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    target = load_target_reference(reference_path)
    reference = (
        None if reference_mode == "hold_initial" else build_reference(target, max_steps)
    )
    if target is None:
        raise ValueError(f"could not load target reference from {reference_path}")
    if reference_mode != "hold_initial" and reference is None:
        raise ValueError(f"could not build trajectory reference from {reference_path}")
    env_config_path = Path(eval_cfg.get("env_config") or "configs/env_default.yaml")
    if not env_config_path.is_absolute():
        env_config_path = ROOT / env_config_path
    base_config = load_yaml(env_config_path)
    metadata = _find_metadata(run_dir, ctx)
    seed = int(args.seed)
    return {
        "reference_mode": reference_mode,
        "max_steps": max_steps,
        "base_config": base_config,
        "reference_path": reference_path,
        "target": target,
        "reference": reference,
        "metadata": metadata,
        "eval_env_config": eval_cfg,
        "env_specs": eval_specs,
        "model_path": model_path,
        "run_dir": run_dir,
        "configs_path": configs_path,
        "vecnormalize_path": vecnormalize_path,
        "seed": int(seed),
        "reward_weights": ctx.get("reward_weights"),
        "reset_perturbation": eval_cfg.get("reset_perturbation"),
        "output_dir": output_dir,
    }


def _make_eval_env(ctx: dict[str, Any], env_spec: dict[str, Any]):
    from environment import (
        Action7DTo12DWrapper,
        HFMSimulator,
        NormalizedAction7DTo12DWrapper,
    )
    from train.legacy_rewards import SemifinalDenseReward
    from train.legacy_train_ppo import (
        HoldInitialResetWrapper,
        ReferenceResetWrapper,
        make_env_config,
    )
    import gymnasium as gym

    class RolloutInfoWrapper(gym.Wrapper):

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info or {})
            reward_fn = getattr(self.unwrapped, "reward_fn", None)
            components = getattr(reward_fn, "last_components", None)
            if isinstance(components, dict):
                info["reward_components"] = dict(components)
            if isinstance(observation, dict):
                info["rollout_observation"] = {
                    "Ip": np.asarray(
                        observation.get("Ip", [np.nan]), dtype=np.float64
                    ).reshape(-1),
                    "reference_Ip": np.asarray(
                        observation.get("reference_Ip", [np.nan]), dtype=np.float64
                    ).reshape(-1),
                    "lcfs_points": np.asarray(
                        observation.get("lcfs_points", np.full((32, 2), np.nan)),
                        dtype=np.float64,
                    ),
                    "reference_lcfs_points": np.asarray(
                        observation.get(
                            "reference_lcfs_points", np.full((32, 2), np.nan)
                        ),
                        dtype=np.float64,
                    ),
                }
            return (observation, reward, terminated, truncated, info)

    cfg = make_env_config(
        ctx["base_config"],
        port=int(env_spec["port"]),
        shot_id=str(env_spec["shot_id"]),
        max_steps=int(ctx["max_steps"]),
        reference_mode=str(ctx["reference_mode"]),
    )
    reward_fn = SemifinalDenseReward(
        target=ctx["target"], reward_weights=ctx["reward_weights"]
    )
    env = HFMSimulator(cfg, reward_fn=reward_fn)
    if ctx["reference_mode"] == "hold_initial":
        env = HoldInitialResetWrapper(
            env,
            shot_id=str(env_spec["shot_id"]),
            reset_perturbation=ctx["reset_perturbation"],
            seed=int(env_spec["seed"]),
        )
    else:
        env = ReferenceResetWrapper(env, reference=ctx["reference"])
    metadata = ctx["metadata"]
    if metadata.action_units == "normalized":
        env = NormalizedAction7DTo12DWrapper(env)
    else:
        env = Action7DTo12DWrapper(env)
    env = RolloutInfoWrapper(env)
    if metadata.feature_mode == "physical_v1":
        from train.features import PhysicalFeatureHistoryWrapper

        env = PhysicalFeatureHistoryWrapper(env, metadata=metadata)
    else:
        from train.features import FixedFeatureFlattenWrapper

        env = FixedFeatureFlattenWrapper(env, metadata=metadata)
    return env


def _make_vec_env(ctx: dict[str, Any]):
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required: pip install stable-baselines3 torch"
        ) from exc
    env_fns = [
        lambda spec=env_spec: _make_eval_env(ctx, spec) for env_spec in ctx["env_specs"]
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vecnormalize_path = ctx["vecnormalize_path"]
    if vecnormalize_path is not None:
        if not vecnormalize_path.exists():
            raise FileNotFoundError(
                f"VecNormalize state not found: {vecnormalize_path}"
            )
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    reward_component_keys = _reward_component_keys()
    base_columns = [
        "episode",
        "env_index",
        "step",
        "env_reward",
        "ip",
        "reference_ip",
        "terminated",
        "truncated",
        "shot_id",
        "port",
        "hfm_timeout",
    ]
    action_columns = sorted(
        {key for row in rows for key in row if key.startswith("action_")}
    )
    extra_component_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_columns
            and key not in action_columns
            and (key not in reward_component_keys)
        }
    )
    columns = [
        *base_columns,
        *reward_component_keys,
        *action_columns,
        *extra_component_columns,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required: pip install stable-baselines3 torch"
        ) from exc
    ctx = _load_context(args)
    output_dir = Path(ctx["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(ctx["model_path"]), device=str(args.device))
    vec_env = _make_vec_env(ctx)
    rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    env_specs = list(ctx["env_specs"])
    num_envs = len(env_specs)
    active = np.ones(num_envs, dtype=bool)
    total_rewards = np.zeros(num_envs, dtype=np.float64)
    step_counts = np.zeros(num_envs, dtype=np.int64)
    max_steps = int(ctx["max_steps"])
    lcfs_points = np.full((num_envs, max_steps, 32, 2), np.nan, dtype=np.float64)
    reference_lcfs_points = np.full(
        (num_envs, max_steps, 32, 2), np.nan, dtype=np.float64
    )
    ip_values = np.full((num_envs, max_steps), np.nan, dtype=np.float64)
    reference_ip_values = np.full((num_envs, max_steps), np.nan, dtype=np.float64)
    observation_valid = np.zeros((num_envs, max_steps), dtype=bool)
    try:
        obs = vec_env.reset()
        for _ in range(max_steps):
            if not np.any(active):
                break
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            next_obs, reward, done, infos = vec_env.step(action)
            reward_arr = np.asarray(reward, dtype=np.float64).reshape(-1)
            done_arr = np.asarray(done, dtype=bool).reshape(-1)
            for env_idx, env_spec in enumerate(env_specs):
                if not active[env_idx]:
                    continue
                info = infos[env_idx] if infos and env_idx < len(infos) else {}
                env_reward = _safe_float(reward_arr[env_idx])
                total_rewards[env_idx] += env_reward
                step_counts[env_idx] += 1
                step_index = int(step_counts[env_idx] - 1)
                components = (
                    info.get("reward_components", {}) if isinstance(info, dict) else {}
                )
                rollout_obs = (
                    info.get("rollout_observation", {})
                    if isinstance(info, dict)
                    else {}
                )
                ip_value = (
                    _safe_float(
                        np.asarray(
                            rollout_obs.get("Ip", [np.nan]), dtype=np.float64
                        ).reshape(-1)[0],
                        default=np.nan,
                    )
                    if isinstance(rollout_obs, dict)
                    else np.nan
                )
                reference_ip_value = (
                    _safe_float(
                        np.asarray(
                            rollout_obs.get("reference_Ip", [np.nan]), dtype=np.float64
                        ).reshape(-1)[0],
                        default=np.nan,
                    )
                    if isinstance(rollout_obs, dict)
                    else np.nan
                )
                if isinstance(rollout_obs, dict) and step_index < max_steps:
                    lcfs = np.asarray(
                        rollout_obs.get("lcfs_points", np.full((32, 2), np.nan)),
                        dtype=np.float64,
                    )
                    ref_lcfs = np.asarray(
                        rollout_obs.get(
                            "reference_lcfs_points", np.full((32, 2), np.nan)
                        ),
                        dtype=np.float64,
                    )
                    if lcfs.shape == (32, 2):
                        lcfs_points[env_idx, step_index] = lcfs
                    if ref_lcfs.shape == (32, 2):
                        reference_lcfs_points[env_idx, step_index] = ref_lcfs
                    ip_values[env_idx, step_index] = ip_value
                    reference_ip_values[env_idx, step_index] = reference_ip_value
                    observation_valid[env_idx, step_index] = True
                row: dict[str, Any] = {
                    "episode": env_idx,
                    "env_index": env_idx,
                    "step": int(info.get("step", step_counts[env_idx])),
                    "env_reward": env_reward,
                    "ip": ip_value,
                    "reference_ip": reference_ip_value,
                    "terminated": bool(info.get("failure", False)),
                    "truncated": bool(done_arr[env_idx])
                    and (not bool(info.get("failure", False))),
                    "shot_id": info.get("shot_id", env_spec["shot_id"]),
                    "port": int(env_spec["port"]),
                    "hfm_timeout": bool(info.get("hfm_timeout", False)),
                }
                if isinstance(components, dict):
                    for key, value in components.items():
                        row[str(key)] = _safe_float(value)
                if bool(args.save_actions):
                    action_arr = np.asarray(action[env_idx]).reshape(-1)
                    for i, value in enumerate(action_arr):
                        row[f"action_{i}"] = _safe_float(value)
                rows.append(row)
                if bool(done_arr[env_idx]):
                    active[env_idx] = False
                    episode_summaries.append(
                        {
                            "episode": env_idx,
                            "env_index": env_idx,
                            "steps": int(step_counts[env_idx]),
                            "return": float(total_rewards[env_idx]),
                            "shot_id": str(env_spec["shot_id"]),
                            "port": int(env_spec["port"]),
                            "done": True,
                        }
                    )
            obs = next_obs
        for env_idx, env_spec in enumerate(env_specs):
            if not active[env_idx]:
                continue
            episode_summaries.append(
                {
                    "episode": env_idx,
                    "env_index": env_idx,
                    "steps": int(step_counts[env_idx]),
                    "return": float(total_rewards[env_idx]),
                    "shot_id": str(env_spec["shot_id"]),
                    "port": int(env_spec["port"]),
                    "done": False,
                }
            )
    finally:
        vec_env.close()
    csv_path = output_dir / args.csv_name
    _write_csv(csv_path, rows)
    observations_path = output_dir / "rollout_observations.npz"
    np.savez_compressed(
        observations_path,
        lcfs_points=lcfs_points,
        reference_lcfs_points=reference_lcfs_points,
        ip=ip_values,
        reference_ip=reference_ip_values,
        valid=observation_valid,
        env_specs_json=np.asarray(
            json.dumps(_jsonable(ctx["env_specs"]), ensure_ascii=False)
        ),
    )
    returns = [item["return"] for item in episode_summaries]
    summary = {
        "model_path": str(ctx["model_path"]),
        "run_dir": str(ctx["run_dir"]),
        "configs_path": str(ctx["configs_path"]),
        "vecnormalize_path": (
            str(ctx["vecnormalize_path"]) if ctx["vecnormalize_path"] else None
        ),
        "reference_path": str(ctx["reference_path"]) if ctx["reference_path"] else None,
        "reference_mode": ctx["reference_mode"],
        "eval_env_config": ctx["eval_env_config"],
        "env_specs": ctx["env_specs"],
        "episodes": episode_summaries,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "csv_path": str(csv_path),
        "observations_path": str(observations_path),
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"model: {ctx['model_path']}")
    print(f"configs: {ctx['configs_path']}")
    if ctx["vecnormalize_path"]:
        print(f"vecnormalize: {ctx['vecnormalize_path']}")
    print(f"saved step rewards: {csv_path}")
    print(f"saved rollout observations: {observations_path}")
    print(f"saved summary: {summary_path}")
    print(
        f"mean_return={summary['mean_return']:.6f}, episodes={len(episode_summaries)}"
    )
    return 0


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Folder name under results/, e.g. 20260615_123135_f2b_direct",
    )
    parser.add_argument(
        "--eval_env_config",
        type=str,
        required=True,
        help="config file name under rollout/configs",
    )
    parser.add_argument(
        "--checkpoint_step", type=int, default=-1, help="-1 for the latest checkpoint"
    )
    parser.add_argument("--eval_episode_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--csv-name", type=str, default="step_reward_components.csv")
    parser.add_argument("--summary-name", type=str, default="summary.json")
    parser.add_argument("--save-actions", action="store_true")
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument(
        "--deterministic", dest="deterministic", action="store_true", default=True
    )
    deterministic.add_argument(
        "--stochastic", dest="deterministic", action="store_false"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy and save every step reward component."
    )
    add_args(parser)
    return evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
