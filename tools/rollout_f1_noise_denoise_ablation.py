"""Compare F1 rollout score with and without the learned denoising layer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list  # noqa: E402
from score.evaluate_core import evaluate  # noqa: E402
from training.distill_noise_student import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_VECNORMALIZE,
    FeatureSequenceBuilder,
    build_raw_env,
    load_vecnormalize_stats,
    noise_feature_mask,
    normalize_obs,
    parse_float_list,
    parse_int_spec,
)
from training.features import FeatureMetadata  # noqa: E402
from training.references import build_reference, load_target_reference, normalize_task  # noqa: E402
from training.train_ppo import _rooted_path, load_train_config, load_yaml, parse_ports  # noqa: E402


DEFAULT_DENOISE_MODEL = PROJECT_ROOT / "results/f1_direct_target_noise_distill/ppo_f1_student_latest.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/f1_noise_denoise_ablation"
DEFAULT_TARGET = PROJECT_ROOT / "evaluation/results/target.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 noise rollout A/B test for denoising feature extractor.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--denoise-model", type=Path, default=DEFAULT_DENOISE_MODEL)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ports", type=str, default=None)
    parser.add_argument("--shots", type=str, default="13844_500,13844_600")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--noise-stds", type=str, default="0.1, 0.2, 0.3, 0.4, 0.5")
    parser.add_argument("--noise-feature-groups", type=str, default="ip,rz,lcfs,history_ip,history_rz")
    parser.add_argument(
        "--perturb-mode",
        choices=["off", "manual", "config"],
        default="off",
        help="off disables reset perturbation; manual uses CLI ranges; config uses reset_perturbation from --train-config.",
    )
    parser.add_argument("--perturb-apply-mode", choices=["all", "one"], default="all")
    parser.add_argument("--signeo-scale-range", type=str, default="0.5,1.5")
    parser.add_argument("--bp-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--q0-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--bp-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--q0-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict-rollout", action="store_true")
    return parser.parse_args()



def _range_pair(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError(f"range must contain exactly two numbers, got {spec!r}")
    lo, hi = values
    if lo > hi:
        raise ValueError(f"range low must be <= high, got {spec!r}")
    return [lo, hi]


def manual_reset_perturbation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_mode": str(args.perturb_apply_mode),
        "signeo_scale_range": _range_pair(args.signeo_scale_range),
        "bp_scale_range": _range_pair(args.bp_scale_range),
        "q0_scale_range": _range_pair(args.q0_scale_range),
        "bp_delta_range": _range_pair(args.bp_delta_range),
        "q0_delta_range": _range_pair(args.q0_delta_range),
    }

def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_component(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _build_context(args: argparse.Namespace, metadata: FeatureMetadata) -> dict[str, Any]:
    train_config = load_train_config(_rooted_path(args.train_config))
    task = normalize_task(train_config.get("task", "F1"))
    if task != "F1":
        raise ValueError(f"this ablation script expects F1, got {task}")
    target_ref = load_target_reference(_rooted_path(train_config.get("reference"), DEFAULT_TARGET), task=task)
    max_steps = int(args.max_steps or train_config.get("max_steps") or 300)
    reference = None
    if str(train_config.get("reference_mode", "trajectory")) != "hold_initial":
        reference = build_reference(task, target_ref, max_steps)
    env_config_path = _rooted_path(train_config.get("env_config"), PROJECT_ROOT / "configs/env_default.yaml")
    return {
        "train_config": train_config,
        "task": task,
        "target_ref": target_ref,
        "reference": reference,
        "base_config": load_yaml(env_config_path),
        "max_steps": max_steps,
        "metadata": metadata,
    }


def rollout_one(
    *,
    model,
    label: str,
    noise_std: float,
    ctx: dict[str, Any],
    shot_id: str,
    seed: int,
    port: int,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    clip_obs: float,
    eps: float,
    target_all: dict[str, Any],
    feature_noise_mask: np.ndarray,
) -> dict[str, Any]:
    metadata: FeatureMetadata = ctx["metadata"]
    env = build_raw_env(
        train_config=ctx["train_config"],
        base_config=ctx["base_config"],
        reference=ctx["reference"],
        target=ctx["target_ref"],
        shot_id=shot_id,
        port=port,
        max_steps=int(ctx["max_steps"]),
        seed=seed,
        action_units=metadata.action_units,
        reset_perturbation=ctx.get("reset_perturbation"),
    )
    builder = FeatureSequenceBuilder(metadata)
    rng = np.random.default_rng(seed + int(round(float(noise_std) * 1_000_000)) + (0 if label == "teacher" else 10_000))
    obs_list: list[dict[str, Any]] = []
    component_rows: list[dict[str, float]] = []
    rewards: list[float] = []
    error: str | None = None
    try:
        raw_obs, _ = env.reset(seed=seed)
        feature = builder.reset(dict(raw_obs))
        for _step in range(1, int(ctx["max_steps"]) + 1):
            clean_norm = normalize_obs(feature, obs_mean, obs_var, clip_obs, eps)
            noisy_norm = clean_norm + (
                rng.normal(0.0, float(noise_std), size=clean_norm.shape).astype(np.float32)
                * feature_noise_mask
            )
            action, _ = model.predict(noisy_norm.reshape(1, -1), deterministic=True)
            action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            raw_next, reward, terminated, truncated, info = env.step(action7)
            obs_list.append(dict(raw_next))
            rewards.append(float(reward))
            components = dict((info or {}).get("reward_components", {}) or {})
            component_rows.append({k: float(v) for k, v in components.items() if isinstance(v, (int, float, np.number))})
            feature = builder.step(action7, dict(raw_next))
            if terminated or truncated:
                break
    except Exception as exc:
        if ctx.get("strict_rollout"):
            raise
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    trajectory = extract_semifinal_trajectory_from_obs_list(obs_list, task_id="F1", target_obs=target_all.get("F1"))
    result = evaluate({"F1": {"trajectory": trajectory, "timeout": False}}, target_all, task_ids=["F1"])
    metric_scores = result["task_metric_scores"].get("F1", {})
    return {
        "policy": label,
        "noise_std": float(noise_std),
        "shot_id": shot_id,
        "seed": int(seed),
        "n_steps": len(obs_list),
        "f_surv": len(obs_list) / max(float(ctx["max_steps"]), 1.0),
        "score_f1": float(result["task_scores"].get("F1", 0.0)),
        "score_ip": float(metric_scores.get("Ip", 0.0)),
        "score_lcfs": float(metric_scores.get("LCFS", 0.0)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "mean_ip_abs_error_ka": _mean_component(component_rows, "ip_abs_error") / 1000.0,
        "mean_lcfs_cm": _mean_component(component_rows, "lcfs_cm"),
        "mean_topology_score": _mean_component(component_rows, "topology_score"),
        "perturb_mode": str(ctx.get("perturb_mode", "off")),
        "error": error or "",
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(row["policy"], row["noise_std"]) for row in rows}, key=lambda x: (float(x[1]), str(x[0])))
    for policy, noise_std in groups:
        group = [row for row in rows if row["policy"] == policy and row["noise_std"] == noise_std]
        out.append(
            {
                "policy": policy,
                "noise_std": noise_std,
                "episodes": len(group),
                "failures": sum(1 for row in group if row.get("error")),
                "score_f1_mean": float(np.mean([row["score_f1"] for row in group])),
                "score_f1_std": float(np.std([row["score_f1"] for row in group])),
                "score_ip_mean": float(np.mean([row["score_ip"] for row in group])),
                "score_lcfs_mean": float(np.mean([row["score_lcfs"] for row in group])),
                "f_surv_mean": float(np.mean([row["f_surv"] for row in group])),
                "ip_abs_error_ka_mean": float(np.nanmean([row["mean_ip_abs_error_ka"] for row in group])),
                "lcfs_cm_mean": float(np.nanmean([row["mean_lcfs_cm"] for row in group])),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    args.output_dir = _rooted_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    metadata = FeatureMetadata.load(_rooted_path(args.metadata))
    ctx = _build_context(args, metadata)
    ctx["strict_rollout"] = bool(args.strict_rollout)
    ctx["perturb_mode"] = str(args.perturb_mode)
    if str(args.perturb_mode) == "config":
        ctx["reset_perturbation"] = ctx["train_config"].get("reset_perturbation") or {"enabled": False}
    elif str(args.perturb_mode) == "manual":
        ctx["reset_perturbation"] = manual_reset_perturbation(args)
    else:
        ctx["reset_perturbation"] = {"enabled": False}
    obs_mean, obs_var, clip_obs, eps = load_vecnormalize_stats(_rooted_path(args.vecnormalize))
    target_all = json.loads(_rooted_path(args.target).read_text(encoding="utf-8"))
    shots = [part.strip() for part in str(args.shots).split(",") if part.strip()]
    seeds = parse_int_spec(args.seeds)
    noise_stds = parse_float_list(args.noise_stds)
    feature_noise_mask = noise_feature_mask(metadata, args.noise_feature_groups).astype(np.float32)
    print(
        f"noise feature groups={args.noise_feature_groups} "
        f"dims={int(feature_noise_mask.sum())}/{feature_noise_mask.size}",
        flush=True,
    )
    port = parse_ports(args.ports or ctx["train_config"].get("ports") or "5558")[0]
    print(f"perturb_mode={args.perturb_mode} reset_perturbation={ctx['reset_perturbation']}", flush=True)

    models = {
        "teacher": PPO.load(str(_rooted_path(args.teacher_model)), device=args.device),
        "denoise": PPO.load(str(_rooted_path(args.denoise_model)), device=args.device),
    }
    rows: list[dict[str, Any]] = []
    for noise_std in noise_stds:
        for label, model in models.items():
            for shot_index, shot_id in enumerate(shots):
                for seed in seeds:
                    episode_seed = int(seed) + 0 * shot_index
                    print(f"rollout policy={label} noise={noise_std} shot={shot_id} seed={episode_seed}", flush=True)
                    row = rollout_one(
                        model=model,
                        label=label,
                        noise_std=float(noise_std),
                        ctx=ctx,
                        shot_id=shot_id,
                        seed=episode_seed,
                        port=port,
                        obs_mean=obs_mean,
                        obs_var=obs_var,
                        clip_obs=clip_obs,
                        eps=eps,
                        target_all=target_all,
                        feature_noise_mask=feature_noise_mask,
                    )
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)

    summary = aggregate(rows)
    _write_csv(args.output_dir / "rollout_rows.csv", rows)
    _write_csv(args.output_dir / "rollout_summary.csv", summary)
    (args.output_dir / "rollout_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (args.output_dir / "rollout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"summary: {args.output_dir / 'rollout_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
