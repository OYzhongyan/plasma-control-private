"""Evaluate F1 policy sensitivity to perturbed target LCFS shapes.

This script keeps the rollout initial shot separate from the target LCFS source:
target LCFS variants are sampled from reset observations of target-source shots
(for example 15892_400/500), then injected as F1 trajectory targets while the
policy is rolled out on the requested F1 rollout shots.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from environment import HFMSimulator  # noqa: E402
from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list  # noqa: E402
from score import config as score_config  # noqa: E402
from score.evaluate_core import evaluate  # noqa: E402
from training.distill_noise_student import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_VECNORMALIZE,
    FeatureSequenceBuilder,
    build_raw_env,
    load_vecnormalize_stats,
    normalize_obs,
    parse_float_list,
    parse_int_spec,
)
from training.features import FeatureMetadata  # noqa: E402
from training.references import TargetReference, build_reference, load_target_reference, normalize_task  # noqa: E402
from training.train_ppo import (  # noqa: E402
    ResetPerturbationSampler,
    _rooted_path,
    load_train_config,
    load_yaml,
    make_env_config,
    parse_ports,
)


DEFAULT_MODEL = PROJECT_ROOT / "results/f1_direct_target_progress_regression_continue_lcfs_targets/ppo_f1_latest.zip"
DEFAULT_TARGET = PROJECT_ROOT / "evaluation/results/target.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/f1_target_lcfs_ablation"


@dataclass
class TargetVariant:
    variant_id: str
    source_shot: str
    source_seed: int
    reset_params: dict[str, float]
    lcfs: np.ndarray
    target_lcfs_delta_cm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 rollout ablation for target LCFS perturbations.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-label", type=str, default="policy")
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ports", type=str, default=None)
    parser.add_argument("--rollout-shots", type=str, default="13844_500,13844_600")
    parser.add_argument("--rollout-seeds", type=str, default="0")
    parser.add_argument("--target-source-shots", type=str, default="15892_400,15892_500")
    parser.add_argument("--target-seeds", type=str, default="0-3")
    parser.add_argument("--include-official-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-perturb-apply-mode", choices=["all", "one"], default="all")
    parser.add_argument("--target-signeo-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--target-bp-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--target-q0-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--target-bp-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--target-q0-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--rollout-reset-perturb", action="store_true")
    parser.add_argument("--rollout-perturb-config", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--strict-rollout", action="store_true")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _range_pair(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError(f"range must contain exactly two numbers, got {spec!r}")
    lo, hi = values
    if lo > hi:
        raise ValueError(f"range low must be <= high, got {spec!r}")
    return [lo, hi]


def _split_csv(spec: str) -> list[str]:
    values = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty csv spec: {spec!r}")
    return values


def target_reset_perturbation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_mode": str(args.target_perturb_apply_mode),
        "signeo_scale_range": _range_pair(args.target_signeo_scale_range),
        "bp_scale_range": _range_pair(args.target_bp_scale_range),
        "q0_scale_range": _range_pair(args.target_q0_scale_range),
        "bp_delta_range": _range_pair(args.target_bp_delta_range),
        "q0_delta_range": _range_pair(args.target_q0_delta_range),
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def lcfs_rms_delta_cm(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float64).reshape(32, 2)
    arr_b = np.asarray(b, dtype=np.float64).reshape(32, 2)
    return float(np.sqrt(np.mean(np.sum((arr_a - arr_b) ** 2, axis=1))) * 100.0)


def build_context(args: argparse.Namespace, metadata: FeatureMetadata) -> dict[str, Any]:
    train_config = load_train_config(_rooted_path(args.train_config))
    task = normalize_task(train_config.get("task", "F1"))
    if task != "F1":
        raise ValueError(f"this script expects F1, got {task}")
    max_steps = int(args.max_steps or train_config.get("max_steps") or 300)
    env_config_path = _rooted_path(train_config.get("env_config"), PROJECT_ROOT / "configs/env_default.yaml")
    target_path = _rooted_path(args.target)
    target_all = json.loads(target_path.read_text(encoding="utf-8"))
    official_ref = load_target_reference(target_path, task="F1")
    return {
        "train_config": train_config,
        "base_config": load_yaml(env_config_path),
        "max_steps": max_steps,
        "metadata": metadata,
        "target_all": target_all,
        "official_ref": official_ref,
        "strict_rollout": bool(args.strict_rollout),
    }


def sample_target_variants(
    *,
    args: argparse.Namespace,
    ctx: dict[str, Any],
    port: int,
) -> list[TargetVariant]:
    variants: list[TargetVariant] = []
    official_lcfs = np.asarray(ctx["target_all"]["F1"]["lcfs_points"], dtype=np.float64)
    if args.include_official_target:
        variants.append(
            TargetVariant(
                variant_id="official",
                source_shot="official",
                source_seed=-1,
                reset_params={},
                lcfs=official_lcfs.copy(),
                target_lcfs_delta_cm=0.0,
            )
        )

    perturb = target_reset_perturbation(args)
    for source_shot in _split_csv(args.target_source_shots):
        for source_seed in parse_int_spec(args.target_seeds):
            sampler = ResetPerturbationSampler(
                shot_id=source_shot,
                reset_perturbation=perturb,
                seed=int(source_seed),
            )
            reset_params = sampler.sample()
            cfg = make_env_config(
                ctx["base_config"],
                port=port,
                shot_id=source_shot,
                max_steps=1,
                reference_mode="hold_initial",
            )
            env = HFMSimulator(cfg, reward_fn=None)
            try:
                obs, _info = env.reset(
                    seed=int(source_seed),
                    options={"reference_mode": "hold", "reset_params": reset_params},
                )
                lcfs = np.asarray(obs["lcfs_points"], dtype=np.float64).reshape(32, 2)
            except Exception:
                env.close()
                if args.strict_rollout:
                    raise
                continue
            finally:
                env.close()
            variant_id = f"{source_shot}_seed{source_seed}"
            variants.append(
                TargetVariant(
                    variant_id=variant_id,
                    source_shot=source_shot,
                    source_seed=int(source_seed),
                    reset_params={k: float(v) for k, v in reset_params.items()},
                    lcfs=lcfs,
                    target_lcfs_delta_cm=lcfs_rms_delta_cm(lcfs, official_lcfs),
                )
            )
            print(
                f"target variant={variant_id} reset={reset_params} "
                f"delta_cm={variants[-1].target_lcfs_delta_cm:.3f}",
                flush=True,
            )
    if not variants:
        raise RuntimeError("no target LCFS variants were sampled")
    return variants


def variant_target_and_reference(ctx: dict[str, Any], variant: TargetVariant) -> tuple[dict[str, Any], TargetReference, dict[str, Any]]:
    target_all = copy.deepcopy(ctx["target_all"])
    target_all["F1"]["lcfs_points"] = np.asarray(variant.lcfs, dtype=np.float64).tolist()
    official_ref: TargetReference = ctx["official_ref"]
    target_ref = TargetReference(lcfs=np.asarray(variant.lcfs, dtype=np.float64), ip_final=official_ref.ip_final)
    reference = build_reference("F1", target_ref, int(ctx["max_steps"]))
    return target_all, target_ref, reference


def coil_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    icoil = np.asarray(trajectory.get("Icoil", []), dtype=np.float64)
    if icoil.ndim != 2 or icoil.shape[0] == 0:
        return {
            "max_coil_margin": float("nan"),
            "mean_coil_margin": float("nan"),
            "first_overlimit_step": -1,
            "overlimit_steps": 0,
        }
    limits = score_config.COIL_LIMITS_A[: icoil.shape[1]]
    margins = np.max(np.abs(icoil) / limits[None, :], axis=1)
    over = margins > 1.0
    first = int(np.argmax(over) + 1) if np.any(over) else -1
    return {
        "max_coil_margin": float(np.max(margins)),
        "mean_coil_margin": float(np.mean(margins)),
        "first_overlimit_step": first,
        "overlimit_steps": int(np.sum(over)),
    }


def rollout_one(
    *,
    model,
    model_label: str,
    ctx: dict[str, Any],
    variant: TargetVariant,
    shot_id: str,
    seed: int,
    port: int,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    clip_obs: float,
    eps: float,
    rollout_reset_perturbation: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: FeatureMetadata = ctx["metadata"]
    target_all, target_ref, reference = variant_target_and_reference(ctx, variant)
    env = build_raw_env(
        train_config=ctx["train_config"],
        base_config=ctx["base_config"],
        reference=reference,
        target=target_ref,
        shot_id=shot_id,
        port=port,
        max_steps=int(ctx["max_steps"]),
        seed=seed,
        action_units=metadata.action_units,
        reset_perturbation=rollout_reset_perturbation,
    )
    builder = FeatureSequenceBuilder(metadata)
    obs_list: list[dict[str, Any]] = []
    rewards: list[float] = []
    error = ""
    try:
        raw_obs, _ = env.reset(seed=seed)
        feature = builder.reset(dict(raw_obs))
        for _step in range(1, int(ctx["max_steps"]) + 1):
            norm = normalize_obs(feature, obs_mean, obs_var, clip_obs, eps)
            action, _ = model.predict(norm.reshape(1, -1), deterministic=True)
            action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            raw_next, reward, terminated, truncated, _info = env.step(action7)
            obs_list.append(dict(raw_next))
            rewards.append(float(reward))
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
    row = {
        "model": model_label,
        "variant_id": variant.variant_id,
        "target_source_shot": variant.source_shot,
        "target_source_seed": variant.source_seed,
        "target_lcfs_delta_cm": variant.target_lcfs_delta_cm,
        "target_reset_params": json.dumps(variant.reset_params, sort_keys=True),
        "rollout_shot": shot_id,
        "rollout_seed": int(seed),
        "n_steps": len(obs_list),
        "f_surv": len(obs_list) / max(float(ctx["max_steps"]), 1.0),
        "score_f1": float(result["task_scores"].get("F1", 0.0)),
        "score_ip": float(metric_scores.get("Ip", 0.0)),
        "score_lcfs": float(metric_scores.get("LCFS", 0.0)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "error": error,
    }
    row.update(coil_metrics(trajectory))
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(row["variant_id"], row["target_source_shot"]) for row in rows})
    for variant_id, source_shot in groups:
        group = [row for row in rows if row["variant_id"] == variant_id and row["target_source_shot"] == source_shot]

        def mean(key: str) -> float:
            vals = [float(row[key]) for row in group if key in row and np.isfinite(float(row[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        out.append(
            {
                "variant_id": variant_id,
                "target_source_shot": source_shot,
                "episodes": len(group),
                "failures": sum(1 for row in group if row.get("error")),
                "target_lcfs_delta_cm": mean("target_lcfs_delta_cm"),
                "score_f1_mean": mean("score_f1"),
                "score_ip_mean": mean("score_ip"),
                "score_lcfs_mean": mean("score_lcfs"),
                "f_surv_mean": mean("f_surv"),
                "max_coil_margin_mean": mean("max_coil_margin"),
                "max_coil_margin_max": float(np.nanmax([float(row["max_coil_margin"]) for row in group])),
                "overlimit_episode_count": sum(1 for row in group if int(row.get("first_overlimit_step", -1)) > 0),
                "first_overlimit_step_min": min(
                    [int(row["first_overlimit_step"]) for row in group if int(row.get("first_overlimit_step", -1)) > 0],
                    default=-1,
                ),
            }
        )
    return out


def save_target_variants(output_dir: Path, variants: list[TargetVariant]) -> None:
    payload = [
        {
            "variant_id": v.variant_id,
            "source_shot": v.source_shot,
            "source_seed": v.source_seed,
            "reset_params": v.reset_params,
            "target_lcfs_delta_cm": v.target_lcfs_delta_cm,
            "lcfs_points": v.lcfs,
        }
        for v in variants
    ]
    (output_dir / "target_lcfs_variants.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "target_lcfs_variants.npz",
        variant_ids=np.asarray([v.variant_id for v in variants]),
        source_shots=np.asarray([v.source_shot for v in variants]),
        source_seeds=np.asarray([v.source_seed for v in variants], dtype=np.int64),
        target_lcfs_delta_cm=np.asarray([v.target_lcfs_delta_cm for v in variants], dtype=np.float64),
        lcfs_points=np.stack([v.lcfs for v in variants], axis=0),
    )


def plot_target_variants(output_dir: Path, variants: list[TargetVariant]) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    official = next((v for v in variants if v.variant_id == "official"), variants[0])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(official.lcfs[:, 0], official.lcfs[:, 1], color="black", linewidth=2.5, label="official target")
    for variant in variants:
        if variant.variant_id == "official":
            continue
        ax.plot(
            variant.lcfs[:, 0],
            variant.lcfs[:, 1],
            linewidth=1.2,
            alpha=0.75,
            label=f"{variant.variant_id} ({variant.target_lcfs_delta_cm:.1f} cm)",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("F1 target LCFS variants")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "target_lcfs_variants.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir = _rooted_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    metadata = FeatureMetadata.load(_rooted_path(args.metadata))
    ctx = build_context(args, metadata)
    obs_mean, obs_var, clip_obs, eps = load_vecnormalize_stats(_rooted_path(args.vecnormalize))
    port = parse_ports(args.ports or ctx["train_config"].get("ports") or "5558")[0]
    variants = sample_target_variants(args=args, ctx=ctx, port=port)
    save_target_variants(args.output_dir, variants)
    if args.plot:
        plot_target_variants(args.output_dir, variants)

    model = PPO.load(str(_rooted_path(args.model)), device=args.device)
    rollout_shots = _split_csv(args.rollout_shots)
    rollout_seeds = parse_int_spec(args.rollout_seeds)
    rollout_reset_perturbation: dict[str, Any] | None = {"enabled": False}
    if args.rollout_reset_perturb:
        if args.rollout_perturb_config:
            rollout_reset_perturbation = json.loads(args.rollout_perturb_config)
        else:
            rollout_reset_perturbation = ctx["train_config"].get("reset_perturbation") or {"enabled": False}
    print(f"rollout_reset_perturbation={rollout_reset_perturbation}", flush=True)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        for shot_id in rollout_shots:
            for seed in rollout_seeds:
                print(
                    f"rollout variant={variant.variant_id} source={variant.source_shot} "
                    f"shot={shot_id} seed={seed}",
                    flush=True,
                )
                row = rollout_one(
                    model=model,
                    model_label=str(args.model_label),
                    ctx=ctx,
                    variant=variant,
                    shot_id=shot_id,
                    seed=int(seed),
                    port=port,
                    obs_mean=obs_mean,
                    obs_var=obs_var,
                    clip_obs=clip_obs,
                    eps=eps,
                    rollout_reset_perturbation=rollout_reset_perturbation,
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
    print(f"rows: {args.output_dir / 'rollout_rows.csv'}")
    print(f"summary: {args.output_dir / 'rollout_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
