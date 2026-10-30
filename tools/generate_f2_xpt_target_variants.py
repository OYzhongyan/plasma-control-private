"""Generate F2 XPT target variants from HFM reset observations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import HFMSimulator  # noqa: E402
from environment.xpt_utils import extract_target_strike_points, extract_target_xpoints  # noqa: E402
from training.train_ppo import ResetPerturbationSampler, _rooted_path, load_yaml, make_env_config, parse_ports  # noqa: E402


def parse_int_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty integer spec: {spec!r}")
    return out


def split_csv(spec: str) -> list[str]:
    out = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not out:
        raise ValueError(f"empty csv spec: {spec!r}")
    return out


def range_pair(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError(f"range must contain two values: {spec!r}")
    lo, hi = values
    if lo > hi:
        raise ValueError(f"range low must be <= high: {spec!r}")
    return [lo, hi]


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def reset_obs(base_config: dict[str, Any], *, port: int, shot_id: str, seed: int, reset_params: dict[str, float]) -> dict[str, Any]:
    cfg = make_env_config(base_config, port=port, shot_id=shot_id, max_steps=1, reference_mode="hold_initial")
    env = HFMSimulator(cfg, reward_fn=None)
    try:
        obs, _ = env.reset(seed=int(seed), options={"reference_mode": "hold", "reset_params": dict(reset_params)})
        return dict(obs)
    finally:
        env.close()


def build_official_variant(target_path: Path) -> dict[str, Any]:
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target_task = target.get("F2a") or target.get("F2")
    if not isinstance(target_task, dict):
        raise ValueError(f"target file does not contain F2a/F2 data: {target_path}")
    x_main = np.asarray(target_task["Xpt_main"], dtype=np.float64).reshape(2, 2)
    x_sec = np.asarray(target_task["Xpt_sec"], dtype=np.float64).reshape(2, 2)
    strike = np.asarray(target_task["strike"], dtype=np.float64).reshape(8, 2)
    r_x = np.asarray([x_sec[0, 0], x_main[0, 0], x_main[1, 0], x_sec[1, 0]], dtype=np.float64)
    z_x = np.asarray([x_sec[0, 1], x_main[0, 1], x_main[1, 1], x_sec[1, 1]], dtype=np.float64)
    x_valid = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    strike_valid = np.ones(8, dtype=np.float64)
    return {
        "variant_id": "public_official_target",
        "source_shot": "public_official_target",
        "source_seed": -1,
        "reset_params": {},
        "lcfs_points": np.asarray(target_task["lcfs_points"], dtype=np.float64).reshape(32, 2),
        "reference_rX": r_x,
        "reference_zX": z_x,
        "reference_x_valid": x_valid,
        "reference_strike_r": strike[:, 0],
        "reference_strike_z": strike[:, 1],
        "reference_strike_valid": strike_valid,
        "reference_nX": 4.0,
        "reference_n_strike": 8.0,
    }


def build_variant(variant_id: str, source_shot: str, seed: int, reset_params: dict[str, float], obs: dict[str, Any]) -> dict[str, Any]:
    r_x, z_x, x_valid = extract_target_xpoints(obs, slots=4)
    r_s, z_s, strike_valid, n_strike = extract_target_strike_points(obs, strike_slots=8)
    return {
        "variant_id": variant_id,
        "source_shot": source_shot,
        "source_seed": int(seed),
        "reset_params": {k: float(v) for k, v in reset_params.items()},
        "lcfs_points": np.asarray(obs["lcfs_points"], dtype=np.float64).reshape(32, 2),
        "reference_rX": np.asarray(r_x, dtype=np.float64),
        "reference_zX": np.asarray(z_x, dtype=np.float64),
        "reference_x_valid": np.asarray(x_valid, dtype=np.float64),
        "reference_strike_r": np.asarray(r_s, dtype=np.float64),
        "reference_strike_z": np.asarray(z_s, dtype=np.float64),
        "reference_strike_valid": np.asarray(strike_valid, dtype=np.float64),
        "reference_nX": float(np.sum(np.asarray(x_valid) > 0.5)),
        "reference_n_strike": float(n_strike),
    }


def plot_variants(variants: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 8.0))
    colors = ["black", "#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    for idx, variant in enumerate(variants):
        color = colors[idx % len(colors)]
        variant_id = str(variant.get("variant_id", f"variant_{idx}"))
        lcfs = np.asarray(variant["lcfs_points"], dtype=np.float64).reshape(32, 2)
        r_x = np.asarray(variant["reference_rX"], dtype=np.float64).reshape(-1)
        z_x = np.asarray(variant["reference_zX"], dtype=np.float64).reshape(-1)
        x_valid = np.asarray(variant.get("reference_x_valid", np.ones_like(r_x)), dtype=np.float64).reshape(-1)
        r_s = np.asarray(variant["reference_strike_r"], dtype=np.float64).reshape(-1)
        z_s = np.asarray(variant["reference_strike_z"], dtype=np.float64).reshape(-1)
        s_valid = np.asarray(variant.get("reference_strike_valid", np.ones_like(r_s)), dtype=np.float64).reshape(-1)
        lw = 2.4 if idx == 0 else 1.3
        alpha = 1.0 if idx == 0 else 0.75
        ax.plot(lcfs[:, 0], lcfs[:, 1], color=color, linewidth=lw, alpha=alpha, label=variant_id)
        x_mask = x_valid > 0.5
        if np.any(x_mask):
            ax.scatter(r_x[x_mask], z_x[x_mask], marker="x", s=70, color=color, linewidths=2.0, alpha=alpha)
        s_mask = s_valid > 0.5
        if np.any(s_mask):
            ax.scatter(r_s[s_mask], z_s[s_mask], marker="o", s=28, facecolors="none", edgecolors=color, alpha=alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("Generated F2 XPT target variants")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _build_jobs(
    *,
    total_count: int,
    include_public: bool,
    source_shots: list[str],
    seed: int,
    perturbation: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    current_count = 1 if include_public else 0
    order = current_count
    for shot_id in source_shots:
        if current_count + len(jobs) >= total_count:
            break
        jobs.append({
            "order": order,
            "variant_id": f"{shot_id}_official_reset",
            "shot_id": shot_id,
            "seed": -1,
            "reset_params": {},
        })
        order += 1
    per_shot_counts = {shot_id: 0 for shot_id in source_shots}
    perturb_index = 0
    while current_count + len(jobs) < total_count:
        shot_id = source_shots[perturb_index % len(source_shots)]
        job_seed = int(seed) + perturb_index
        sampler = ResetPerturbationSampler(shot_id=shot_id, reset_perturbation=perturbation, seed=job_seed)
        per_shot_idx = per_shot_counts[shot_id]
        per_shot_counts[shot_id] += 1
        jobs.append({
            "order": order,
            "variant_id": f"{shot_id}_perturb{per_shot_idx}",
            "shot_id": shot_id,
            "seed": job_seed,
            "reset_params": sampler.sample(),
        })
        order += 1
        perturb_index += 1
    return jobs


def _run_jobs(
    *,
    jobs: list[dict[str, Any]],
    ports: list[int],
    base_config: dict[str, Any],
    strict: bool,
) -> list[dict[str, Any]]:
    results: list[tuple[int, dict[str, Any]]] = []
    if not jobs:
        return []
    n_ports = max(1, len(ports))

    def worker(job: dict[str, Any], port: int) -> tuple[int, dict[str, Any]]:
        obs = reset_obs(
            base_config,
            port=port,
            shot_id=str(job["shot_id"]),
            seed=max(int(job["seed"]), 0),
            reset_params=dict(job["reset_params"]),
        )
        variant = build_variant(str(job["variant_id"]), str(job["shot_id"]), int(job["seed"]), dict(job["reset_params"]), obs)
        return int(job["order"]), variant

    for start in range(0, len(jobs), n_ports):
        batch = jobs[start:start + n_ports]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(worker, job, int(ports[idx])): job
                for idx, job in enumerate(batch)
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    order, variant = future.result()
                except Exception as exc:
                    if strict:
                        raise
                    print(f"skip {job['variant_id']}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                results.append((order, variant))
                print(
                    f"generated {variant['variant_id']} nX={variant['reference_nX']:.0f} "
                    f"n_strike={variant['reference_n_strike']:.0f} reset={variant['reset_params']}",
                    flush=True,
                )
    return [variant for _, variant in sorted(results, key=lambda item: item[0])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate F2 XPT target variants from reset state.")
    parser.add_argument("--output", type=Path, default=ROOT / "training/targets/f2_xpt_13906_target_variants.json")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env_default.yaml")
    parser.add_argument("--target", type=Path, default=ROOT / "evaluation/results/target.json")
    parser.add_argument("--ports", type=str, default="5558")
    parser.add_argument("--source-shot", type=str, default="13906_500", help="Backward-compatible single source shot.")
    parser.add_argument("--source-shots", type=str, default=None, help="Comma-separated source shots. Overrides --source-shot when set.")
    parser.add_argument("--total-count", type=int, default=64, help="Total number of target variants in the output JSON, including official targets.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic perturbation sampling.")
    parser.add_argument("--include-public-official-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--signeo-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--bp-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--q0-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output = _rooted_path(args.output)
    env_config = _rooted_path(args.env_config)
    target_path = _rooted_path(args.target)
    if output is None or env_config is None or target_path is None:
        raise ValueError("paths could not be resolved")
    output.parent.mkdir(parents=True, exist_ok=True)

    total_count = int(args.total_count)
    if total_count <= 0:
        raise ValueError("--total-count must be positive")

    base_config = load_yaml(env_config)
    ports = parse_ports(args.ports)
    source_shots = split_csv(args.source_shots if args.source_shots is not None else args.source_shot)
    perturbation = {
        "enabled": True,
        "apply_mode": "all",
        "signeo_scale_range": range_pair(args.signeo_scale_range),
        "bp_scale_range": range_pair(args.bp_scale_range),
        "q0_scale_range": range_pair(args.q0_scale_range),
    }

    variants: list[dict[str, Any]] = []
    if args.include_public_official_target:
        variants.append(build_official_variant(target_path))

    jobs = _build_jobs(
        total_count=total_count,
        include_public=bool(args.include_public_official_target),
        source_shots=source_shots,
        seed=int(args.seed),
        perturbation=perturbation,
    )
    variants.extend(_run_jobs(
        jobs=jobs,
        ports=ports,
        base_config=base_config,
        strict=bool(args.strict),
    ))
    variants = variants[:total_count]

    payload = {
        "schema_version": 1,
        "task": "F2",
        "description": "F2 XPT target variants generated from HFM reset LCFS/X points/strike points.",
        "source_shots": source_shots,
        "seed": int(args.seed),
        "total_count": int(total_count),
        "perturbation": perturbation,
        "variants": variants,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(f"wrote {len(variants)} variants: {output}", flush=True)
    if args.plot:
        plot_output = _rooted_path(args.plot_output) if args.plot_output is not None else output.with_suffix(".png")
        if plot_output is None:
            raise ValueError("plot output path could not be resolved")
        plot_variants(variants, plot_output)
        print(f"wrote XPT plot: {plot_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
