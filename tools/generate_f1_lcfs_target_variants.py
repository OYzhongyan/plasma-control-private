"""Generate F1 LCFS target variants from HFM reset observations."""

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


def lcfs_delta_cm(lcfs: np.ndarray, public_lcfs: np.ndarray) -> float:
    lcfs = np.asarray(lcfs, dtype=np.float64).reshape(32, 2)
    public_lcfs = np.asarray(public_lcfs, dtype=np.float64).reshape(32, 2)
    return float(np.sqrt(np.mean(np.sum((lcfs - public_lcfs) ** 2, axis=1))) * 100.0)


def plot_lcfs_variants(variants: list[dict[str, Any]], *, output: Path, public_lcfs: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    public_arr = np.asarray(public_lcfs, dtype=np.float64).reshape(32, 2)
    ax.plot(public_arr[:, 0], public_arr[:, 1], color="black", linewidth=2.5, label="public official target")

    for variant in variants:
        variant_id = str(variant.get("variant_id", "variant"))
        if variant_id == "public_official_target":
            continue
        lcfs = np.asarray(variant["lcfs_points"], dtype=np.float64).reshape(32, 2)
        delta_cm = float(variant.get("target_lcfs_delta_cm", lcfs_delta_cm(lcfs, public_arr)))
        source_shot = str(variant.get("source_shot", ""))
        label = f"{variant_id} ({delta_cm:.1f} cm)"
        if source_shot and source_shot not in variant_id:
            label = f"{source_shot}: {label}"
        ax.plot(lcfs[:, 0], lcfs[:, 1], linewidth=1.2, alpha=0.78, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("Generated F1 target LCFS variants")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def reset_lcfs(base_config: dict[str, Any], *, port: int, shot_id: str, seed: int, reset_params: dict[str, float]) -> np.ndarray:
    cfg = make_env_config(base_config, port=port, shot_id=shot_id, max_steps=1, reference_mode="hold_initial")
    env = HFMSimulator(cfg, reward_fn=None)
    try:
        obs, _ = env.reset(seed=int(seed), options={"reference_mode": "hold", "reset_params": dict(reset_params)})
        return np.asarray(obs["lcfs_points"], dtype=np.float64).reshape(32, 2)
    finally:
        env.close()


def _public_official_variant(public_lcfs: np.ndarray) -> dict[str, Any]:
    return {
        "variant_id": "public_official_target",
        "source_shot": "public_official_target",
        "source_seed": -1,
        "reset_params": {},
        "target_lcfs_delta_cm": 0.0,
        "lcfs_points": public_lcfs,
    }


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
    public_lcfs: np.ndarray,
    strict: bool,
) -> list[dict[str, Any]]:
    results: list[tuple[int, dict[str, Any]]] = []
    if not jobs:
        return []
    n_ports = max(1, len(ports))

    def worker(job: dict[str, Any], port: int) -> tuple[int, dict[str, Any]]:
        lcfs = reset_lcfs(
            base_config,
            port=port,
            shot_id=str(job["shot_id"]),
            seed=max(int(job["seed"]), 0),
            reset_params=dict(job["reset_params"]),
        )
        variant = {
            "variant_id": str(job["variant_id"]),
            "source_shot": str(job["shot_id"]),
            "source_seed": int(job["seed"]),
            "reset_params": {k: float(v) for k, v in dict(job["reset_params"]).items()},
            "target_lcfs_delta_cm": lcfs_delta_cm(lcfs, public_lcfs),
            "lcfs_points": lcfs,
        }
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
                print(f"generated {variant['variant_id']} reset={variant['reset_params']}", flush=True)
    return [variant for _, variant in sorted(results, key=lambda item: item[0])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate F1 LCFS target variants from reset LCFS.")
    parser.add_argument("--output", type=Path, default=ROOT / "training/targets/f1_lcfs_15892_target_variants_v2.json")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env_default.yaml")
    parser.add_argument("--target", type=Path, default=ROOT / "evaluation/results/target.json")
    parser.add_argument("--ports", type=str, default="5558")
    parser.add_argument("--source-shots", type=str, default="15892_400,15892_500")
    parser.add_argument("--total-count", type=int, default=64, help="Total number of target variants in the output JSON, including official targets.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic perturbation sampling.")
    parser.add_argument("--include-public-official-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--signeo-scale-range", type=str, default="0.85,1.15")
    parser.add_argument("--bp-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--q0-scale-range", type=str, default="0.9,1.1")
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
    source_shots = split_csv(args.source_shots)
    public_lcfs = np.asarray(json.loads(target_path.read_text(encoding="utf-8"))["F1"]["lcfs_points"], dtype=np.float64)
    perturbation = {
        "enabled": True,
        "apply_mode": "all",
        "signeo_scale_range": range_pair(args.signeo_scale_range),
        "bp_scale_range": range_pair(args.bp_scale_range),
        "q0_scale_range": range_pair(args.q0_scale_range),
    }

    variants: list[dict[str, Any]] = []
    if args.include_public_official_target:
        variants.append(_public_official_variant(public_lcfs))

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
        public_lcfs=public_lcfs,
        strict=bool(args.strict),
    ))
    variants = variants[:total_count]

    payload = {
        "schema_version": 1,
        "task": "F1",
        "description": "F1 LCFS target variants generated from HFM reset LCFS.",
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
        plot_lcfs_variants(variants, output=plot_output, public_lcfs=public_lcfs)
        print(f"wrote LCFS plot: {plot_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
