"""Convenience wrapper for rolling out F2 SB3 policies from a train config."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from train.legacy_train_ppo import _rooted_path, load_train_config
from tools import rollout_stage0_policy


def _rooted(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out an F2 SB3 policy, inferring paths from the training config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vecnormalize", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shots", nargs="*", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--episodes-per-shot", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--policy-mode", choices=["model", "zero"], default="model")
    parser.add_argument("--perturb-mode", choices=["none", "config"], default="none")
    parser.add_argument("--reference-variant-index", type=int, default=None)
    parser.add_argument("--infer-result", type=Path, default=None)
    parser.add_argument("--score-output", type=Path, default=None)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def inferred_paths(train_config_path: Path) -> tuple[Path, Path, Path]:
    train_config = load_train_config(train_config_path)
    output_dir = _rooted_path(
        train_config.get("output_dir"), PROJECT_ROOT / "results/f2_policy"
    )
    if output_dir is None:
        raise ValueError("could not infer output_dir from train config")
    model = output_dir / "ppo_f2_latest.zip"
    vecnormalize = output_dir / "vecnormalize.pkl"
    rollout_dir = output_dir / "rollout_analysis"
    return (model, vecnormalize, rollout_dir)


def main() -> int:
    args = parse_args()
    train_config = _rooted(args.train_config)
    if train_config is None:
        raise ValueError("train-config path could not be resolved")
    model_default, vec_default, out_default = inferred_paths(train_config)
    model = _rooted(args.model) or model_default
    if args.vecnormalize is None:
        vecnormalize = vec_default
    else:
        vec_spec = str(args.vecnormalize).strip().lower()
        vecnormalize = (
            None
            if vec_spec in {"", "none", "null", "false"}
            else _rooted(args.vecnormalize)
        )
    output_dir = _rooted(args.output_dir) or out_default
    infer_result = (
        _rooted(args.infer_result)
        if args.infer_result
        else output_dir / "infer_result.json"
    )
    score_output = (
        _rooted(args.score_output)
        if args.score_output
        else output_dir / "score_summary.json"
    )
    forwarded = [
        "--train-config",
        str(train_config),
        "--model",
        str(model),
        "--output-dir",
        str(output_dir),
        "--episodes-per-shot",
        str(args.episodes_per_shot),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--policy-mode",
        str(args.policy_mode),
        "--perturb-mode",
        str(args.perturb_mode),
    ]
    if vecnormalize is None:
        forwarded.extend(["--vecnormalize", "none"])
    else:
        forwarded.extend(["--vecnormalize", str(vecnormalize)])
    if args.shots:
        forwarded.append("--shots")
        forwarded.extend((str(shot) for shot in args.shots))
    if args.port is not None:
        forwarded.extend(["--port", str(args.port)])
    if args.max_steps is not None:
        forwarded.extend(["--max-steps", str(args.max_steps)])
    if not args.deterministic:
        forwarded.append("--no-deterministic")
    if args.reference_variant_index is not None:
        forwarded.extend(
            ["--reference-variant-index", str(args.reference_variant_index)]
        )
    forwarded.append("--score" if args.score else "--no-score")
    forwarded.extend(["--infer-result", str(infer_result)])
    forwarded.extend(["--score-output", str(score_output)])
    if args.validate_only:
        forwarded.append("--validate-only")
    sys.argv = ["rollout_stage0_policy.py", *forwarded]
    return int(rollout_stage0_policy.main())


if __name__ == "__main__":
    raise SystemExit(main())
