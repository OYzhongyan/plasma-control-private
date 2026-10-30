"""Export a trained SB3 policy to the submission ONNX layout."""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.features import FeatureMetadata  # noqa: E402
from training.references import normalize_task  # noqa: E402


class OnnxablePolicy(torch.nn.Module):
    """Minimal wrapper around an SB3 actor-critic policy."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        actions, _, _ = self.policy(observation, deterministic=True)
        return actions


def default_out_path(task: str) -> Path:
    task = normalize_task(task)
    name = "policy1.onnx" if task == "F1" else "policy2.onnx"
    return ROOT / "submission" / "model" / name


def metadata_path_for_onnx(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def vecnormalize_stats_path_for_onnx(path: Path) -> Path:
    return path.with_name(f"{path.stem}_vecnormalize_stats.npz")


def vecnormalize_source_candidates(model_path: Path, explicit_path: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    parent = model_path.parent
    stem = model_path.stem
    candidates.extend(
        [
            parent / "vecnormalize_stats.npz",
            parent / "vecnormalize.pkl",
            model_path.with_name(f"{stem}_vecnormalize_stats.npz"),
            model_path.with_name(f"{stem}.vecnormalize_stats.npz"),
            model_path.with_name(f"{stem}_vecnormalize.pkl"),
            model_path.with_name(f"{stem}.vecnormalize.pkl"),
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _write_vecnormalize_stats_from_object(vecnormalize: object, out_path: Path) -> None:
    obs_rms = getattr(vecnormalize, "obs_rms", None)
    if obs_rms is None:
        raise ValueError("VecNormalize object does not contain obs_rms")
    values: dict[str, np.ndarray] = {
        "obs_mean": np.asarray(obs_rms.mean, dtype=np.float64),
        "obs_var": np.asarray(obs_rms.var, dtype=np.float64),
        "clip_obs": np.asarray(getattr(vecnormalize, "clip_obs", 10.0), dtype=np.float64),
        "epsilon": np.asarray(getattr(vecnormalize, "epsilon", 1e-8), dtype=np.float64),
    }
    ret_rms = getattr(vecnormalize, "ret_rms", None)
    if ret_rms is not None:
        values["ret_mean"] = np.asarray(ret_rms.mean, dtype=np.float64)
        values["ret_var"] = np.asarray(ret_rms.var, dtype=np.float64)
        values["clip_reward"] = np.asarray(getattr(vecnormalize, "clip_reward", 10.0), dtype=np.float64)
    np.savez(out_path, **values)


def export_vecnormalize_stats(model_path: Path, out_path: Path, explicit_path: Path | None = None) -> Path | None:
    """Copy or convert VecNormalize statistics into the submission model directory."""

    stats_out = vecnormalize_stats_path_for_onnx(out_path)
    for source in vecnormalize_source_candidates(model_path, explicit_path):
        if not source.is_file():
            continue
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        suffixes = "".join(source.suffixes)
        if source.suffix == ".npz" or suffixes.endswith(".stats.npz"):
            shutil.copyfile(source, stats_out)
        elif source.suffix == ".pkl":
            with open(source, "rb") as f:
                vecnormalize = pickle.load(f)
            _write_vecnormalize_stats_from_object(vecnormalize, stats_out)
        else:
            continue
        print(f"vecnormalize stats: {stats_out} (source: {source})")
        return stats_out
    print("warning: VecNormalize stats not found; ONNX inference will use raw observations.")
    return None


def export_onnx(model_path: Path, out_path: Path, metadata: FeatureMetadata) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to export SB3 models") from exc

    model = PPO.load(str(model_path), device="cpu")
    obs_shape = model.observation_space.shape
    action_shape = model.action_space.shape
    if obs_shape is None:
        raise ValueError("model observation space must have a fixed shape")
    action_dim = int(action_shape[0]) if action_shape is not None and len(action_shape) else 7
    action_mode = "7D" if action_dim == 7 else "12D"
    metadata = replace(metadata, action_dim=action_dim, action_mode=action_mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_obs = torch.zeros((1, *obs_shape), dtype=torch.float32, device="cpu")
    module = OnnxablePolicy(model.policy).to("cpu")
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            sample_obs,
            str(out_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    metadata.save(metadata_path_for_onnx(out_path))
    print(f"exported: {out_path}")
    print(f"metadata: {metadata_path_for_onnx(out_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export semifinal PPO policy to ONNX")
    parser.add_argument("--task", choices=["F1", "F2"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON from training; defaults to task feature metadata.",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=None,
        help=(
            "Optional VecNormalize .pkl or stats .npz. If omitted, export searches beside "
            "the model for vecnormalize_stats.npz or vecnormalize.pkl."
        ),
    )
    args = parser.parse_args(argv)

    task = normalize_task(args.task)
    out_path = args.out or default_out_path(task)
    metadata = FeatureMetadata.load(args.metadata) if args.metadata else FeatureMetadata.for_task(task)
    export_onnx(args.model, out_path, metadata)
    export_vecnormalize_stats(args.model, out_path, args.vecnormalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
