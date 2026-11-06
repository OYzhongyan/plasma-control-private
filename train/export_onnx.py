"""Export a trained SB3 policy and automatically package one platform upload."""

from __future__ import annotations
import argparse
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from submission.history import REFERENCE_FEATURE_MODE, reference_metadata, validate_reference_metadata
from submission.model_bundle import create_bundle


class OnnxablePolicy(torch.nn.Module):
    """Minimal wrapper around an SB3 actor-critic policy."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        actions, _, _ = self.policy(observation, deterministic=True)
        return actions


def default_out_path() -> Path:
    name = "policy2.onnx"
    return ROOT / "submission" / "model" / name


def metadata_path_for_onnx(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def vecnormalize_stats_path_for_onnx(path: Path) -> Path:
    return path.with_name(f"{path.stem}_vecnormalize_stats.npz")


def vecnormalize_source_candidates(
    model_path: Path, explicit_path: Path | None = None
) -> list[Path]:
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
        "clip_obs": np.asarray(
            getattr(vecnormalize, "clip_obs", 10.0), dtype=np.float64
        ),
        "epsilon": np.asarray(
            getattr(vecnormalize, "epsilon", 1e-08), dtype=np.float64
        ),
    }
    ret_rms = getattr(vecnormalize, "ret_rms", None)
    if ret_rms is not None:
        values["ret_mean"] = np.asarray(ret_rms.mean, dtype=np.float64)
        values["ret_var"] = np.asarray(ret_rms.var, dtype=np.float64)
        values["clip_reward"] = np.asarray(
            getattr(vecnormalize, "clip_reward", 10.0), dtype=np.float64
        )
    np.savez(out_path, **values)


def export_vecnormalize_stats(
    model_path: Path, out_path: Path, explicit_path: Path | None = None
) -> Path | None:
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
    print(
        "warning: VecNormalize stats not found; ONNX inference will use raw observations."
    )
    return None


def load_export_metadata(
    model_path: Path,
    metadata_path: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    """Recover the preprocessing contract saved alongside a model/checkpoint."""
    if metadata_path is not None:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    directories = [model_path.parent, model_path.parent.parent]
    if config_path is None:
        for directory in directories:
            candidate = directory / "f2_features.metadata.json"
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        config_path = next(
            (
                directory / "configs.pkl"
                for directory in directories
                if (directory / "configs.pkl").is_file()
            ),
            None,
        )
    if config_path is None:
        raise ValueError("reference_only_v1 metadata not found; provide matching --metadata or --config")
    if config_path.suffix == ".pkl":
        with config_path.open("rb") as stream:
            config = pickle.load(stream)
    else:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    features = config.get("features", {})
    actions = config.get("actions", {})
    normalization = config.get("vecnormalize", {})
    if features.get("observation_mode") == REFERENCE_FEATURE_MODE:
        return reference_metadata(
            action_mode=str(actions.get("mode", "absolute")),
            norm_obs=bool(normalization.get("enabled", True) and normalization.get("norm_obs", True)),
        )
    raise ValueError("Export requires reference_only_v1 training metadata; retrain the model with the current wrapper")


def export_onnx(
    model_path: Path, out_path: Path, metadata: dict
) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required to export SB3 models"
        ) from exc
    model = PPO.load(str(model_path), device="cpu")
    obs_shape = model.observation_space.shape
    action_shape = model.action_space.shape
    if obs_shape is None:
        raise ValueError("model observation space must have a fixed shape")
    action_dim = (
        int(action_shape[0]) if action_shape is not None and len(action_shape) else 7
    )
    metadata = dict(metadata)
    if metadata.get("feature_mode") != REFERENCE_FEATURE_MODE:
        raise ValueError("Export requires reference_only_v1 metadata; retrain the model with the current wrapper")
    if len(obs_shape) != 1 or int(metadata["obs_dim"]) != int(obs_shape[0]):
        raise ValueError(
            f"Model input {obs_shape} does not match metadata obs_dim={metadata.get('obs_dim')}. "
            "Provide the model's training configs.pkl with --config or matching --metadata."
        )
    if action_dim != 7:
        raise ValueError("The normalized action wrapper requires a 7D actor")
    validate_reference_metadata(metadata)
    metadata["action_dim"] = action_dim
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
    metadata_path_for_onnx(out_path).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"exported: {out_path}")
    print(f"metadata: {metadata_path_for_onnx(out_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export semifinal PPO policy to ONNX")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional matching metadata JSON; otherwise detected from the training directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training configs.pkl or YAML, needed if the model was moved away from its run directory.",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=None,
        help="Optional VecNormalize .pkl or stats .npz. If omitted, export searches beside the model for vecnormalize_stats.npz or vecnormalize.pkl.",
    )
    packaging = parser.add_mutually_exclusive_group()
    packaging.add_argument(
        "--bundle", type=Path,
        help="Upload ZIP destination; defaults to results/submission_model.zip in the project.",
    )
    packaging.add_argument(
        "--no-bundle", action="store_true",
        help="Only export ONNX and preprocessing files for development, without creating an upload ZIP.",
    )
    args = parser.parse_args(argv)
    out_path = args.out or default_out_path()
    metadata = load_export_metadata(args.model, args.metadata, args.config)
    export_onnx(args.model, out_path, metadata)
    stats = export_vecnormalize_stats(args.model, out_path, args.vecnormalize)
    if metadata.get("requires_vecnormalize"):
        if stats is None:
            raise ValueError("This model requires VecNormalize observation statistics")
        with np.load(stats) as data:
            if any(
                np.asarray(data[key]).size != int(metadata["obs_dim"])
                for key in ("obs_mean", "obs_var")
            ):
                raise ValueError(
                    "VecNormalize statistics do not match the exported observation layout"
                )
    if not args.no_bundle:
        if stats is None:
            raise ValueError("A model upload requires matching VecNormalize statistics; use --vecnormalize")
        # Stage only this export's outputs, including custom --out filenames.
        # Never pick up old files from submission/model when a source is missing.
        with tempfile.TemporaryDirectory(prefix="plasma-model-export-") as temporary:
            staging = Path(temporary)
            for source, name in (
                (out_path, "policy2.onnx"),
                (metadata_path_for_onnx(out_path), "policy2.metadata.json"),
                (stats, "policy2_vecnormalize_stats.npz"),
            ):
                shutil.copyfile(source, staging / name)
            package = create_bundle(staging, args.bundle or ROOT / "results" / "submission_model.zip")
        print(f"Model package ready: {package.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
