"""Export a complete controller and atomically package one platform upload.

Only this local export process imports training code or reads pickle files.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
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
from train.model_metadata import REFERENCE_FEATURE_MODE, reference_metadata
from submission.controller_contract import make_manifest, MODEL_NAME, MANIFEST_NAME
from submission.model_bundle import create_bundle


def default_out_path():
    return ROOT / "submission/model" / MODEL_NAME


def metadata_path_for_onnx(path):
    return Path(path).with_suffix(".json")


def load_export_metadata(
    model_path: Path,
    metadata_path: Path | None = None,
    config_path: Path | None = None,
    *, allow_custom: bool = False,
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
        if allow_custom:
            return {}
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
    if allow_custom:
        return dict(config.get("metadata", config))
    raise ValueError("Export requires reference_only_v1 training metadata; retrain the model with the current wrapper")


def normalization_candidates(model_path):
    stem = model_path.stem
    return [model_path.with_name(stem + "_vecnormalize_stats.npz"),
            model_path.with_name(stem + "_vecnormalize.pkl"),
            model_path.with_name(stem + ".vecnormalize_stats.npz"),
            model_path.with_name(stem + ".vecnormalize.pkl"),
            model_path.parent / "vecnormalize_stats.npz",
            model_path.parent / "vecnormalize.pkl"]


def load_normalization(model_path, explicit_path=None):
    candidates = [explicit_path] if explicit_path is not None else normalization_candidates(model_path)
    if explicit_path is not None and not explicit_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {explicit_path}")
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        return None
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            values = {k: np.array(data[k]) for k in ("obs_mean", "obs_var", "epsilon", "clip_obs") if k in data}
    elif source.suffix == ".pkl":
        with source.open("rb") as stream:
            obj = pickle.load(stream)
        values = dict(obs_mean=np.array(obj.obs_rms.mean), obs_var=np.array(obj.obs_rms.var),
                      epsilon=np.array(obj.epsilon), clip_obs=np.array(obj.clip_obs))
    else:
        raise ValueError("VecNormalize must be a local .pkl or numeric .npz")
    if not {"obs_mean", "obs_var"} <= set(values):
        raise ValueError("VecNormalize requires obs_mean and obs_var")
    for name, value in values.items():
        if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"VecNormalize {name} must contain finite real numbers")
    if np.any(values["obs_var"] < 0):
        raise ValueError("VecNormalize variance cannot be negative")
    for name in ("epsilon", "clip_obs"):
        if name in values and (values[name].size != 1 or float(values[name].reshape(-1)[0]) <= 0):
            raise ValueError(f"VecNormalize {name} must be one positive number")
    return values


def verify_controller(module, manifest, directory):
    import onnxruntime as ort
    from submission.inference_common import BaseOnnxPolicy
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = BaseOnnxPolicy(model_dir=directory, session_options=options, raise_on_error=True)
    rng = np.random.default_rng(7)
    for episode in range(2):
        runtime.reset()
        state = (torch.zeros((1, manifest["state_dim"]),
                            dtype=getattr(torch, manifest["state_dtype"])) if manifest["state_dim"] else None)
        for step in range(3):
            observations = {f["name"]: rng.uniform(.1, 1., size=f["shape"]).astype(np.float32)
                            for f in manifest["observations"]}
            inputs = [torch.from_numpy(observations[f["name"]][None]) for f in manifest["observations"]]
            if state is not None:
                inputs.append(state)
            with torch.no_grad():
                outputs = module(*inputs)
            if state is not None:
                action, state = outputs
            else:
                action = outputs
            np.testing.assert_allclose(runtime.act(observations), action.detach().cpu().numpy()[0],
                                       atol=2e-4, rtol=2e-5, err_msg="Torch/ONNX controller parity failed")
            if state is not None:
                np.testing.assert_allclose(runtime.state, state.detach().cpu().numpy(), atol=1e-6, rtol=2e-5)


def export_onnx(model_path, out_path, metadata, *, normalization=None,
                controller_factory="train.controller_export:build_controller"):
    from stable_baselines3 import PPO
    from train.controller_export import ControllerSpec
    model_path, out_path = Path(model_path), Path(out_path)
    if out_path.suffix.lower() != ".onnx":
        raise ValueError("Controller output must end in .onnx")
    model = PPO.load(str(model_path), device="cpu")
    if normalization is None:
        normalization = load_normalization(model_path)
    module_name, separator, factory_name = controller_factory.partition(":")
    if not separator:
        raise ValueError("--controller-factory must be module:function")
    factory = getattr(importlib.import_module(module_name), factory_name)
    spec = factory(model, metadata, normalization)
    if not isinstance(spec, ControllerSpec):
        raise ValueError("Controller factory must return ControllerSpec")
    manifest = make_manifest(spec.observations, spec.state_dim, spec.state_dtype)
    module = spec.module.to("cpu").eval()
    inputs = tuple(torch.zeros((1, *field["shape"]), dtype=torch.float32) for field in manifest["observations"])
    names = [field["name"] for field in manifest["observations"]]
    outputs = ["action"]
    if spec.state_dim:
        inputs += (torch.zeros((1, spec.state_dim), dtype=getattr(torch, spec.state_dtype)),)
        names.append("state")
        outputs.append("next_state")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plasma-controller-export-", dir=out_path.parent) as temp:
        staged = Path(temp)
        with torch.no_grad():
            torch.onnx.export(module, inputs, str(staged / MODEL_NAME),
                              input_names=names, output_names=outputs,
                              dynamic_axes={name: {0: "batch"} for name in names + outputs},
                              opset_version=17, dynamo=False)
        (staged / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_controller(module, manifest, staged)
        os.replace(staged / MODEL_NAME, out_path)
        os.replace(staged / MANIFEST_NAME, metadata_path_for_onnx(out_path))
    print(f"controller: {out_path}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metadata", type=Path, help="Local training metadata, not the controller manifest")
    parser.add_argument("--config", type=Path, help="Matching training configs.pkl or YAML")
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--controller-factory", default="train.controller_export:build_controller",
                        help="Local-only module:function returning ControllerSpec; never uploaded")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bundle", type=Path)
    group.add_argument("--no-bundle", action="store_true")
    args = parser.parse_args(argv)
    metadata = load_export_metadata(args.model, args.metadata, args.config,
                                   allow_custom=args.controller_factory != "train.controller_export:build_controller")
    normalization = load_normalization(args.model, args.vecnormalize)
    out_path = args.out or default_out_path()
    export_onnx(args.model, out_path, metadata, normalization=normalization,
                controller_factory=args.controller_factory)
    if not args.no_bundle:
        with tempfile.TemporaryDirectory(prefix="plasma-controller-package-") as temp:
            staging = Path(temp)
            shutil.copyfile(out_path, staging / MODEL_NAME)
            shutil.copyfile(metadata_path_for_onnx(out_path), staging / MANIFEST_NAME)
            package = create_bundle(staging, args.bundle or ROOT / "results/submission_model.zip")
        print(f"Model package ready: {package.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
