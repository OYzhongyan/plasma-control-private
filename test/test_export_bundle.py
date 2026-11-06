"""The export CLI produces a single, fresh upload without a second command."""
import json
from pathlib import Path
import sys
import zipfile

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnx")
    from submission.history import reference_metadata

    directory = tmp_path_factory.mktemp("bundle_source")
    metadata = reference_metadata("delta")

    class Flat(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (90,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    model = sb3.PPO("MlpPolicy", Flat(), n_steps=2, batch_size=2,
                    policy_kwargs={"net_arch": [8, 8]}, device="cpu", seed=7)
    model.save(directory / "model.zip")
    (directory / "f2_features.metadata.json").write_text(json.dumps(metadata))
    np.savez(directory / "vecnormalize_stats.npz", obs_mean=np.zeros(90),
             obs_var=np.ones(90), epsilon=1e-8, clip_obs=10.)
    return directory


def test_export_automatically_packages_current_files(trained_model, tmp_path, monkeypatch):
    from train import export_onnx as exporter
    from submission.model_bundle import unpack_bundle

    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    assert exporter.main(["--model", str(trained_model / "model.zip")]) == 0
    package = tmp_path / "results/submission_model.zip"
    extracted = unpack_bundle(package, tmp_path / "server_model")
    for name in ("policy2.onnx", "policy2.metadata.json", "policy2_vecnormalize_stats.npz"):
        assert (extracted / name).read_bytes() == (tmp_path / "submission/model" / name).read_bytes()


def test_export_custom_paths_keep_fixed_archive_names(trained_model, tmp_path):
    from train.export_onnx import main

    package = tmp_path / "team.zip"
    assert main(["--model", str(trained_model / "model.zip"),
                 "--out", str(tmp_path / "actor.onnx"), "--bundle", str(package)]) == 0
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {
            "policy2.onnx", "policy2.metadata.json", "policy2_vecnormalize_stats.npz"}
        assert archive.read("policy2.onnx") == (tmp_path / "actor.onnx").read_bytes()


def test_missing_current_stats_never_packages_stale_export(trained_model, tmp_path, monkeypatch):
    from train import export_onnx as exporter

    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    source = tmp_path / "moved"
    source.mkdir()
    (source / "model.zip").write_bytes((trained_model / "model.zip").read_bytes())
    (source / "f2_features.metadata.json").write_bytes(
        (trained_model / "f2_features.metadata.json").read_bytes())
    output = tmp_path / "submission/model"
    output.mkdir(parents=True)
    (output / "policy2_vecnormalize_stats.npz").write_bytes(
        (trained_model / "vecnormalize_stats.npz").read_bytes())
    package = tmp_path / "results/submission_model.zip"
    package.parent.mkdir()
    package.write_bytes(b"previous upload")
    with pytest.raises(ValueError, match="VecNormalize"):
        exporter.main(["--model", str(source / "model.zip")])
    assert package.read_bytes() == b"previous upload"


def test_development_export_can_skip_packaging(trained_model, tmp_path, monkeypatch):
    from train import export_onnx as exporter

    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    assert exporter.main(["--model", str(trained_model / "model.zip"), "--no-bundle"]) == 0
    assert (tmp_path / "submission/model/policy2.onnx").is_file()
    assert not (tmp_path / "results/submission_model.zip").exists()
