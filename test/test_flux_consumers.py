"""Full-checkpoint consumers preserve the flux contract; actor export stays flat."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def flux_metadata():
    from submission.history import observation_metadata
    return observation_metadata("delta", critic_type="flux_cnn")


def flux_space():
    return gym.spaces.Dict({
        "actor_obs": gym.spaces.Box(-np.inf, np.inf, (812,), dtype=np.float32),
        "flux_history": gym.spaces.Box(-np.inf, np.inf, (4, 66, 65), dtype=np.float32),
        "flux_mask": gym.spaces.Box(0, 1, (4,), dtype=np.float32),
    })


def test_flux_metadata_is_complete_and_rejects_changed_shapes():
    from submission.history import validate_observation_metadata
    metadata = flux_metadata()
    assert metadata["feature_mode"] == "cnn_flux_critic_v1"
    assert metadata["actor_feature_mode"] == "historical_observation_v1"
    assert metadata["observation_shapes"] == {
        "actor_obs": [812], "flux_history": [4, 66, 65], "flux_mask": [4]}
    validate_observation_metadata(metadata)
    metadata["observation_shapes"]["flux_history"] = [4, 65, 66]
    with pytest.raises(ValueError, match="observation_shapes"):
        validate_observation_metadata(metadata)


def test_saved_flux_metadata_restores_eval_interface_without_critic_config(tmp_path):
    from omegaconf import OmegaConf
    from train.evaluate import configure_model_interface
    metadata = flux_metadata()
    (tmp_path / "f2_features.metadata.json").write_text(json.dumps(metadata))
    cfg = OmegaConf.create({"features": {"history_steps": 1, "observation_mode": "reference_only_v1"},
                            "actions": {"mode": "absolute"}})
    OmegaConf.set_struct(cfg, True)
    model = SimpleNamespace(observation_space=flux_space(),
                            action_space=gym.spaces.Box(-1, 1, (7,), dtype=np.float32))
    assert configure_model_interface(cfg, model, tmp_path / "model.zip") == metadata
    assert cfg.critic.type == "flux_cnn"
    assert cfg.features.observation_mode == "historical_observation_v1"
    assert cfg.features.history_steps == 4
    assert cfg.actions.mode == "delta"
    model.observation_space["flux_history"] = gym.spaces.Box(-np.inf, np.inf, (4, 65, 66), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions"):
        configure_model_interface(cfg, model, tmp_path / "model.zip")


def test_dict_normalization_snapshots_keep_actor_export_statistics(tmp_path):
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from train.callbacks import save_vecnormalize_stats
    from train.export_onnx import _write_vecnormalize_stats_from_object
    actor, flux = RunningMeanStd(shape=(812,)), RunningMeanStd(shape=())
    actor.update(np.arange(1624).reshape(2, 812))
    flux.update(np.array([2., 5., 7.]))
    norm = SimpleNamespace(obs_rms={"actor_obs": actor}, flux_rms=flux,
                           clip_obs=10., epsilon=1e-8, ret_rms=None)
    for save, name in ((save_vecnormalize_stats, "snapshot.npz"),
                       (_write_vecnormalize_stats_from_object, "export.npz")):
        path = tmp_path / name
        save(norm, path)
        with np.load(path) as data:
            np.testing.assert_array_equal(data["obs_mean"], actor.mean)
            np.testing.assert_array_equal(data["obs_var"], actor.var)
            assert data["flux_mean"].shape == ()
            assert data["flux_count"] == flux.count


def test_config_metadata_recovery_identifies_full_flux_checkpoint(tmp_path):
    from train.export_onnx import load_export_metadata
    config = {"features": {"observation_mode": "historical_observation_v1", "history_steps": 4},
              "critic": {"type": "flux_cnn"}, "actions": {"mode": "delta"}}
    path = tmp_path / "configs.pkl"
    path.write_bytes(pickle.dumps(config))
    assert load_export_metadata(tmp_path / "model.zip", config_path=path) == flux_metadata()


@pytest.fixture(scope="module")
def full_checkpoint(tmp_path_factory):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train.flux_observation import FluxObservationWrapper, FluxVecNormalize
    from train.flux_critic_policy import FluxCriticPolicy
    from train.wrappers import NormalizedAction7DTo12DWrapper
    from test_submission_history import Replay

    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("flux_checkpoint")
    norm = FluxVecNormalize(DummyVecEnv([lambda: FluxObservationWrapper(
        NormalizedAction7DTo12DWrapper(Replay(), action_mode="delta"), 4,
        include_xpt=True, action_mode="delta")]))
    norm.reset()
    for _ in range(4):
        norm.step(np.full((1, 7), .1, np.float32))
    norm.training = False
    norm.norm_reward = False
    model = PPO(FluxCriticPolicy, norm, n_steps=2, batch_size=2, seed=7, device="cpu")
    model.save(directory / "model.zip")
    norm.save(str(directory / "vecnormalize.pkl"))
    (directory / "f2_features.metadata.json").write_text(json.dumps(flux_metadata()))
    return directory, model, norm


def test_full_checkpoint_adapter_matches_training_sequence_and_frozen_stats(full_checkpoint):
    from submission.checkpoint_policy import CheckpointPolicy
    from test_submission_history import observation
    directory, model, norm = full_checkpoint
    policy = CheckpointPolicy(directory / "model.zip")
    assert policy.ready, policy.load_error
    assert policy.status()["obs_dim"] == 812
    teacher = norm.venv.envs[0]
    before = policy.normalization.flux_rms.count
    for _ in range(2):
        policy.reset()
        expected, _ = teacher.reset()
        for index in range(7):
            action, _ = model.predict(norm.normalize_obs(expected), deterministic=True)
            physical = policy.act(observation(index))
            expected, _, _, _, info = teacher.step(action)
            np.testing.assert_allclose(physical, info["physical_action"], atol=1e-4, rtol=1e-5)
            if index == 6:
                assert all(value.dtype == np.float32 for value in norm.normalize_obs(expected).values())
    assert policy.normalization.flux_rms.count == before


def test_full_checkpoint_rejects_corrupt_flux_statistics(full_checkpoint, tmp_path):
    from submission.checkpoint_policy import CheckpointPolicy
    directory, _, normalization = full_checkpoint
    stats = tmp_path / "corrupt.pkl"
    previous = normalization.flux_rms.var
    try:
        normalization.flux_rms.var = np.asarray(np.nan)
        normalization.save(str(stats))
    finally:
        normalization.flux_rms.var = previous
    policy = CheckpointPolicy(directory / "model.zip", vecnormalize_path=stats)
    assert not policy.ready
    assert "flux" in policy.load_error.lower()


def test_actor_only_onnx_roundtrip_has_flat_metadata_and_no_convolution(full_checkpoint, tmp_path):
    import onnx
    import onnxruntime as ort
    from train.export_onnx import main
    directory, model, _ = full_checkpoint
    out = tmp_path / "actor.onnx"
    assert main(["--model", str(directory / "model.zip"), "--out", str(out), "--no-bundle"]) == 0
    metadata = json.loads(out.with_suffix(".metadata.json").read_text())
    assert metadata["feature_mode"] == "historical_observation_v1"
    assert metadata["obs_dim"] == 812
    assert "observation_shapes" not in metadata
    assert json.loads((directory / "f2_features.metadata.json").read_text())["feature_mode"] == "cnn_flux_critic_v1"
    assert "Conv" not in {node.op_type for node in onnx.load(out).graph.node}
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actor = np.random.default_rng(31).normal(size=(3, 812)).astype(np.float32)
    obs = {"actor_obs": actor, "flux_history": np.zeros((3, 4, 66, 65), np.float32),
           "flux_mask": np.zeros((3, 4), np.float32)}
    import torch
    with torch.no_grad():
        expected = model.policy.get_distribution({key: torch.as_tensor(value) for key, value in obs.items()}).get_actions(deterministic=True).numpy()
    np.testing.assert_allclose(session.run(None, {"observation": actor})[0], expected, atol=2e-6)
    with np.load(out.with_name("actor_vecnormalize_stats.npz")) as stats:
        assert stats["obs_mean"].shape == (812,)
