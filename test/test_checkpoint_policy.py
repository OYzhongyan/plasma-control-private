"""A checkpoint agent reuses actual training wrappers and frozen normalization."""
import json
import pickle
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def observation(index=0):
    shapes = {"reference_Ip": (1,), "reference_lX": (1,), "reference_lcfs_points": (32, 2),
              "reference_rX": (4,), "reference_zX": (4,), "reference_strike_r": (8,), "reference_strike_z": (8,)}
    result, offset = {}, 0
    for key, shape in shapes.items():
        size = int(np.prod(shape))
        result[key] = (np.arange(offset, offset + size) + index).reshape(shape).astype(np.float64)
        offset += size
    return result


def save_model(directory, dimension=90):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    directory.mkdir(parents=True, exist_ok=True)
    class Flat(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (dimension,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
    norm = VecNormalize(DummyVecEnv([Flat]))
    norm.obs_rms.mean = np.arange(dimension) + .1
    norm.obs_rms.var = np.linspace(.1, 5, dimension)
    norm.training = False
    model = sb3.PPO("MlpPolicy", norm, n_steps=2, batch_size=2, seed=7,
                    policy_kwargs={"net_arch": [8, 8]}, device="cpu")
    model.save(directory / "model.zip")
    norm.save(str(directory / "vecnormalize.pkl"))
    (directory / "f2_features.metadata.json").write_text(json.dumps({
        "action_mode": "delta", "obs_dim": 90, "observation_normalization": True}))
    return model, norm


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkpoint_agent")
    model, norm = save_model(directory)
    return directory, model, norm


@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_checkpoint_matches_actual_training_wrappers_and_reset(source, tmp_path, mode):
    from submission.checkpoint_policy import CheckpointPolicy
    from train.wrappers import ReferenceObservationWrapper, NormalizedAction7DTo12DWrapper
    directory, model, norm = source
    local = tmp_path / "run"
    local.mkdir()
    (local / "model.zip").write_bytes((directory / "model.zip").read_bytes())
    (local / "vecnormalize.pkl").write_bytes((directory / "vecnormalize.pkl").read_bytes())
    (local / "f2_features.metadata.json").write_text(json.dumps({"action_mode": mode, "observation_normalization": True}))
    policy = CheckpointPolicy(model_path=local / "model.zip")
    assert policy.ready, policy.load_error
    class Sink(gym.Env):
        observation_space = gym.spaces.Dict({})
        action_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        def reset(self, **kwargs):
            return {}, {}
    encoder = ReferenceObservationWrapper(Sink())
    actions = NormalizedAction7DTo12DWrapper(Sink(), action_mode=mode)
    for _ in range(2):
        policy.reset()
        actions.reset()
        before = policy.normalization.obs_rms.mean.copy()
        for index in range(8):
            obs = observation(index)
            vector = encoder.convert_obs_dict_to_vector(obs)
            action, _ = model.predict(norm.normalize_obs(vector), deterministic=True)
            np.testing.assert_allclose(policy.act(obs), actions.action(action), atol=1e-4, rtol=1e-5)
        np.testing.assert_array_equal(policy.normalization.obs_rms.mean, before)


def test_default_adapter_uses_modified_wrapper_and_actual_model_dimension(tmp_path, monkeypatch):
    from submission.checkpoint_policy import CheckpointPolicy
    from train import wrappers
    model, norm = save_model(tmp_path, dimension=3)
    class ThreeFeatures(gym.ObservationWrapper):
        def __init__(self, env, **kwargs):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        def observation(self, obs):
            return np.array([obs["Ip"][0], obs["R"][0], obs["Z"][0]], np.float32)
    monkeypatch.setattr(wrappers, "ReferenceObservationWrapper", ThreeFeatures)
    policy = CheckpointPolicy(model_path=tmp_path / "model.zip")
    assert policy.ready, policy.load_error
    obs = {"Ip": [3.], "R": [.6], "Z": [.1]}
    raw, _ = model.predict(norm.normalize_obs(np.array([3., .6, .1], np.float32)), deterministic=True)
    expected = raw * (np.array([1499, 175, 175, 175, 175, 175, 80]) * .16)
    np.testing.assert_allclose(policy.act(obs), expected[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]], rtol=1e-5, atol=1e-4)


def test_missing_matching_normalization_never_silently_runs_raw(source, tmp_path):
    from submission.checkpoint_policy import CheckpointPolicy
    directory, _, _ = source
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes((directory / "model.zip").read_bytes())
    policy = CheckpointPolicy(model_path=checkpoint)
    assert not policy.ready and "VecNormalize" in policy.load_error
    assert str(tmp_path) not in json.dumps(policy.status())


def test_latest_checkpoint_pairs_its_own_statistics(tmp_path):
    from submission.checkpoint_policy import resolve_checkpoint
    directory = tmp_path / "run/checkpoints"
    directory.mkdir(parents=True)
    for step in (20, 100):
        (directory / f"ppo_f2_{step}_steps.zip").write_bytes(b"model")
        (directory / f"ppo_f2_{step}_steps_vecnormalize.pkl").write_bytes(b"matching")
    model, stats = resolve_checkpoint(directory.parent)
    assert model.name == "ppo_f2_100_steps.zip"
    assert stats.name == "ppo_f2_100_steps_vecnormalize.pkl"


def test_wrong_statistics_shape_is_rejected(source, tmp_path):
    from submission.checkpoint_policy import CheckpointPolicy
    directory, _, norm = source
    (tmp_path / "model.zip").write_bytes((directory / "model.zip").read_bytes())
    original_mean, original_var = norm.obs_rms.mean, norm.obs_rms.var
    try:
        norm.obs_rms.mean, norm.obs_rms.var = np.zeros(1), np.ones(1)
        norm.save(str(tmp_path / "vecnormalize.pkl"))
    finally:
        norm.obs_rms.mean, norm.obs_rms.var = original_mean, original_var
    policy = CheckpointPolicy(model_path=tmp_path / "model.zip")
    assert not policy.ready and "shape" in policy.load_error


def test_checkpoint_with_normalization_disabled_can_use_saved_vecnormalize(source, tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from submission.checkpoint_policy import CheckpointPolicy
    directory, model, _ = source
    (tmp_path / "model.zip").write_bytes((directory / "model.zip").read_bytes())
    class Flat(gym.Env):
        observation_space = model.observation_space
        action_space = model.action_space
    norm = VecNormalize(DummyVecEnv([Flat]), norm_obs=False)
    norm.save(str(tmp_path / "vecnormalize.pkl"))
    (tmp_path / "f2_features.metadata.json").write_text(json.dumps({"observation_normalization": False}))
    policy = CheckpointPolicy(model_path=tmp_path / "model.zip")
    assert policy.ready, policy.load_error
