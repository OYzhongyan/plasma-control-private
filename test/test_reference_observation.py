"""Reference-only actor/critic inputs never depend on state or action history."""
import copy
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.wrappers import HistoricalOBservationWrapper


def target(offset=0.):
    return {
        "reference_Ip": np.array([offset]),
        "reference_lX": np.array([offset + 1]),
        "reference_lcfs_points": (np.arange(2, 66) + offset).reshape(32, 2),
        "reference_rX": np.arange(66, 70) + offset,
        "reference_zX": np.arange(70, 74) + offset,
        "reference_strike_r": np.arange(74, 82) + offset,
        "reference_strike_z": np.arange(82, 90) + offset,
    }


class ReferenceEnv(gym.Env):
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return target(), {"reset": True}

    def step(self, action):
        self.index += 1
        return target(self.index), 1., self.index == 8, False, {"index": self.index}


@pytest.mark.parametrize("history_steps", [0, 1, 4])
@pytest.mark.parametrize("action_mode", ["absolute", "delta"])
def test_reference_only_reset_and_step_use_exact_current_90_features(history_steps, action_mode):
    env = HistoricalOBservationWrapper(ReferenceEnv(), history_steps,
                                       include_xpt=True, action_mode=action_mode)
    assert env.observation_space.shape == (90,)
    obs, info = env.reset()
    np.testing.assert_array_equal(obs, np.arange(90, dtype=np.float32))
    assert env.observation_space.contains(obs) and info == {"reset": True}
    for index in range(1, 4):
        obs, reward, terminated, truncated, info = env.step(np.full(7, (-1.) ** index))
        np.testing.assert_array_equal(obs, np.arange(90, dtype=np.float32) + index)
        assert (reward, terminated, truncated, info) == (1., False, False, {"index": index})
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.arange(90, dtype=np.float32))


def test_measured_state_other_reference_fields_and_action_history_cannot_leak_into_input():
    env = HistoricalOBservationWrapper(ReferenceEnv(), 4, include_xpt=False)
    observation = target()
    observation.update(Ip=np.array([np.nan]), lcfs_points=np.full((32, 2), np.nan),
                       I_PF=np.full(12, np.inf), reference_R=np.array([99.]),
                       reference_x_valid=np.zeros(4), reference_nX=np.array([6]))
    np.testing.assert_array_equal(env.convert_obs_dict_to_vector(observation), np.arange(90))


def test_sensor_delay_and_noise_keep_the_current_reference():
    from train.wrappers import ObservationDelayNoiseWrapper

    class SensorEnv(ReferenceEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            return {**obs, "Ip": np.array([100.])}, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return {**obs, "Ip": np.array([100. + self.index])}, reward, terminated, truncated, info

    delayed = ObservationDelayNoiseWrapper(SensorEnv(), delay_steps=1,
                                           noise_std={"reference_Ip": 10000.}, seed=7)
    env = HistoricalOBservationWrapper(delayed)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.arange(90, dtype=np.float32))
    obs, *_ = env.step(np.zeros(7))
    np.testing.assert_array_equal(obs, np.arange(90, dtype=np.float32) + 1)
    assert delayed._obs_hist[0]["Ip"][0] == 100.


@pytest.mark.parametrize("key,value", [
    ("reference_rX", np.zeros(3)),
    ("reference_lcfs_points", np.zeros((31, 2))),
    ("reference_Ip", np.array([np.nan])),
])
def test_invalid_reference_fails_with_field_name(key, value):
    env = HistoricalOBservationWrapper(ReferenceEnv(), 0)
    observation = target()
    observation[key] = value
    with pytest.raises(ValueError, match=key):
        env.convert_obs_dict_to_vector(observation)


def test_reference_metadata_describes_zero_history_and_rejects_reordered_fields():
    from submission.history import ReferencePolicyAdapter, reference_metadata
    metadata = reference_metadata("delta")
    assert metadata["feature_mode"] == "reference_only_v1"
    assert metadata["obs_dim"] == 90
    assert metadata["history_steps"] == metadata["action_history_steps"] == 0
    adapter = ReferencePolicyAdapter(metadata)
    np.testing.assert_array_equal(adapter.encode(target()), np.arange(90))
    first = adapter.decode(np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(adapter.encode(target()), np.arange(90))
    second = adapter.decode(np.ones(7, dtype=np.float32))
    np.testing.assert_allclose(second, 2 * first)
    adapter.reset()
    np.testing.assert_allclose(adapter.decode(np.ones(7, dtype=np.float32)), first)
    invalid = copy.deepcopy(metadata)
    invalid["reference_keys"].reverse()
    with pytest.raises(ValueError, match="reference_keys"):
        ReferencePolicyAdapter(invalid)


def test_export_config_requires_explicit_reference_only_layout(tmp_path):
    from train.export_onnx import load_export_metadata
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"features": {"observation_mode": "reference_only_v1"},
                                     "actions": {"mode": "delta"}}))
    metadata = load_export_metadata(tmp_path / "model.zip", config_path=config)
    assert metadata["obs_dim"] == 90 and metadata["action_mode"] == "delta"
    config.write_text(yaml.safe_dump({"features": {"history_steps": 4}, "actions": {"mode": "delta"}}))
    with pytest.raises(ValueError, match="reference_only_v1"):
        load_export_metadata(tmp_path / "model.zip", config_path=config)


@pytest.fixture(scope="module")
def reference_export(tmp_path_factory):
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from submission.history import reference_metadata
    from train.export_onnx import main
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    directory = tmp_path_factory.mktemp("reference_export")
    env = VecNormalize(DummyVecEnv([
        lambda: HistoricalOBservationWrapper(ReferenceEnv(), 4, include_xpt=True)]))
    model = sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=4, n_epochs=2,
                    policy_kwargs={"net_arch": [8, 8]}, seed=2, device="cpu")
    import torch
    actor_before = model.policy.action_net.weight.detach().clone()
    critic_before = model.policy.value_net.weight.detach().clone()
    model.learn(16)
    assert not torch.equal(actor_before, model.policy.action_net.weight)
    assert not torch.equal(critic_before, model.policy.value_net.weight)
    assert model.policy.mlp_extractor.policy_net[0].in_features == 90
    assert model.policy.mlp_extractor.value_net[0].in_features == 90
    model.save(directory / "model.zip")
    env.save(str(directory / "vecnormalize.pkl"))
    (directory / "f2_features.metadata.json").write_text(json.dumps(reference_metadata("absolute")))
    main(["--model", str(directory / "model.zip"), "--out", str(directory / "policy2.onnx"),
          "--bundle", str(directory / "submission_model.zip")])
    env.training = False
    return directory, model, env


def test_trained_90_feature_actor_critic_export_and_inference_match(reference_export):
    from submission.inference2 import Policy
    from train.wrappers import NormalizedAction7DTo12DWrapper
    directory, model, env = reference_export
    policy = Policy(directory)
    assert policy.ready, policy.load_error
    assert policy.metadata["obs_dim"] == 90
    actions = NormalizedAction7DTo12DWrapper(ReferenceEnv())
    for index in range(3):
        vector = (np.arange(90, dtype=np.float32) + index).reshape(1, -1)
        normalized = env.normalize_obs(vector)
        action, _ = model.predict(normalized, deterministic=True)
        np.testing.assert_allclose(policy.act(target(index)), actions.action(action[0]), atol=1e-4, rtol=1e-5)


def test_export_rejects_changed_reference_order_before_writing(reference_export, tmp_path):
    from submission.history import reference_metadata
    from train.export_onnx import export_onnx
    directory, _, _ = reference_export
    metadata = reference_metadata("absolute")
    metadata["reference_keys"].reverse()
    with pytest.raises(ValueError, match="reference_keys"):
        export_onnx(directory / "model.zip", tmp_path / "bad.onnx", metadata)
    assert not (tmp_path / "bad.onnx").exists()


def test_evaluation_uses_the_models_action_mode(reference_export, tmp_path):
    from omegaconf import OmegaConf
    from submission.history import reference_metadata
    from train.evaluate import configure_model_interface
    _, model, _ = reference_export
    (tmp_path / "f2_features.metadata.json").write_text(json.dumps(reference_metadata("delta")))
    cfg = OmegaConf.create({"features": {"observation_mode": "reference_only_v1", "history_steps": 4},
                           "actions": {"mode": "absolute"}})
    configure_model_interface(cfg, model, tmp_path / "model.zip")
    assert cfg.actions.mode == "delta"
    assert cfg.features.history_steps == 0


def test_evaluation_requires_matching_normalization_when_metadata_requires_it(tmp_path):
    from submission.history import reference_metadata
    from train.evaluate import resolve_evaluation_normalization
    with pytest.raises(FileNotFoundError, match="VecNormalize"):
        resolve_evaluation_normalization(reference_metadata("delta"), None)
    stats = tmp_path / "vecnormalize.pkl"
    stats.write_bytes(b"matching training stats")
    assert resolve_evaluation_normalization(reference_metadata("delta"), stats) == stats
    assert resolve_evaluation_normalization(reference_metadata("delta", norm_obs=False), stats) is None
