"""Check complete exported-controller parity with the original training pipeline."""
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    directory = tmp_path_factory.mktemp("controller_source")
    class Flat(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (90,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
        def reset(self, **kwargs):
            return np.zeros(90, np.float32), {}
    norm = VecNormalize(DummyVecEnv([Flat]))
    norm.obs_rms.mean = np.linspace(-5., 5., 90)
    norm.obs_rms.var = np.linspace(.1, 3., 90)
    norm.training = False
    model = sb3.PPO("MlpPolicy", norm, n_steps=2, batch_size=2, seed=7,
                    policy_kwargs={"net_arch": [8, 8]}, device="cpu")
    model.save(directory / "model.zip")
    norm.save(str(directory / "vecnormalize.pkl"))
    return directory, model, norm


def targets(index):
    shapes = {"reference_Ip": (1,), "reference_lX": (1,), "reference_lcfs_points": (32, 2),
              "reference_rX": (4,), "reference_zX": (4,), "reference_strike_r": (8,),
              "reference_strike_z": (8,)}
    offset, values = 0, {}
    for key, shape in shapes.items():
        size = int(np.prod(shape))
        values[key] = (np.sin(np.arange(offset, offset + size) + index) * 6.).reshape(shape)
        offset += size
    return values


@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_full_controller_matches_ppo_normalization_wrappers_and_reset(trained, tmp_path, mode):
    from train.model_metadata import reference_metadata
    from submission.model_bundle import unpack_bundle
    from submission.inference_common import BaseOnnxPolicy
    from train.export_onnx import main
    from train.wrappers import ReferenceObservationWrapper, NormalizedAction7DTo12DWrapper
    directory, model, norm = trained
    metadata = tmp_path / "training.json"
    metadata.write_text(json.dumps(reference_metadata(mode)))
    bundle = tmp_path / "submission_model.zip"
    assert main(["--model", str(directory / "model.zip"), "--metadata", str(metadata),
                 "--out", str(tmp_path / "controller.onnx"), "--bundle", str(bundle)]) == 0
    runtime = BaseOnnxPolicy(model_dir=unpack_bundle(bundle, tmp_path / "server"))
    assert runtime.ready, runtime.load_error
    assert runtime.metadata["state_dim"] == (7 if mode == "delta" else 0)
    assert len(runtime.metadata["observations"]) == 7
    assert {p.name for p in (tmp_path / "server").iterdir()} == {"controller.onnx", "controller.json"}
    class Sink(gym.Env):
        observation_space = gym.spaces.Dict({})
        action_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        def reset(self, **kwargs):
            return {}, {}
    encoder = ReferenceObservationWrapper(Sink())
    decoder = NormalizedAction7DTo12DWrapper(Sink(), action_mode=mode)
    for episode in range(2):
        runtime.reset()
        decoder.reset()
        for step in range(25):
            obs = targets(step)
            vector = encoder.convert_obs_dict_to_vector(obs).reshape(1, -1)
            raw, _ = model.predict(norm.normalize_obs(vector), deterministic=True)
            expected = decoder.action(raw[0])
            np.testing.assert_allclose(runtime.act(obs), expected, rtol=2e-5, atol=2e-4)


def test_custom_factory_embeds_different_features_and_state(trained, tmp_path, monkeypatch):
    from train.export_onnx import main
    from submission.model_bundle import unpack_bundle
    from submission.inference_common import BaseOnnxPolicy
    directory, _, _ = trained
    module = tmp_path / "custom_controller.py"
    module.write_text('''
import torch
from train.controller_export import ControllerSpec
class Controller(torch.nn.Module):
    def forward(self, Ip, R, state):
        value = Ip * 2. + R + state
        return value.expand(-1, 12), state + Ip
def build(model, metadata, normalization):
    return ControllerSpec(Controller(), ('Ip', 'R'), state_dim=1)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    metadata = tmp_path / "custom.json"
    metadata.write_text('{"custom_features": true}')
    bundle = tmp_path / "custom.zip"
    main(["--model", str(directory / "model.zip"), "--metadata", str(metadata),
          "--controller-factory", "custom_controller:build", "--out", str(tmp_path / "actor.onnx"),
          "--bundle", str(bundle)])
    policy = BaseOnnxPolicy(model_dir=unpack_bundle(bundle, tmp_path / "server"))
    assert policy.ready, policy.load_error
    np.testing.assert_allclose(policy.act({"Ip": [2.], "R": [3.]}), np.full(12, 7.))
    np.testing.assert_allclose(policy.act({"Ip": [2.], "R": [3.]}), np.full(12, 9.))
    policy.reset()
    np.testing.assert_allclose(policy.act({"Ip": [2.], "R": [3.]}), np.full(12, 7.))
