"""Check episode state, exported actor parity and HTTP failure reporting."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.history import ReferencePolicyAdapter, reference_metadata
from train.references import build_reference, default_reference_path
from train.wrappers import ReferenceObservationWrapper, NormalizedAction7DTo12DWrapper


def observation(index=0):
    reference = build_reference(default_reference_path())[0]
    obs = {"reference_" + k: v for k, v in reference.items() if k != "Ip"}
    obs.update(
        Ip=np.array([499000.0 - 1300 * index]),
        reference_Ip=np.array([499000.0 - 1000 * index]),
        lX=np.array([index % 2]),
        I_PF=np.arange(12, dtype=float) * (100 + index),
        lcfs_points=np.asarray(reference["lcfs_points"]) + index * 0.001,
        rX=np.r_[reference["rX"], [0, 0]],
        zX=np.r_[reference["zX"], [0, 0]],
        FX=np.ones(6),
        nX=np.array([4]),
        FA=0.0,
        FB=1.0,
        rx=np.linspace(0.5, 2.5, 66),
        zx=np.linspace(-2, 2, 65),
    )
    r, z = np.meshgrid(obs["rx"], obs["zx"], indexing="ij")
    obs["Fx"] = ((r - 1.5) ** 2 - z**2).reshape(-1)
    return obs


class Replay(gym.Env):
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.index = 0
        return observation(), {}

    def step(self, action):
        self.index += 1
        self.physical_action = np.asarray(action).copy()
        return observation(self.index), 0.0, False, False, {}


@pytest.mark.parametrize("mode", ["absolute", "delta"])
@pytest.mark.parametrize("steps", [1, 4])
def test_received_observations_match_training_wrappers_and_reset(mode, steps):
    env = Replay()
    action_wrapper = NormalizedAction7DTo12DWrapper(env, action_mode=mode)
    teacher = ReferenceObservationWrapper(
        action_wrapper, steps, include_xpt=True, action_mode=mode
    )
    adapter = ReferencePolicyAdapter(reference_metadata(mode))
    expected, _ = teacher.reset()
    for i in range(10):
        np.testing.assert_array_equal(adapter.encode(observation(i)), expected)
        raw = np.linspace(-1.4, 1.4, 7, dtype=np.float32) * (-1 if i % 2 else 1)
        actual_action = adapter.decode(raw)
        expected, _, _, _, _ = teacher.step(np.clip(raw, -1, 1))
        np.testing.assert_array_equal(actual_action, env.physical_action)
    adapter.reset()
    expected, _ = teacher.reset()
    np.testing.assert_array_equal(adapter.encode(observation()), expected)
    actual_action = adapter.decode(np.ones(7, dtype=np.float32))
    teacher.step(np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(actual_action, env.physical_action)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from train.export_onnx import main
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    directory = tmp_path_factory.mktemp("history_export")
    metadata = reference_metadata("delta")

    class Flat(gym.Env):
        observation_space = gym.spaces.Box(
            -np.inf, np.inf, (metadata["obs_dim"],), dtype=np.float32
        )
        action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    model = sb3.PPO(
        "MlpPolicy",
        Flat(),
        n_steps=2,
        batch_size=2,
        policy_kwargs={"net_arch": [8, 8]},
        device="cpu",
        seed=7,
    )
    model.save(directory / "ppo_f2_latest.zip")
    norm = VecNormalize(DummyVecEnv([Flat]), norm_obs=True)
    norm.obs_rms.mean = (
        ReferencePolicyAdapter(metadata).encode(observation()).astype(np.float64) + 0.1
    )
    norm.obs_rms.var = np.linspace(0.1, 5.0, metadata["obs_dim"])
    norm.save(str(directory / "vecnormalize.pkl"))
    config = {
        "features": {"observation_mode": "reference_only_v1", "history_steps": 0},
        "actions": {"mode": "delta"},
        "vecnormalize": {"enabled": True, "norm_obs": True},
    }
    with (directory / "configs.pkl").open("wb") as stream:
        pickle.dump(config, stream)
    onnx = directory / "policy2.onnx"
    main(["--model", str(directory / "ppo_f2_latest.zip"), "--out", str(onnx),
          "--bundle", str(directory / "submission_model.zip")])
    return directory, model, norm


def test_exported_policy_matches_sb3_over_multiple_steps_and_episodes(exported):
    from submission.inference2 import Policy
    from submission.inference_common import apply_obs_normalization

    directory, model, norm = exported
    policy = Policy(directory, backend="onnx")
    assert policy.ready, policy.load_error
    assert policy.metadata["obs_dim"] == 90
    assert policy.obs_normalization is not None
    env = Replay()
    actions = NormalizedAction7DTo12DWrapper(env, action_mode="delta")
    teacher = ReferenceObservationWrapper(
        actions, 4, include_xpt=True, action_mode="delta"
    )
    for _ in range(2):
        policy.reset()
        features, _ = teacher.reset()
        for i in range(6):
            normalized = norm.normalize_obs(features[None])[0]
            np.testing.assert_allclose(
                apply_obs_normalization(features, policy.obs_normalization),
                normalized,
                rtol=1e-7,
                atol=1e-8,
            )
            action, _ = model.predict(normalized, deterministic=True)
            actual = policy.act(observation(i))
            features, _, _, _, _ = teacher.step(action)
            np.testing.assert_allclose(
                actual, env.physical_action, rtol=1e-5, atol=1e-4
            )
    norm.close()


def test_http_health_act_and_invalid_input(exported):
    from submission.service2 import create_app

    directory, _, _ = exported
    client = create_app(directory, backend="onnx", allow_legacy=True).test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json["obs_dim"] == 90
    assert client.post("/reset", json={"episode_id": "F2b"}).status_code == 200
    response = client.post(
        "/act",
        json={
            "observation": {k: np.asarray(v).tolist() for k, v in observation().items()}
        },
    )
    assert response.status_code == 200
    assert len(response.json["action"]) == 12
    assert np.any(np.asarray(response.json["action"]) != 0)
    assert client.post("/reset", json={}).status_code == 200
    response = client.post("/act", json={"observation": {}})
    assert response.status_code == 500
    assert response.json["ok"] is False


def test_invalid_artifacts_fail_readiness_instead_of_zero_actions(tmp_path):
    from submission.inference2 import Policy
    from submission.service2 import create_app

    policy = Policy(tmp_path, backend="onnx")
    assert not policy.ready
    with pytest.raises(RuntimeError, match="No ONNX"):
        policy.act(observation())
    assert create_app(tmp_path, backend="onnx", allow_legacy=True).test_client().get("/health").status_code == 503


def test_export_rejects_incorrect_metadata(exported, tmp_path):
    from train.export_onnx import export_onnx

    directory, _, _ = exported
    with pytest.raises(ValueError, match="does not match metadata"):
        export_onnx(
            directory / "ppo_f2_latest.zip", tmp_path / "wrong.onnx", {**reference_metadata("delta"), "obs_dim": 216}
        )
    assert not (tmp_path / "wrong.onnx").exists()


def test_missing_normalization_and_wrong_dimension_are_not_ready(exported, tmp_path):
    import shutil
    from submission.inference2 import Policy

    directory, _, _ = exported
    shutil.copyfile(directory / "policy2.onnx", tmp_path / "policy2.onnx")
    metadata_path = tmp_path / "policy2.metadata.json"
    metadata = json.loads((directory / "policy2.metadata.json").read_text())
    metadata_path.write_text(json.dumps(metadata))
    policy = Policy(tmp_path, backend="onnx")
    assert not policy.ready
    assert "VecNormalize" in policy.load_error
    metadata.update(obs_dim=216, requires_vecnormalize=False)
    metadata_path.write_text(json.dumps(metadata))
    policy = Policy(tmp_path, backend="onnx")
    assert not policy.ready
    assert "layout" in policy.load_error
