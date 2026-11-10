"""Flux history preserves actor inputs and uses one scalar set of statistics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from test_submission_history import Replay, observation
from train.wrappers import (
    NormalizedAction7DTo12DWrapper,
    ObservationDelayNoiseWrapper,
    ObservationWrapper,
)


class FluxReplay(Replay):
    def __init__(self, invalid_at=None):
        self.resets = 0
        self.steps = 0
        self.invalid_at = invalid_at or {}

    def _observation(self, index):
        obs = observation(index)
        obs["Fx"] = np.arange(4290, dtype=np.float32) + index * 10000
        if index in self.invalid_at:
            obs["Fx"] = self.invalid_at[index]
        return obs

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.index = 0
        return self._observation(0), {"reset_marker": self.resets}

    def step(self, action):
        self.steps += 1
        self.index += 1
        return self._observation(self.index), float(self.index), False, False, {}


def wrap_replay(wrapper_cls, *, mode="delta", delay=0, include_xpt=True, raw=None):
    raw = raw or FluxReplay()
    delayed = ObservationDelayNoiseWrapper(raw, delay_steps=delay, noise_std={}, seed=0)
    actions = NormalizedAction7DTo12DWrapper(delayed, action_mode=mode)
    return wrapper_cls(actions, 4, include_xpt=include_xpt, action_mode=mode), raw


@pytest.mark.parametrize("mode", ["absolute", "delta"])
@pytest.mark.parametrize("delay", [0, 1])
def test_flux_history_keeps_baseline_actor_physical_actions_and_delay(mode, delay):
    from train.flux_observation import FluxObservationWrapper

    baseline, _ = wrap_replay(ObservationWrapper, mode=mode, delay=delay)
    wrapper, raw = wrap_replay(FluxObservationWrapper, mode=mode, delay=delay)
    for episode in range(2):
        actual, info = wrapper.reset()
        expected, _ = baseline.reset()
        assert info["reset_marker"] == episode + 1
        np.testing.assert_array_equal(actual["actor_obs"], expected)
        assert actual["actor_obs"].dtype == expected.dtype
        np.testing.assert_array_equal(actual["flux_mask"], [0, 0, 0, 1])
        np.testing.assert_array_equal(actual["flux_history"][:3], 0)
        assert actual["flux_history"][3, 1, 0] == 65
        assert actual["flux_history"][3, 0, 1] == 1
        frames = [np.arange(4290, dtype=np.float32).reshape(66, 65)]
        for step in range(1, 7):
            action = np.linspace(-0.7, 0.7, 7, dtype=np.float32) * (-1) ** step
            actual, reward, terminated, truncated, info = wrapper.step(action)
            expected, expected_reward, *_ = baseline.step(action)
            np.testing.assert_array_equal(actual["actor_obs"], expected)
            assert reward == expected_reward and not terminated and not truncated
            delayed_index = max(0, step - delay)
            frames.append(np.arange(4290, dtype=np.float32).reshape(66, 65) + delayed_index * 10000)
            valid = min(step + 1, 4)
            np.testing.assert_array_equal(actual["flux_mask"], [0] * (4 - valid) + [1] * valid)
            np.testing.assert_array_equal(actual["flux_history"][-valid:], np.stack(frames[-valid:]))
            if mode == "delta":
                np.testing.assert_array_equal(actual["actor_obs"][-12:], info["physical_action"])
            assert actual["flux_history"].dtype == np.float32
            assert actual["flux_mask"].dtype == np.float32
            assert wrapper.observation_space["actor_obs"].shape == actual["actor_obs"].shape
    assert raw.resets == 2 and raw.steps == 12
    if mode == "delta":
        assert actual["actor_obs"].shape == (812,)


@pytest.mark.parametrize("invalid", [None, np.zeros(3), np.full(4290, np.nan), np.full(4290, np.inf)])
def test_invalid_flux_masks_whole_frame_and_preserves_good_history(invalid):
    from train.flux_observation import FluxObservationWrapper

    wrapper, _ = wrap_replay(FluxObservationWrapper, include_xpt=False, raw=FluxReplay({1: invalid}))
    wrapper.reset()
    obs, *_ = wrapper.step(np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(obs["flux_mask"], [0, 0, 1, 0])
    np.testing.assert_array_equal(obs["flux_history"][-1], 0)
    assert np.all(np.isfinite(obs["flux_history"]))
    obs, *_ = wrapper.step(np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(obs["flux_mask"], [0, 1, 0, 1])
    assert obs["flux_history"][-1, 1, 0] == 20065


@pytest.mark.parametrize("invalid", [None, np.zeros(3)])
def test_malformed_flux_required_by_actor_keeps_baseline_failure(invalid):
    from train.flux_observation import FluxObservationWrapper

    baseline, _ = wrap_replay(ObservationWrapper, raw=FluxReplay({1: invalid}))
    wrapper, _ = wrap_replay(FluxObservationWrapper, raw=FluxReplay({1: invalid}))
    baseline.reset()
    wrapper.reset()
    with pytest.raises(ValueError, match="valid rx .* zx .* Fx"):
        baseline.step(np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match="valid rx .* zx .* Fx"):
        wrapper.step(np.zeros(7, dtype=np.float32))


@pytest.mark.parametrize("invalid", [np.full(4290, np.nan), np.full(4290, np.inf)])
def test_nonfinite_full_size_flux_is_masked_with_real_actor_xpt_encoder(invalid):
    from train.flux_observation import FluxObservationWrapper

    baseline, _ = wrap_replay(ObservationWrapper, raw=FluxReplay({1: invalid}))
    wrapper, _ = wrap_replay(FluxObservationWrapper, raw=FluxReplay({1: invalid}))
    baseline.reset()
    wrapper.reset()
    with np.errstate(invalid="ignore"):
        expected, *_ = baseline.step(np.zeros(7, dtype=np.float32))
        actual, *_ = wrapper.step(np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(actual["actor_obs"], expected)
    np.testing.assert_array_equal(actual["flux_mask"], [0, 0, 1, 0])
    np.testing.assert_array_equal(actual["flux_history"][-1], 0)


class NumericFluxEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "actor_obs": gym.spaces.Box(-np.inf, np.inf, (812,), dtype=np.float32),
        "flux_history": gym.spaces.Box(-np.inf, np.inf, (4, 66, 65), dtype=np.float32),
        "flux_mask": gym.spaces.Box(0, 1, (4,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    def __init__(self, offset=0):
        self.episode = -1
        self.offset = offset

    def current(self):
        flux = np.zeros((4, 66, 65), dtype=np.float32)
        mask = np.zeros(4, dtype=np.float32)
        for slot, value in enumerate(self.frames[-4:], start=4 - len(self.frames[-4:])):
            flux[slot] = value
            mask[slot] = 1
        return {"actor_obs": np.arange(812, dtype=np.float32) / 10 + self.frames[-1],
                "flux_history": flux, "flux_mask": mask}

    def reset(self, *, seed=None, options=None):
        self.episode += 1
        self.index = 0
        self.frames = [2 + self.episode * 4 + self.offset]
        return self.current(), {}

    def step(self, action):
        self.index += 1
        self.frames.append(self.frames[-1] * 2)
        return self.current(), float(self.index), False, self.index == 2, {}


class ActorOnly(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["actor_obs"]

    def observation(self, obs):
        return obs["actor_obs"]


def make_vec(cls, **kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return cls(DummyVecEnv([NumericFluxEnv, lambda: NumericFluxEnv(offset=10)]), **kwargs)


def test_actor_and_reward_normalization_match_baseline_independent_statistics():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train.flux_observation import FluxVecNormalize

    flux = make_vec(FluxVecNormalize, gamma=0.995)
    baseline = VecNormalize(DummyVecEnv([lambda: ActorOnly(NumericFluxEnv()),
                                       lambda: ActorOnly(NumericFluxEnv(offset=10))]), gamma=0.995)
    np.testing.assert_array_equal(flux.reset()["actor_obs"], baseline.reset())
    assert flux.norm_obs_keys == ["actor_obs"]
    assert list(flux.obs_rms) == ["actor_obs"]
    for _ in range(6):
        obs, rewards, dones, infos = flux.step(np.zeros((2, 7), dtype=np.float32))
        expected, expected_rewards, expected_dones, expected_infos = baseline.step(np.zeros((2, 7), dtype=np.float32))
        np.testing.assert_array_equal(obs["actor_obs"], expected)
        np.testing.assert_array_equal(rewards, expected_rewards)
        np.testing.assert_array_equal(dones, expected_dones)
        np.testing.assert_array_equal(flux.obs_rms["actor_obs"].mean, baseline.obs_rms.mean)
        np.testing.assert_array_equal(flux.obs_rms["actor_obs"].var, baseline.obs_rms.var)
        assert flux.ret_rms.count == baseline.ret_rms.count
        for index, done in enumerate(dones):
            if done:
                np.testing.assert_array_equal(infos[index]["terminal_observation"]["actor_obs"],
                                              expected_infos[index]["terminal_observation"])


def test_physical_actor_normalization_matches_baseline_without_raw_float32_cast():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train.flux_observation import FluxObservationWrapper, FluxVecNormalize

    # SubprocVecEnv stacks raw float64 actor features. Give DummyVecEnv float64
    # buffers here so this local test retains the same precision at its boundary.
    baseline, _ = wrap_replay(ObservationWrapper, delay=1)
    treatment, _ = wrap_replay(FluxObservationWrapper, delay=1)
    baseline.observation_space = gym.spaces.Box(-np.inf, np.inf, (812,), dtype=np.float64)
    treatment.observation_space["actor_obs"] = gym.spaces.Box(-np.inf, np.inf, (812,), dtype=np.float64)
    baseline = VecNormalize(DummyVecEnv([lambda: baseline]), gamma=0.995)
    treatment = FluxVecNormalize(DummyVecEnv([lambda: treatment]), gamma=0.995)
    np.testing.assert_array_equal(treatment.reset()["actor_obs"], baseline.reset())
    for step in range(6):
        action = np.linspace(-0.7, 0.7, 7, dtype=np.float32)[None] * (-1) ** step
        actual, reward, *_ = treatment.step(action)
        expected, expected_reward, *_ = baseline.step(action)
        np.testing.assert_array_equal(actual["actor_obs"], expected)
        np.testing.assert_array_equal(reward, expected_reward)
        np.testing.assert_array_equal(treatment.obs_rms["actor_obs"].mean, baseline.obs_rms.mean)
        np.testing.assert_array_equal(treatment.obs_rms["actor_obs"].var, baseline.obs_rms.var)


def test_scalar_flux_stats_update_main_observations_once_and_terminal_uses_current_stats():
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train.flux_observation import FluxVecNormalize

    norm = FluxVecNormalize(DummyVecEnv([NumericFluxEnv]), clip_obs=0.5)
    initial = norm.reset()
    pixels = 4290
    assert norm.flux_rms.mean.shape == () and norm.flux_rms.var.shape == ()
    assert norm.flux_rms.count == pytest.approx(1e-4 + pixels)
    np.testing.assert_array_equal(initial["flux_history"][:, :3], 0)
    raw_initial = norm.get_original_obs()
    norm.normalize_obs(raw_initial)
    norm.normalize_obs(raw_initial)
    assert norm.flux_rms.count == pytest.approx(1e-4 + pixels)
    norm.step(np.zeros((1, 7)))
    assert norm.flux_rms.count == pytest.approx(1e-4 + 3 * pixels)
    obs, _, dones, infos = norm.step(np.zeros((1, 7)))
    assert dones[0]
    # Main reset values seen: 2, then history (2,4), then new episode's 6.
    count = 1e-4 + 4 * pixels
    mean = 14 * pixels / count
    variance = (60 * pixels + 1e-4) / count - mean**2
    assert norm.flux_rms.count == pytest.approx(count)
    assert norm.flux_rms.mean == pytest.approx(mean)
    assert norm.flux_rms.var == pytest.approx(variance)
    terminal = infos[0]["terminal_observation"]
    np.testing.assert_array_equal(terminal["flux_mask"], [0, 1, 1, 1])
    np.testing.assert_array_equal(terminal["flux_history"][0], 0)
    expected = np.clip((np.array([2, 4, 8]) - mean) / np.sqrt(variance + norm.epsilon), -0.5, 0.5)
    np.testing.assert_allclose(terminal["flux_history"][1:, 0, 0], expected)
    np.testing.assert_array_equal(obs["flux_mask"], [[0, 0, 0, 1]])
    np.testing.assert_array_equal(obs["flux_history"][0, :3], 0)
    assert np.max(np.abs(obs["flux_history"])) <= 0.5


@pytest.mark.parametrize("norm_obs", [True, False])
def test_frozen_serialized_normalizer_keeps_flux_statistics_and_raw_padding(tmp_path, norm_obs):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train.flux_observation import FluxVecNormalize

    norm = make_vec(FluxVecNormalize, norm_obs=norm_obs)
    norm.reset()
    norm.step(np.zeros((2, 7)))
    norm.training = False
    norm.norm_reward = False
    stats = (norm.flux_rms.mean.copy(), norm.flux_rms.var.copy(), norm.flux_rms.count)
    path = tmp_path / "vecnormalize.pkl"
    norm.save(str(path))
    restored = VecNormalize.load(str(path), DummyVecEnv([NumericFluxEnv, lambda: NumericFluxEnv(offset=10)]))
    assert isinstance(restored, FluxVecNormalize)
    assert restored.norm_obs_keys == ["actor_obs"]
    assert not restored.training
    for _ in range(2):
        result = restored.reset()
        raw = restored.get_original_obs()
        np.testing.assert_array_equal(result["flux_mask"], raw["flux_mask"])
        np.testing.assert_array_equal(result["flux_history"][:, :3], 0)
        np.testing.assert_allclose(restored.unnormalize_obs(result)["flux_history"], raw["flux_history"], atol=1e-5)
        restored.step(np.zeros((2, 7)))
        np.testing.assert_array_equal(restored.flux_rms.mean, stats[0])
        np.testing.assert_array_equal(restored.flux_rms.var, stats[1])
        assert restored.flux_rms.count == stats[2]
    if not norm_obs:
        assert stats[2] == 1e-4
        np.testing.assert_array_equal(result["flux_history"], raw["flux_history"])
