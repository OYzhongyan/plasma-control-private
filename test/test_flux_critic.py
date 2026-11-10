"""Real SB3 contracts for the actor-preserving, flux-only value branch."""
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def spaces():
    actor = gym.spaces.Box(-np.inf, np.inf, (812,), np.float32)
    observation = gym.spaces.Dict({
        "actor_obs": actor,
        "flux_history": gym.spaces.Box(-np.inf, np.inf, (4, 66, 65), np.float32),
        "flux_mask": gym.spaces.Box(0, 1, (4,), np.float32),
    })
    return actor, observation, gym.spaces.Box(-1, 1, (7,), np.float32)


def make_policy():
    from train.flux_critic_policy import FluxCriticPolicy
    _, observation, action = spaces()
    return FluxCriticPolicy(observation, action, lambda _: 1e-4,
                            net_arch=dict(pi=[256, 256], vf=[256, 256]),
                            activation_fn=torch.nn.Tanh, log_std_init=-1)


def batch(size=2):
    generator = torch.Generator().manual_seed(19)
    return {"actor_obs": torch.randn(size, 812, generator=generator),
            "flux_history": torch.randn(size, 4, 66, 65, generator=generator),
            "flux_mask": torch.tensor([[0., 0., 1., 1.]]).repeat(size, 1)}


def actor_parameters(policy):
    return [*policy.mlp_extractor.policy_net.parameters(),
            *policy.action_net.parameters(), policy.log_std]


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_actor_initialization_and_sampling_rng_equal_baseline():
    # Extra initialization outside the isolated RNG context breaks exact parity.
    actor, _, action = spaces()
    torch.manual_seed(0)
    baseline = ActorCriticPolicy(actor, action, lambda _: 1e-4,
                                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                activation_fn=torch.nn.Tanh, log_std_init=-1)
    expected_rng = torch.get_rng_state().clone()
    observations = batch()
    expected_actions = baseline.get_distribution(observations["actor_obs"]).get_actions()
    torch.manual_seed(0)
    policy = make_policy()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for actual, expected in zip(actor_parameters(policy), actor_parameters(baseline)):
        assert torch.equal(actual, expected)
    assert torch.equal(policy.get_distribution(observations).get_actions(), expected_actions)
    assert sum(p.numel() for p in actor_parameters(policy)) == 275726
    assert sum(p.numel() for p in policy.flux_critic.parameters()) == 989185
    assert sum(p.numel() for p in policy.parameters()) == 1264911
    assert not list(policy.features_extractor.parameters())
    optimized = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in policy.parameters()}


def test_action_only_prediction_never_runs_cnn_or_uses_flux():
    policy = make_policy()
    observations = batch()
    changed = {key: value.clone() for key, value in observations.items()}
    changed["flux_history"].fill_(234.)
    changed["flux_mask"].fill_(1.)
    def reject_critic(*_):
        raise AssertionError("Action inference executed flux critic")
    handle = policy.flux_critic.register_forward_pre_hook(reject_critic)
    try:
        torch.manual_seed(12)
        before = policy._predict(observations)
        torch.manual_seed(12)
        after = policy._predict(changed)
        assert torch.equal(before, after)
        numpy_obs = {key: value.numpy() for key, value in observations.items()}
        actual, _ = policy.predict(numpy_obs, deterministic=True)
        expected = policy._predict(observations, deterministic=True).detach().numpy()
        np.testing.assert_array_equal(actual, np.clip(expected, -1, 1))
    finally:
        handle.remove()


def test_value_loss_reaches_critic_and_never_actor():
    policy = make_policy()
    observations = batch()
    actions, values, log_prob = policy(observations)
    evaluated, evaluated_log_prob, entropy = policy.evaluate_actions(observations, actions)
    assert actions.shape == (2, 7)
    assert values.shape == (2, 1)
    assert log_prob.shape == entropy.shape == (2,)
    assert torch.equal(values, evaluated)
    assert torch.equal(log_prob, evaluated_log_prob)
    assert torch.equal(policy.predict_values(observations), values)
    policy.zero_grad(set_to_none=True)
    ((values - 3.) ** 2).mean().backward()
    assert all(parameter.grad is None for parameter in actor_parameters(policy))
    assert all(parameter.grad is not None for parameter in policy.flux_critic.parameters())
    assert any(torch.count_nonzero(p.grad) for p in policy.flux_critic.parameters())


def test_invalid_flux_is_masked_but_valid_flux_and_time_order_affect_values():
    policy = make_policy()
    observations = batch()
    changed = {key: value.clone() for key, value in observations.items()}
    changed["flux_history"][:, :2] = 100000.
    before = policy.predict_values(observations)
    assert torch.equal(before, policy.predict_values(changed))
    changed["flux_history"][:, 2:] = 100.
    assert not torch.allclose(before, policy.predict_values(changed))
    swapped = {key: value.clone() for key, value in observations.items()}
    swapped["flux_history"] = swapped["flux_history"].flip(1)
    swapped["flux_mask"] = swapped["flux_mask"].flip(1)
    assert not torch.allclose(before, policy.predict_values(swapped))


class TinyFluxEnv(gym.Env):
    """500-step-style time limit shortened to exercise bootstrap cheaply."""
    _, observation_space, action_space = spaces()

    def __init__(self, true_failure=False):
        self.true_failure = true_failure

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {}

    def observation(self):
        return {"actor_obs": np.full(812, self.steps / 10, np.float32),
                "flux_history": np.full((4, 66, 65), self.steps / 10, np.float32),
                "flux_mask": np.ones(4, np.float32)}

    def step(self, action):
        self.steps += 1
        ended = self.steps == 2
        return (self.observation(), 1., ended and self.true_failure,
                ended and not self.true_failure, {})


@pytest.mark.parametrize("true_failure", [False, True])
def test_real_ppo_timeout_bootstrap_and_checkpoint_reload(tmp_path, true_failure):
    from train.flux_critic_policy import FluxCriticPolicy
    env = DummyVecEnv([lambda: TinyFluxEnv(true_failure=true_failure)])
    model = PPO(FluxCriticPolicy, env, n_steps=2, batch_size=2,
                n_epochs=1, seed=0, device="cpu", gamma=.995)
    original = [p.detach().clone() for p in model.policy.parameters()]
    # Collect using the actual policy and timeout path, then check terminal value.
    terminal = TinyFluxEnv()
    terminal.reset()
    terminal.step(np.zeros(7))
    final_obs, *_ = terminal.step(np.zeros(7))
    terminal_tensors, _ = model.policy.obs_to_tensor(final_obs)
    expected_bootstrap = model.policy.predict_values(terminal_tensors).item()
    model.learn(total_timesteps=2)
    np.testing.assert_allclose(model.rollout_buffer.rewards[-1, 0],
                               1 if true_failure else 1 + .995 * expected_bootstrap,
                               rtol=1e-6)
    assert any(not torch.equal(a, b) for a, b in zip(original, model.policy.parameters()))
    model.save(tmp_path / "flux.zip")
    restored = PPO.load(tmp_path / "flux.zip", env=env, device="cpu")
    assert type(restored.policy) is FluxCriticPolicy
    observations, _ = TinyFluxEnv().reset()
    expected, _ = model.predict(observations, deterministic=True)
    actual, _ = restored.predict(observations, deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    tensor_obs, _ = restored.policy.obs_to_tensor(observations)
    assert torch.equal(model.policy.predict_values(tensor_obs),
                       restored.policy.predict_values(tensor_obs))
    # Optimizer state must include the replacement critic after loading.
    restored.learn(total_timesteps=2)
    env.close()
