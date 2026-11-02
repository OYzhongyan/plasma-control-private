"""Streaming raw rewards must sum to the unchanged offline F2b score."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.eval.score import config, evaluate
from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list
from train.references import build_reference, default_reference_path


def base_observation():
    reference = build_reference(default_reference_path())[0]
    obs = {"reference_" + key: value for key, value in reference.items() if key != "Ip"}
    obs.update(Ip=np.array([499000.]), reference_Ip=np.array([123.]), lX=np.array([1]),
               I_PF=np.zeros(12), lcfs_points=np.asarray(reference["lcfs_points"]),
               rX=np.r_[reference["rX"], [0., 0.]], zX=np.r_[reference["zX"], [0., 0.]],
               FX=np.ones(6), nX=np.array([4]), FA=0., FB=1.,
               rx=np.linspace(.5, 2.5, 66), zx=np.linspace(-2., 2., 65), Fx=np.zeros(4290))
    frame = extract_semifinal_trajectory_from_obs_list([obs], target_obs=obs)
    target = {key: frame[source][0].copy() for key, source in
              (("lcfs_points", "lcfs_per_step"), ("Xpt_main", "Xpt_main"), ("Xpt_sec", "Xpt_sec"), ("strike", "strike"))}
    return obs, target


def target_observation(target):
    main, secondary = target["Xpt_main"], target["Xpt_sec"]
    points = np.asarray([secondary[0], main[0], main[1], secondary[1]])
    return {"reference_rX": points[:, 0], "reference_zX": points[:, 1],
            "reference_strike_r": target["strike"][:, 0], "reference_strike_z": target["strike"][:, 1]}


@pytest.mark.parametrize("case", ["perfect", "offset", "short", "limiter", "topology", "current", "coil", "coil_equal", "timing", "timeout", "failure", "long"])
def test_reward_sum_matches_offline_score(case):
    from train.rewards import CompetitionReward
    base, target = base_observation()
    refs = target_observation(target)
    reward_fn = CompetitionReward(target=target)
    n = 7 if case == "short" else 520 if case == "long" else 500
    obs_list, rewards = [], []
    times = []
    for index in range(n):
        obs = copy.deepcopy(base)
        obs.update(refs)
        obs["Ip"] = np.array([config.build_Ip_ref()[min(index, 499)]])
        info = {"step": index + 1}
        if case == "offset": obs["lcfs_points"][0, 0] += .02
        elif case == "limiter": obs["lX"] = np.array([0])
        elif case == "topology": obs["nX"] = np.array([2])
        elif case == "current" and index == 12: obs["Ip"] += 60000.
        elif case == "coil" and index == 12: obs["I_PF"][0] = 45001.
        elif case == "coil_equal": obs["I_PF"][0] = 45000.
        elif case == "timing":
            info["inference_time_ms"] = 1.
            times.append(1.)
        if case == "timeout" and index == n - 1: info["timeout"] = True
        if case == "failure" and index == n - 1: obs["failure"] = np.array([1])
        value, terminated = reward_fn(obs, np.zeros(12), terminated=(case == "failure" and index == n-1), truncated=index == n-1, info=info)
        assert terminated == (case == "failure" and index == n-1)
        obs_list.append(obs); rewards.append(value)
    trajectory = extract_semifinal_trajectory_from_obs_list(obs_list, target_obs=refs)
    if times: trajectory["inference_time_ms"] = times
    expected = evaluate({"F2b": {"trajectory": trajectory, "timeout": case == "timeout"}}, {"F2b": target})
    assert sum(rewards) == pytest.approx(expected["total_score"], abs=1e-10)
    if case in {"perfect", "coil_equal", "long"}:
        assert sum(rewards) == pytest.approx(30.)
        assert rewards[0] == pytest.approx(.06)
    if case in {"timing", "timeout"}: assert min(rewards) < 0  # reverse earlier credit


def test_reset_discards_coil_and_global_penalties():
    from train.rewards import CompetitionReward, SemifinalDenseReward
    assert SemifinalDenseReward is CompetitionReward
    obs, target = base_observation(); obs.update(target_observation(target))
    reward = CompetitionReward(target=target, reward_weights={"failure_penalty": -5., "use_competition_score": False}, score_lower_bound=False)
    bad = copy.deepcopy(obs); bad["I_PF"][0] = 50000.
    assert reward(bad, np.zeros(12), info={"step": 1})[0] == 0.
    assert reward(obs, np.zeros(12), info={"step": 2})[0] == 0.
    reward.reset_state()
    assert reward.last_components == {}
    assert reward(obs, np.zeros(12), info={"step": 1})[0] == pytest.approx(.06)


def test_parallel_environments_do_not_share_reward_state():
    from train.rewards import CompetitionReward
    obs, target = base_observation()
    first, second = CompetitionReward(target=target), CompetitionReward(target=target)
    bad = copy.deepcopy(obs); bad["I_PF"][0] = 50000.
    assert first(bad, np.zeros(12), info={"step": 1})[0] == 0.
    assert second(obs, np.zeros(12), info={"step": 1})[0] == pytest.approx(.06)


def test_implicit_target_comes_from_reference_not_actual_shape():
    from train.rewards import CompetitionReward
    obs, _ = base_observation()
    first = CompetitionReward()(obs, np.zeros(12), info={"step": 1})[0]
    obs["lcfs_points"] = obs["lcfs_points"] + .5
    changed = CompetitionReward()(obs, np.zeros(12), info={"step": 1})[0]
    assert first - changed == pytest.approx(6./500)


def test_environment_reset_clears_reward_even_mid_episode(monkeypatch):
    import environment.hfm_simulator as simulator
    from train.rewards import CompetitionReward
    obs, target = base_observation()
    class Engine:
        def reset(self, **kwargs): return {}
    monkeypatch.setattr(simulator, "HFMSocketPredictor", lambda config: Engine())
    monkeypatch.setattr(simulator, "_build_reference_trajectory", lambda **kwargs: {"Ip": [499000.]})
    monkeypatch.setattr(simulator, "_obs_dict_from_raw", lambda **kwargs: copy.deepcopy(obs))
    reward = CompetitionReward(target=target)
    bad = copy.deepcopy(obs); bad["I_PF"][0] = 50000.
    reward(bad, np.zeros(12), info={"step": 1})
    env = simulator.HFMSimulator({"max_steps": 500}, reward_fn=reward)
    env.reset()
    assert reward.last_components == {}
    assert reward(obs, np.zeros(12), info={"step": 1})[0] == pytest.approx(.06)


def test_saved_hfm_observations_match_saved_trajectory_score():
    from train.rewards import CompetitionReward, _target_from_observation
    import json
    directory = ROOT / "results/flow_check_20261030/train_hydra/eval"
    if not (directory / "rollout_observations.npz").is_file():
        pytest.skip("Optional saved HFM rollout is not present")
    saved = np.load(directory / "rollout_observations.npz")
    trajectory = json.loads((directory / "infer_result.json").read_text())["F2b"]["trajectory"]
    reward = CompetitionReward()
    values = []
    target = None
    n = int(saved["valid"][0].sum())
    for index in range(n):
        obs = {"Ip": np.array([saved["ip"][0, index]]), "reference_Ip": np.array([saved["reference_ip"][0, index]]),
               "lX": np.array([trajectory["lX"][index]]), "I_PF": trajectory["Icoil"][index],
               "FA": trajectory["psia"][index], "FB": saved["fb"][0, index],
               "lcfs_points": saved["lcfs_points"][0, index], "reference_lcfs_points": saved["reference_lcfs_points"][0, index],
               "rX": saved["rX"][0, index], "zX": saved["zX"][0, index], "nX": np.array([saved["nX"][0, index]]),
               "rx": saved["rx"], "zx": saved["zx"], "Fx": saved["fx"][0, index],
               "reference_rX": saved["ref_rX"][0, index], "reference_zX": saved["ref_zX"][0, index],
               "reference_strike_r": saved["ref_strike_r"][0, index], "reference_strike_z": saved["ref_strike_z"][0, index]}
        # NPZ omits raw FX; reconstruct matched values from the saved trajectory.
        fluxes = np.array([trajectory["psiX_sec"][index][0], *trajectory["psiX_main"][index], trajectory["psiX_sec"][index][1]])
        obs["FX"] = np.zeros(6)
        for j in range(int(obs["nX"][0])):
            dist = (saved["x_slot_r"][0, index] - obs["rX"][j])**2 + (saved["x_slot_z"][0, index] - obs["zX"][j])**2
            obs["FX"][j] = fluxes[int(np.argmin(dist))]
        if target is None: target = _target_from_observation(obs)
        values.append(reward(obs, np.zeros(12), truncated=index==n-1, info={"step": index+1})[0])
    expected = evaluate({"F2b": {"trajectory": trajectory, "timeout": False}}, {"F2b": target})
    assert sum(values) == pytest.approx(expected["total_score"], abs=1e-10)


def test_training_factory_and_ppo_update_with_competition_reward(monkeypatch):
    gym = pytest.importorskip("gymnasium")
    PPO = pytest.importorskip("stable_baselines3").PPO
    import train.train_ppo as entrypoint
    from train.rewards import CompetitionReward
    base, _ = base_observation()

    class ReplaySimulator(gym.Env):
        observation_space = gym.spaces.Dict({})
        action_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)
        def __init__(self, config, reward_fn):
            assert isinstance(reward_fn, CompetitionReward)
            self.reward_fn = reward_fn
        def reset(self, *, seed=None, options=None):
            self.index = 0
            self.reward_fn.reset_state()
            return copy.deepcopy(base), {}
        def step(self, action):
            self.index += 1
            obs = copy.deepcopy(base)
            obs["Ip"] = np.array([config.build_Ip_ref()[self.index-1]])
            info = {"step": self.index}
            reward, terminated = self.reward_fn(obs, action, truncated=self.index == 8, info=info)
            return obs, reward, terminated, self.index == 8, info

    monkeypatch.setattr(entrypoint, "HFMSimulator", ReplaySimulator)
    env = entrypoint.make_env({}, port=2223, shot_id="15892_400", max_steps=8,
                              reference_mode="trajectory", references=build_reference(default_reference_path(), max_steps=8),
                              action_mode="delta", obs_history_step=4, reward_weights={})()
    try:
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(7))
        assert sum(info["reward_components"][f"F2b_{name}"] for name in config.metric_weights()) == pytest.approx(reward)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs={"net_arch": [8]}, device="cpu", verbose=0, seed=0)
        model.learn(total_timesteps=16)
        assert model.num_timesteps == 16
        assert model._n_updates == 2
    finally:
        env.close()
