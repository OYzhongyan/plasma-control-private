"""Offline tests for the modular semifinal training baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from environment import NormalizedAction7DTo12DWrapper  # noqa: E402
from environment.preprocessing import action_7d_to_12d  # noqa: E402
from training.features import (  # noqa: E402
    FeatureMetadata,
    PhysicalFeatureHistoryWrapper,
    flatten_observation,
    physical_f2_xpt_feature_vector,
    physical_feature_vector,
)
from training.references import (  # noqa: E402
    build_hold_initial_reference_from_obs,
    build_reference,
    default_reference_path,
    load_target_reference,
)
from training.rewards import SemifinalDenseReward  # noqa: E402
from training.reward_f2 import F2XPTTargetReward  # noqa: E402
from training.train_ppo import (  # noqa: E402
    HoldInitialResetWrapper,
    ReferenceResetWrapper,
    RewardInfoWrapper,
    ResetPerturbationSampler,
    _build_training_context,
    assign_shots_to_envs,
    build_ppo_init_kwargs,
    load_train_config,
    ports_for_envs,
)
import inference_common as submission_inference  # noqa: E402


def _base_obs() -> dict:
    rx = np.linspace(0.25, 1.35, 66)
    zx = np.linspace(-1.7, 1.7, 65)
    return {
        "Ip": np.array([500_000.0]),
        "R": np.array([0.8]),
        "Z": np.array([0.0]),
        "lX": np.array([0.0]),
        "reference_Ip": np.array([500_000.0]),
        "reference_R": np.array([0.8]),
        "reference_Z": np.array([0.0]),
        "reference_lX": np.array([0.0]),
        "lcfs_points": np.zeros((32, 2)),
        "reference_lcfs_points": np.zeros((32, 2)),
        "I_PF": np.zeros(12),
        "Rmax": np.array([1.2]),
        "Rmin": np.array([0.3]),
        "aminor": np.array([0.4]),
        "deltal": np.array([0.0]),
        "deltau": np.array([0.0]),
        "kappa": np.array([1.6]),
        "rc": np.array([0.8]),
        "zc": np.array([0.0]),
        "FA": np.array([0.0]),
        "FB": np.array([0.0]),
        "failure": np.array([0]),
        "rX": np.array([0.7, 0.6, 0.6, 0.7, 0.0, 0.0]),
        "zX": np.array([1.4, 1.0, -1.0, -1.4, 0.0, 0.0]),
        "FX": np.zeros(6),
        "nX": np.array([4]),
        "Fx": np.zeros(66 * 65),
        "rx": rx,
        "zx": zx,
        "reference_rX": np.array([0.7, 0.6, 0.6, 0.7]),
        "reference_zX": np.array([1.4, 1.0, -1.0, -1.4]),
        "reference_x_valid": np.ones(4),
        "reference_strike_r": np.linspace(0.3, 1.2, 8),
        "reference_strike_z": np.linspace(-1.6, 1.6, 8),
        "reference_strike_valid": np.ones(8),
        "reference_nX": np.array([4]),
        "reference_n_strike": np.array([8]),
    }


def test_reference_shapes():
    f1 = load_target_reference(default_reference_path("F1"))
    ref1 = build_reference("F1", f1, 300)
    assert np.asarray(ref1["Ip"]).shape == (300,)
    assert np.asarray(ref1["lcfs_points"]).shape == (300, 32, 2)
    assert "R" not in ref1
    assert "Z" not in ref1

    f1_public = load_target_reference(ROOT / "evaluation" / "results" / "target.json", task="F1")
    ref_public = build_reference("F1", f1_public, 300)
    assert np.asarray(ref_public["Ip"])[99] == 500_000.0
    assert np.asarray(ref_public["Ip"])[100] == 498_500.0
    assert np.asarray(ref_public["Ip"])[-1] == 200_000.0
    assert np.asarray(ref_public["lcfs_points"]).shape == (300, 32, 2)

    f2 = load_target_reference(default_reference_path("F2"))
    ref2 = build_reference("F2", f2, 500)
    assert np.asarray(ref2["Ip"]).shape == (500,)
    assert np.asarray(ref2["rX"]).shape == (4,)
    assert np.asarray(ref2["strike_r"]).shape == (8,)


def test_feature_dims_and_flatten():
    obs = _base_obs()
    meta1 = FeatureMetadata.for_task("F1")
    meta2 = FeatureMetadata.for_task("F2")
    flat1 = flatten_observation(obs, meta1.keys, meta1.shapes)
    flat2 = flatten_observation(obs, meta2.keys, meta2.shapes)
    assert flat1.shape == (meta1.obs_dim,)
    assert flat2.shape == (meta2.obs_dim,)
    assert meta2.obs_dim > meta1.obs_dim


def test_stage0_config_loading_and_assignment():
    cfg_path = ROOT / "training" / "configs" / "f1_stage0a_parallel_hold_no_perturb.json"
    cfg = load_train_config(cfg_path)
    assert cfg["task"] == "F1"
    assert cfg["reference_mode"] == "hold_initial"
    shots = assign_shots_to_envs(cfg["shot_groups"], cfg["num_envs"], "13844_500")
    assert shots[:4] == ["13844_500", "13844_600", "15892_400", "15892_500"]
    first_port = int(cfg["ports"])
    assert ports_for_envs(cfg["ports"], cfg["num_envs"])[:4] == [
        first_port,
        first_port + 1,
        first_port + 2,
        first_port + 3,
    ]
    ctx = _build_training_context(type("Args", (), {
        "train_config": cfg_path,
        "task": None,
        "config": None,
        "reference": None,
        "ports": None,
        "shot_id": None,
        "max_steps": None,
        "total_timesteps": None,
        "save_dir": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "gamma": None,
        "seed": None,
    })())
    assert ctx["metadata"].action_units == "normalized"
    assert ctx["ppo"]["device"] == "auto"
    assert ctx["ppo"]["progress_bar"] is True
    assert ctx["vecnormalize"]["enabled"] is True
    assert ctx["vecnormalize"]["norm_reward"] is True
    assert ctx["checkpoint"]["enabled"] is True
    assert ctx["reward_log_interval"] > 0


def test_stage1_official_target_config():
    cfg_path = ROOT / "training" / "configs" / "f1_stage1.json"
    cfg = load_train_config(cfg_path)
    assert cfg["task"] == "F1"
    assert cfg["stage"] == "f1_stage1"
    assert cfg["reference_mode"] == "trajectory"
    assert cfg["max_steps"] == 300
    assert cfg["reward_weights"]["r_score"] == 0.0
    assert cfg["reward_weights"]["z_score"] == 0.0
    assert cfg["reward_weights"]["limiter_transition_score"] > 0.0
    ctx = _build_training_context(type("Args", (), {
        "train_config": cfg_path,
        "task": None,
        "config": None,
        "reference": None,
        "ports": None,
        "shot_id": None,
        "max_steps": None,
        "total_timesteps": None,
        "save_dir": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "gamma": None,
        "seed": None,
    })())
    assert ctx["reference_mode"] == "trajectory"
    assert ctx["max_steps"] == 300
    assert ctx["reference_path"].name == "target.json"
    assert ctx["metadata"].action_units == "normalized"
    assert np.asarray(ctx["reference"]["Ip"]).shape == (300,)
    assert np.asarray(ctx["reference"]["lcfs_points"]).shape == (300, 32, 2)
    assert "R" not in ctx["reference"]
    assert "Z" not in ctx["reference"]
    shots = [spec["shot_id"] for spec in ctx["env_specs"]]
    assert sorted(set(shots)) == ["13844_500", "13844_600"]


def test_ppo_config_builds_policy_kwargs():
    ppo = {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "clip_range": 0.15,
        "ent_coef": 0.001,
        "vf_coef": 0.4,
        "max_grad_norm": 0.6,
        "target_kl": 0.02,
        "net_arch": [64, 32],
        "activation_fn": "relu",
        "log_std_init": -2.0,
        "device": "auto",
        "progress_bar": True,
        "seed": 7,
    }
    policy, kwargs = build_ppo_init_kwargs(ppo, max_steps=80)
    assert policy == "MlpPolicy"
    assert kwargs["n_steps"] == 80
    assert kwargs["gae_lambda"] == 0.9
    assert kwargs["target_kl"] == 0.02
    assert kwargs["policy_kwargs"]["net_arch"] == {"pi": [64, 32], "vf": [64, 32]}
    assert kwargs["policy_kwargs"]["log_std_init"] == -2.0


def test_stage0b_reset_perturbation_config():
    cfg_path = ROOT / "training" / "configs" / "f1_stage0b_parallel_hold_perturb.json"
    cfg = load_train_config(cfg_path)
    perturb = cfg["reset_perturbation"]
    assert cfg["stage"] == "stage0b_hold_perturb"
    assert cfg["num_envs"] == 64
    assert cfg["max_steps"] == 200
    assert cfg["ppo"]["n_steps"] == 128
    assert cfg["ppo"]["batch_size"] == 1024
    assert cfg["ppo"]["n_epochs"] == 6
    assert perturb["enabled"] is True
    assert perturb["signeo_scale_range"] != [1.0, 1.0]
    assert perturb["bp_delta_range"] != [0.0, 0.0]
    assert perturb["q0_delta_range"] != [0.0, 0.0]


def test_f2_stage0_configs_include_xpt_hold():
    cfg0a = load_train_config(ROOT / "training" / "configs" / "f2_stage0a_parallel_hold_no_perturb.json")
    cfg0b = load_train_config(ROOT / "training" / "configs" / "f2_stage0b_parallel_hold_perturb.json")
    for cfg in (cfg0a, cfg0b):
        assert cfg["task"] == "F2"
        assert cfg["reference_mode"] == "hold_initial"
        assert cfg["num_envs"] == 64
        assert cfg["shot_groups"]["xpt_hold"] == ["13906_500"]
        shots = assign_shots_to_envs(cfg["shot_groups"], cfg["num_envs"], "13906_500")
        assert sorted(set(shots)) == ["13844_500", "13844_600", "13906_500", "15892_400", "15892_500"]
        assert shots[:5] == ["13844_500", "13844_600", "15892_400", "15892_500", "13906_500"]

    assert cfg0a["reset_perturbation"]["enabled"] is False
    assert cfg0b["reset_perturbation"]["enabled"] is True
    assert cfg0b["max_steps"] == 200
    assert cfg0b["ppo"]["n_steps"] == 128
    assert cfg0b["ppo"]["batch_size"] == 1024
    assert cfg0b["resume"]["enabled"] is True
    assert cfg0b["resume"]["model_path"] == "results/f2_stage0a_parallel_hold/ppo_f2_latest.zip"
    assert cfg0b["resume"]["vecnormalize_path"] == "results/f2_stage0a_parallel_hold/vecnormalize.pkl"


def test_f2_direct_target_xpt_configs():
    cfg_paths = [
        ROOT / "training" / "configs" / "f2a_direct_target_xpt.json",
        ROOT / "training" / "configs" / "f2b_direct_target_xpt.json",
    ]
    expected_shots = [
        ["13844_500", "13844_600"],
        ["15892_400", "15892_500"],
    ]
    for cfg_path, shots_expected in zip(cfg_paths, expected_shots):
        cfg = load_train_config(cfg_path)
        assert cfg["task"] == "F2"
        assert cfg["reward_type"] == "f2_target_v1"
        assert cfg["reference_mode"] == "trajectory"
        assert cfg["max_steps"] == 500
        assert cfg["num_envs"] == 48
        assert cfg["ppo"]["total_timesteps"] == 9830400
        assert cfg["ppo"]["n_steps"] == 256
        assert cfg["ppo"]["batch_size"] == 2048
        assert cfg["features"]["mode"] == "physical_f2_xpt_v1"
        assert cfg["resume"]["enabled"] is False
        ctx = _build_training_context(type("Args", (), {
            "train_config": cfg_path,
            "task": None,
            "config": None,
            "reference": None,
            "ports": None,
            "shot_id": None,
            "max_steps": None,
            "total_timesteps": None,
            "save_dir": None,
            "learning_rate": None,
            "n_steps": None,
            "batch_size": None,
            "n_epochs": None,
            "gamma": None,
            "gae_lambda": None,
            "ent_coef": None,
            "clip_range": None,
            "target_kl": None,
            "max_grad_norm": None,
            "vf_coef": None,
            "log_std_init": None,
            "net_arch": None,
            "activation_fn": None,
            "device": None,
            "reward_log_interval": None,
            "seed": None,
            "progress_bar": None,
        })())
        assert ctx["reward_type"] == "f2_target_v1"
        assert ctx["metadata"].feature_mode == "physical_f2_xpt_v1"
        assert ctx["metadata"].action_units == "normalized"
        assert ctx["reference_path"].name == "target.json"
        assert sorted(set(spec["shot_id"] for spec in ctx["env_specs"])) == shots_expected
        assert ctx["env_specs"][0]["port"] == 5558
        assert ctx["env_specs"][-1]["port"] == 5605


def test_f2a_dense_progress_regression_config():
    cfg_path = ROOT / "training" / "configs" / "f2a_direct_target_progress_regression.json"
    cfg = load_train_config(cfg_path)
    assert cfg["task"] == "F2"
    assert cfg["stage"] == "f2a_direct_target_progress_regression"
    assert cfg["reward_type"] == "semifinal_dense"
    assert cfg["reference_mode"] == "trajectory"
    assert cfg["features"]["mode"] == "physical_f2_xpt_v1"
    assert cfg["num_envs"] == 48
    assert cfg["reward_weights"]["topology_score"] == 0.0
    assert cfg["reward_weights"]["nx_score"] > 0.0
    assert cfg["reward_weights"]["x_score"] > 0.0

    ctx = _build_training_context(type("Args", (), {
        "train_config": cfg_path,
        "task": None,
        "config": None,
        "reference": None,
        "ports": None,
        "shot_id": None,
        "max_steps": None,
        "total_timesteps": None,
        "save_dir": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "gamma": None,
        "gae_lambda": None,
        "ent_coef": None,
        "clip_range": None,
        "target_kl": None,
        "max_grad_norm": None,
        "vf_coef": None,
        "log_std_init": None,
        "net_arch": None,
        "activation_fn": None,
        "device": None,
        "reward_log_interval": None,
        "seed": None,
        "progress_bar": None,
    })())
    assert ctx["reward_type"] == "semifinal_dense"
    assert ctx["metadata"].feature_mode == "physical_f2_xpt_v1"
    assert ctx["metadata"].action_units == "normalized"
    assert ctx["reference_path"].name == "target.json"
    assert sorted(set(spec["shot_id"] for spec in ctx["env_specs"])) == [
        "13844_500",
        "13844_600",
    ]
    assert ctx["env_specs"][0]["port"] == 5558
    assert ctx["env_specs"][-1]["port"] == 5605


def test_f1_precision_perturb_config():
    cfg_path = ROOT / "training" / "configs" / "f1_direct_target_precision_perturb.json"
    cfg = load_train_config(cfg_path)
    assert cfg["task"] == "F1"
    assert cfg["reference_mode"] == "trajectory"
    assert cfg["reset_perturbation"]["enabled"] is True
    assert cfg["reset_perturbation"]["signeo_scale_range"] == [0.75, 1.25]
    assert cfg["reset_perturbation"]["bp_scale_range"] == [0.95, 1.05]
    assert cfg["reset_perturbation"]["q0_scale_range"] == [0.95, 1.05]
    assert cfg["reward_weights"]["ip_score_shape"] == "exp_relative"
    assert cfg["reward_weights"]["lcfs_precision_shape"] == "exp"

    ctx = _build_training_context(type("Args", (), {
        "train_config": cfg_path,
        "task": None,
        "config": None,
        "reference": None,
        "ports": None,
        "shot_id": None,
        "max_steps": None,
        "total_timesteps": None,
        "save_dir": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "gamma": None,
        "gae_lambda": None,
        "ent_coef": None,
        "clip_range": None,
        "target_kl": None,
        "max_grad_norm": None,
        "vf_coef": None,
        "log_std_init": None,
        "net_arch": None,
        "activation_fn": None,
        "device": None,
        "reward_log_interval": None,
        "seed": None,
        "progress_bar": None,
    })())
    assert ctx["metadata"].feature_mode == "physical_v1"
    assert ctx["metadata"].action_units == "normalized"
    assert ctx["resume"]["enabled"] is True
    assert ctx["reference_path"].name == "target.json"


def test_reset_perturbation_sampler_supports_scale_ranges():
    sampler = ResetPerturbationSampler(
        shot_id="13844_500",
        reset_perturbation={
            "enabled": True,
            "signeo_scale_range": [0.75, 1.25],
            "bp_scale_range": [0.95, 1.05],
            "q0_scale_range": [0.95, 1.05],
        },
        seed=0,
    )
    seen: set[str] = set()
    for _ in range(80):
        params = sampler.sample()
        assert len(params) == 1
        key, value = next(iter(params.items()))
        seen.add(key)
        if key == "signeo":
            assert 7.5e6 * 0.75 <= value <= 7.5e6 * 1.25
        elif key == "bp":
            assert 0.21 * 0.95 <= value <= 0.21 * 1.05
        elif key == "q0":
            assert 1.5 * 0.95 <= value <= 1.5 * 1.05
        else:
            raise AssertionError(key)
    assert seen == {"signeo", "bp", "q0"}


def test_reference_reset_wrapper_injects_trajectory_perturbation():
    class _ResetEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.last_options = None

        def reset(self, *, seed=None, options=None):
            self.last_options = dict(options or {})
            return np.zeros(1, dtype=np.float32), {}

    env = _ResetEnv()
    wrapper = ReferenceResetWrapper(
        env,
        reference={"Ip": np.zeros(3)},
        shot_id="13844_500",
        reset_perturbation={
            "enabled": True,
            "signeo_scale_range": [0.75, 1.25],
        },
        seed=1,
    )
    wrapper.reset()
    assert env.last_options["reference_mode"] == "trajectory"
    assert np.allclose(env.last_options["reference"]["Ip"], np.zeros(3))
    assert set(env.last_options["reset_params"]) == {"signeo"}
    signeo = env.last_options["reset_params"]["signeo"]
    assert 7.5e6 * 0.75 <= signeo <= 7.5e6 * 1.25


def test_hold_initial_reset_perturbation_samples_single_key():
    class _ResetEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    wrapper = HoldInitialResetWrapper(
        _ResetEnv(),
        shot_id="13844_500",
        reset_perturbation={
            "enabled": True,
            "signeo_scale_range": [0.98, 1.02],
            "bp_delta_range": [-0.005, 0.005],
            "q0_delta_range": [-0.03, 0.03],
        },
        seed=123,
    )
    for _ in range(20):
        params = wrapper._sample_reset_params()
        assert len(params) == 1
        assert set(params).issubset({"signeo", "bp", "q0"})


def test_hold_initial_reference_shapes():
    obs = _base_obs()
    obs["lX"] = np.array([1.0])
    ref = build_hold_initial_reference_from_obs(obs, 8)
    assert np.asarray(ref["Ip"]).shape == (8,)
    assert np.asarray(ref["R"]).shape == (8,)
    assert np.asarray(ref["Z"]).shape == (8,)
    assert np.asarray(ref["lX"])[0] == 1.0
    assert np.asarray(ref["lcfs_points"]).shape == (8, 32, 2)


def test_physical_feature_dim_and_lcfs_absolute_error():
    obs = _base_obs()
    obs["lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["reference_lcfs_points"] = np.ones((32, 2), dtype=np.float32) * 0.03
    base = physical_feature_vector(obs, "F1")
    meta = FeatureMetadata.for_task("F1", mode="physical_v1", history_steps=4)
    assert meta.obs_dim == base.size + 4 * (7 + 1 + 2 + 12)
    assert "lcfs_error_3cm" in meta.keys
    assert np.isfinite(base).all()


def test_physical_f1_no_ref_rz_feature_mode():
    obs = _base_obs()
    meta_old = FeatureMetadata.for_task("F1", mode="physical_v1", history_steps=4)
    meta_new = FeatureMetadata.for_task("F1", mode="physical_f1_no_ref_rz_v1", history_steps=4)
    removed = {"reference_r", "r_error_3cm", "reference_z", "z_error_3cm"}

    assert removed.isdisjoint(meta_new.keys)
    assert removed.issubset(meta_old.keys)
    assert meta_new.obs_dim == meta_old.obs_dim - 4

    train_flat = physical_feature_vector(obs, "F1", mode="physical_f1_no_ref_rz_v1")
    assert train_flat.shape == (meta_new.obs_dim - 4 * (7 + 1 + 2 + 12),)
    assert np.isfinite(train_flat).all()

    sub_flat = submission_inference.flatten_observation(obs, meta_new.to_dict())
    assert sub_flat.shape == (meta_new.obs_dim,)
    assert np.isfinite(sub_flat).all()


def test_physical_f2_xpt_v1_feature_mode_is_stable():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F2"))
    obs["reference_rX"] = target.rX.copy()
    obs["reference_zX"] = target.zX.copy()
    obs["reference_x_valid"] = np.ones(4)
    obs["reference_nX"] = np.array([4])
    obs["reference_strike_r"] = target.strike_r.copy()
    obs["reference_strike_z"] = target.strike_z.copy()
    obs["reference_strike_valid"] = np.ones(8)

    meta = FeatureMetadata.for_task("F2", mode="physical_f2_xpt_v1", history_steps=4)
    base = physical_f2_xpt_feature_vector(obs)
    assert meta.feature_mode == "physical_f2_xpt_v1"
    assert meta.obs_dim == base.size + 4 * (7 + 1 + 2 + 12)
    assert "xpt_position_error_3cm" in meta.keys
    assert "x_error_3cm" not in meta.keys
    assert np.isfinite(base).all()

    for nx in (0, 2, 4, 5):
        obs["nX"] = np.array([nx])
        vec = physical_f2_xpt_feature_vector(obs)
        assert vec.shape == (base.size,)
        assert np.isfinite(vec).all()


def test_submission_physical_f2_xpt_v1_matches_training_flatten():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F2"))
    obs["reference_rX"] = target.rX.copy()
    obs["reference_zX"] = target.zX.copy()
    obs["reference_x_valid"] = np.ones(4)
    obs["reference_nX"] = np.array([4])
    obs["reference_strike_r"] = target.strike_r.copy()
    obs["reference_strike_z"] = target.strike_z.copy()
    obs["reference_strike_valid"] = np.ones(8)
    meta = FeatureMetadata.for_task("F2", mode="physical_f2_xpt_v1", history_steps=0)
    train_vec = physical_f2_xpt_feature_vector(obs)
    infer_vec = submission_inference.flatten_observation(obs, meta.to_dict())
    assert train_vec.shape == infer_vec.shape
    assert np.allclose(train_vec, infer_vec)


def test_reward_is_finite():
    obs = _base_obs()
    f1_target = load_target_reference(default_reference_path("F1"))
    f1_reward = SemifinalDenseReward("F1", f1_target)
    value1 = f1_reward(obs, np.zeros(12), info={"step": 1})
    assert np.isfinite(value1)

    f2_target = load_target_reference(default_reference_path("F2"))
    obs["lX"] = np.array([1.0])
    obs["reference_lX"] = np.array([1.0])
    f2_reward = SemifinalDenseReward("F2", f2_target)
    value2 = f2_reward(obs, np.zeros(12), info={"step": 1})
    assert np.isfinite(value2)


def test_f2_xpt_target_reward_components_are_monotonic_and_finite():
    target = load_target_reference(default_reference_path("F2"))
    obs = _base_obs()
    obs["reference_Ip"] = np.array([500_000.0])
    obs["reference_lX"] = np.array([1.0])
    obs["lX"] = np.array([1.0])
    obs["reference_lcfs_points"] = target.lcfs.copy()
    obs["lcfs_points"] = target.lcfs + np.array([0.10, 0.0])
    obs["reference_rX"] = target.rX.copy()
    obs["reference_zX"] = target.zX.copy()
    obs["reference_x_valid"] = np.ones(4)
    obs["reference_nX"] = np.array([4])
    obs["reference_strike_r"] = target.strike_r.copy()
    obs["reference_strike_z"] = target.strike_z.copy()
    obs["reference_strike_valid"] = np.ones(8)
    obs["rX"] = np.concatenate([target.rX + 0.10, np.zeros(2)])
    obs["zX"] = np.concatenate([target.zX, np.zeros(2)])
    obs["FX"] = np.array([0.02, 0.02, 0.02, 0.02, 0.0, 0.0])
    obs["FB"] = np.array([0.0])
    obs["nX"] = np.array([2])

    reward = F2XPTTargetReward(
        target,
        reward_weights={
            "failure_penalty": -100.0,
            "alive_offset": 0.0,
            "score_scale": 1.0,
            "ip_score": 0.0,
            "lcfs_component": 1.0,
            "x_count_score": 1.0,
            "x_position_component": 1.0,
            "x_flux_component": 1.0,
            "topology_score": 0.0,
            "coil_penalty": 0.0,
            "ip_error_boundary_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    initial = dict(reward.last_components)

    obs["lcfs_points"] = target.lcfs + np.array([0.03, 0.0])
    obs["rX"] = np.concatenate([target.rX + 0.01, np.zeros(2)])
    obs["zX"] = np.concatenate([target.zX, np.zeros(2)])
    obs["FX"] = np.zeros(6)
    obs["nX"] = np.array([4])
    reward(obs, np.zeros(12), info={"step": 2})
    improved = reward.last_components
    assert improved["lcfs_component"] > initial["lcfs_component"]
    assert improved["x_count_score"] > initial["x_count_score"]
    assert improved["x_position_component"] > initial["x_position_component"]
    assert improved["x_flux_component"] > initial["x_flux_component"]
    assert np.isfinite(improved["reward"])

    failure_reward = reward(obs, np.zeros(12), terminated=True, info={"step": 3})
    assert failure_reward == -100.0
    assert "lcfs_component" not in reward.last_components


def test_reward_uses_reference_lx_for_stage0_topology():
    obs = _base_obs()
    obs["lX"] = np.array([1.0])
    obs["reference_lX"] = np.array([1.0])
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "topology_score": 1.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    assert reward.last_components["topology_score"] == 1.0

    obs["reference_lX"] = np.array([0.0])
    reward(obs, np.zeros(12), info={"step": 2})
    assert reward.last_components["topology_score"] == 0.0


def test_reward_lcfs_score_uses_absolute_coordinate_rms():
    obs = _base_obs()
    obs["lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["reference_lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["reference_lcfs_points"][:, 0] = 0.03
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "lcfs_absolute_score": 1.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    assert abs(reward.last_components["lcfs_absolute_score"]) < 1e-6


def test_reward_lcfs_progress_precision_score():
    obs = _base_obs()
    obs["reference_lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"][:, 0] = 0.045
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "lcfs_absolute_score": 1.0,
            "lcfs_score_mode": "progress_precision",
            "lcfs_precision_cm": 3.0,
            "lcfs_progress_weight": 0.7,
            "lcfs_precision_weight": 0.3,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    assert reward.last_components["lcfs_cm"] > 4.4
    assert reward.last_components["lcfs_absolute_score"] == 0.0

    obs["lcfs_points"][:, 0] = 0.04
    reward(obs, np.zeros(12), info={"step": 2})
    assert reward.last_components["lcfs_progress_score"] > 0.0
    assert reward.last_components["lcfs_absolute_score"] > 0.0


def test_reward_exp_precision_and_ip_relative_score_are_monotonic():
    obs = _base_obs()
    obs["reference_lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"][:, 0] = 0.03
    obs["reference_Ip"] = np.array([500_000.0])
    obs["Ip"] = np.array([530_000.0])
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "ip_score": 1.0,
            "ip_score_shape": "exp_relative",
            "ip_relative_sigma": 0.03,
            "lcfs_absolute_score": 1.0,
            "lcfs_score_mode": "progress_precision",
            "lcfs_precision_shape": "exp",
            "lcfs_precision_sigma_cm": 1.5,
            "lcfs_precision_power": 2.0,
            "lcfs_progress_weight": 0.0,
            "lcfs_precision_weight": 1.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "ip_error_boundary_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    far = dict(reward.last_components)
    assert np.isclose(far["ip_relative_error"], 0.06)

    obs["lcfs_points"][:, 0] = 0.01
    obs["Ip"] = np.array([505_000.0])
    reward(obs, np.zeros(12), info={"step": 2})
    near = dict(reward.last_components)
    assert near["lcfs_precision_score"] > far["lcfs_precision_score"]
    assert near["ip_score"] > far["ip_score"]
    assert near["ip_relative_error"] < far["ip_relative_error"]


def test_reward_lcfs_regression_penalty():
    obs = _base_obs()
    obs["reference_lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"] = np.zeros((32, 2), dtype=np.float32)
    obs["lcfs_points"][:, 0] = 0.04
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "lcfs_absolute_score": 1.0,
            "lcfs_score_mode": "progress_precision",
            "lcfs_regression_penalty": 0.3,
            "lcfs_regression_margin_cm": 1.0,
            "lcfs_regression_scale_cm": 5.0,
            "lcfs_regression_penalty_shape": "huber",
            "lcfs_regression_huber_delta": 1.0,
            "lcfs_regression_penalty_cap": 1.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
            "failure_penalty": -100.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    assert reward.last_components["lcfs_initial_cm"] > 3.9
    assert reward.last_components["lcfs_regression_penalty"] == 0.0

    obs["lcfs_points"][:, 0] = 0.049
    reward(obs, np.zeros(12), info={"step": 2})
    assert reward.last_components["lcfs_regression_penalty"] == 0.0

    obs["lcfs_points"][:, 0] = 0.09
    reward(obs, np.zeros(12), info={"step": 3})
    assert 0.0 < reward.last_components["lcfs_regression_penalty"] < 1.0

    obs["lcfs_points"][:, 0] = 0.50
    reward(obs, np.zeros(12), info={"step": 4})
    assert reward.last_components["lcfs_regression_penalty_raw"] > 1.0
    assert reward.last_components["lcfs_regression_penalty"] == 1.0

    failure_reward = reward(obs, np.zeros(12), terminated=True, info={"step": 5})
    assert failure_reward == -100.0
    assert "lcfs_regression_penalty" not in reward.last_components


def test_f2_dense_reward_x_position_progress_regression():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F2"))
    obs["reference_lX"] = np.array([1.0])
    obs["lX"] = np.array([0.0])
    obs["reference_lcfs_points"] = target.lcfs.copy()
    obs["lcfs_points"] = target.lcfs.copy()
    obs["reference_rX"] = target.rX.copy()
    obs["reference_zX"] = target.zX.copy()
    obs["reference_x_valid"] = np.ones(4)
    obs["reference_nX"] = np.array([4])
    obs["rX"] = np.concatenate([target.rX + 0.08, np.zeros(2)])
    obs["zX"] = np.concatenate([target.zX, np.zeros(2)])
    obs["nX"] = np.array([4])

    reward = SemifinalDenseReward(
        "F2",
        target,
        reward_weights={
            "ip_score": 0.0,
            "lcfs_absolute_score": 0.0,
            "x_score": 1.0,
            "x_position_score_mode": "progress_precision",
            "x_position_precision_cm": 3.0,
            "x_position_progress_weight": 0.7,
            "x_position_precision_weight": 0.3,
            "x_position_regression_penalty": 0.25,
            "x_position_regression_margin_cm": 1.0,
            "x_position_regression_scale_cm": 5.0,
            "x_position_regression_penalty_shape": "huber",
            "x_position_regression_huber_delta": 1.0,
            "x_position_regression_penalty_cap": 1.0,
            "topology_score": 0.0,
            "nx_score": 0.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "ip_error_boundary_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    initial = dict(reward.last_components)
    assert initial["topology_score"] == 0.0
    assert initial["x_position_initial_cm"] > 7.0
    assert initial["x_score"] == 0.0

    obs["rX"] = np.concatenate([target.rX + 0.04, np.zeros(2)])
    reward(obs, np.zeros(12), info={"step": 2})
    improved = dict(reward.last_components)
    assert improved["x_position_progress_score"] > 0.0
    assert improved["x_score"] > initial["x_score"]

    obs["rX"] = np.concatenate([target.rX + 0.25, np.zeros(2)])
    reward(obs, np.zeros(12), info={"step": 3})
    assert reward.last_components["x_position_regression_penalty"] > 0.0

    obs["rX"] = np.concatenate([target.rX + 1.00, np.zeros(2)])
    reward(obs, np.zeros(12), info={"step": 4})
    assert reward.last_components["x_position_regression_penalty"] == 1.0


def test_limiter_transition_score_requires_topology_lcfs_and_ip_safety():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "limiter_transition_score": 1.0,
            "score_scale": 1.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
            "ip_error_hard_limit_a": 50_000.0,
        },
    )
    reward(obs, np.zeros(12), info={"step": 1})
    assert reward.last_components["ip_safe_score"] == 1.0
    assert reward.last_components["limiter_transition_score"] == 1.0

    obs["lX"] = np.array([1.0])
    reward(obs, np.zeros(12), info={"step": 2})
    assert reward.last_components["topology_score"] == 0.0
    assert reward.last_components["limiter_transition_score"] == 0.0

    obs["lX"] = np.array([0.0])
    obs["Ip"] = np.array([550_000.0])
    reward(obs, np.zeros(12), info={"step": 3})
    assert reward.last_components["ip_safe_score"] == 0.0
    assert reward.last_components["limiter_transition_score"] == 0.0


def test_ip_error_boundary_penalty():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "ip_score": 1.0,
            "score_scale": 0.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "ip_error_boundary_penalty": 1.0,
            "ip_error_soft_limit_a": 40_000.0,
            "ip_error_hard_limit_a": 50_000.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    obs["Ip"] = np.array([540_000.0])
    reward(obs, np.zeros(12), info={"step": 1})
    assert reward.last_components["ip_error_boundary_penalty"] == 0.0

    obs["Ip"] = np.array([545_000.0])
    reward(obs, np.zeros(12), info={"step": 2})
    assert 0.0 < reward.last_components["ip_error_boundary_penalty"] < 1.0

    obs["Ip"] = np.array([550_000.0])
    reward(obs, np.zeros(12), info={"step": 3})
    assert reward.last_components["ip_error_boundary_penalty"] == 1.0


def test_ip_error_boundary_huber_cap_penalty():
    obs = _base_obs()
    target = load_target_reference(default_reference_path("F1"))
    reward = SemifinalDenseReward(
        "F1",
        target,
        reward_weights={
            "score_scale": 0.0,
            "alive_offset": 0.0,
            "coil_penalty": 0.0,
            "ip_error_boundary_penalty": 1.0,
            "ip_error_soft_limit_a": 40_000.0,
            "ip_error_hard_limit_a": 50_000.0,
            "ip_error_boundary_penalty_shape": "huber",
            "ip_error_boundary_huber_delta": 1.0,
            "ip_error_boundary_penalty_cap": 4.0,
            "action_l2_penalty": 0.0,
            "action_smooth_penalty": 0.0,
        },
    )
    obs["Ip"] = np.array([545_000.0])
    reward(obs, np.zeros(12), info={"step": 1})
    assert np.isclose(reward.last_components["ip_error_boundary_penalty"], 0.25)

    obs["Ip"] = np.array([560_000.0])
    reward(obs, np.zeros(12), info={"step": 2})
    assert np.isclose(reward.last_components["ip_error_boundary_penalty"], 3.0)

    obs["Ip"] = np.array([620_000.0])
    reward(obs, np.zeros(12), info={"step": 3})
    assert reward.last_components["ip_error_boundary_penalty_raw"] > 4.0
    assert reward.last_components["ip_error_boundary_penalty"] == 4.0


def test_action_mapping():
    action7 = np.arange(7, dtype=np.float32)
    action12 = action_7d_to_12d(action7)
    assert action12.shape == (12,)
    assert np.allclose(action12[[1, 2]], [1, 1])
    assert np.allclose(action12[[9, 10]], [5, 5])


def test_normalized_action_mapping():
    class _ActionEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(12,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    wrapper = NormalizedAction7DTo12DWrapper(_ActionEnv())
    assert np.allclose(wrapper.action_space.low, -1.0)
    assert np.allclose(wrapper.action_space.high, 1.0)
    action12 = wrapper.action(np.array([2.0, -2.0, 0.5, 0.0, 1.0, -1.0, 0.25]))
    assert action12.shape == (12,)
    assert action12[0] == 1499
    assert np.allclose(action12[[1, 2]], [-230, -230])
    assert np.allclose(action12[[3, 4]], [86, 86])


def test_reward_info_wrapper_exposes_reward_components():
    class _Reward:
        def __init__(self):
            self.last_components = {"reward": 1.25, "ip_score": 0.5}

    class _RewardInfoEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.reward_fn = _Reward()
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.25, False, False, {}

    env = RewardInfoWrapper(_RewardInfoEnv())
    _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert info["reward_components"] == {"reward": 1.25, "ip_score": 0.5}


class _DummyDictEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({})
        self.step_count = 0

    def _obs(self) -> dict:
        obs = _base_obs()
        obs["Ip"] = np.array([500_000.0 + 1000.0 * self.step_count])
        obs["R"] = np.array([0.8 + 0.001 * self.step_count])
        obs["Z"] = np.array([0.001 * self.step_count])
        obs["I_PF"] = np.ones(12) * self.step_count
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        return self._obs(), 0.0, False, False, {}


def test_history_wrapper_reset_and_step_are_deterministic():
    meta = FeatureMetadata.for_task("F1", mode="physical_v1", history_steps=2)
    env = PhysicalFeatureHistoryWrapper(_DummyDictEnv(), metadata=meta)
    obs0, _ = env.reset(seed=1)
    history_dim = 2 * (7 + 1 + 2 + 12)
    assert obs0.shape == (meta.obs_dim,)
    assert np.allclose(obs0[-history_dim:], 0.0)

    action = np.ones(7, dtype=np.float32)
    obs1, *_ = env.step(action)
    env2 = PhysicalFeatureHistoryWrapper(_DummyDictEnv(), metadata=meta)
    env2.reset(seed=1)
    obs1b, *_ = env2.step(action)
    assert np.allclose(obs1, obs1b)
    assert not np.allclose(obs1[-history_dim:], 0.0)


def run_all():
    test_reference_shapes()
    test_feature_dims_and_flatten()
    test_stage0_config_loading_and_assignment()
    test_stage1_official_target_config()
    test_ppo_config_builds_policy_kwargs()
    test_stage0b_reset_perturbation_config()
    test_f2_stage0_configs_include_xpt_hold()
    test_f2_direct_target_xpt_configs()
    test_f2a_dense_progress_regression_config()
    test_f1_precision_perturb_config()
    test_reset_perturbation_sampler_supports_scale_ranges()
    test_reference_reset_wrapper_injects_trajectory_perturbation()
    test_hold_initial_reset_perturbation_samples_single_key()
    test_hold_initial_reference_shapes()
    test_physical_feature_dim_and_lcfs_absolute_error()
    test_physical_f1_no_ref_rz_feature_mode()
    test_physical_f2_xpt_v1_feature_mode_is_stable()
    test_submission_physical_f2_xpt_v1_matches_training_flatten()
    test_reward_is_finite()
    test_f2_xpt_target_reward_components_are_monotonic_and_finite()
    test_reward_uses_reference_lx_for_stage0_topology()
    test_reward_lcfs_score_uses_absolute_coordinate_rms()
    test_reward_lcfs_progress_precision_score()
    test_reward_exp_precision_and_ip_relative_score_are_monotonic()
    test_reward_lcfs_regression_penalty()
    test_f2_dense_reward_x_position_progress_regression()
    test_limiter_transition_score_requires_topology_lcfs_and_ip_safety()
    test_ip_error_boundary_penalty()
    test_ip_error_boundary_huber_cap_penalty()
    test_action_mapping()
    test_normalized_action_mapping()
    test_reward_info_wrapper_exposes_reward_components()
    test_history_wrapper_reset_and_step_are_deterministic()
    print("test_training_modules: all passed")


if __name__ == "__main__":
    run_all()
