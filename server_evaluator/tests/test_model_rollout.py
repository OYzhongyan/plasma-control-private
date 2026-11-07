"""Exercise real ONNX inference and scoring against a fake HFM socket boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def reference_metadata(mode):
    # Independent snapshot of the public main interface, not the server helper.
    shapes = {"reference_Ip": [1], "reference_lX": [1], "reference_lcfs_points": [32, 2],
              "reference_rX": [4], "reference_zX": [4], "reference_strike_r": [8],
              "reference_strike_z": [8]}
    return dict(version=3, feature_mode="reference_only_v1", history_steps=0,
                include_xpt=True, action_history_steps=0, action_history_dim=0,
                reference_keys=list(shapes), reference_shapes=shapes, obs_dim=90,
                action_dim=7, action_mode=mode, action_units="normalized",
                observation_normalization=True, requires_vecnormalize=True)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metadata = reference_metadata("delta")
    _write_json(model_dir / "policy2.metadata.json", metadata)
    np.savez(model_dir / "policy2_vecnormalize_stats.npz",
             obs_mean=np.zeros(90), obs_var=np.ones(90),
             clip_obs=np.array(10.), epsilon=np.array(1e-8))
    output = helper.make_tensor("constant", TensorProto.FLOAT, [1, 7], [.1] * 7)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["action"], value=output)], "test",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 90])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 7])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save_model(model, model_dir / "policy2.onnx")
    theta = np.arange(32) * 2 * np.pi / 32
    contour = np.stack([.6 + .1 * np.cos(theta), .2 * np.sin(theta)], axis=1).tolist()
    target = {"F2b": {"lcfs_points": contour,
                      "Xpt_main": [[.5, .3], [.5, -.3]],
                      "Xpt_sec": [[.6, .4], [.6, -.4]],
                      "strike": [[.6, .5]] * 8}}
    _write_json(tmp_path / "target.json", target)
    (tmp_path / "env.yaml").write_text("predictor:\n  host: wrong\n  port: 9\nmax_steps: 1\n", encoding="utf-8")
    (tmp_path / "reference.yaml").write_text(
        "shot_id: '15892_400'\nmax_steps: 1\ntarget_json: /wrong.json\nreference_mode: hold\n",
        encoding="utf-8")
    config = dict(target_path=str(tmp_path / "target.json"),
                  env_config=str(tmp_path / "env.yaml"),
                  reference_config=str(tmp_path / "reference.yaml"),
                  obs_perturb_config=None, hfm_host="127.0.0.1", hfm_port=15559,
                  socket_timeout=45., seed=123, onnx_threads=1)
    return model_dir, config, target


@pytest.fixture
def fake_hfm(monkeypatch):
    import environment

    class Simulator:
        instances = []
        fail_reset = False
        fail_step = False
        terminal_nan = False

        def __init__(self, config):
            self.config = config
            self.engine = SimpleNamespace(_power_supply=None)
            self.closed = False
            self.steps = 0
            self.actions = []
            self.instances.append(self)

        def observation(self):
            ref = self.options.get("reference") or {}
            contour = np.asarray(ref.get("lcfs_points", np.zeros((500, 32, 2))))[0]
            obs = {"Ip": np.array([500000. - self.steps * 1000]),
                   "R": np.array([.6]), "Z": np.array([0.]), "lX": np.array([1.]),
                   "I_PF": np.zeros(12), "lcfs_points": contour,
                   "nX": np.array([0.]), "FA": np.array([0.]), "FB": np.array([1.]),
                   "rX": np.zeros(6), "zX": np.zeros(6), "FX": np.zeros(6)}
            obs["reference_Ip"] = np.array([np.asarray(ref.get("Ip", [500000.]))[min(self.steps, 499)]])
            obs["reference_lX"] = np.array([1.])
            obs["reference_lcfs_points"] = contour.copy()
            for key, size in (("rX", 4), ("zX", 4), ("strike_r", 8), ("strike_z", 8)):
                obs["reference_" + key] = np.asarray(ref.get(key, np.zeros(size)))
            if self.terminal_nan and self.steps == 2:
                obs["lcfs_points"] = np.full((32, 2), np.nan)
            return obs

        def reset(self, *, seed=None, options=None):
            self.seed = seed
            self.options = options or {}
            if self.fail_reset:
                raise ConnectionError("HFM reset unavailable")
            return self.observation(), {}

        def step(self, action):
            if self.fail_step:
                raise TimeoutError("HFM STEP timed out")
            self.actions.append(np.asarray(action).copy())
            self.steps += 1
            return self.observation(), 0., self.steps == 2, False, {}

        def close(self):
            self.closed = True

    monkeypatch.setattr(environment, "HFMSimulator", Simulator)
    return Simulator


@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_main_metadata_encoding_and_actions_are_accepted(artifacts, mode):
    from server_evaluator.model_rollout import _load_policy
    model_dir, config, _ = artifacts
    _write_json(model_dir / "policy2.metadata.json", reference_metadata(mode))
    policy = _load_policy(model_dir, config)
    shapes = reference_metadata(mode)["reference_shapes"]
    observation, offset = {}, 0
    for key, shape in shapes.items():
        count = int(np.prod(shape))
        observation[key] = np.arange(offset, offset + count).reshape(shape)
        offset += count
    observation["Ip"] = np.array([np.nan])
    np.testing.assert_array_equal(policy.reference_adapter.encode(observation), np.arange(90))
    scale = np.array([1499, 230, 172, 172, 348, 348, 270]) if mode == "absolute" else (
        np.array([1499, 175, 175, 175, 175, 175, 80]) * .16)
    expected = (.1 * scale)[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]]
    np.testing.assert_allclose(policy.act(observation), expected, rtol=1e-6)
    np.testing.assert_allclose(policy.act(observation), expected * (2 if mode == "delta" else 1), rtol=1e-6)
    policy.reset()
    np.testing.assert_allclose(policy.act(observation), expected, rtol=1e-6)
    assert policy.session.get_session_options().intra_op_num_threads == 1


@pytest.mark.parametrize("field,value", [
    ("feature_mode", "historical_observation_v1"), ("history_steps", 4),
    ("obs_dim", 812), ("version", 2), ("action_units", "physical_voltage"),
    ("requires_vecnormalize", False), ("observation_normalization", 1),
    ("reference_keys", []), ("reference_shapes", {}), ("action_history_dim", 12),
])
def test_metadata_rejects_mismatched_main_contract(artifacts, field, value):
    from server_evaluator.model_rollout import _validate_metadata
    from server_evaluator.grader import InvalidSubmission
    model_dir, _, _ = artifacts
    metadata = reference_metadata("delta")
    metadata[field] = value
    _write_json(model_dir / "policy2.metadata.json", metadata)
    with pytest.raises(InvalidSubmission, match=field):
        _validate_metadata(model_dir)


def test_power_supply_uses_main_delay_ranges_and_seed():
    from environment.power_supply import PowerSupplyModel, DELAY_RANGE_PF, DELAY_RANGE_VS
    assert DELAY_RANGE_PF == (.002, .005)
    assert DELAY_RANGE_VS == (0., .001)
    first, second = PowerSupplyModel(seed=7), PowerSupplyModel(seed=7)
    first.reset()
    second.reset()
    for index in range(20):
        command = np.full(12, index * 40.)
        np.testing.assert_array_equal(first.step(command), second.step(command))
        assert np.all((first.delay_s[:11] >= .002) & (first.delay_s[:11] <= .005))
        assert 0 <= first.delay_s[11] <= .001
        np.testing.assert_array_equal(first.delay_steps, np.round(first.delay_s / .001))


def test_real_main_export_bundle_loads_and_drives_scoring(artifacts, fake_hfm, tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train.export_onnx import main as export
    from submission.model_bundle import unpack_bundle
    from submission.history import ReferencePolicyAdapter
    from server_evaluator.model_rollout import _load_policy, run_model

    class Flat(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (90,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    normalization = VecNormalize(DummyVecEnv([Flat]))
    # Non-identity statistics catch missing/double normalization at inference.
    normalization.obs_rms.mean = np.linspace(-1, 1, 90)
    normalization.obs_rms.var = np.linspace(.5, 2, 90)
    normalization.training = False
    model = sb3.PPO("MlpPolicy", normalization, n_steps=2, batch_size=2,
                    policy_kwargs={"net_arch": [8, 8]}, device="cpu", seed=7)
    directory = tmp_path / "main_export"
    directory.mkdir()
    model.save(directory / "model.zip")
    normalization.save(str(directory / "vecnormalize.pkl"))
    _write_json(directory / "f2_features.metadata.json", reference_metadata("absolute"))
    export(["--model", str(directory / "model.zip"), "--out", str(directory / "policy2.onnx"),
            "--bundle", str(directory / "submission_model.zip")])
    model_dir = unpack_bundle(directory / "submission_model.zip", tmp_path / "unpacked")
    _, config, _ = artifacts
    policy = _load_policy(model_dir, config)
    result = run_model(model_dir, config, tmp_path / "trajectory.json")
    assert result["status"] == "completed"
    assert set(result["metrics_scores"]) == {"Ip", "LCFS", "X", "strike", "psiX", "X2", "psiX2"}
    actual = fake_hfm.instances[-1]
    actual.steps = 0
    observation = actual.observation()
    adapter = ReferencePolicyAdapter(policy.metadata)
    vector = adapter.encode(observation).reshape(1, -1)
    action, _ = model.predict(normalization.normalize_obs(vector), deterministic=True)
    expected = adapter.decode(action)
    np.testing.assert_allclose(actual.actions[0], expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(policy.act(observation), expected, rtol=1e-5, atol=1e-4)


def test_direct_rollout_scores_real_post_step_observations_and_overrides_target(artifacts, fake_hfm, tmp_path):
    from server_evaluator.model_rollout import run_model
    model_dir, config, _ = artifacts
    output = tmp_path / "trajectory.json"
    result = run_model(model_dir, config, output)
    assert result["status"] == "completed"
    assert result["metrics_scores"]["Ip"] == pytest.approx(.024)
    saved = json.loads(output.read_text())
    assert saved["F2b"]["trajectory"]["Ip"] == [499000., 498000.]
    assert saved["F2b"]["f_surv"] == .004
    assert "inference_time_ms" not in saved["F2b"]["trajectory"]
    assert len(fake_hfm.instances) == 2
    assert all(env.closed and env.seed == 123 for env in fake_hfm.instances)
    actual = fake_hfm.instances[1]
    assert actual.config["max_steps"] == 500
    assert actual.config["predictor"]["host"] == "127.0.0.1"
    assert actual.config["predictor"]["port"] == 15559
    assert actual.options["reference_mode"] == "trajectory"
    assert actual.options["reference"]["Ip"][:2].tolist() == [499000., 498000.]
    assert actual.options["reference"]["rX"].tolist() == [.6, .5, .5, .6]
    assert actual.actions[0].shape == (12,)
    assert actual.actions[0][0] == pytest.approx(23.984)
    assert actual.actions[1][0] == pytest.approx(47.968)


def test_terminal_missing_geometry_keeps_original_score_and_audit_null(artifacts, fake_hfm, tmp_path):
    from server_evaluator.model_rollout import run_model
    from evaluation.eval.score import evaluate
    model_dir, config, target = artifacts
    fake_hfm.terminal_nan = True
    output = tmp_path / "trajectory.json"
    result = run_model(model_dir, config, output)
    saved = json.loads(output.read_text())
    assert saved["F2b"]["trajectory"]["lcfs_per_step"][1] == [[None, None]] * 32
    assert result["status"] == "completed"
    assert result["metrics_scores"]["Ip"] == pytest.approx(.024)
    assert result["metrics_scores"]["LCFS"] == pytest.approx(.012)
    assert result["metrics_scores"] == pytest.approx(evaluate(saved, target)["task_metric_scores"]["F2b"])
    assert all(env.closed for env in fake_hfm.instances)


@pytest.mark.parametrize("delay", [{"delay_steps": 1}, {"delay": {"enabled": True, "steps": 2}}])
def test_observation_delay_is_explicitly_rejected(artifacts, fake_hfm, tmp_path, delay):
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import GradingUnavailable
    model_dir, config, _ = artifacts
    path = tmp_path / "perturb.yaml"
    _write_json(path, {"enabled": True, **delay})
    config["obs_perturb_config"] = str(path)
    with pytest.raises(GradingUnavailable, match="delay"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


def test_noise_without_delay_only_changes_policy_observations(artifacts, fake_hfm, tmp_path):
    from server_evaluator.model_rollout import run_model
    model_dir, config, _ = artifacts
    path = tmp_path / "perturb.yaml"
    _write_json(path, {"enabled": True, "delay_steps": 0,
                       "noise": {"enabled": True, "std": {"Ip": 100.}}})
    config["obs_perturb_config"] = str(path)
    output = tmp_path / "trajectory.json"
    result = run_model(model_dir, config, output)
    assert result["metrics_scores"]["Ip"] == pytest.approx(.024)
    assert json.loads(output.read_text())["F2b"]["trajectory"]["Ip"] == [499000., 498000.]


def test_phase_file_finishes_at_scoring_with_actual_step_count(artifacts, fake_hfm, tmp_path):
    from server_evaluator.model_rollout import run_model
    model_dir, config, _ = artifacts
    phase_path = tmp_path / "phase.json"
    config["_progress_path"] = str(phase_path)
    run_model(model_dir, config, tmp_path / "trajectory.json")
    assert json.loads(phase_path.read_text()) == {"phase": "scoring", "step": 2}


def test_phase_write_failure_is_infrastructure_error_before_simulation(artifacts, fake_hfm, tmp_path):
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import GradingUnavailable
    model_dir, config, _ = artifacts
    config["_progress_path"] = str(tmp_path)  # Cannot replace a directory with a JSON file.
    with pytest.raises(GradingUnavailable, match="phase"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


@pytest.mark.parametrize("failure", ["fail_reset", "fail_step"])
def test_hfm_failure_is_unavailable_and_closes_environment(artifacts, fake_hfm, tmp_path, failure):
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import GradingUnavailable
    model_dir, config, _ = artifacts
    setattr(fake_hfm, failure, True)
    with pytest.raises(GradingUnavailable, match="HFM"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert fake_hfm.instances and all(env.closed for env in fake_hfm.instances)


@pytest.mark.parametrize("field,value", [("feature_mode", "raw"), ("obs_dim", 216), ("action_dim", 12)])
def test_invalid_metadata_rejected_before_simulation(artifacts, fake_hfm, tmp_path, field, value):
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import InvalidSubmission
    model_dir, config, _ = artifacts
    path = model_dir / "policy2.metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    _write_json(path, metadata)
    with pytest.raises(InvalidSubmission):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


@pytest.mark.parametrize("kind", ["nan", "negative_variance", "object"])
def test_bad_normalization_stats_rejected_without_pickle(artifacts, fake_hfm, tmp_path, kind):
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import InvalidSubmission
    model_dir, config, _ = artifacts
    mean, var = np.zeros(90), np.ones(90)
    if kind == "nan": mean[0] = np.nan
    if kind == "negative_variance": var[0] = -1
    if kind == "object": mean = np.array([{"bad": 1}] * 90, dtype=object)
    np.savez(model_dir / "policy2_vecnormalize_stats.npz", obs_mean=mean, obs_var=var)
    with pytest.raises(InvalidSubmission):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


def test_external_tensor_in_nested_graph_rejected_before_runtime(artifacts, fake_hfm, tmp_path):
    import onnx
    from onnx import helper, TensorProto
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import InvalidSubmission
    model_dir, config, _ = artifacts
    model = onnx.load_model(model_dir / "policy2.onnx")
    tensor = TensorProto(name="bad", data_type=TensorProto.FLOAT, dims=[1])
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../../secret")
    child = helper.make_graph([helper.make_node("Constant", [], ["x"], value=tensor)],
                              "nested", [], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])])
    model.graph.node.append(helper.make_node("If", ["condition"], ["nested"], then_branch=child, else_branch=child))
    (model_dir / "policy2.onnx").write_bytes(model.SerializeToString())
    with pytest.raises(InvalidSubmission, match="external"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


def test_model_nan_output_is_rejected_before_simulation(artifacts, fake_hfm, tmp_path):
    import onnx
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import InvalidSubmission
    model_dir, config, _ = artifacts
    model = onnx.load_model(model_dir / "policy2.onnx")
    model.graph.node[0].attribute[0].t.float_data[0] = float("nan")
    (model_dir / "policy2.onnx").write_bytes(model.SerializeToString())
    with pytest.raises(InvalidSubmission, match="finite"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


def test_torch_dynamic_output_dimension_is_probed_without_accumulating_actions(artifacts):
    import onnx
    from server_evaluator.model_rollout import _load_policy
    model_dir, config, _ = artifacts
    model = onnx.load_model(model_dir / "policy2.onnx")
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_param = "Reshapeaction_dim_1"
    (model_dir / "policy2.onnx").write_bytes(model.SerializeToString())
    policy = _load_policy(model_dir, config)
    assert policy.ready
    np.testing.assert_array_equal(policy.reference_adapter.actions.current_abs_action, np.zeros(7))


@pytest.mark.parametrize("dynamic", [False, True])
def test_eight_action_model_rejected_even_with_dynamic_declaration(artifacts, dynamic):
    import onnx
    from onnx import helper, TensorProto
    from server_evaluator.model_rollout import _load_policy
    from server_evaluator.grader import InvalidSubmission
    model_dir, config, _ = artifacts
    model = onnx.load_model(model_dir / "policy2.onnx")
    model.graph.node[0].attribute[0].t.CopyFrom(helper.make_tensor("constant", TensorProto.FLOAT, [1, 8], [.1] * 8))
    dimension = model.graph.output[0].type.tensor_type.shape.dim[1]
    if dynamic:
        dimension.dim_param = "unknown_actions"
    else:
        dimension.dim_value = 8
    (model_dir / "policy2.onnx").write_bytes(model.SerializeToString())
    with pytest.raises(InvalidSubmission):
        _load_policy(model_dir, config)


def test_model_act_error_closes_both_environments(artifacts, fake_hfm, tmp_path, monkeypatch):
    from submission.inference_common import BaseOnnxPolicy
    from server_evaluator.model_rollout import run_model
    from server_evaluator.grader import InvalidSubmission
    def bad_act(self, observation):
        raise ValueError("bad inference output")
    monkeypatch.setattr(BaseOnnxPolicy, "act", bad_act)
    model_dir, config, _ = artifacts
    with pytest.raises(InvalidSubmission, match="inference.*step 1"):
        run_model(model_dir, config, tmp_path / "trajectory.json")
    assert len(fake_hfm.instances) == 2
    assert all(env.closed for env in fake_hfm.instances)


def test_existing_project_export_passes_preflight_if_available():
    from server_evaluator.model_rollout import _load_policy
    model_dir = Path(__file__).resolve().parents[2] / "submission" / "model"
    if not (model_dir / "policy2.onnx").is_file():
        pytest.skip("project model export not present")
    metadata = json.loads((model_dir / "policy2.metadata.json").read_text())
    if metadata.get("feature_mode") != "reference_only_v1":
        pytest.skip("project export predates the current reference-only interface")
    policy = _load_policy(model_dir, {"onnx_threads": 1})
    assert policy.ready
    np.testing.assert_array_equal(policy.reference_adapter.actions.current_abs_action, np.zeros(7))


def test_missing_onnx_runtime_is_server_failure(artifacts, fake_hfm, tmp_path, monkeypatch):
    from server_evaluator import model_rollout
    from server_evaluator.grader import GradingUnavailable
    original = model_rollout.importlib.import_module
    def missing(name):
        if name == "onnxruntime":
            raise ImportError("runtime package missing")
        return original(name)
    monkeypatch.setattr(model_rollout.importlib, "import_module", missing)
    model_dir, config, _ = artifacts
    with pytest.raises(GradingUnavailable, match="runtime package missing"):
        model_rollout.run_model(model_dir, config, tmp_path / "trajectory.json")
    assert not fake_hfm.instances


def test_cli_writes_readable_failure_envelope_atomically(tmp_path, monkeypatch):
    from server_evaluator import model_rollout
    from server_evaluator.grader import InvalidSubmission
    _write_json(tmp_path / "config.json", {})
    monkeypatch.setattr(model_rollout, "_apply_memory_limit", lambda config: None)
    def invalid(*args):
        raise InvalidSubmission("invalid test model")
    monkeypatch.setattr(model_rollout, "run_model", invalid)
    result = tmp_path / "result.json"
    code = model_rollout.main(["--model-dir", str(tmp_path), "--config", str(tmp_path / "config.json"),
                              "--trajectory", str(tmp_path / "trajectory.json"), "--result", str(result)])
    assert code == 0
    assert json.loads(result.read_text()) == {"kind": "invalid", "error": "invalid test model"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("artifact", ["metadata", "stats", "onnx"])
def test_artifact_io_failure_is_server_failure(artifacts, artifact, monkeypatch):
    from server_evaluator import model_rollout
    from server_evaluator.grader import GradingUnavailable
    model_dir, config, _ = artifacts
    if artifact == "metadata":
        original = Path.read_text
        def denied(path, *args, **kwargs):
            if path.name == "policy2.metadata.json":
                raise PermissionError("artifact access denied")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", denied)
    elif artifact == "onnx":
        original = Path.read_bytes
        def denied(path, *args, **kwargs):
            if path.name == "policy2.onnx":
                raise PermissionError("artifact access denied")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_bytes", denied)
    else:
        def denied(*args, **kwargs):
            raise PermissionError("artifact access denied")
        monkeypatch.setattr(model_rollout.np, "load", denied)
    with pytest.raises(GradingUnavailable, match="access denied"):
        model_rollout._load_policy(model_dir, config)


def test_memory_limit_uses_existing_hard_ceiling_without_touching_real_process(monkeypatch):
    from server_evaluator import model_rollout
    calls = []
    fake = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1,
                           getrlimit=lambda _: (-1, 2 * 1024 ** 3),
                           setrlimit=lambda which, limits: calls.append((which, limits)))
    monkeypatch.setattr(model_rollout.sys, "platform", "linux")
    original = model_rollout.importlib.import_module
    monkeypatch.setattr(model_rollout.importlib, "import_module", lambda name: fake if name == "resource" else original(name))
    model_rollout._apply_memory_limit({"memory_limit_mb": 4096})
    assert calls == [(9, (2 * 1024 ** 3, 2 * 1024 ** 3))]


def test_memory_limit_setup_failure_is_infrastructure_failure(monkeypatch):
    from server_evaluator import model_rollout
    from server_evaluator.grader import GradingUnavailable
    def denied(*args):
        raise PermissionError("cannot set memory limit")
    fake = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1,
                           getrlimit=lambda _: (-1, -1), setrlimit=denied)
    monkeypatch.setattr(model_rollout.sys, "platform", "linux")
    original = model_rollout.importlib.import_module
    monkeypatch.setattr(model_rollout.importlib, "import_module", lambda name: fake if name == "resource" else original(name))
    with pytest.raises(GradingUnavailable, match="memory limit"):
        model_rollout._apply_memory_limit({"memory_limit_mb": 4096})


def test_cli_memory_error_is_invalid_model(tmp_path, monkeypatch):
    from server_evaluator import model_rollout
    _write_json(tmp_path / "config.json", {})
    monkeypatch.setattr(model_rollout, "_apply_memory_limit", lambda config: None)
    def exhausted(*args):
        raise MemoryError("allocation failed")
    monkeypatch.setattr(model_rollout, "run_model", exhausted)
    result_path = tmp_path / "result.json"
    model_rollout.main(["--model-dir", str(tmp_path), "--config", str(tmp_path / "config.json"),
                        "--trajectory", str(tmp_path / "trajectory.json"), "--result", str(result_path)])
    result = json.loads(result_path.read_text())
    assert result["kind"] == "invalid"
    assert "memory" in result["error"].lower()


def test_atomic_json_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    from server_evaluator import model_rollout
    original = model_rollout.os.replace
    attempts = []
    def transient(source, destination):
        attempts.append(1)
        if len(attempts) == 1:
            error = PermissionError("phase reader holds the destination")
            error.winerror = 32
            raise error
        return original(source, destination)
    monkeypatch.setattr(model_rollout.os, "replace", transient)
    path = tmp_path / "phase.json"
    model_rollout._atomic_json(path, {"phase": "hfm_step", "step": 3}, sync=False)
    assert json.loads(path.read_text()) == {"phase": "hfm_step", "step": 3}
    assert len(attempts) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_windows_phase_write_failure_is_bounded_and_cleans_temp(tmp_path, monkeypatch):
    from server_evaluator import model_rollout
    from server_evaluator.grader import GradingUnavailable
    attempts = []
    def denied(source, destination):
        attempts.append(1)
        error = PermissionError("permanent phase write denial")
        error.winerror = 5
        raise error
    monkeypatch.setattr(model_rollout.os, "replace", denied)
    monkeypatch.setattr(model_rollout.time, "sleep", lambda seconds: None)
    with pytest.raises(GradingUnavailable, match="permanent phase write denial"):
        model_rollout._phase({"_progress_path": str(tmp_path / "phase.json")}, "inference", 1)
    assert len(attempts) == 5
    assert not list(tmp_path.glob("*.tmp"))
