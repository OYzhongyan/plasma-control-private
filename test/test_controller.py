"""Portable controllers consume simulator fields and never import training code."""
import copy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def manifest(state_dim=0):
    return dict(version=1, interface="hfm_controller_v1", observations=[
        {"name": "Ip", "shape": [1]}], state_dim=state_dim, state_dtype="float32",
        action_dim=12, action_units="voltage_setpoint", reset_state="zeros")


def write_controller(directory, *, stateful=False):
    import onnx
    from onnx import TensorProto as T, helper as h
    directory.mkdir(parents=True, exist_ok=True)
    inputs = [h.make_tensor_value_info("Ip", T.FLOAT, [1, 1])]
    outputs = [h.make_tensor_value_info("action", T.FLOAT, [1, 12])]
    nodes = [h.make_node("Concat", ["Ip"] * 12, ["raw"], axis=1)]
    if stateful:
        inputs += [h.make_tensor_value_info("state", T.FLOAT, [1, 12])]
        outputs += [h.make_tensor_value_info("next_state", T.FLOAT, [1, 12])]
        nodes += [h.make_node("Add", ["raw", "state"], ["next_state"]),
                  h.make_node("Identity", ["next_state"], ["action"])]
    else:
        nodes += [h.make_node("Identity", ["raw"], ["action"])]
    model = h.make_model(h.make_graph(nodes, "controller", inputs, outputs),
                         opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save_model(model, directory / "controller.onnx")
    (directory / "controller.json").write_text(json.dumps(manifest(12 if stateful else 0)))
    return directory


def test_contract_supports_requested_fields_and_optional_state():
    from submission.controller_contract import validate_manifest, observation_inputs
    value = manifest()
    validate_manifest(value)
    values = observation_inputs({"Ip": [123.], "unrequested": [np.nan]}, value)
    assert list(values) == ["Ip"]
    assert values["Ip"].dtype == np.float32 and values["Ip"].shape == (1, 1)
    np.testing.assert_array_equal(values["Ip"], [[123.]])
    with pytest.raises(ValueError, match="Ip"):
        observation_inputs({}, value)
    with pytest.raises(ValueError, match="Ip"):
        observation_inputs({"Ip": [1., 2.]}, value)


@pytest.mark.parametrize("field,value", [
    ("version", True), ("interface", "reference_only_v1"),
    ("observations", [{"name": "bad", "shape": [1]}]),
    ("observations", [{"name": "Ip", "shape": [2]}]),
    ("observations", [{"name": "Ip", "shape": [True]}]),
    ("observations", [{"name": "Ip", "shape": [1]}] * 2),
    ("state_dim", -1), ("state_dim", True), ("state_dim", 16385),
    ("state_dtype", "object"), ("action_dim", 7), ("action_units", "normalized"),
    ("reset_state", "pickle"),
])
def test_invalid_transport_manifest_is_rejected(field, value):
    from submission.controller_contract import validate_manifest
    config = manifest()
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_manifest(config)


@pytest.mark.parametrize("stateful", [False, True])
def test_runtime_transports_inputs_and_resets_explicit_state(tmp_path, stateful):
    from submission.inference_common import BaseOnnxPolicy
    directory = write_controller(tmp_path / "controller", stateful=stateful)
    policy = BaseOnnxPolicy(model_dir=directory)
    assert policy.ready, policy.load_error
    np.testing.assert_array_equal(policy.act({"Ip": [2.]}), np.full(12, 2.))
    np.testing.assert_array_equal(policy.act({"Ip": [3.]}), np.full(12, 5. if stateful else 3.))
    policy.reset()
    np.testing.assert_array_equal(policy.act({"Ip": [3.]}), np.full(12, 3.))


def test_runtime_failure_does_not_advance_state(tmp_path):
    from submission.inference_common import BaseOnnxPolicy
    policy = BaseOnnxPolicy(model_dir=write_controller(tmp_path / "controller", stateful=True))
    policy.act({"Ip": [2.]})
    with pytest.raises(ValueError, match="finite"):
        policy.act({"Ip": [np.nan]})
    np.testing.assert_array_equal(policy.act({"Ip": [3.]}), np.full(12, 5.))


def test_package_is_two_data_files_and_runtime_cannot_import_train(tmp_path):
    from submission.model_bundle import create_bundle, unpack_bundle
    directory = write_controller(tmp_path / "controller", stateful=True)
    archive = create_bundle(directory, tmp_path / "model.zip")
    model = unpack_bundle(archive, tmp_path / "unpacked")
    assert {p.name for p in model.iterdir()} == {"controller.onnx", "controller.json"}
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'train' or fullname.startswith('train.'):
            raise AssertionError('Training code imported: ' + fullname)
sys.meta_path.insert(0, Block())
from submission.inference_common import BaseOnnxPolicy
p = BaseOnnxPolicy(model_dir=sys.argv[1])
assert p.ready, p.load_error
assert p.act({'Ip': [3.]}).tolist() == [3.] * 12
assert not any(n == 'train' or n.startswith('train.') for n in sys.modules)
'''
    proc = subprocess.run([sys.executable, "-B", "-c", script, str(model)], cwd=ROOT,
                          capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert proc.returncode == 0, proc.stderr


def test_published_field_schema_matches_simulator():
    from submission.controller_contract import OBSERVATION_SHAPES
    from environment.hfm_simulator import _build_observation_space
    expected = {k: tuple(v.shape) for k, v in _build_observation_space().spaces.items()}
    assert OBSERVATION_SHAPES == expected


def test_http_service_uses_complete_controller_and_reset(tmp_path):
    from submission.service2 import create_app
    client = create_app(write_controller(tmp_path / "model", stateful=True)).test_client()
    assert client.get("/health").status_code == 200
    for value in (3., 6.):
        response = client.post("/act", json={"observation": {"Ip": [3.]}})
        assert response.status_code == 200 and response.json["action"] == [value] * 12
    assert client.post("/reset", json={}).status_code == 200
    assert client.post("/act", json={"observation": {"Ip": [3.]}}).json["action"] == [3.] * 12
    assert client.post("/act", json={"observation": {}}).status_code == 500


def test_missing_controller_fails_readiness_without_zero_action_fallback(tmp_path):
    from submission.service2 import create_app
    from submission.inference2 import Policy
    policy = Policy(tmp_path)
    assert not policy.ready
    with pytest.raises(RuntimeError):
        policy.act({"Ip": [3.]})
    assert create_app(tmp_path).test_client().get("/health").status_code == 503
