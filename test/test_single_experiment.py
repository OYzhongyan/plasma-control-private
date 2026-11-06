"""Exercise the single-experiment interfaces without a simulator or model."""

import ast
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_training_and_evaluation_configs_need_no_experiment_selector():
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "train/configs")
    ):
        training = compose(config_name="base")
        evaluation = compose(config_name="eval")
    for cfg in (training, evaluation):
        assert "task" not in cfg
        assert cfg.max_steps == 500
        assert list(cfg.shot_groups.limiter_start) == ["15892_400", "15892_500"]
    assert evaluation.eval.shots is None


@pytest.mark.parametrize("checkpoint", [None, 123456])
def test_existing_checkpoint_filenames_still_resolve(tmp_path, checkpoint):
    from train.evaluate import resolve_model_paths

    relative = (
        "ppo_f2_latest.zip"
        if checkpoint is None
        else f"checkpoints/ppo_f2_{checkpoint}_steps.zip"
    )
    model = tmp_path / relative
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()
    assert resolve_model_paths(tmp_path, checkpoint) == (model, None)


@pytest.mark.parametrize(
    "mode", ["physical_v1", "physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"]
)
def test_train_and_submission_features_match_without_selector(mode):
    from train.features import (
        FeatureMetadata,
        physical_feature_vector,
        physical_f2_xpt_feature_vector,
    )
    from submission.inference_common import flatten_observation
    from train.references import build_reference, default_reference_path

    reference = build_reference(default_reference_path())[0]
    observation = {
        "reference_" + key: value for key, value in reference.items() if key != "Ip"
    }
    observation.update({key: value for key, value in reference.items() if key != "Ip"})
    observation.update(
        Ip=np.array([450000.0]),
        reference_Ip=np.array([400000.0]),
        R=np.array([1.8]),
        reference_R=np.array([1.75]),
        Z=np.array([0.02]),
        reference_Z=np.array([0.0]),
        I_PF=np.arange(12) * 100.0,
    )
    metadata = FeatureMetadata.create(mode=mode)
    assert "task" not in metadata.to_dict()
    features = (
        physical_feature_vector(observation)
        if mode == "physical_v1"
        else physical_f2_xpt_feature_vector(
            observation, no_ref_rz=mode.endswith("no_ref_rz_v1")
        )
    )
    np.testing.assert_allclose(
        features,
        flatten_observation(observation, metadata.to_dict()),
        rtol=1e-6,
        atol=1e-6,
    )
    # Metadata from existing exports may still carry the now-ignored label.
    old_metadata = {**metadata.to_dict(), "task": "F2"}
    assert FeatureMetadata.from_dict(old_metadata) == metadata


def test_reference_runner_passes_no_experiment_selector(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "eval_files"))
    runner = importlib.import_module("run_test")
    output = tmp_path / "infer_result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_test.py",
            "--no-build",
            "--config",
            str(ROOT / "configs/env_default.yaml"),
            "--reference-config",
            str(ROOT / "eval_files/references.yaml"),
            "--service-url",
            "http://localhost:12345",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(runner, "_load_target_json", lambda _: None)
    monkeypatch.setattr(
        runner.submission_client, "wait_for_health", lambda _: {"ok": True}
    )
    received = {}

    def rollout(**kwargs):
        received.update(kwargs)
        return {"trajectory": {}, "timeout": False}

    monkeypatch.setattr(runner.subtask_runner, "run_rollout", rollout)
    runner.main()
    assert received["reference_spec"]["shot_id"] == "15892_400"
    assert received["service_url"] == "http://localhost:12345"
    assert not {"task", "task_id", "task_ids"}.intersection(received)
    assert set(json.loads(output.read_text(encoding="utf-8"))) == {"F2b"}


def test_legacy_rollout_context_needs_only_config_and_no_selector():
    from argparse import Namespace
    from train.legacy_train_ppo import _build_training_context

    ctx = _build_training_context(
        Namespace(train_config=ROOT / "train/configs/f2b_direct_target_xpt.json")
    )
    assert "task" not in ctx
    assert ctx["max_steps"] == 500
    assert len(ctx["reference"]["Ip"]) == 500


def test_runtime_functions_and_cli_have_no_experiment_selectors():
    for directory in (
        "train",
        "tools",
        "submission",
        "evaluation",
        "eval_files",
        "rollout",
    ):
        for path in (ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arguments = (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    )
                    assert not {"task", "task_id", "task_ids"}.intersection(
                        a.arg for a in arguments
                    ), path
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):
                    flags = {a.value for a in node.args if isinstance(a, ast.Constant)}
                    assert not {
                        "--task",
                        "--task-id",
                        "--tasks",
                        "--task-config",
                    }.intersection(flags), path
