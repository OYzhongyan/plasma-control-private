"""Regression checks for retained F2b entrypoints, targets and scoring."""

from pathlib import Path
import importlib
import json
import sys
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.references import (
    build_reference,
    default_reference_path,
    load_target_reference,
)


def test_default_hydra_configuration_only_assigns_limiter_shots():
    from train.train_ppo import assign_shots_to_envs

    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "train/configs")
    ):
        cfg = compose(config_name="base")
    assert "task" not in cfg
    assert set(cfg.shot_groups) == {"limiter_start"}
    shots = assign_shots_to_envs(dict(cfg.shot_groups), 32, str(cfg.shot_id))
    assert set(shots) == {"15892_400", "15892_500"}


def test_legacy_reference_interface_matches_hydra():
    path = default_reference_path()
    target = load_target_reference(path)
    legacy = build_reference(target, 1000)
    current = build_reference(path, max_steps=1000)[0]
    for key in current:
        np.testing.assert_array_equal(legacy[key], current[key])


def test_official_shared_target_and_f2b_target_are_equivalent(tmp_path):
    from train.evaluate import build_scoring_target

    target = build_scoring_target(build_reference(default_reference_path())[0])
    target = {k: np.asarray(v).tolist() for k, v in target.items()}
    paths = []
    for key in ("F2b", "F2a"):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps({key: target}), encoding="utf-8")
        paths.append(path)
    refs = [build_reference(p)[0] for p in paths]
    assert refs[0] == refs[1]


def test_scoring_runs_f2b_only_and_retains_30_point_scale():
    from evaluation.eval.score import config
    from evaluation.eval.score.evaluate_core import evaluate

    assert config.TASK_IDS == ["F2b"]
    assert sum(config.metric_weights().values()) == 30
    result = evaluate({"F2b": {"trajectory": {}, "timeout": False}}, {"F2b": {}})
    assert result["task_scores"] == {"F2b": 0.0}


@pytest.mark.parametrize(
    "name",
    [
        "train.legacy_train_ppo",
        "train.export_onnx",
        "tools.rollout_f2_policy",
        "tools.generate_f2_xpt_target_variants",
    ],
)
def test_retained_tools_import_without_old_training_package(name):
    importlib.import_module(name)
