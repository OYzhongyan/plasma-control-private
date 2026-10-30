"""Reference horizons used by the Hydra training and evaluation entrypoints."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.hfm_simulator import _build_reference_trajectory  # noqa: E402
from training.references import build_reference, default_reference_path  # noqa: E402


@pytest.mark.parametrize("task,default_steps", [("F1", 300), ("F2", 500)])
def test_extended_reference_preserves_schedule_and_holds_final_target(task, default_steps):
    path = default_reference_path(task)
    original = build_reference(path, task)[0]
    extended = build_reference(path, task, max_steps=1000)[0]

    assert len(original["Ip"]) == default_steps
    assert len(extended["Ip"]) == 1000
    np.testing.assert_array_equal(extended["Ip"][:default_steps], original["Ip"])
    np.testing.assert_array_equal(extended["Ip"][default_steps:], original["Ip"][-1])
    for key in original.keys() - {"Ip"}:
        np.testing.assert_array_equal(extended[key], original[key])


@pytest.mark.parametrize("task", ["F1", "F2"])
def test_shortened_reference_preserves_prefix(task):
    path = default_reference_path(task)
    original = build_reference(path, task)[0]
    shortened = build_reference(path, task, max_steps=50)[0]
    np.testing.assert_array_equal(shortened["Ip"], original["Ip"][:50])


@pytest.mark.parametrize("max_steps", [0, -1])
def test_reference_rejects_nonpositive_horizon(max_steps):
    with pytest.raises(ValueError, match="max_steps must be positive"):
        build_reference(default_reference_path("F2"), "F2", max_steps=max_steps)


@pytest.mark.parametrize("task", ["f1", "f2"])
@pytest.mark.parametrize("max_steps", [128, 1000])
def test_hydra_horizon_reaches_environment_reference(monkeypatch, tmp_path, task, max_steps):
    from hydra import compose, initialize_config_dir

    monkeypatch.syspath_prepend(str(ROOT / "train_v2"))
    entrypoint = importlib.import_module("train_v2.train_ppo")
    monkeypatch.setattr(entrypoint, "_resolve_save_dir", lambda: tmp_path)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "train_v2/configs")):
        cfg = compose(config_name="base", overrides=[f"task={task}", f"max_steps={max_steps}"])
    cfg, references, _, base_config, _ = entrypoint.build_training_context(cfg)
    env_config = entrypoint.make_env_config(
        base_config,
        port=2223,
        shot_id=str(cfg.shot_id),
        max_steps=int(cfg.max_steps),
        reference_mode=str(cfg.target.reference_mode),
    )
    assert env_config["max_steps"] == max_steps
    for reference in references:
        # Validate the same trajectory coercion used by HFMSimulator.reset(),
        # without starting a socket connection or a simulator process.
        raw = {"rB": np.zeros(32), "zB": np.zeros(32)}
        trajectory = _build_reference_trajectory(
            reference_mode="trajectory",
            reference_spec=reference,
            raw=raw,
            max_steps=env_config["max_steps"],
        )
        assert trajectory["Ip"].shape == (max_steps,)
        assert trajectory["lcfs_points"].shape == (max_steps, 32, 2)
        if task == "f2":
            assert trajectory["rX"].shape == (max_steps, 4)
            assert trajectory["strike_r"].shape == (max_steps, 8)
