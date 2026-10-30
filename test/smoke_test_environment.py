"""Smoke test: basic package creation and wrapper behavior."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
import yaml

from environment import (
    Action7DTo12DWrapper,
    DictObsFlattenWrapper,
    HFMSimulator,
    action_7d_to_12d,
)


def test_7d_mapping():
    a7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    a12 = action_7d_to_12d(a7)
    assert a12.shape == (12,), a12.shape
    print("action_7d_to_12d ok")


def test_env_creation():
    config_path = root / "configs" / "env_default.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = HFMSimulator(config)
    assert env.action_space.shape == (12,)
    assert env.observation_space is not None
    obs_spaces = env.observation_space.spaces
    assert "rB" in obs_spaces
    assert "zB" in obs_spaces
    assert "lcfs_points" in obs_spaces
    assert "reference_lcfs_points" in obs_spaces
    assert obs_spaces["rB"].shape == (32,)
    assert obs_spaces["zB"].shape == (32,)
    assert obs_spaces["lcfs_points"].shape == (32, 2)
    assert obs_spaces["reference_lcfs_points"].shape == (32, 2)
    env.close()
    print("HFMSimulator creation ok")


def test_wrappers():
    config_path = root / "configs" / "env_default.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = HFMSimulator(config)
    env = DictObsFlattenWrapper(env, keys=["Ip", "R", "Z", "Rmax", "Rmin", "kappa"])
    env = Action7DTo12DWrapper(env)
    assert env.action_space.shape == (7,)
    assert env.observation_space.shape is not None
    env.close()
    print("wrappers ok")


if __name__ == "__main__":
    test_7d_mapping()
    test_env_creation()
    test_wrappers()
    print("smoke_test_environment passed.")
