"""Build noisy policy observations for local robustness checks.

Only the observation payload sent to the submission service is perturbed. The
true trajectory used for scoring is still recorded from the environment by
``subtask_runner``.

Observation delay is intentionally disabled here. The delay mentioned by the
competition is power-supply signal delay, and that is already implemented in
the environment power-supply model.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "obs_perturb.yaml"


def load_obs_perturb_config(path: str | None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        return {"enabled": False}
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {"enabled": False}


def make_observation_rng(base_seed: int) -> np.random.Generator:
    """Keep the original F2b random stream without an experiment selector."""
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed) & 4294967295, 0])
    )


def delay_steps_from_config(cfg: dict[str, Any]) -> int:
    """Compatibility shim: local observation delay is always disabled."""
    _ = cfg
    return 0


def select_delayed_base(
    hist: list[dict[str, np.ndarray]], step_index: int, delay_steps: int
) -> dict[str, np.ndarray]:
    """Return the current observation copy; delay arguments are ignored."""
    _ = delay_steps
    if not hist:
        return {}
    src = hist[min(max(int(step_index), 0), len(hist) - 1)]
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in src.items()}


def build_policy_observation(
    base: dict[str, Any], rng: np.random.Generator, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Copy ``base`` and add configured zero-mean Gaussian noise per key."""
    out: dict[str, Any] = {}
    for k, v in base.items():
        out[k] = np.asarray(v, dtype=np.float64).copy()
    noise_block = cfg.get("noise") or {}
    if bool(noise_block.get("enabled", False)):
        std_map = noise_block.get("std") or {}
        for key, std in std_map.items():
            if key not in out:
                continue
            std_value = float(std)
            if std_value <= 0.0:
                continue
            arr = np.asarray(out[key], dtype=np.float64)
            out[key] = arr + rng.normal(0.0, std_value, size=arr.shape)
    return out


def snapshot_obs_arrays(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Save a numpy copy of one true observation."""
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}


def serialize_obs_for_record(
    obs: dict[str, Any], max_flat: int = 128
) -> dict[str, Any]:
    """Compress one observation into a JSON-serializable debug record."""
    row: dict[str, Any] = {}
    for k, v in obs.items():
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size == 0:
            continue
        if a.size == 1:
            row[k] = float(a[0])
        elif a.size <= max_flat:
            row[k] = [float(x) for x in a.tolist()]
        else:
            row[f"{k}__mean"] = float(np.mean(a))
            row[f"{k}__std"] = float(np.std(a))
    return row
