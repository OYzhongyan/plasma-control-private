#
# shot_id -> fixed files + default reset params.
# L_addr / LX_addr are fixed at env init and are not changed on reset().
# reset() only exposes signeo / bp / q0.
#

import os
from pathlib import Path

import yaml


def _resolve_shot_config_path() -> Path:
    candidates = []
    env_path = os.environ.get("SAISDATA_SHOTS_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    pkg_root = Path(__file__).resolve().parents[1]
    candidates.append(pkg_root / "configs" / "shots.yaml")
    saisdata_root = Path(__file__).resolve().parents[2]
    candidates.append(saisdata_root / "inference" / "shots.yaml")
    saisdata_env = os.environ.get("SAISDATA_ROOT", "/saisdata/33")
    candidates.append(Path(saisdata_env) / "inference" / "shots.yaml")
    candidates.append(Path("/saisdata/inference/shots.yaml"))

    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


SHOT_CONFIG_PATH = _resolve_shot_config_path()


def _normalize_number(value):
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return value
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def _load_shot_registry() -> dict[str, dict]:
    with open(SHOT_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Shot config must be a mapping, got {type(data).__name__}")
    registry: dict[str, dict] = {}
    for key, value in data.items():
        spec = dict(value)
        defaults = spec.get("reset_defaults", {})
        spec["reset_defaults"] = {name: _normalize_number(val) for name, val in defaults.items()}
        registry[str(key)] = spec
    return registry


SHOT_REGISTRY: dict[str, dict] = _load_shot_registry()

REFERENCE_KEYS = ["Ip", "R", "Z", "lX", "lcfs_points"]


def get_shot_spec(shot_id: str) -> dict:
    if shot_id not in SHOT_REGISTRY:
        raise KeyError(f"Unknown shot_id: {shot_id}. Available: {list(SHOT_REGISTRY.keys())}")
    return dict(SHOT_REGISTRY[shot_id])


def get_fge_init_config_for_shot(
    shot_id: str,
    signeo: float | None = None,
    bp: float | None = None,
    q0: float | None = None,
) -> dict:
    """Build init config with fixed L_addr/LX_addr and internal fixed flags."""
    spec = get_shot_spec(shot_id)
    defaults = spec.get("reset_defaults", {})
    base = {
        "L_addr": spec["L_addr"],
        "LX_addr": spec["LX_addr"],
        "signeo": defaults.get("signeo") if signeo is None else signeo,
        "bp": defaults.get("bp") if bp is None else bp,
        "q0": defaults.get("q0") if q0 is None else q0,
    }
    return base
