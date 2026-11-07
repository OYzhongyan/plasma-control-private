"""Organizer-owned simulator field transport; no training imports or feature logic."""
from __future__ import annotations
import json
from pathlib import Path

INTERFACE = "hfm_controller_v1"
MODEL_NAME = "controller.onnx"
MANIFEST_NAME = "controller.json"
MAX_STATE_DIM = 16384
OBSERVATION_SHAPES = {
    "Ip": (1,), "R": (1,), "Z": (1,), "I_PF": (12,),
    "Rmax": (1,), "Rmin": (1,), "aminor": (1,), "deltal": (1,),
    "deltau": (1,), "kappa": (1,), "rc": (1,), "zc": (1,),
    "FA": (1,), "FB": (1,), "Bm": (100,), "nrx": (1,), "nzx": (1,),
    "Fx": (4290,), "rx": (66,), "zx": (65,), "rX": (6,), "zX": (6,),
    "nX": (1,), "lX": (1,), "rB": (32,), "zB": (32,), "FX": (6,),
    "lcfs_points": (32, 2), "reference_Ip": (1,), "reference_R": (1,),
    "reference_Z": (1,), "reference_lX": (1,), "reference_lcfs_points": (32, 2),
    "reference_rX": (4,), "reference_zX": (4,), "reference_x_valid": (4,),
    "reference_strike_r": (8,), "reference_strike_z": (8,),
    "reference_strike_valid": (8,), "reference_nX": (1,),
    "reference_n_strike": (1,), "failure": (1,),
}


def make_manifest(fields, state_dim=0, state_dtype="float32"):
    value = dict(version=1, interface=INTERFACE,
                 observations=[{"name": name, "shape": list(OBSERVATION_SHAPES[name])} for name in fields],
                 state_dim=state_dim, state_dtype=state_dtype, action_dim=12,
                 action_units="voltage_setpoint", reset_state="zeros")
    validate_manifest(value)
    return value


def validate_manifest(value):
    if not isinstance(value, dict):
        raise ValueError("controller manifest must be an object")
    for key, expected in {"version": 1, "interface": INTERFACE, "action_dim": 12,
                          "action_units": "voltage_setpoint", "reset_state": "zeros"}.items():
        if type(value.get(key)) is not type(expected) or value[key] != expected:
            raise ValueError(f"Invalid controller {key}: expected {expected!r}")
    if type(value.get("state_dim")) is not int or not 0 <= value["state_dim"] <= MAX_STATE_DIM:
        raise ValueError(f"state_dim must be an integer from 0 to {MAX_STATE_DIM}")
    if value.get("state_dtype") not in ("float32", "float64"):
        raise ValueError("state_dtype must be float32 or float64")
    fields = value.get("observations")
    if not isinstance(fields, list) or not 1 <= len(fields) <= len(OBSERVATION_SHAPES):
        raise ValueError("observations must be a nonempty list of simulator fields")
    seen = set()
    for field in fields:
        if not isinstance(field, dict) or set(field) != {"name", "shape"}:
            raise ValueError("observations entries require name and shape")
        name, shape = field["name"], field["shape"]
        if not isinstance(name, str) or name not in OBSERVATION_SHAPES or name in seen:
            raise ValueError(f"Invalid or duplicate observations field: {name!r}")
        if (not isinstance(shape, list) or any(type(n) is not int for n in shape)
                or tuple(shape) != OBSERVATION_SHAPES[name]):
            raise ValueError(f"observations shape for {name} must be {OBSERVATION_SHAPES[name]}")
        seen.add(name)


def read_manifest(path: str | Path):
    def unique(pairs):
        result = {}
        for key, val in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = val
        return result
    def reject(number):
        raise ValueError(f"Non-finite manifest number: {number}")
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("controller manifest exceeds 1 MiB")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=reject)
    validate_manifest(value)
    return value


def observation_inputs(observation, manifest):
    import math
    import numpy as np
    result = {}
    for field in manifest["observations"]:
        name, shape = field["name"], field["shape"]
        if name not in observation:
            raise ValueError(f"Missing simulator observation: {name}")
        raw = np.asarray(observation[name])
        if raw.dtype.kind not in "biuf" or raw.size != math.prod(shape):
            raise ValueError(f"Simulator observation {name} must have shape {shape}")
        result[name] = np.asarray(raw, dtype=np.float32).reshape((1, *shape))
    return result
