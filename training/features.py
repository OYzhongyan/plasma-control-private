"""Feature definitions shared by training, export metadata, and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from environment.xpt_utils import extract_xpt_observation_pack

from .references import normalize_task

F1_FEATURE_KEYS = [
    "Ip",
    "R",
    "Z",
    "lX",
    "reference_Ip",
    "reference_R",
    "reference_Z",
    "reference_lX",
    "lcfs_points",
    "reference_lcfs_points",
    "I_PF",
    "Rmax",
    "Rmin",
    "aminor",
    "deltal",
    "deltau",
    "kappa",
    "rc",
    "zc",
    "FA",
    "FB",
    "failure",
]

F2_FEATURE_KEYS = [
    *F1_FEATURE_KEYS,
    "rX",
    "zX",
    "FX",
    "nX",
    "reference_rX",
    "reference_zX",
    "reference_x_valid",
    "reference_strike_r",
    "reference_strike_z",
    "reference_strike_valid",
    "reference_nX",
    "reference_n_strike",
]

FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    "Ip": (1,),
    "R": (1,),
    "Z": (1,),
    "lX": (1,),
    "reference_Ip": (1,),
    "reference_R": (1,),
    "reference_Z": (1,),
    "reference_lX": (1,),
    "lcfs_points": (32, 2),
    "reference_lcfs_points": (32, 2),
    "I_PF": (12,),
    "Rmax": (1,),
    "Rmin": (1,),
    "aminor": (1,),
    "deltal": (1,),
    "deltau": (1,),
    "kappa": (1,),
    "rc": (1,),
    "zc": (1,),
    "FA": (1,),
    "FB": (1,),
    "failure": (1,),
    "rX": (6,),
    "zX": (6,),
    "FX": (6,),
    "nX": (1,),
    "reference_rX": (4,),
    "reference_zX": (4,),
    "reference_x_valid": (4,),
    "reference_strike_r": (8,),
    "reference_strike_z": (8,),
    "reference_strike_valid": (8,),
    "reference_nX": (1,),
    "reference_n_strike": (1,),
}

COIL_LIMITS_A = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float32)
ACTION_SCALE_7D_V = np.asarray([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)

PHYSICAL_F1_FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    "ip": (1,),
    "reference_ip": (1,),
    "ip_error_50ka": (1,),
    "r": (1,),
    "reference_r": (1,),
    "r_error_3cm": (1,),
    "z": (1,),
    "reference_z": (1,),
    "z_error_3cm": (1,),
    "lx": (1,),
    "reference_lx": (1,),
    "lx_error": (1,),
    "lcfs_points_absolute": (32, 2),
    "reference_lcfs_points_absolute": (32, 2),
    "lcfs_error_3cm": (32, 2),
    "i_pf_norm": (12,),
    "coil_margin": (1,),
    "rmax": (1,),
    "rmin": (1,),
    "aminor": (1,),
    "deltal": (1,),
    "deltau": (1,),
    "kappa": (1,),
    "rc": (1,),
    "zc": (1,),
    "fa": (1,),
    "fb": (1,),
    "failure": (1,),
}

PHYSICAL_F1_NO_REF_RZ_FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    key: shape
    for key, shape in PHYSICAL_F1_FEATURE_SHAPES.items()
    if key not in {"reference_r", "r_error_3cm", "reference_z", "z_error_3cm"}
}

PHYSICAL_NO_REF_RZ_MODES = {
    "physical_f1_no_ref_rz_v1",
    "physical_f2_xpt_no_ref_rz_v1",
}

PHYSICAL_F2_EXTRA_FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    "rx": (6,),
    "zx": (6,),
    "fx": (6,),
    "nx": (1,),
    "reference_rx": (4,),
    "reference_zx": (4,),
    "x_error_3cm": (4, 2),
    "reference_x_valid": (4,),
    "reference_strike_r": (8,),
    "reference_strike_z": (8,),
    "reference_strike_valid": (8,),
    "reference_nx": (1,),
    "reference_n_strike": (1,),
}

PHYSICAL_F2_XPT_EXTRA_FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    "xpt_nx_norm": (1,),
    "xpt_nx_error": (1,),
    "xpt_x_valid": (4,),
    "xpt_r": (4,),
    "xpt_z": (4,),
    "xpt_position_error_3cm": (4, 2),
    "xpt_flux_error_0p003": (4,),
    "reference_xpt_r": (4,),
    "reference_xpt_z": (4,),
    "reference_xpt_valid": (4,),
    "reference_nx": (1,),
}


def feature_keys_for_task(task: str) -> list[str]:
    return list(F1_FEATURE_KEYS if normalize_task(task) == "F1" else F2_FEATURE_KEYS)


def feature_shapes_for_keys(keys: Sequence[str]) -> dict[str, list[int]]:
    return {key: list(FEATURE_SHAPES.get(key, (1,))) for key in keys}


def feature_dim(keys: Sequence[str]) -> int:
    return int(sum(np.prod(FEATURE_SHAPES.get(key, (1,))) for key in keys))


def _history_feature_shapes(history_steps: int) -> dict[str, tuple[int, ...]]:
    history_steps = max(0, int(history_steps))
    return {
        "history_action_7d": (history_steps, 7),
        "history_delta_ip": (history_steps, 1),
        "history_delta_rz": (history_steps, 2),
        "history_delta_i_pf": (history_steps, 12),
    }


def physical_feature_shapes_for_task(
    task: str,
    history_steps: int = 0,
    mode: str = "physical_v1",
) -> dict[str, list[int]]:
    task = normalize_task(task)
    mode = str(mode or "physical_v1")
    if mode in PHYSICAL_NO_REF_RZ_MODES:
        shapes: dict[str, tuple[int, ...]] = dict(PHYSICAL_F1_NO_REF_RZ_FEATURE_SHAPES)
    else:
        shapes = dict(PHYSICAL_F1_FEATURE_SHAPES)
    if task == "F2" and mode in {"physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"}:
        shapes.update(PHYSICAL_F2_XPT_EXTRA_FEATURE_SHAPES)
    elif task == "F2":
        shapes.update(PHYSICAL_F2_EXTRA_FEATURE_SHAPES)
    if int(history_steps) > 0:
        shapes.update(_history_feature_shapes(history_steps))
    return {key: list(shape) for key, shape in shapes.items()}


def physical_feature_keys_for_task(
    task: str,
    history_steps: int = 0,
    mode: str = "physical_v1",
) -> list[str]:
    return list(physical_feature_shapes_for_task(task, history_steps, mode=mode).keys())


def _scalar(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = observation.get(key)
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float(default)
    out = float(arr[0])
    return out if np.isfinite(out) else float(default)


def _array(observation: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    return _coerce_array(observation.get(key), shape, dtype=np.float32)


def _clip_finite(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def physical_feature_vector(
    observation: dict[str, Any],
    task: str,
    mode: str = "physical_v1",
    dtype=np.float32,
) -> np.ndarray:
    """Build the stage input features with fixed physical normalization.

    LCFS terms intentionally stay in device absolute coordinates. The error term
    uses the public 3 cm LCFS scoring scale.
    """

    task = normalize_task(task)
    mode = str(mode or "physical_v1")
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    r = _scalar(observation, "R")
    ref_r = _scalar(observation, "reference_R", r)
    z = _scalar(observation, "Z")
    ref_z = _scalar(observation, "reference_Z", z)
    lx = _scalar(observation, "lX")
    ref_lx = _scalar(observation, "reference_lX", lx)
    lcfs = _array(observation, "lcfs_points", (32, 2))
    ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
    i_pf = _array(observation, "I_PF", (12,))

    features: dict[str, np.ndarray] = {
        "ip": np.asarray([ip / 500_000.0], dtype=np.float32),
        "reference_ip": np.asarray([ref_ip / 500_000.0], dtype=np.float32),
        "ip_error_50ka": np.asarray([(ip - ref_ip) / 50_000.0], dtype=np.float32),
        "r": np.asarray([r], dtype=np.float32),
        "reference_r": np.asarray([ref_r], dtype=np.float32),
        "r_error_3cm": np.asarray([(r - ref_r) / 0.03], dtype=np.float32),
        "z": np.asarray([z], dtype=np.float32),
        "reference_z": np.asarray([ref_z], dtype=np.float32),
        "z_error_3cm": np.asarray([(z - ref_z) / 0.03], dtype=np.float32),
        "lx": np.asarray([lx], dtype=np.float32),
        "reference_lx": np.asarray([ref_lx], dtype=np.float32),
        "lx_error": np.asarray([lx - ref_lx], dtype=np.float32),
        "lcfs_points_absolute": lcfs,
        "reference_lcfs_points_absolute": ref_lcfs,
        "lcfs_error_3cm": (lcfs - ref_lcfs) / 0.03,
        "i_pf_norm": i_pf / COIL_LIMITS_A,
        "coil_margin": np.asarray([np.max(np.abs(i_pf) / COIL_LIMITS_A)], dtype=np.float32),
        "rmax": np.asarray([_scalar(observation, "Rmax")], dtype=np.float32),
        "rmin": np.asarray([_scalar(observation, "Rmin")], dtype=np.float32),
        "aminor": np.asarray([_scalar(observation, "aminor")], dtype=np.float32),
        "deltal": np.asarray([_scalar(observation, "deltal")], dtype=np.float32),
        "deltau": np.asarray([_scalar(observation, "deltau")], dtype=np.float32),
        "kappa": np.asarray([_scalar(observation, "kappa")], dtype=np.float32),
        "rc": np.asarray([_scalar(observation, "rc")], dtype=np.float32),
        "zc": np.asarray([_scalar(observation, "zc")], dtype=np.float32),
        "fa": np.asarray([_scalar(observation, "FA")], dtype=np.float32),
        "fb": np.asarray([_scalar(observation, "FB")], dtype=np.float32),
        "failure": np.asarray([_scalar(observation, "failure")], dtype=np.float32),
    }

    if task == "F2":
        rx = _array(observation, "rX", (6,))
        zx = _array(observation, "zX", (6,))
        ref_rx = _array(observation, "reference_rX", (4,))
        ref_zx = _array(observation, "reference_zX", (4,))
        features.update(
            {
                "rx": rx,
                "zx": zx,
                "fx": _array(observation, "FX", (6,)),
                "nx": np.asarray([_scalar(observation, "nX")], dtype=np.float32),
                "reference_rx": ref_rx,
                "reference_zx": ref_zx,
                "x_error_3cm": np.stack(
                    [(rx[:4] - ref_rx) / 0.03, (zx[:4] - ref_zx) / 0.03],
                    axis=-1,
                ),
                "reference_x_valid": _array(observation, "reference_x_valid", (4,)),
                "reference_strike_r": _array(observation, "reference_strike_r", (8,)),
                "reference_strike_z": _array(observation, "reference_strike_z", (8,)),
                "reference_strike_valid": _array(observation, "reference_strike_valid", (8,)),
                "reference_nx": np.asarray([_scalar(observation, "reference_nX")], dtype=np.float32),
                "reference_n_strike": np.asarray(
                    [_scalar(observation, "reference_n_strike")],
                    dtype=np.float32,
                ),
            }
        )

    keys = physical_feature_keys_for_task(task, history_steps=0, mode=mode)
    return np.concatenate([_clip_finite(features[key]).reshape(-1) for key in keys]).astype(dtype)


def _xpt_pack_from_observation(observation: dict[str, Any]):
    ref_rx = _array(observation, "reference_rX", (4,))
    ref_zx = _array(observation, "reference_zX", (4,))
    ref_sr = _array(observation, "reference_strike_r", (8,))
    ref_sz = _array(observation, "reference_strike_z", (8,))
    try:
        return extract_xpt_observation_pack(
            observation,
            ref_rx,
            ref_zx,
            ref_sr,
            ref_sz,
        )
    except Exception:
        return {
            "nX": int(_scalar(observation, "nX")),
            "x_r": np.zeros(4, dtype=np.float32),
            "x_z": np.zeros(4, dtype=np.float32),
            "x_flux_diff": np.zeros(4, dtype=np.float32),
            "x_valid": np.zeros(4, dtype=np.float32),
        }


def physical_f2_xpt_feature_vector(
    observation: dict[str, Any],
    *,
    no_ref_rz: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Build F2 XPT features with reference-matched fixed X point slots."""

    base_map = _physical_f1_feature_map(observation)
    ref_rx = _array(observation, "reference_rX", (4,))
    ref_zx = _array(observation, "reference_zX", (4,))
    ref_valid = _array(observation, "reference_x_valid", (4,))
    reference_nx = _scalar(observation, "reference_nX", float(np.sum(ref_valid > 0.5)))
    if reference_nx <= 0.0:
        reference_nx = 4.0
    pack = _xpt_pack_from_observation(observation)
    x_r = np.asarray(pack["x_r"], dtype=np.float32).reshape(4)
    x_z = np.asarray(pack["x_z"], dtype=np.float32).reshape(4)
    x_valid = np.asarray(pack["x_valid"], dtype=np.float32).reshape(4)
    x_flux = np.asarray(pack["x_flux_diff"], dtype=np.float32).reshape(4)
    nx = float(pack["nX"])

    features: dict[str, np.ndarray] = {
        **base_map,
        "xpt_nx_norm": np.asarray([nx / max(reference_nx, 1.0)], dtype=np.float32),
        "xpt_nx_error": np.asarray([(nx - reference_nx) / max(reference_nx, 1.0)], dtype=np.float32),
        "xpt_x_valid": x_valid,
        "xpt_r": x_r,
        "xpt_z": x_z,
        "xpt_position_error_3cm": np.stack(
            [(x_r - ref_rx) / 0.03, (x_z - ref_zx) / 0.03],
            axis=-1,
        ),
        "xpt_flux_error_0p003": x_flux / 0.003,
        "reference_xpt_r": ref_rx,
        "reference_xpt_z": ref_zx,
        "reference_xpt_valid": ref_valid,
        "reference_nx": np.asarray([reference_nx], dtype=np.float32),
    }
    mode = "physical_f2_xpt_no_ref_rz_v1" if no_ref_rz else "physical_f2_xpt_v1"
    keys = physical_feature_keys_for_task("F2", history_steps=0, mode=mode)
    return np.concatenate([_clip_finite(features[key]).reshape(-1) for key in keys]).astype(dtype)


def _physical_f1_feature_map(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    r = _scalar(observation, "R")
    ref_r = _scalar(observation, "reference_R", r)
    z = _scalar(observation, "Z")
    ref_z = _scalar(observation, "reference_Z", z)
    lx = _scalar(observation, "lX")
    ref_lx = _scalar(observation, "reference_lX", lx)
    lcfs = _array(observation, "lcfs_points", (32, 2))
    ref_lcfs = _array(observation, "reference_lcfs_points", (32, 2))
    i_pf = _array(observation, "I_PF", (12,))

    return {
        "ip": np.asarray([ip / 500_000.0], dtype=np.float32),
        "reference_ip": np.asarray([ref_ip / 500_000.0], dtype=np.float32),
        "ip_error_50ka": np.asarray([(ip - ref_ip) / 50_000.0], dtype=np.float32),
        "r": np.asarray([r], dtype=np.float32),
        "reference_r": np.asarray([ref_r], dtype=np.float32),
        "r_error_3cm": np.asarray([(r - ref_r) / 0.03], dtype=np.float32),
        "z": np.asarray([z], dtype=np.float32),
        "reference_z": np.asarray([ref_z], dtype=np.float32),
        "z_error_3cm": np.asarray([(z - ref_z) / 0.03], dtype=np.float32),
        "lx": np.asarray([lx], dtype=np.float32),
        "reference_lx": np.asarray([ref_lx], dtype=np.float32),
        "lx_error": np.asarray([lx - ref_lx], dtype=np.float32),
        "lcfs_points_absolute": lcfs,
        "reference_lcfs_points_absolute": ref_lcfs,
        "lcfs_error_3cm": (lcfs - ref_lcfs) / 0.03,
        "i_pf_norm": i_pf / COIL_LIMITS_A,
        "coil_margin": np.asarray([np.max(np.abs(i_pf) / COIL_LIMITS_A)], dtype=np.float32),
        "rmax": np.asarray([_scalar(observation, "Rmax")], dtype=np.float32),
        "rmin": np.asarray([_scalar(observation, "Rmin")], dtype=np.float32),
        "aminor": np.asarray([_scalar(observation, "aminor")], dtype=np.float32),
        "deltal": np.asarray([_scalar(observation, "deltal")], dtype=np.float32),
        "deltau": np.asarray([_scalar(observation, "deltau")], dtype=np.float32),
        "kappa": np.asarray([_scalar(observation, "kappa")], dtype=np.float32),
        "rc": np.asarray([_scalar(observation, "rc")], dtype=np.float32),
        "zc": np.asarray([_scalar(observation, "zc")], dtype=np.float32),
        "fa": np.asarray([_scalar(observation, "FA")], dtype=np.float32),
        "fb": np.asarray([_scalar(observation, "FB")], dtype=np.float32),
        "failure": np.asarray([_scalar(observation, "failure")], dtype=np.float32),
    }


def _coerce_array(value: Any, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    size = int(np.prod(shape))
    if value is None:
        return np.zeros(shape, dtype=dtype)
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    out = np.zeros(size, dtype=dtype)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(shape)


def flatten_observation(
    observation: dict[str, Any],
    keys: Sequence[str],
    shapes: dict[str, Sequence[int]] | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Flatten selected dict observation keys with stable missing-key shapes."""
    parts: list[np.ndarray] = []
    for key in keys:
        shape = tuple(shapes[key]) if shapes and key in shapes else FEATURE_SHAPES.get(key, (1,))
        parts.append(_coerce_array(observation.get(key), shape, dtype=dtype).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype)


@dataclass(frozen=True)
class FeatureMetadata:
    task: str
    keys: list[str]
    shapes: dict[str, list[int]]
    action_mode: str = "7D"
    action_dim: int = 7
    action_units: str = "physical_voltage"
    feature_mode: str = "raw"
    history_steps: int = 0

    @classmethod
    def for_task(
        cls,
        task: str,
        mode: str = "raw",
        history_steps: int = 0,
        action_units: str = "physical_voltage",
    ) -> "FeatureMetadata":
        task = normalize_task(task)
        mode = str(mode or "raw")
        history_steps = max(0, int(history_steps))
        action_units = str(action_units or "physical_voltage")
        physical_modes = {
            "physical_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
            "physical_f1_no_ref_rz_v1",
        }
        if mode in physical_modes:
            if mode in {"physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"} and task != "F2":
                raise ValueError(f"feature mode {mode!r} is only valid for F2")
            if mode == "physical_f1_no_ref_rz_v1" and task != "F1":
                raise ValueError("feature mode 'physical_f1_no_ref_rz_v1' is only valid for F1")
            keys = physical_feature_keys_for_task(task, history_steps=history_steps, mode=mode)
            shapes = physical_feature_shapes_for_task(task, history_steps=history_steps, mode=mode)
            return cls(
                task=task,
                keys=keys,
                shapes=shapes,
                action_units=action_units,
                feature_mode=mode,
                history_steps=history_steps,
            )

        keys = feature_keys_for_task(task)
        return cls(
            task=task,
            keys=keys,
            shapes=feature_shapes_for_keys(keys),
            action_units=action_units,
            feature_mode="raw",
            history_steps=0,
        )

    @property
    def obs_dim(self) -> int:
        return int(sum(np.prod(self.shapes[key]) for key in self.keys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "task": self.task,
            "keys": list(self.keys),
            "shapes": dict(self.shapes),
            "obs_dim": self.obs_dim,
            "action_mode": self.action_mode,
            "action_dim": self.action_dim,
            "action_units": self.action_units,
            "feature_mode": self.feature_mode,
            "history_steps": self.history_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureMetadata":
        keys = [str(key) for key in data["keys"]]
        shapes = {str(k): [int(x) for x in v] for k, v in data["shapes"].items()}
        return cls(
            task=normalize_task(data["task"]),
            keys=keys,
            shapes=shapes,
            action_mode=str(data.get("action_mode", "7D")),
            action_dim=int(data.get("action_dim", 7)),
            action_units=str(data.get("action_units", "physical_voltage")),
            feature_mode=str(data.get("feature_mode", "raw")),
            history_steps=int(data.get("history_steps", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureMetadata":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FixedFeatureFlattenWrapper(gym.ObservationWrapper):
    """Flatten dict observations according to a ``FeatureMetadata`` object."""

    def __init__(self, env: gym.Env, metadata: FeatureMetadata, dtype=np.float32):
        super().__init__(env)
        self.metadata = metadata
        self.dtype = dtype
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(metadata.obs_dim,),
            dtype=dtype,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        return flatten_observation(
            observation,
            keys=self.metadata.keys,
            shapes=self.metadata.shapes,
            dtype=self.dtype,
        )


class PhysicalFeatureHistoryWrapper(gym.Wrapper):
    """Convert dict observations to physical-v1 features with fixed history slots."""

    def __init__(self, env: gym.Env, metadata: FeatureMetadata, dtype=np.float32):
        super().__init__(env)
        if metadata.feature_mode not in {
            "physical_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
            "physical_f1_no_ref_rz_v1",
        }:
            raise ValueError(
                "PhysicalFeatureHistoryWrapper requires feature_mode='physical_v1', "
                "'physical_f2_xpt_v1', 'physical_f2_xpt_no_ref_rz_v1', "
                "or 'physical_f1_no_ref_rz_v1'"
            )
        self.metadata = metadata
        self.dtype = dtype
        self.history_steps = max(0, int(metadata.history_steps))
        self._last_observation: dict[str, Any] | None = None
        self._reset_history()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(metadata.obs_dim,),
            dtype=dtype,
        )

    def _reset_history(self) -> None:
        self._history_action = np.zeros((self.history_steps, 7), dtype=np.float32)
        self._history_delta_ip = np.zeros((self.history_steps, 1), dtype=np.float32)
        self._history_delta_rz = np.zeros((self.history_steps, 2), dtype=np.float32)
        self._history_delta_i_pf = np.zeros((self.history_steps, 12), dtype=np.float32)

    @staticmethod
    def _push(buffer: np.ndarray, value: np.ndarray) -> np.ndarray:
        if buffer.shape[0] == 0:
            return buffer
        out = np.roll(buffer, shift=-1, axis=0)
        out[-1] = value
        return out

    def _history_vector(self) -> np.ndarray:
        if self.history_steps <= 0:
            return np.zeros(0, dtype=self.dtype)
        return np.concatenate(
            [
                self._history_action.reshape(-1),
                self._history_delta_ip.reshape(-1),
                self._history_delta_rz.reshape(-1),
                self._history_delta_i_pf.reshape(-1),
            ]
        ).astype(self.dtype)

    def _build_observation(self, observation: dict[str, Any]) -> np.ndarray:
        if self.metadata.feature_mode in {"physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"}:
            base = physical_f2_xpt_feature_vector(
                observation,
                no_ref_rz=self.metadata.feature_mode == "physical_f2_xpt_no_ref_rz_v1",
                dtype=self.dtype,
            )
        else:
            base = physical_feature_vector(
                observation,
                self.metadata.task,
                mode=self.metadata.feature_mode,
                dtype=self.dtype,
            )
        return np.concatenate([base, self._history_vector()]).astype(self.dtype)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_observation = observation
        self._reset_history()
        return self._build_observation(observation), info

    def step(self, action: np.ndarray):
        prev = self._last_observation
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.history_steps > 0:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            action7 = np.zeros(7, dtype=np.float32)
            n_action = min(7, action_arr.size)
            if n_action > 0:
                action7[:n_action] = action_arr[:n_action]
            if self.metadata.action_units == "normalized":
                action_norm = np.clip(action7, -1.0, 1.0)
            else:
                action_norm = action7 / ACTION_SCALE_7D_V

            if prev is None:
                delta_ip = np.zeros(1, dtype=np.float32)
                delta_rz = np.zeros(2, dtype=np.float32)
                delta_i_pf = np.zeros(12, dtype=np.float32)
            else:
                delta_ip = np.asarray(
                    [(_scalar(observation, "Ip") - _scalar(prev, "Ip")) / 50_000.0],
                    dtype=np.float32,
                )
                delta_rz = np.asarray(
                    [
                        (_scalar(observation, "R") - _scalar(prev, "R")) / 0.03,
                        (_scalar(observation, "Z") - _scalar(prev, "Z")) / 0.03,
                    ],
                    dtype=np.float32,
                )
                delta_i_pf = (
                    _array(observation, "I_PF", (12,)) - _array(prev, "I_PF", (12,))
                ) / COIL_LIMITS_A

            self._history_action = self._push(self._history_action, _clip_finite(action_norm))
            self._history_delta_ip = self._push(self._history_delta_ip, _clip_finite(delta_ip))
            self._history_delta_rz = self._push(self._history_delta_rz, _clip_finite(delta_rz))
            self._history_delta_i_pf = self._push(
                self._history_delta_i_pf,
                _clip_finite(delta_i_pf),
            )

        self._last_observation = observation
        return self._build_observation(observation), reward, terminated, truncated, info
