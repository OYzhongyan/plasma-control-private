"""Shared inference helpers for semifinal submission policies."""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from environment.xpt_utils import extract_xpt_observation_pack
except Exception:
    extract_xpt_observation_pack = None
ACTION_DIM = 12
ACTION_DIM_7 = 7
ACTION_7D_TO_12D_INDEX = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
ACTION_LOW_12D = np.array(
    [-1499, -230, -172, -172, -348, -348, -270, -270, -348, -348, -270, -270],
    dtype=np.float32,
)
ACTION_HIGH_12D = np.array(
    [1499, 230, 172, 172, 348, 348, 270, 270, 348, 348, 270, 270], dtype=np.float32
)
ACTION_LOW_7D = np.array([-1499, -230, -172, -172, -348, -348, -270], dtype=np.float32)
ACTION_HIGH_7D = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)
ACTION_7D_FROM_12D_INDEX = [0, 1, 3, 5, 7, 9, 11]
COIL_LIMITS_A = np.array([45000.0] + [14000.0] * 10 + [4000.0], dtype=np.float32)
BASE_FEATURE_KEYS = [
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
    *BASE_FEATURE_KEYS,
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
FEATURE_SHAPES = {
    "Ip": [1],
    "R": [1],
    "Z": [1],
    "lX": [1],
    "reference_Ip": [1],
    "reference_R": [1],
    "reference_Z": [1],
    "reference_lX": [1],
    "lcfs_points": [32, 2],
    "reference_lcfs_points": [32, 2],
    "I_PF": [12],
    "Rmax": [1],
    "Rmin": [1],
    "aminor": [1],
    "deltal": [1],
    "deltau": [1],
    "kappa": [1],
    "rc": [1],
    "zc": [1],
    "FA": [1],
    "FB": [1],
    "failure": [1],
    "rX": [6],
    "zX": [6],
    "FX": [6],
    "nX": [1],
    "reference_rX": [4],
    "reference_zX": [4],
    "reference_x_valid": [4],
    "reference_strike_r": [8],
    "reference_strike_z": [8],
    "reference_strike_valid": [8],
    "reference_nX": [1],
    "reference_n_strike": [1],
    "xpt_nx_norm": [1],
    "xpt_nx_error": [1],
    "xpt_x_valid": [4],
    "xpt_r": [4],
    "xpt_z": [4],
    "xpt_position_error_3cm": [4, 2],
    "xpt_flux_error_0p003": [4],
    "reference_xpt_r": [4],
    "reference_xpt_z": [4],
    "reference_xpt_valid": [4],
    "reference_nx": [1],
}


def default_metadata() -> dict[str, Any]:
    keys = F2_FEATURE_KEYS
    shapes = {key: FEATURE_SHAPES[key] for key in keys}
    obs_dim = int(sum((np.prod(shapes[key]) for key in keys)))
    return {
        "version": 1,
        "keys": keys,
        "shapes": shapes,
        "obs_dim": obs_dim,
        "action_mode": "7D",
        "action_dim": 7,
        "action_units": "physical_voltage",
        "feature_mode": "raw",
        "history_steps": 0,
    }


def action_7d_to_12d(action_7d: np.ndarray) -> np.ndarray:
    action_7d = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    if action_7d.size != ACTION_DIM_7:
        raise ValueError(f"action_7d must have size 7, got {action_7d.size}")
    action_7d = np.clip(action_7d, ACTION_LOW_7D, ACTION_HIGH_7D)
    return action_7d[ACTION_7D_TO_12D_INDEX]


def normalized_action_7d_to_12d(action_7d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_norm = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    if action_norm.size != ACTION_DIM_7:
        raise ValueError(
            f"normalized action_7d must have size 7, got {action_norm.size}"
        )
    action_norm = np.clip(action_norm, -1.0, 1.0)
    action_physical = action_norm * ACTION_HIGH_7D
    return (action_7d_to_12d(action_physical), action_norm)


def _coerce(value: Any, shape: list[int], dtype=np.float32) -> np.ndarray:
    size = int(np.prod(shape))
    out = np.zeros(size, dtype=dtype)
    if value is not None:
        flat = np.asarray(value, dtype=dtype).reshape(-1)
        n = min(size, flat.size)
        if n > 0:
            out[:n] = flat[:n]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(shape)


def _scalar(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    arr = _coerce(observation.get(key), [1]).reshape(-1)
    if arr.size == 0:
        return float(default)
    value = float(arr[0])
    return value if np.isfinite(value) else float(default)


def _array(observation: dict[str, Any], key: str, shape: list[int]) -> np.ndarray:
    return _coerce(observation.get(key), shape)


def _xpt_pack(observation: dict[str, Any]) -> dict[str, Any]:
    ref_rx = _array(observation, "reference_rX", [4])
    ref_zx = _array(observation, "reference_zX", [4])
    ref_sr = _array(observation, "reference_strike_r", [8])
    ref_sz = _array(observation, "reference_strike_z", [8])
    if extract_xpt_observation_pack is not None:
        try:
            return extract_xpt_observation_pack(
                observation, ref_rx, ref_zx, ref_sr, ref_sz
            )
        except Exception:
            pass
    return {
        "nX": int(_scalar(observation, "nX")),
        "x_r": np.zeros(4, dtype=np.float32),
        "x_z": np.zeros(4, dtype=np.float32),
        "x_flux_diff": np.zeros(4, dtype=np.float32),
        "x_valid": np.zeros(4, dtype=np.float32),
    }


def _physical_feature_map(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    r = _scalar(observation, "R")
    ref_r = _scalar(observation, "reference_R", r)
    z = _scalar(observation, "Z")
    ref_z = _scalar(observation, "reference_Z", z)
    lx = _scalar(observation, "lX")
    ref_lx = _scalar(observation, "reference_lX", lx)
    lcfs = _array(observation, "lcfs_points", [32, 2])
    ref_lcfs = _array(observation, "reference_lcfs_points", [32, 2])
    i_pf = _array(observation, "I_PF", [12])
    out: dict[str, np.ndarray] = {
        "ip": np.array([ip / 500000.0], dtype=np.float32),
        "reference_ip": np.array([ref_ip / 500000.0], dtype=np.float32),
        "ip_error_50ka": np.array([(ip - ref_ip) / 50000.0], dtype=np.float32),
        "r": np.array([r], dtype=np.float32),
        "reference_r": np.array([ref_r], dtype=np.float32),
        "r_error_3cm": np.array([(r - ref_r) / 0.03], dtype=np.float32),
        "z": np.array([z], dtype=np.float32),
        "reference_z": np.array([ref_z], dtype=np.float32),
        "z_error_3cm": np.array([(z - ref_z) / 0.03], dtype=np.float32),
        "lx": np.array([lx], dtype=np.float32),
        "reference_lx": np.array([ref_lx], dtype=np.float32),
        "lx_error": np.array([lx - ref_lx], dtype=np.float32),
        "lcfs_points_absolute": lcfs,
        "reference_lcfs_points_absolute": ref_lcfs,
        "lcfs_error_3cm": (lcfs - ref_lcfs) / 0.03,
        "i_pf_norm": i_pf / COIL_LIMITS_A,
        "coil_margin": np.array(
            [np.max(np.abs(i_pf) / COIL_LIMITS_A)], dtype=np.float32
        ),
        "rmax": np.array([_scalar(observation, "Rmax")], dtype=np.float32),
        "rmin": np.array([_scalar(observation, "Rmin")], dtype=np.float32),
        "aminor": np.array([_scalar(observation, "aminor")], dtype=np.float32),
        "deltal": np.array([_scalar(observation, "deltal")], dtype=np.float32),
        "deltau": np.array([_scalar(observation, "deltau")], dtype=np.float32),
        "kappa": np.array([_scalar(observation, "kappa")], dtype=np.float32),
        "rc": np.array([_scalar(observation, "rc")], dtype=np.float32),
        "zc": np.array([_scalar(observation, "zc")], dtype=np.float32),
        "fa": np.array([_scalar(observation, "FA")], dtype=np.float32),
        "fb": np.array([_scalar(observation, "FB")], dtype=np.float32),
        "failure": np.array([_scalar(observation, "failure")], dtype=np.float32),
    }
    rx = _array(observation, "rX", [6])
    zx = _array(observation, "zX", [6])
    ref_rx = _array(observation, "reference_rX", [4])
    ref_zx = _array(observation, "reference_zX", [4])
    ref_valid = _array(observation, "reference_x_valid", [4])
    reference_nx = _scalar(observation, "reference_nX", float(np.sum(ref_valid > 0.5)))
    if reference_nx <= 0.0:
        reference_nx = 4.0
    pack = _xpt_pack(observation)
    x_r = np.asarray(pack["x_r"], dtype=np.float32).reshape(4)
    x_z = np.asarray(pack["x_z"], dtype=np.float32).reshape(4)
    x_valid = np.asarray(pack["x_valid"], dtype=np.float32).reshape(4)
    x_flux = np.asarray(pack["x_flux_diff"], dtype=np.float32).reshape(4)
    nx = float(pack["nX"])
    out.update(
        {
            "rx": rx,
            "zx": zx,
            "fx": _array(observation, "FX", [6]),
            "nx": np.array([_scalar(observation, "nX")], dtype=np.float32),
            "reference_rx": ref_rx,
            "reference_zx": ref_zx,
            "x_error_3cm": np.stack(
                [(rx[:4] - ref_rx) / 0.03, (zx[:4] - ref_zx) / 0.03], axis=-1
            ),
            "reference_x_valid": _array(observation, "reference_x_valid", [4]),
            "reference_strike_r": _array(observation, "reference_strike_r", [8]),
            "reference_strike_z": _array(observation, "reference_strike_z", [8]),
            "reference_strike_valid": _array(
                observation, "reference_strike_valid", [8]
            ),
            "reference_nx": np.array(
                [_scalar(observation, "reference_nX")], dtype=np.float32
            ),
            "reference_n_strike": np.array(
                [_scalar(observation, "reference_n_strike")], dtype=np.float32
            ),
            "xpt_nx_norm": np.array([nx / max(reference_nx, 1.0)], dtype=np.float32),
            "xpt_nx_error": np.array(
                [(nx - reference_nx) / max(reference_nx, 1.0)], dtype=np.float32
            ),
            "xpt_x_valid": x_valid,
            "xpt_r": x_r,
            "xpt_z": x_z,
            "xpt_position_error_3cm": np.stack(
                [(x_r - ref_rx) / 0.03, (x_z - ref_zx) / 0.03], axis=-1
            ),
            "xpt_flux_error_0p003": x_flux / 0.003,
            "reference_xpt_r": ref_rx,
            "reference_xpt_z": ref_zx,
            "reference_xpt_valid": ref_valid,
            "reference_nx": np.array([reference_nx], dtype=np.float32),
        }
    )
    return out


def flatten_observation(
    observation: dict[str, Any],
    metadata: dict[str, Any],
    history: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if metadata.get("feature_mode") in {
        "physical_v1",
        "physical_f2_xpt_v1",
        "physical_f2_xpt_no_ref_rz_v1",
    }:
        feature_map = _physical_feature_map(observation)
        if history:
            feature_map.update(history)
        parts = []
        shapes = metadata.get("shapes", {})
        for key in metadata.get("keys", []):
            shape = list(shapes.get(key, FEATURE_SHAPES.get(key, [1])))
            parts.append(_coerce(feature_map.get(key), shape).reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32)
    parts = []
    shapes = metadata.get("shapes", {})
    for key in metadata.get("keys", []):
        shape = list(shapes.get(key, FEATURE_SHAPES.get(key, [1])))
        parts.append(_coerce(observation.get(key), shape).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_metadata(model_path: Path | None) -> dict[str, Any]:
    if model_path is not None:
        candidate = model_path.with_suffix(".metadata.json")
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
    return default_metadata()


def _vecnormalize_candidates(model_path: Path) -> list[Path]:
    stem = model_path.stem
    return [
        model_path.with_name(f"{stem}_vecnormalize_stats.npz"),
        model_path.with_name(f"{stem}.vecnormalize_stats.npz"),
        model_path.with_name(f"{stem}_vecnormalize.npz"),
        model_path.with_name(f"{stem}.vecnormalize.npz"),
        model_path.parent / "vecnormalize_stats.npz",
    ]


def load_vecnormalize_stats(
    model_path: Path | None, obs_dim: int
) -> dict[str, Any] | None:
    if model_path is None:
        return None
    obs_dim = int(obs_dim)
    if obs_dim <= 0:
        return None
    for candidate in _vecnormalize_candidates(model_path):
        if not candidate.is_file():
            continue
        try:
            with np.load(candidate) as data:
                if "obs_mean" not in data or "obs_var" not in data:
                    continue
                mean = np.asarray(data["obs_mean"], dtype=np.float32).reshape(-1)
                var = np.asarray(data["obs_var"], dtype=np.float32).reshape(-1)
                if mean.size != obs_dim or var.size != obs_dim:
                    continue
                clip_obs = (
                    float(np.asarray(data["clip_obs"]).reshape(-1)[0])
                    if "clip_obs" in data
                    else 10.0
                )
                epsilon = (
                    float(np.asarray(data["epsilon"]).reshape(-1)[0])
                    if "epsilon" in data
                    else 1e-08
                )
                if not np.isfinite(clip_obs) or clip_obs <= 0.0:
                    clip_obs = 10.0
                if not np.isfinite(epsilon) or epsilon <= 0.0:
                    epsilon = 1e-08
                return {
                    "path": str(candidate),
                    "mean": mean,
                    "var": np.maximum(var, 0.0),
                    "clip_obs": clip_obs,
                    "epsilon": epsilon,
                }
        except Exception:
            continue
    return None


def apply_obs_normalization(
    obs: np.ndarray, stats: dict[str, Any] | None
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if stats is None:
        return obs
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    var = np.asarray(stats["var"], dtype=np.float32).reshape(-1)
    if mean.size != obs.size or var.size != obs.size:
        return obs
    epsilon = float(stats.get("epsilon", 1e-08))
    clip_obs = float(stats.get("clip_obs", 10.0))
    normalized = (obs - mean) / np.sqrt(var + epsilon)
    return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32)


class BaseOnnxPolicy:
    """ONNX-backed policy with zero-action fallback."""

    def __init__(self, *, model_dir: str | Path, preferred_name: str):
        self.model_dir = Path(model_dir)
        self.preferred_name = preferred_name
        self.session = None
        self.input_name = None
        self.output_name = None
        self.metadata = default_metadata()
        self.obs_normalization: dict[str, Any] | None = None
        self._last_observation: dict[str, Any] | None = None
        self._last_action7: np.ndarray | None = None
        self._episode_initial_rz: tuple[float, float] | None = None
        self._history: dict[str, np.ndarray] = {}
        self._try_load_model()
        self._reset_history()

    def _resolve_model_path(self) -> Path | None:
        preferred = self.model_dir / self.preferred_name
        if preferred.exists():
            return preferred
        fallback = self.model_dir / "policy.onnx"
        if fallback.exists():
            return fallback
        candidates = sorted(self.model_dir.glob("*.onnx"))
        return candidates[0] if candidates else None

    def _try_load_model(self) -> None:
        model_path = self._resolve_model_path()
        self.metadata = load_metadata(model_path)
        self.obs_normalization = load_vecnormalize_stats(
            model_path, int(self.metadata.get("obs_dim", 0))
        )
        if model_path is None:
            return
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception:
            self.session = None
            self.input_name = None
            self.output_name = None

    def _reset_history(self) -> None:
        steps = max(0, int(self.metadata.get("history_steps", 0)))
        self._history = {
            "history_action_7d": np.zeros((steps, 7), dtype=np.float32),
            "history_delta_ip": np.zeros((steps, 1), dtype=np.float32),
            "history_delta_rz": np.zeros((steps, 2), dtype=np.float32),
            "history_delta_i_pf": np.zeros((steps, 12), dtype=np.float32),
        }
        self._last_observation = None
        self._last_action7 = None
        self._episode_initial_rz = None

    def _sanitize_reference_rz(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.metadata.get("feature_mode") not in {
            "physical_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
        }:
            return observation
        r = _scalar(observation, "R")
        z = _scalar(observation, "Z")
        if self._episode_initial_rz is None and np.isfinite(r) and np.isfinite(z):
            self._episode_initial_rz = (float(r), float(z))
        if self._episode_initial_rz is None:
            return observation
        out = dict(observation)
        missing_reference_rz = (
            "reference_R" not in observation or "reference_Z" not in observation
        )
        ref_r = _scalar(observation, "reference_R", 0.0)
        ref_z = _scalar(observation, "reference_Z", 0.0)
        ref_invalid = (
            missing_reference_rz
            or abs(float(ref_r)) <= 1e-06
            or (not np.isfinite(ref_r))
            or (not np.isfinite(ref_z))
        )
        if ref_invalid:
            init_r, init_z = self._episode_initial_rz
            out["reference_R"] = np.asarray([init_r], dtype=np.float32)
            out["reference_Z"] = np.asarray([init_z], dtype=np.float32)
        task_default_lx = 1.0
        ref_lx = _scalar(observation, "reference_lX", task_default_lx)
        if "reference_lX" not in observation or not np.isfinite(ref_lx):
            out["reference_lX"] = np.asarray([task_default_lx], dtype=np.float32)
        return out

    @staticmethod
    def _push_history(buffer: np.ndarray, value: np.ndarray) -> np.ndarray:
        if buffer.shape[0] == 0:
            return buffer
        out = np.roll(buffer, shift=-1, axis=0)
        out[-1] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _update_physical_history(self, observation: dict[str, Any]) -> None:
        if self.metadata.get("feature_mode") not in {
            "physical_v1",
            "physical_f2_xpt_v1",
            "physical_f2_xpt_no_ref_rz_v1",
        }:
            return
        if self._last_observation is None or self._last_action7 is None:
            return
        prev = self._last_observation
        delta_ip = np.array(
            [(_scalar(observation, "Ip") - _scalar(prev, "Ip")) / 50000.0],
            dtype=np.float32,
        )
        delta_rz = np.array(
            [
                (_scalar(observation, "R") - _scalar(prev, "R")) / 0.03,
                (_scalar(observation, "Z") - _scalar(prev, "Z")) / 0.03,
            ],
            dtype=np.float32,
        )
        delta_i_pf = (
            _array(observation, "I_PF", [12]) - _array(prev, "I_PF", [12])
        ) / COIL_LIMITS_A
        action_norm = self._last_action7
        self._history["history_action_7d"] = self._push_history(
            self._history["history_action_7d"], action_norm
        )
        self._history["history_delta_ip"] = self._push_history(
            self._history["history_delta_ip"], delta_ip
        )
        self._history["history_delta_rz"] = self._push_history(
            self._history["history_delta_rz"], delta_rz
        )
        self._history["history_delta_i_pf"] = self._push_history(
            self._history["history_delta_i_pf"], delta_i_pf
        )

    def _action7_from_output(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if action.size == ACTION_DIM_7:
            if self.metadata.get("action_units") == "normalized":
                return normalized_action_7d_to_12d(action)
            action7 = np.clip(action.astype(np.float32), ACTION_LOW_7D, ACTION_HIGH_7D)
            return (
                action_7d_to_12d(action7),
                np.clip(action7 / ACTION_HIGH_7D, -1.0, 1.0),
            )
        if action.size == ACTION_DIM:
            action12 = np.clip(
                action.astype(np.float32), ACTION_LOW_12D, ACTION_HIGH_12D
            )
            return (
                action12,
                np.clip(action12[ACTION_7D_FROM_12D_INDEX] / ACTION_HIGH_7D, -1.0, 1.0),
            )
        raise ValueError(f"Policy must return 12D or 7D action, got size {action.size}")

    def reset(self) -> None:
        self._reset_history()
        return None

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self.session is None or self.input_name is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        observation = self._sanitize_reference_rz(observation)
        self._update_physical_history(observation)
        obs = flatten_observation(observation, self.metadata, self._history)
        obs = apply_obs_normalization(obs, self.obs_normalization).reshape(1, -1)
        try:
            action = self.session.run([self.output_name], {self.input_name: obs})[0]
        except Exception:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action12, action7 = self._action7_from_output(action)
        self._last_observation = observation
        self._last_action7 = action7
        return action12.astype(np.float32)
