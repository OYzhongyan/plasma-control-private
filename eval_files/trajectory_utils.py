"""轨迹与参考曲线工具：从 obs 列表提取轨迹、构建 task reference、序列化。"""

from __future__ import annotations
import json
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def _load_xpt_utils():
    root = Path(__file__).resolve().parents[1]
    path = root / "environment" / "xpt_utils.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_semifinal_xpt_utils", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_XPT_UTILS = _load_xpt_utils()
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_scalar(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _resolve_project_path(path_value: str | Path | None, default: str | Path) -> Path:
    path = Path(path_value) if path_value is not None else Path(default)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _fit_series_to_steps(values: Any, max_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(max_steps, float(arr), dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size >= max_steps:
        return flat[:max_steps].astype(np.float64)
    if flat.size == 0:
        return np.zeros(max_steps, dtype=np.float64)
    pad = np.full(max_steps - flat.size, float(flat[-1]), dtype=np.float64)
    return np.concatenate([flat.astype(np.float64), pad], axis=0)


def _load_official_target_block(
    reference_spec: dict[str, Any]
) -> dict[str, Any] | None:
    if reference_spec.get("target_source") != "official_target":
        return None
    target_path = _resolve_project_path(
        reference_spec.get("target_json"), "evaluation/results/target.json"
    )
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("F2b", data.get("F2a", data.get("F2")))


def _official_xpt_reference_fields(
    target_block: dict[str, Any]
) -> dict[str, np.ndarray]:
    if not {"Xpt_main", "Xpt_sec", "strike"}.issubset(target_block):
        return {}
    x_main = np.asarray(target_block["Xpt_main"], dtype=np.float64)
    x_sec = np.asarray(target_block["Xpt_sec"], dtype=np.float64)
    strike = np.asarray(target_block["strike"], dtype=np.float64)
    return {
        "rX": np.asarray(
            [x_sec[0, 0], x_main[0, 0], x_main[1, 0], x_sec[1, 0]], dtype=np.float64
        ),
        "zX": np.asarray(
            [x_sec[0, 1], x_main[0, 1], x_main[1, 1], x_sec[1, 1]], dtype=np.float64
        ),
        "x_valid": np.ones(4, dtype=np.float64),
        "strike_r": strike[:, 0].astype(np.float64),
        "strike_z": strike[:, 1].astype(np.float64),
        "strike_valid": np.ones(strike.shape[0], dtype=np.float64),
        "nX": np.asarray([4.0], dtype=np.float64),
        "n_strike": np.asarray([float(strike.shape[0])], dtype=np.float64),
    }


def build_official_target_observation(
    reference_spec: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    """Build a target-like observation from the public scoring target JSON."""
    target_block = _load_official_target_block(reference_spec)
    if target_block is None:
        return None
    lx = float(
        reference_spec.get("target_lX", 0.0 if "Xpt_main" not in target_block else 1.0)
    )
    ip = _fit_series_to_steps(target_block.get("Ip", [0.0]), 1)[0]
    obs: dict[str, np.ndarray] = {
        "Ip": np.asarray([ip], dtype=np.float64),
        "lX": np.asarray([lx], dtype=np.float64),
        "lcfs_points": np.asarray(target_block["lcfs_points"], dtype=np.float64),
    }
    xpt = _official_xpt_reference_fields(target_block)
    if xpt:
        obs.update(
            {
                "reference_rX": xpt["rX"],
                "reference_zX": xpt["zX"],
                "reference_x_valid": xpt["x_valid"],
                "reference_strike_r": xpt["strike_r"],
                "reference_strike_z": xpt["strike_z"],
                "reference_strike_valid": xpt["strike_valid"],
                "reference_nX": xpt["nX"],
                "reference_n_strike": xpt["n_strike"],
            }
        )
    return obs


def extract_trajectory_from_obs_list(
    obs_list: List[Dict[str, Any]],
    *,
    ip_key: str = "Ip",
    r_key: str = "R",
    z_key: str = "Z",
    lcfs_key: Optional[str] = None,
) -> Dict[str, Any]:
    """从观测列表提取 trajectory：Ip, R0, Z0 及可选的 lcfs_per_step。"""
    if not obs_list:
        return {"Ip": np.array([]), "R0": np.array([]), "Z0": np.array([])}
    Ip, R0, Z0 = ([], [], [])
    lcfs_per_step = []
    for obs in obs_list:
        o = obs if isinstance(obs, dict) else {}
        ip = _to_scalar(o.get(ip_key))
        if ip is not None:
            Ip.append(ip)
        r = _to_scalar(o.get(r_key))
        if r is not None:
            R0.append(r)
        z = _to_scalar(o.get(z_key))
        if z is not None:
            Z0.append(z)
        if lcfs_key and lcfs_key in o and (o[lcfs_key] is not None):
            lcfs_per_step.append(np.asarray(o[lcfs_key], dtype=float))
    out = {
        "Ip": np.array(Ip, dtype=float),
        "R0": np.array(R0, dtype=float),
        "Z0": np.array(Z0, dtype=float),
    }
    if lcfs_per_step:
        out["lcfs_per_step"] = lcfs_per_step
    return out


def _array_or_zeros(
    obs: dict[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    value = obs.get(key)
    if value is None:
        return np.zeros(shape, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != int(np.prod(shape)):
        return np.zeros(shape, dtype=np.float64)
    return arr.reshape(shape)


def _extract_lx(obs: dict[str, Any]) -> int:
    if "lX" in obs:
        val = _to_scalar(obs.get("lX"))
        if val is not None:
            return int(round(val))
    nx = _to_scalar(obs.get("nX"), 0.0)
    return int((nx or 0.0) > 0.5)


def _target_xpt_refs(
    target_obs: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_obs is None or _XPT_UTILS is None:
        return (
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
        )
    if "reference_rX" in target_obs and "reference_zX" in target_obs:
        return (
            np.asarray(target_obs["reference_rX"], dtype=np.float64).reshape(4),
            np.asarray(target_obs["reference_zX"], dtype=np.float64).reshape(4),
            np.asarray(
                target_obs.get("reference_strike_r", np.zeros(8)), dtype=np.float64
            ).reshape(8),
            np.asarray(
                target_obs.get("reference_strike_z", np.zeros(8)), dtype=np.float64
            ).reshape(8),
        )
    try:
        r_x, z_x, _ = _XPT_UTILS.extract_target_xpoints(target_obs, slots=4)
        r_s, z_s, _, _ = _XPT_UTILS.extract_target_strike_points(
            target_obs, strike_slots=8
        )
    except Exception:
        return (
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
            np.zeros(8, dtype=np.float64),
        )
    return (
        np.asarray(r_x, dtype=np.float64),
        np.asarray(z_x, dtype=np.float64),
        np.asarray(r_s, dtype=np.float64),
        np.asarray(z_s, dtype=np.float64),
    )


def _is_xpt_target_task(
    reference_spec: dict[str, Any], target_obs: dict[str, Any] | None
) -> bool:
    ip_curve = reference_spec.get("curves", {}).get("Ip", {})
    return True
    n_x = _to_scalar((target_obs or {}).get("nX"), 0.0) or 0.0
    return int(round(n_x)) == 4


def _target_xpt_reference_fields(
    reference_spec: dict[str, Any], target_obs: dict[str, Any] | None
) -> dict[str, np.ndarray]:
    if (
        target_obs is None
        or _XPT_UTILS is None
        or (not _is_xpt_target_task(reference_spec, target_obs))
    ):
        return {}
    if "reference_rX" in target_obs and "reference_zX" in target_obs:
        x_valid = np.asarray(
            target_obs.get("reference_x_valid", np.ones(4)), dtype=np.float64
        )
        strike_valid = np.asarray(
            target_obs.get("reference_strike_valid", np.ones(8)), dtype=np.float64
        )
        return {
            "rX": np.asarray(target_obs["reference_rX"], dtype=np.float64),
            "zX": np.asarray(target_obs["reference_zX"], dtype=np.float64),
            "x_valid": x_valid,
            "strike_r": np.asarray(
                target_obs.get("reference_strike_r", np.zeros(8)), dtype=np.float64
            ),
            "strike_z": np.asarray(
                target_obs.get("reference_strike_z", np.zeros(8)), dtype=np.float64
            ),
            "strike_valid": strike_valid,
            "nX": np.asarray([float(np.sum(x_valid > 0.5))], dtype=np.float64),
            "n_strike": np.asarray(
                [float(np.sum(strike_valid > 0.5))], dtype=np.float64
            ),
        }
    try:
        r_x, z_x, x_valid = _XPT_UTILS.extract_target_xpoints(target_obs, slots=4)
        r_s, z_s, s_valid, n_s = _XPT_UTILS.extract_target_strike_points(
            target_obs, strike_slots=8
        )
    except Exception:
        return {}
    n_x = _to_scalar(target_obs.get("nX"), float(np.sum(x_valid))) or float(
        np.sum(x_valid)
    )
    return {
        "rX": np.asarray(r_x, dtype=np.float64),
        "zX": np.asarray(z_x, dtype=np.float64),
        "x_valid": np.asarray(x_valid, dtype=np.float64),
        "strike_r": np.asarray(r_s, dtype=np.float64),
        "strike_z": np.asarray(z_s, dtype=np.float64),
        "strike_valid": np.asarray(s_valid, dtype=np.float64),
        "nX": np.asarray([n_x], dtype=np.float64),
        "n_strike": np.asarray([n_s], dtype=np.float64),
    }


def extract_semifinal_trajectory_from_obs_list(
    obs_list: List[Dict[str, Any]], *, target_obs: dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Extract trajectory fields required by the semi-final F2b scorer."""
    if not obs_list:
        return {
            "Ip": np.array([], dtype=np.float64),
            "lX": np.array([], dtype=np.float64),
            "nX": np.array([], dtype=np.float64),
            "Icoil": np.zeros((0, 12), dtype=np.float64),
            "psia": np.array([], dtype=np.float64),
            "psib": np.array([], dtype=np.float64),
            "lcfs_per_step": [],
            "Xpt_main": np.zeros((0, 2, 2), dtype=np.float64),
            "Xpt_sec": np.zeros((0, 2, 2), dtype=np.float64),
            "strike": np.zeros((0, 8, 2), dtype=np.float64),
            "psiX_main": np.zeros((0, 2), dtype=np.float64),
            "psiX_sec": np.zeros((0, 2), dtype=np.float64),
        }
    ref_r_x, ref_z_x, ref_r_s, ref_z_s = _target_xpt_refs(target_obs)
    primary = (
        tuple(getattr(_XPT_UTILS, "X_PRIMARY_SLOTS", (1, 2))) if _XPT_UTILS else (1, 2)
    )
    secondary = (
        tuple(getattr(_XPT_UTILS, "X_SECONDARY_SLOTS", (0, 3)))
        if _XPT_UTILS
        else (0, 3)
    )
    ip, lx, nx, icoil, psia, psib = ([], [], [], [], [], [])
    lcfs_per_step = []
    xpt_main, xpt_sec, strike = ([], [], [])
    psix_main, psix_sec = ([], [])
    for obs in obs_list:
        o = obs if isinstance(obs, dict) else {}
        ip.append(_to_scalar(o.get("Ip"), 0.0) or 0.0)
        lx.append(_extract_lx(o))
        nx_val = int(round(_to_scalar(o.get("nX"), 0.0) or 0.0))
        nx.append(nx_val)
        icoil.append(_array_or_zeros(o, "I_PF", (12,)))
        psia.append(_to_scalar(o.get("FA"), 0.0) or 0.0)
        psib.append(_to_scalar(o.get("FB"), 0.0) or 0.0)
        lcfs_per_step.append(_array_or_zeros(o, "lcfs_points", (32, 2)))
        if _XPT_UTILS is not None:
            try:
                pack = _XPT_UTILS.extract_xpt_observation_pack(
                    o, ref_r_x, ref_z_x, ref_r_s, ref_z_s, fx_order="C"
                )
            except Exception:
                pack = None
        else:
            pack = None
        if pack is not None:
            x_r = np.asarray(pack["x_r"], dtype=np.float64)
            x_z = np.asarray(pack["x_z"], dtype=np.float64)
            x_fx = np.asarray(pack["x_fx"], dtype=np.float64)
            xpt_main.append(np.column_stack([x_r[list(primary)], x_z[list(primary)]]))
            xpt_sec.append(
                np.column_stack([x_r[list(secondary)], x_z[list(secondary)]])
            )
            strike.append(
                np.column_stack(
                    [
                        np.asarray(pack["strike_r"], dtype=np.float64),
                        np.asarray(pack["strike_z"], dtype=np.float64),
                    ]
                )
            )
            psix_main.append(x_fx[list(primary)])
            psix_sec.append(x_fx[list(secondary)])
        else:
            xpt_main.append(np.zeros((2, 2), dtype=np.float64))
            xpt_sec.append(np.zeros((2, 2), dtype=np.float64))
            strike.append(np.zeros((8, 2), dtype=np.float64))
            fb = psib[-1]
            psix_main.append(np.full(2, fb, dtype=np.float64))
            psix_sec.append(np.full(2, fb, dtype=np.float64))
    return {
        "Ip": np.asarray(ip, dtype=np.float64),
        "lX": np.asarray(lx, dtype=np.float64),
        "nX": np.asarray(nx, dtype=np.float64),
        "Icoil": np.asarray(icoil, dtype=np.float64),
        "psia": np.asarray(psia, dtype=np.float64),
        "psib": np.asarray(psib, dtype=np.float64),
        "lcfs_per_step": lcfs_per_step,
        "Xpt_main": np.asarray(xpt_main, dtype=np.float64),
        "Xpt_sec": np.asarray(xpt_sec, dtype=np.float64),
        "strike": np.asarray(strike, dtype=np.float64),
        "psiX_main": np.asarray(psix_main, dtype=np.float64),
        "psiX_sec": np.asarray(psix_sec, dtype=np.float64),
    }


def _piecewise_linear(
    max_steps: int, anchors: list[int], values: np.ndarray
) -> np.ndarray:
    x = np.arange(max_steps)
    return np.interp(
        x, np.asarray(anchors, dtype=int), np.asarray(values, dtype=np.float64)
    )


def _repeat_initial(initial_value: np.ndarray, max_steps: int) -> np.ndarray:
    arr = np.asarray(initial_value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(max_steps, float(arr), dtype=np.float64)
    return np.repeat(arr[None, ...], max_steps, axis=0)


def _semifinal_ip_reference(max_steps: int) -> np.ndarray:
    t = np.arange(max_steps, dtype=np.float64) + 1.0
    pass
    ref = np.where(t <= 100.0, 500000.0 - t * 1000.0, 400000.0)
    return np.clip(ref, 400000.0, 500000.0)
    raise ValueError("Unsupported semifinal Ip task_id: F2b")


def _build_curve_series(
    initial_value: Any,
    curve_spec: dict[str, Any],
    max_steps: int,
    *,
    target_value: Any = None,
) -> np.ndarray:
    mode = curve_spec.get("mode", "hold_initial")
    initial_arr = np.asarray(initial_value, dtype=np.float64)
    if mode == "hold_initial":
        return _repeat_initial(initial_arr, max_steps)
    if mode == "hold_target":
        if target_value is None:
            raise ValueError("hold_target requires target_value")
        return _repeat_initial(np.asarray(target_value, dtype=np.float64), max_steps)
    if mode == "linear_to_target":
        if target_value is None:
            raise ValueError("linear_to_target requires target_value")
        target_arr = np.asarray(target_value, dtype=np.float64)
        ramp_start = int(curve_spec.get("ramp_start_step", 0))
        ramp_end = int(curve_spec.get("ramp_end_step", max_steps - 1))
        x = np.arange(max_steps, dtype=np.float64)
        denom = max(float(ramp_end - ramp_start), 1.0)
        w = np.clip((x - float(ramp_start)) / denom, 0.0, 1.0)
        return (1.0 - w[(slice(None),) + (None,) * initial_arr.ndim]) * initial_arr + w[
            (slice(None),) + (None,) * target_arr.ndim
        ] * target_arr
    if mode == "anchor_offsets":
        if initial_arr.ndim != 0:
            raise ValueError(
                f"anchor_offsets only supports scalar references, got shape {initial_arr.shape}"
            )
        anchor_steps = list(curve_spec.get("anchor_steps", []))
        offsets = np.asarray(curve_spec.get("offsets", []), dtype=np.float64)
        if len(anchor_steps) != len(offsets):
            raise ValueError("anchor_steps and offsets must have the same length")
        if (
            not anchor_steps
            or anchor_steps[0] != 0
            or anchor_steps[-1] != max_steps - 1
        ):
            raise ValueError("anchor_steps must start at 0 and end at max_steps - 1")
        return _piecewise_linear(max_steps, anchor_steps, float(initial_arr) + offsets)
    if mode == "explicit_series":
        values = np.asarray(curve_spec.get("values"), dtype=np.float64)
        if values.shape[0] != max_steps:
            raise ValueError(
                f"explicit_series length must equal max_steps={max_steps}, got {values.shape}"
            )
        return values
    if mode == "semifinal_ip":
        return _semifinal_ip_reference(max_steps)
    raise ValueError(f"Unsupported curve mode: {mode}")


def build_reference_trajectory(
    reference_spec: dict[str, Any],
    initial_obs: dict[str, Any],
    max_steps: int,
    target_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据 task_spec 和 initial_obs 构建该任务的参考轨迹与 XPT 控制目标。"""
    if reference_spec.get("target_source") == "official_target":
        target_block = _load_official_target_block(reference_spec)
        if target_block is None:
            raise ValueError("official_target task is missing target block")
        ip_ref = _fit_series_to_steps(target_block.get("Ip", []), max_steps)
        lx_t = float(
            reference_spec.get(
                "target_lX", 0.0 if "Xpt_main" not in target_block else 1.0
            )
        )
        lcfs = np.asarray(target_block["lcfs_points"], dtype=np.float64)
        reference: dict[str, Any] = {
            "Ip": ip_ref,
            "lX": np.full(max_steps, lx_t, dtype=np.float64),
            "lcfs_points": np.repeat(lcfs[None, :, :], max_steps, axis=0),
        }
        reference.update(_official_xpt_reference_fields(target_block))
        return reference
    curves = reference_spec.get("curves", {})
    ip0 = np.asarray(initial_obs["Ip"], dtype=np.float64).reshape(-1)[0]
    r0 = np.asarray(initial_obs["R"], dtype=np.float64).reshape(-1)[0]
    z0 = np.asarray(initial_obs["Z"], dtype=np.float64).reshape(-1)[0]
    lx0 = _to_scalar(initial_obs.get("lX"), 0.0) or 0.0
    lcfs0 = np.asarray(initial_obs.get("lcfs_points"), dtype=np.float64)
    target_obs = target_obs or initial_obs
    ip_t = np.asarray(target_obs["Ip"], dtype=np.float64).reshape(-1)[0]
    r_t = np.asarray(target_obs["R"], dtype=np.float64).reshape(-1)[0]
    z_t = np.asarray(target_obs["Z"], dtype=np.float64).reshape(-1)[0]
    default_lx_t = _to_scalar(target_obs.get("lX"), lx0) or lx0
    lx_t = float(reference_spec.get("target_lX", default_lx_t))
    lcfs_t = np.asarray(target_obs.get("lcfs_points"), dtype=np.float64)
    ip_ref = _build_curve_series(
        ip0, curves.get("Ip", {"mode": "hold_initial"}), max_steps, target_value=ip_t
    )
    r_ref = _build_curve_series(
        r0, curves.get("R", {"mode": "hold_initial"}), max_steps, target_value=r_t
    )
    z_ref = _build_curve_series(
        z0, curves.get("Z", {"mode": "hold_initial"}), max_steps, target_value=z_t
    )
    lx_ref = _build_curve_series(
        lx0, curves.get("lX", {"mode": "hold_target"}), max_steps, target_value=lx_t
    )
    lcfs_ref = _build_curve_series(
        lcfs0,
        curves.get("lcfs_points", {"mode": "hold_initial"}),
        max_steps,
        target_value=lcfs_t,
    )
    reference = {
        "Ip": ip_ref,
        "R": r_ref,
        "Z": z_ref,
        "lX": lx_ref,
        "lcfs_points": lcfs_ref,
    }
    reference.update(_target_xpt_reference_fields(reference_spec, target_obs))
    return reference


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
