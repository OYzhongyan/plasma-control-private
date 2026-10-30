#
# Copyright @2025 ENN Energy(enn.cn)
#
# HFM Simulator: Gymnasium Env, dict observation, 12D action, optional reward_fn.
# Reference is trajectory-based: hold-initial-equilibrium or user-defined trajectory.
#

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from .hfm_predictor import HFMSocketPredictor
from .shot_registry import REFERENCE_KEYS

RAW_OBSERVATION_SPECS: dict[str, tuple[int, ...]] = {
    # 等离子体全局状态：Ip 为等离子体电流，R/Z 为位置或几何中心坐标。
    "Ip": (1,),
    "R": (1,),
    "Z": (1,),

    # 12 个极向场（PF, poloidal-field）线圈的电流。
    "I_PF": (12,),

    # 等离子体边界/形状的标量几何特征：径向范围、小半径、三角度和拉长率。
    "Rmax": (1,),
    "Rmin": (1,),
    "aminor": (1,),
    "deltal": (1,),
    "deltau": (1,),
    "kappa": (1,),
    "rc": (1,),
    "zc": (1,),

    # 磁通函数的参考值：FA 通常表示磁轴处值，FB 通常表示 LCFS/边界处值。
    # 具体归一化和单位由 HFM 接口定义。
    "FA": (1,),
    "FB": (1,),

    # 磁场/磁通网格相关数据。Bm 是 HFM 输出的 100 维磁场特征；
    # Fx 是二维磁通函数网格展平后的数组。
    "Bm": (100,),
    "nrx": (1,),  # R 方向网格点数
    "nzx": (1,),  # Z 方向网格点数
    "Fx": (4290,),  # 磁通函数网格展平值，通常对应 66*65 个网格点
    "rx": (66,),  # 磁通网格的 R 方向坐标
    "zx": (65,),  # 磁通网格的 Z 方向坐标

    # X-point 候选点：最多 6 个候选位置及其对应的磁通函数值。
    "rX": (6,),  # 候选 X-point 的 R 坐标
    "zX": (6,),  # 候选 X-point 的 Z 坐标
    "nX": (1,),  # 当前检测到的 X-point 数量
    "lX": (1,),  # 磁拓扑类别/拓扑标签

    # LCFS/等离子体边界的离散轮廓点坐标，共 32 个点。
    "rB": (32,),  # 边界点的 R 坐标
    "zB": (32,),  # 边界点的 Z 坐标

    # 6 个候选 X-point 位置对应的磁通函数值。
    "FX": (6,),
}

SCALAR_REFERENCE_KEYS = tuple(key for key in REFERENCE_KEYS if key != "lcfs_points")
LCFS_NUM_POINTS = 32
LCFS_POINT_DIM = 2
LCFS_SERIES_DIM = 3
XPT_REFERENCE_SPECS: dict[str, tuple[int, ...]] = {
    "rX": (4,),
    "zX": (4,),
    "x_valid": (4,),
    "strike_r": (8,),
    "strike_z": (8,),
    "strike_valid": (8,),
    "nX": (1,),
    "n_strike": (1,),
}


def _default_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.array(
        [-1499, -230, -172, -172, -348, -348, -270, -270, -348, -348, -270, -270],
        dtype=np.float32,
    )
    high = np.array(
        [1499, 230, 172, 172, 348, 348, 270, 270, 348, 348, 270, 270],
        dtype=np.float32,
    )
    return low, high


def _box(shape: tuple[int, ...]) -> gym.spaces.Box:
    return gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float64)


def _zeros(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _coerce_raw_value(key: str, value: Any) -> np.ndarray:
    shape = RAW_OBSERVATION_SPECS[key]
    if value is None:
        return _zeros(shape)

    arr = np.asarray(value, dtype=np.float64)
    expected_size = int(np.prod(shape))
    if arr.size != expected_size:
        raise ValueError(
            f"Raw observation key {key} expects shape {shape} (size={expected_size}), got {arr.shape}"
        )
    return arr.reshape(shape)


def _resample_curve(values: np.ndarray, num_points: int = LCFS_NUM_POINTS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == num_points:
        return values
    if values.size == 0:
        return np.zeros((num_points,), dtype=np.float64)
    if values.size == 1:
        return np.full((num_points,), float(values[0]), dtype=np.float64)

    src_x = np.linspace(0.0, 1.0, values.size)
    dst_x = np.linspace(0.0, 1.0, num_points)
    return np.interp(dst_x, src_x, values)


def _empty_lcfs_points() -> np.ndarray:
    return np.zeros((LCFS_NUM_POINTS, LCFS_POINT_DIM), dtype=np.float64)


def _extract_lcfs_points(raw: dict[str, Any]) -> np.ndarray:
    r_boundary = _resample_curve(_coerce_raw_value("rB", raw.get("rB")), LCFS_NUM_POINTS)
    z_boundary = _resample_curve(_coerce_raw_value("zB", raw.get("zB")), LCFS_NUM_POINTS)
    return np.stack([r_boundary, z_boundary], axis=-1)


def _build_observation_space() -> gym.Space:
    spaces: dict[str, gym.Space] = {key: _box(shape) for key, shape in RAW_OBSERVATION_SPECS.items()}
    spaces["lcfs_points"] = _box((LCFS_NUM_POINTS, LCFS_POINT_DIM))
    for key in SCALAR_REFERENCE_KEYS:
        spaces[f"reference_{key}"] = _box((1,))
    spaces["reference_lcfs_points"] = _box((LCFS_NUM_POINTS, LCFS_POINT_DIM))
    for key, shape in XPT_REFERENCE_SPECS.items():
        spaces[f"reference_{key}"] = _box(shape)
    spaces["failure"] = gym.spaces.MultiBinary(1)
    return gym.spaces.Dict(spaces)


def _coerce_scalar_series(values: Any, max_steps: int, default: float) -> np.ndarray:
    if values is None:
        arr = np.full((max_steps,), default, dtype=np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full((max_steps,), float(arr[0]), dtype=np.float64)
        elif arr.size != max_steps:
            raise ValueError(f"Reference length must equal max_steps={max_steps}, got {arr.size}")
    return arr


def _normalize_lcfs_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != LCFS_POINT_DIM or arr.shape[1] != LCFS_POINT_DIM:
        raise ValueError(f"LCFS points must have shape (N, {LCFS_POINT_DIM}), got {arr.shape}")
    if arr.shape[0] == LCFS_NUM_POINTS:
        return arr

    r_boundary = _resample_curve(arr[:, 0], LCFS_NUM_POINTS)
    z_boundary = _resample_curve(arr[:, 1], LCFS_NUM_POINTS)
    return np.stack([r_boundary, z_boundary], axis=-1)


def _coerce_lcfs_series(
    values: Any,
    max_steps: int,
    default: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=np.float64).copy()

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == LCFS_POINT_DIM:
        arr = np.repeat(_normalize_lcfs_points(arr)[None, ...], max_steps, axis=0)
    elif arr.ndim == LCFS_SERIES_DIM and arr.shape[0] == max_steps and arr.shape[2] == LCFS_POINT_DIM:
        if arr.shape[1] != LCFS_NUM_POINTS:
            arr = np.stack(
                [_normalize_lcfs_points(step_points) for step_points in arr],
                axis=0,
            )
    else:
        raise ValueError(
            f"reference_lcfs_points must have shape ({max_steps}, N, {LCFS_POINT_DIM}) "
            f"or (N, {LCFS_POINT_DIM}), got {arr.shape}"
        )

    if arr.shape != (max_steps, LCFS_NUM_POINTS, LCFS_POINT_DIM):
        raise ValueError(
            f"reference_lcfs_points must have shape ({max_steps}, {LCFS_NUM_POINTS}, {LCFS_POINT_DIM}), "
            f"got {arr.shape}"
        )
    return arr


def _coerce_vector_series(
    values: Any,
    max_steps: int,
    shape: tuple[int, ...],
    default: np.ndarray | None = None,
) -> np.ndarray:
    base = np.zeros(shape, dtype=np.float64) if default is None else np.asarray(default, dtype=np.float64)
    if base.shape != shape:
        base = np.zeros(shape, dtype=np.float64)
    if values is None:
        return np.repeat(base[None, ...], max_steps, axis=0)

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == shape:
        return np.repeat(arr[None, ...], max_steps, axis=0)
    if arr.shape == (max_steps,) + shape:
        return arr
    if shape == (1,) and arr.size == 1:
        return np.full((max_steps, 1), float(arr.reshape(-1)[0]), dtype=np.float64)
    raise ValueError(
        f"reference vector must have shape {shape} or ({max_steps}, {', '.join(map(str, shape))}), "
        f"got {arr.shape}"
    )


def _build_hold_reference(raw: dict[str, Any], max_steps: int) -> dict[str, np.ndarray]:
    lcfs_points = _extract_lcfs_points(raw)
    reference = {
        key: np.full(
            (max_steps,),
            float(_coerce_raw_value(key, raw.get(key)).reshape(-1)[0]),
            dtype=np.float64,
        )
        for key in SCALAR_REFERENCE_KEYS
    }
    reference["lcfs_points"] = np.repeat(lcfs_points[None, ...], max_steps, axis=0)
    for key, shape in XPT_REFERENCE_SPECS.items():
        reference[key] = np.zeros((max_steps,) + shape, dtype=np.float64)
    return reference


def _build_reference_trajectory(
    reference_mode: str,
    reference_spec: dict[str, Any] | None,
    raw: dict[str, Any],
    max_steps: int,
) -> dict[str, np.ndarray]:
    hold_ref = _build_hold_reference(raw, max_steps)
    if reference_mode == "hold":
        return hold_ref

    if reference_mode != "trajectory":
        raise ValueError(f"Unsupported reference_mode: {reference_mode}")

    reference_spec = reference_spec or {}
    reference = {
        key: _coerce_scalar_series(reference_spec.get(key), max_steps, float(hold_ref[key][0]))
        for key in SCALAR_REFERENCE_KEYS
    }
    reference["lcfs_points"] = _coerce_lcfs_series(
        reference_spec.get("lcfs_points"),
        max_steps,
        hold_ref["lcfs_points"],
    )
    for key, shape in XPT_REFERENCE_SPECS.items():
        reference[key] = _coerce_vector_series(
            reference_spec.get(key),
            max_steps,
            shape,
            hold_ref[key][0],
        )
    return reference


def _obs_dict_from_raw(
    raw: dict[str, Any],
    reference: dict[str, np.ndarray],
    reference_index: int,
) -> dict[str, Any]:
    out = {key: _coerce_raw_value(key, raw.get(key)) for key in RAW_OBSERVATION_SPECS}
    out["lcfs_points"] = _extract_lcfs_points(raw)

    ref_idx = int(min(max(reference_index, 0), len(reference["Ip"]) - 1))
    for key in SCALAR_REFERENCE_KEYS:
        out[f"reference_{key}"] = np.array([reference[key][ref_idx]], dtype=np.float64)
    out["reference_lcfs_points"] = np.asarray(reference["lcfs_points"][ref_idx], dtype=np.float64)
    for key in XPT_REFERENCE_SPECS:
        out[f"reference_{key}"] = np.asarray(reference[key][ref_idx], dtype=np.float64)
    out["failure"] = np.array([1 if raw.get("failure", False) else 0], dtype=np.uint8)
    return out


class HFMSimulator(gym.Env):
    """Gymnasium environment for HFM."""

    def __init__(
        self,
        config: dict[str, Any],
        reward_fn: Callable[[dict, np.ndarray], float] | None = None,
    ):
        super().__init__()
        self.config = config
        self.reward_fn = reward_fn
        predictor_config = config.get("predictor", config)
        self.engine = HFMSocketPredictor(config=predictor_config)

        self.max_steps = int(config.get("max_steps", 100))
        self.step_count = 0
        self.shot_id = predictor_config.get("shot_id", "13844")

        reference_cfg = config.get("reference", {})
        self.reference_mode_default = reference_cfg.get("mode", "hold")
        self.reference: dict[str, np.ndarray] | None = None

        action_low, action_high = _default_action_bounds()
        cfg_low = config.get("action_low")
        cfg_high = config.get("action_high")
        if cfg_low is not None:
            action_low = np.asarray(cfg_low, dtype=np.float32)
        if cfg_high is not None:
            action_high = np.asarray(cfg_high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(12,), dtype=np.float32
        )
        self.observation_space = _build_observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        reset_params = options.get("reset_params", {})
        reference_mode = options.get("reference_mode", self.reference_mode_default)
        reference_spec = options.get("reference")

        signeo = reset_params.get("signeo")
        bp = reset_params.get("bp")
        q0 = reset_params.get("q0")

        raw = self.engine.reset(signeo=signeo, bp=bp, q0=q0)
        self.step_count = 0
        self.reference = _build_reference_trajectory(
            reference_mode=reference_mode,
            reference_spec=reference_spec,
            raw=raw,
            max_steps=self.max_steps,
        )

        obs = _obs_dict_from_raw(
            raw=raw,
            reference=self.reference,
            reference_index=self.step_count,
        )
        info = {
            "reset_params": reset_params,
            "shot_id": self.shot_id,
            "reference_mode": reference_mode,
            "reference_keys": list(REFERENCE_KEYS) + list(XPT_REFERENCE_SPECS),
            "reference_length": self.max_steps,
            "reference_trajectory": self.reference,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,):
            action = np.broadcast_to(action, (12,)).copy()

        raw = self.engine.step(action)
        self.step_count += 1

        if self.reference is None:
            raise RuntimeError("Environment must be reset before calling step().")

        obs = _obs_dict_from_raw(
            raw=raw,
            reference=self.reference,
            reference_index=self.step_count,
        )
        terminated = bool(raw.get("failure", False))
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "failure": terminated,
            "shot_id": self.shot_id,
            "reference_keys": list(REFERENCE_KEYS) + list(XPT_REFERENCE_SPECS),
            "reference_index": min(self.step_count, self.max_steps - 1),
            "reference_length": self.max_steps,
            "raw_observation": raw,
        }

        reward = 0.0
        if self.reward_fn is not None:
            reward, terminated = self.reward_fn(
                obs,
                action,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.engine.close()
