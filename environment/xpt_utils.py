# -*- coding: utf-8 -*-
"""
XPT（先进偏滤器）观测提取：从 HFM ``obs`` dict 得到固定语义的 X 点槽位与打击点坐标。

极向场与磁通关系（与 FusionControl / RZIP 约定一致）::

    Br = -(1/R) * ∂ψ/∂Z
    Bz =  (1/R) * ∂ψ/∂R

理想 X 点处 **Br≈0, Bz≈0**，等价于 ∂ψ/∂R≈0 且 ∂ψ/∂Z≈0（在对应 R 上）。
因此向选手提供每槽的 ``dpsi_dr``、``dpsi_dz``（在匹配后的 X 点坐标上插值）作为磁通梯度特征。

打击点（与 FusionControl ``hfm_control_xpt_adjust/xpt_utils`` 一致）：
- **CCW**：在计算域矩形边界上，从 ``(rmax, z_ref)`` 起沿边界逆时针排序（右→上→左→下），
  保证双零/XPT 下 8 个打击点槽位语义稳定。
- **z_exclude_half_width=0.5**：剔除 ``|z| < 0.5`` m 的点，避免中平面贴壁伪交点进入偏滤器腿控点。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np

_N_RX = 66
_N_ZX = 65
_FX_SIZE = _N_RX * _N_ZX
_FLUX_DIFF_SCALE = 1.0
_STRIKE_DEDUP_RTOL = 1e-5
_DEFAULT_STRIKE_SLOTS = 8
_DEFAULT_Z_EXCLUDE = 0.5
# 槽位语义（z 降序）：X1/X2 为主级，X0/X3 为次级
X_PRIMARY_SLOTS = (1, 2)
X_SECONDARY_SLOTS = (0, 3)
# pack_to_vector 字段顺序（固定，勿改）
PACK_VECTOR_LAYOUT = (
    "nX",
    "fb",
    "x_r[4]",
    "x_z[4]",
    "x_fx[4]",
    "x_flux_diff[4]",
    "x_dpsi_dr[4]",
    "x_dpsi_dz[4]",
    "x_valid[4]",
    "strike_r[8]",
    "strike_z[8]",
    "strike_valid[8]",
    "strike_n_use",
    "strike_n_actual",
)


class XptObservationPack(TypedDict):
    """``extract_xpt_observation_pack`` 返回结构。"""

    nX: int
    fb: float
    fx_order: str
    x_r: np.ndarray
    x_z: np.ndarray
    x_fx: np.ndarray
    x_flux_diff: np.ndarray
    x_dpsi_dr: np.ndarray
    x_dpsi_dz: np.ndarray
    x_valid: np.ndarray
    strike_r: np.ndarray
    strike_z: np.ndarray
    strike_valid: np.ndarray
    strike_n_use: int
    strike_n_actual: int


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return float(default)
    return float(arr.flat[0])


def _greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RL 热路径：小矩阵贪心匹配（4/8 槽），避免 scipy 导入开销。"""
    n_rows, n_cols = cost.shape
    used_cols: set[int] = set()
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n_rows):
        best_j: int | None = None
        best = float("inf")
        for j in range(n_cols):
            if j in used_cols:
                continue
            c = cost[i, j]
            if c < best:
                best = c
                best_j = j
        if best_j is not None and best < 1e11:
            used_cols.add(best_j)
            rows.append(i)
            cols.append(best_j)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """小矩阵最小代价一一匹配

    行数多于列数时，匹配所有列到不同的行；否则匹配所有行到不同的列。
    这样 ``nX<slots`` 时由全局距离决定槽位，避免前序槽位贪心抢点。
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if max(n_rows, n_cols) > 16:
        return _greedy_assignment(cost)
    if n_rows > n_cols:
        col_rows, row_cols = _linear_sum_assignment(cost.T)
        return row_cols, col_rows

    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for i in range(n_rows):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, cols) in states.items():
            for j in range(n_cols):
                bit = 1 << j
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_total = total + float(cost[i, j])
                prev = next_states.get(new_mask)
                if prev is None or new_total < prev[0]:
                    next_states[new_mask] = (new_total, cols + (j,))
        states = next_states

    _, best_cols = min(states.values(), key=lambda item: item[0])
    return np.arange(n_rows, dtype=int), np.asarray(best_cols, dtype=int)


def _match_to_reference_slots(
    reference_r: np.ndarray | list[float],
    reference_z: np.ndarray | list[float],
    candidate_r: np.ndarray,
    candidate_z: np.ndarray,
    *,
    slots: int,
    direct_if_equal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    将候选点填入 ``slots`` 个 reference 槽位。

    - 候选个数 == ``slots`` 且 ``direct_if_equal``：按候选已有顺序直接填入（X 为 z 降序，打击点为 CCW）。
    - 否则：对 (reference 槽 × 候选) 做**匈牙利全局最优**匹配，使总距离平方和最小，且每个候选最多匹配一次。
    """
    ref_r = np.asarray(reference_r, dtype=np.float64).ravel()[:slots]
    ref_z = np.asarray(reference_z, dtype=np.float64).ravel()[:slots]
    cand_r = np.asarray(candidate_r, dtype=np.float64).ravel()
    cand_z = np.asarray(candidate_z, dtype=np.float64).ravel()
    n_cand = min(cand_r.size, cand_z.size)

    r_pad = np.zeros(slots, dtype=np.float64)
    z_pad = np.zeros(slots, dtype=np.float64)
    valid = np.zeros(slots, dtype=np.float64)
    if n_cand <= 0:
        return r_pad, z_pad, valid, 0

    if direct_if_equal and n_cand == slots:
        r_pad[:] = cand_r[:slots]
        z_pad[:] = cand_z[:slots]
        valid[:] = 1.0
        return r_pad, z_pad, valid, n_cand

    cost = np.full((slots, n_cand), 1e12, dtype=np.float64)
    for i in range(slots):
        for j in range(n_cand):
            cost[i, j] = (ref_r[i] - cand_r[j]) ** 2 + (ref_z[i] - cand_z[j]) ** 2
    row_ind, col_ind = _linear_sum_assignment(cost)
    for i, j in zip(row_ind, col_ind):
        if 0 <= j < n_cand and cost[i, j] < 1e11:
            r_pad[i] = cand_r[j]
            z_pad[i] = cand_z[j]
            valid[i] = 1.0
    return r_pad, z_pad, valid, n_cand


def extract_nx(obs: dict[str, Any]) -> int:
    """当前步 HFM 报告的有效 X 点个数 ``nX``。"""
    if not isinstance(obs, dict):
        return 0
    r_all = np.asarray(obs.get("rX", []), dtype=np.float64).ravel()
    z_all = np.asarray(obs.get("zX", []), dtype=np.float64).ravel()
    fx_all = np.asarray(obs.get("FX", []), dtype=np.float64).ravel()
    nx_raw = int(_safe_scalar(obs.get("nX", 0), default=0.0))
    return min(max(nx_raw, 0), len(r_all), len(z_all), len(fx_all))


def _sorted_valid_xpoint_arrays(
    obs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """全部有效 X 点按 z 降序；不含槽位截断。"""
    empty = np.zeros(0, dtype=np.float64)
    if not isinstance(obs, dict):
        return empty, empty, empty, 0, 0.0
    r_all = np.asarray(obs.get("rX", []), dtype=np.float64).ravel()
    z_all = np.asarray(obs.get("zX", []), dtype=np.float64).ravel()
    fx_all = np.asarray(obs.get("FX", []), dtype=np.float64).ravel()
    fb = _safe_scalar(obs.get("FB", 0.0), default=0.0)
    nx = extract_nx(obs)
    if nx <= 0:
        return empty, empty, empty, 0, fb
    order = np.argsort(z_all[:nx])[::-1]
    return r_all[:nx][order], z_all[:nx][order], fx_all[:nx][order], nx, fb


def extract_target_xpoints(
    initial_obs: dict[str, Any],
    slots: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 **reset 初态** 提取 reference X 点（z 降序固定 4 槽）。

    复赛任务二：对 ``configs/env_default.yaml`` 中 ``shot_id``（如 ``13906_500``）
    执行一次 ``reset``，用本函数得到 ``reference_rX`` / ``reference_zX``。
    """
    target_rX, target_zX, _, valid, _, _ = assign_xpoints_to_slots(
        initial_obs,
        reference_rX=np.zeros(slots),
        reference_zX=np.zeros(slots),
        slots=slots,
        force_sort_only=True,
    )
    return target_rX, target_zX, valid


def assign_xpoints_to_slots(
    obs: dict[str, Any],
    reference_rX: np.ndarray | list[float],
    reference_zX: np.ndarray | list[float],
    slots: int = 4,
    *,
    force_sort_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    将当前步 X 点填入 ``slots`` 个固定槽位。

    - ``nX == 4``（或 ``force_sort_only=True``）：按 z 降序直接填入 4 槽。
    - ``nX != 4``：候选点先按 z 降序，再对每个 reference 槽位在**未占用**候选中取最近点
      （一一匹配）；不足 4 个有效点时 ``valid=0`` 且坐标为 0。

    返回 ``r_pad, z_pad, fx_pad, valid, nX, fb``，长度均为 ``slots``。
    """
    r_pad = np.zeros(slots, dtype=np.float64)
    z_pad = np.zeros(slots, dtype=np.float64)
    fx_pad = np.zeros(slots, dtype=np.float64)
    valid = np.zeros(slots, dtype=np.float64)

    ref_r = np.asarray(reference_rX, dtype=np.float64).ravel()[:slots]
    ref_z = np.asarray(reference_zX, dtype=np.float64).ravel()[:slots]
    if ref_r.size < slots or ref_z.size < slots:
        raise ValueError("reference_rX/reference_zX must have length >= slots")

    # 按z降序排序
    r_c, z_c, fx_c, nx, fb = _sorted_valid_xpoint_arrays(obs)
    if nx <= 0:
        return r_pad, z_pad, fx_pad, valid, 0, fb

    if force_sort_only or nx == slots:
        # 直接降序排列对应槽位
        n_fill = min(nx, slots)
        r_pad[:n_fill] = r_c[:n_fill]
        z_pad[:n_fill] = z_c[:n_fill]
        fx_pad[:n_fill] = fx_c[:n_fill]
        valid[:n_fill] = 1.0
        return r_pad, z_pad, fx_pad, valid, nx, fb

    # 槽位匹配
    r_m, z_m, v_m, _ = _match_to_reference_slots(
        ref_r, ref_z, r_c, z_c, slots=slots, direct_if_equal=False
    )
    r_pad[:] = r_m
    z_pad[:] = z_m
    valid[:] = v_m
    # 匹配fx磁通
    for i in range(slots):
        if valid[i] > 0.5:
            for j in range(nx):
                if abs(r_c[j] - r_pad[i]) < 1e-12 and abs(z_c[j] - z_pad[i]) < 1e-12:
                    fx_pad[i] = fx_c[j]
                    break
    return r_pad, z_pad, fx_pad, valid, nx, fb


def extract_sorted_xpoints(
    obs: dict[str, Any],
    slots: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """兼容旧接口：仅 z 降序取前 ``slots`` 个，**不做** reference 匹配。"""
    r_pad = np.zeros(slots, dtype=np.float64)
    z_pad = np.zeros(slots, dtype=np.float64)
    fx_pad = np.zeros(slots, dtype=np.float64)
    valid = np.zeros(slots, dtype=np.float64)
    r_c, z_c, fx_c, nx, fb = _sorted_valid_xpoint_arrays(obs)
    n_fill = min(nx, slots)
    if n_fill > 0:
        r_pad[:n_fill] = r_c[:n_fill]
        z_pad[:n_fill] = z_c[:n_fill]
        fx_pad[:n_fill] = fx_c[:n_fill]
        valid[:n_fill] = 1.0
    return r_pad, z_pad, fx_pad, valid, min(nx, slots), fb


def flux_abs_diff(f_x: np.ndarray, fb: float, scale: float = _FLUX_DIFF_SCALE) -> np.ndarray:
    """逐槽 ``|FX - FB| / scale``。"""
    return np.abs(np.asarray(f_x, dtype=np.float64) - np.float64(fb)) / np.float64(scale)


def extract_xpoint_flux(
    obs: dict[str, Any],
    reference_rX: np.ndarray | list[float],
    reference_zX: np.ndarray | list[float],
    slots: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """匹配槽位上的 ``FX`` 与 ``|FX-FB|``。"""
    _, _, fx_pad, valid, _, fb = assign_xpoints_to_slots(
        obs, reference_rX, reference_zX, slots=slots
    )
    diff = flux_abs_diff(fx_pad, fb) * valid
    return fx_pad, diff, fb


def reshape_fx_to_psi(
    Fx: np.ndarray,
    order: Literal["C", "F"] = "C",
) -> np.ndarray:
    fx = np.asarray(Fx, dtype=np.float64).ravel()
    if fx.size != _FX_SIZE:
        raise ValueError(f"Fx must have length {_FX_SIZE}, got {fx.size}")
    return fx.reshape(_N_RX, _N_ZX, order=order)


def interp_psi_bilinear(
    rx: np.ndarray,
    zx: np.ndarray,
    psi: np.ndarray,
    r: float,
    z: float,
) -> float:
    rx = np.asarray(rx, dtype=np.float64).ravel()
    zx = np.asarray(zx, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (rx.size, zx.size):
        raise ValueError(f"psi shape {psi.shape} != ({rx.size}, {zx.size})")

    r = float(np.clip(r, rx[0], rx[-1]))
    z = float(np.clip(z, zx[0], zx[-1]))
    ir = int(np.clip(int(np.searchsorted(rx, r) - 1), 0, rx.size - 2))
    iz = int(np.clip(int(np.searchsorted(zx, z) - 1), 0, zx.size - 2))

    r0, r1 = rx[ir], rx[ir + 1]
    z0, z1 = zx[iz], zx[iz + 1]
    q00, q10 = psi[ir, iz], psi[ir + 1, iz]
    q01, q11 = psi[ir, iz + 1], psi[ir + 1, iz + 1]
    tr = (r - r0) / (r1 - r0) if r1 > r0 else 0.0
    tz = (z - z0) / (z1 - z0) if z1 > z0 else 0.0
    q0 = q00 * (1.0 - tr) + q10 * tr
    q1 = q01 * (1.0 - tr) + q11 * tr
    return float(q0 * (1.0 - tz) + q1 * tz)


def infer_fx_reshape_order(obs: dict[str, Any]) -> Literal["C", "F"]:
    fx = np.asarray(obs.get("Fx", []), dtype=np.float64).ravel()
    if fx.size != _FX_SIZE:
        return "C"
    rx = np.asarray(obs.get("rx", []), dtype=np.float64).ravel()
    zx = np.asarray(obs.get("zx", []), dtype=np.float64).ravel()
    if rx.size != _N_RX or zx.size != _N_ZX:
        return "C"
    r_x = np.asarray(obs.get("rX", []), dtype=np.float64).ravel()
    z_x = np.asarray(obs.get("zX", []), dtype=np.float64).ravel()
    f_x = np.asarray(obs.get("FX", []), dtype=np.float64).ravel()
    n_x = extract_nx(obs)
    if n_x <= 0:
        return "C"

    best: Literal["C", "F"] = "C"
    best_err = float("inf")
    for order in ("C", "F"):
        psi = reshape_fx_to_psi(fx, order=order)
        err_sum = 0.0
        count = 0
        for k in range(n_x):
            try:
                pred = interp_psi_bilinear(rx, zx, psi, float(r_x[k]), float(z_x[k]))
            except Exception:
                err_sum = float("inf")
                break
            err_sum += abs(pred - f_x[k])
            count += 1
        if count and err_sum / count < best_err:
            best_err = err_sum / count
            best = order
    return best


def get_psi_grid(
    obs: dict[str, Any],
    order: Literal["C", "F"] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Literal["C", "F"]]:
    rx = np.asarray(obs.get("rx", []), dtype=np.float64).ravel()
    zx = np.asarray(obs.get("zx", []), dtype=np.float64).ravel()
    fx = np.asarray(obs.get("Fx", []), dtype=np.float64).ravel()
    if rx.size != _N_RX or zx.size != _N_ZX or fx.size != _FX_SIZE:
        raise ValueError("obs must contain valid rx (66), zx (65), Fx (4290)")
    ord_use = infer_fx_reshape_order(obs) if order is None else order
    return rx, zx, reshape_fx_to_psi(fx, order=ord_use), ord_use


def gradient_psi_on_grid(
    rx: np.ndarray,
    zx: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g = np.gradient(np.asarray(psi, dtype=np.float64), np.asarray(rx, dtype=np.float64), np.asarray(zx, dtype=np.float64))
    return g[0], g[1]


def extract_xpoint_psi_gradient(
    obs: dict[str, Any],
    reference_rX: np.ndarray | list[float],
    reference_zX: np.ndarray | list[float],
    slots: int = 4,
    order: Literal["C", "F"] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在匹配后的 X 点槽位上插值 ``∂ψ/∂R``、``∂ψ/∂Z``；无效槽为 0。
    """
    r_pad, z_pad, _, valid, _, _ = assign_xpoints_to_slots(
        obs, reference_rX, reference_zX, slots=slots
    )
    dpsi_dr = np.zeros(slots, dtype=np.float64)
    dpsi_dz = np.zeros(slots, dtype=np.float64)
    if not np.any(valid > 0.5):
        return dpsi_dr, dpsi_dz

    rx, zx, psi, _ = get_psi_grid(obs, order=order)
    gdr, gdz = gradient_psi_on_grid(rx, zx, psi)
    for s in range(slots):
        if valid[s] < 0.5:
            continue
        dpsi_dr[s] = interp_psi_bilinear(rx, zx, gdr, float(r_pad[s]), float(z_pad[s]))
        dpsi_dz[s] = interp_psi_bilinear(rx, zx, gdz, float(r_pad[s]), float(z_pad[s]))
    return dpsi_dr, dpsi_dz


# ---------------------------------------------------------------------------
# 打击点：域边界扫描 ψ=FB（纯 numpy，RL 每步友好；无 matplotlib）
# ---------------------------------------------------------------------------


def _normalize_rx_zx_fx_to_2d(rx: Any, zx: Any, Fx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rx = np.asarray(rx)
    zx = np.asarray(zx)
    Fx = np.asarray(Fx)
    if rx.ndim == 1 and zx.ndim == 1:
        rx_2d, zx_2d = np.meshgrid(rx, zx, indexing="ij")
    else:
        rx_2d, zx_2d = rx, zx
    if Fx.ndim == 1:
        Fx = Fx.reshape(rx_2d.shape)
    rx1 = np.unique(rx_2d[:, 0])
    zx1 = np.unique(zx_2d[0, :])
    return rx_2d, zx_2d, Fx, np.asarray(rx1), np.asarray(zx1)


def _scalar_fb(FB: Any) -> float:
    if isinstance(FB, np.ndarray):
        FB = FB[~np.isnan(FB)][0] if np.any(~np.isnan(FB)) else float(FB.flat[0])
    return float(FB)


def _crossings_1d(coords: np.ndarray, values: np.ndarray, level: float) -> list[float]:
    """一维序列相对 ``level`` 的线性插值过零点坐标。"""
    x = np.asarray(coords, dtype=np.float64).ravel()
    f = np.asarray(values, dtype=np.float64).ravel() - float(level)
    if x.size != f.size or x.size < 2:
        return []
    out: list[float] = []
    for i in range(f.size - 1):
        f0, f1 = f[i], f[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if abs(f0) < 1e-14:
            out.append(float(x[i]))
        if f0 * f1 < 0.0:
            t = -f0 / (f1 - f0)
            out.append(float(x[i] + t * (x[i + 1] - x[i])))
    if abs(f[-1]) < 1e-14:
        out.append(float(x[-1]))
    return out


def extract_strike_points_edges_fast(
    rx: np.ndarray,
    zx: np.ndarray,
    psi: np.ndarray,
    fb: float,
) -> dict[str, np.ndarray]:
    """LCFS (ψ=FB) 与计算域四边交点：沿边界扫描过零，无 matplotlib。"""
    rx = np.asarray(rx, dtype=np.float64).ravel()
    zx = np.asarray(zx, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64)
    fb = float(fb)
    rmin, rmax = float(rx[0]), float(rx[-1])
    zmin, zmax = float(zx[0]), float(zx[-1])
    span = max(rmax - rmin, zmax - zmin, 1e-12)

    top = [np.array([r, zmax], dtype=np.float64) for r in _crossings_1d(rx, psi[:, -1], fb)]
    bottom = [np.array([r, zmin], dtype=np.float64) for r in _crossings_1d(rx, psi[:, 0], fb)]
    left = [np.array([rmin, z], dtype=np.float64) for z in _crossings_1d(zx, psi[0, :], fb)]
    right = [np.array([rmax, z], dtype=np.float64) for z in _crossings_1d(zx, psi[-1, :], fb)]
    return {
        "top": _dedup_points(top, span),
        "bottom": _dedup_points(bottom, span),
        "left": _dedup_points(left, span),
        "right": _dedup_points(right, span),
    }


def _dedup_points(points: list[np.ndarray], span: float) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=float)
    tol = max(float(span) * _STRIKE_DEDUP_RTOL, 1e-12)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol:
            keep.append(i)
    return pts[keep]


def extract_strike_points_edges(
    rx: Any,
    zx: Any,
    Fx: Any,
    FB: Any,
    *,
    order: Literal["C", "F"] = "C",
) -> dict[str, np.ndarray]:
    """LCFS 与域边界交点（默认快速边界扫描）。"""
    rx1 = np.asarray(rx, dtype=np.float64).ravel()
    zx1 = np.asarray(zx, dtype=np.float64).ravel()
    psi = reshape_fx_to_psi(Fx, order=order)
    return extract_strike_points_edges_fast(rx1, zx1, psi, _scalar_fb(FB))


def merge_strike_dict(strikes: dict[str, np.ndarray]) -> np.ndarray:
    chunks = []
    for key in ("top", "bottom", "left", "right"):
        pts = strikes.get(key)
        if pts is not None and len(pts) > 0:
            chunks.append(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
    return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)


def filter_equatorial_band(points: np.ndarray, z_exclude_half_width: float) -> np.ndarray:
    """剔除 |z| < z_exclude_half_width（中平面贴壁伪打击点）。"""
    if points.size == 0:
        return points.reshape(0, 2)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = float(z_exclude_half_width)
    if h <= 0:
        return pts
    return pts[np.abs(pts[:, 1]) >= h - 1e-15]


def _boundary_ccw_s(
    r: float,
    z: float,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    z_ref: float,
    tol: float,
) -> float | None:
    dr = rmax - rmin
    dz = zmax - zmin
    if dr + dz < 1e-15:
        return None
    z0 = float(np.clip(z_ref, zmin, zmax))

    def on_seg(a: float, b: float, x: float) -> bool:
        lo, hi = (a, b) if a <= b else (b, a)
        return lo - tol <= x <= hi + tol

    if abs(r - rmax) <= tol and on_seg(zmin, zmax, z):
        return (z - z0) if z >= z0 - tol else (zmax - z0) + dr + dz + dr + (z - zmin)
    if abs(z - zmax) <= tol and on_seg(rmin, rmax, r):
        return (zmax - z0) + (rmax - r)
    if abs(r - rmin) <= tol and on_seg(zmin, zmax, z):
        return (zmax - z0) + dr + (zmax - z)
    if abs(z - zmin) <= tol and on_seg(rmin, rmax, r):
        return (zmax - z0) + dr + dz + (r - rmin)
    return None


def sort_strikes_ccw(
    points: np.ndarray,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    z_ref: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    span = max(rmax - rmin, zmax - zmin, 1e-12)
    tol = max(span * 1e-5, 1e-9)
    s_list, ok = [], []
    for i in range(len(pts)):
        s = _boundary_ccw_s(pts[i, 0], pts[i, 1], rmin, rmax, zmin, zmax, z_ref, tol)
        s_list.append(0.0 if s is None else s)
        ok.append(s is not None)
    if np.all(ok):
        return pts[np.argsort(s_list)]
    rc, zc = 0.5 * (rmin + rmax), 0.5 * (zmin + zmax)
    ang = np.arctan2(pts[:, 1] - zc, pts[:, 0] - rc)
    theta_ref = np.arctan2(float(np.clip(z_ref, zmin, zmax)) - zc, rmax - rc)
    d = (ang - theta_ref + np.pi) % (2 * np.pi) - np.pi
    return pts[np.argsort(d)]


def extract_strike_points_ordered(
    rx: Any,
    zx: Any,
    Fx: Any,
    FB: Any,
    *,
    z_exclude_half_width: float = _DEFAULT_Z_EXCLUDE,
    z_ref: float = 0.0,
    order: Literal["C", "F"] = "C",
) -> dict[str, Any]:
    rx1 = np.asarray(rx, dtype=np.float64).ravel()
    zx1 = np.asarray(zx, dtype=np.float64).ravel()
    psi = reshape_fx_to_psi(Fx, order=order)
    rmin, rmax = float(rx1.min()), float(rx1.max())
    zmin, zmax = float(zx1.min()), float(zx1.max())
    merged = filter_equatorial_band(
        merge_strike_dict(extract_strike_points_edges_fast(rx1, zx1, psi, _scalar_fb(FB))),
        z_exclude_half_width,
    )
    span = max(rmax - rmin, zmax - zmin, 1e-12)
    tol = max(span * 1e-5, 1e-9)
    dedup: list[np.ndarray] = []
    for p in merged:
        if not dedup or min(np.linalg.norm(p - q) for q in dedup) > tol:
            dedup.append(p)
    pts_u = np.asarray(dedup, dtype=np.float64).reshape(-1, 2) if dedup else np.empty((0, 2), dtype=np.float64)
    ordered = sort_strikes_ccw(pts_u, rmin, rmax, zmin, zmax, z_ref=z_ref)
    n = int(ordered.shape[0])
    return {
        "n_strike": n,
        "r": ordered[:, 0].copy() if n else np.zeros(0, dtype=np.float64),
        "z": ordered[:, 1].copy() if n else np.zeros(0, dtype=np.float64),
        "points_ccw": ordered.copy(),
        "rmin": rmin,
        "rmax": rmax,
        "zmin": zmin,
        "zmax": zmax,
    }


def _strike_candidates_from_obs(
    obs: dict[str, Any],
    *,
    z_exclude_half_width: float,
    z_ref: float,
    order: Literal["C", "F"] = "C",
) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(obs, dict):
        return np.zeros(0), np.zeros(0), 0
    rx, zx, fx, fb = obs.get("rx"), obs.get("zx"), obs.get("Fx"), obs.get("FB", 0.0)
    if rx is None or zx is None or fx is None:
        return np.zeros(0), np.zeros(0), 0
    out = extract_strike_points_ordered(
        rx,
        zx,
        fx,
        fb,
        z_exclude_half_width=z_exclude_half_width,
        z_ref=z_ref,
        order=order,
    )
    r_s = np.asarray(out["r"], dtype=np.float64).ravel()
    z_s = np.asarray(out["z"], dtype=np.float64).ravel()
    return r_s, z_s, int(r_s.size)


def extract_target_strike_points(
    initial_obs: dict[str, Any],
    *,
    strike_slots: int = _DEFAULT_STRIKE_SLOTS,
    z_exclude_half_width: float = _DEFAULT_Z_EXCLUDE,
    z_ref: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    从 reset 初态（如 ``13906_500``）提取 reference 打击点 8 槽（CCW）。

    初态候选数等于 8 时直接采用 CCW 序；多于 8 取 CCW 前 8 个作为 reference 模板。
    """
    r_c, z_c, n = _strike_candidates_from_obs(
        initial_obs, z_exclude_half_width=z_exclude_half_width, z_ref=z_ref
    )
    ref_r = np.zeros(strike_slots, dtype=np.float64)
    ref_z = np.zeros(strike_slots, dtype=np.float64)
    valid = np.zeros(strike_slots, dtype=np.float64)
    n_fill = min(n, strike_slots)
    if n_fill > 0:
        ref_r[:n_fill] = r_c[:n_fill]
        ref_z[:n_fill] = z_c[:n_fill]
        valid[:n_fill] = 1.0
    return ref_r, ref_z, valid, int(np.sum(valid > 0.5))


def extract_strike_slots(
    obs: dict[str, Any],
    reference_rS: np.ndarray | list[float],
    reference_zS: np.ndarray | list[float],
    *,
    strike_slots: int = _DEFAULT_STRIKE_SLOTS,
    z_exclude_half_width: float = _DEFAULT_Z_EXCLUDE,
    z_ref: float = 0.0,
    order: Literal["C", "F"] = "C",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    打击点填入 8 个 reference 槽位。

    - 候选数 == 8：直接 CCW 顺序填入。
    - 否则：对 reference 8 槽做匈牙利匹配（与 X 点 ``nX!=4`` 同理）。
    """
    ref_r = np.asarray(reference_rS, dtype=np.float64).ravel()[:strike_slots]
    ref_z = np.asarray(reference_zS, dtype=np.float64).ravel()[:strike_slots]
    if ref_r.size < strike_slots or ref_z.size < strike_slots:
        raise ValueError("reference_rS/reference_zS must have length >= strike_slots")

    r_c, z_c, n_actual = _strike_candidates_from_obs(
        obs, z_exclude_half_width=z_exclude_half_width, z_ref=z_ref, order=order
    )
    r_pad, z_pad, valid, _ = _match_to_reference_slots(
        ref_r, ref_z, r_c, z_c, slots=strike_slots, direct_if_equal=True
    )
    n_use = int(np.sum(valid > 0.5))
    return r_pad, z_pad, valid, n_use, n_actual


def extract_xpt_observation_pack(
    obs: dict[str, Any],
    reference_rX: np.ndarray | list[float],
    reference_zX: np.ndarray | list[float],
    reference_rS: np.ndarray | list[float],
    reference_zS: np.ndarray | list[float],
    *,
    x_slots: int = 4,
    strike_slots: int = _DEFAULT_STRIKE_SLOTS,
    z_exclude_half_width: float = _DEFAULT_Z_EXCLUDE,
    z_ref: float = 0.0,
    fx_order: Literal["C", "F"] | None = "C",
) -> XptObservationPack:
    """
    一次性提取复赛 XPT 任务常用观测量（供选手预处理或拼进策略输入）。

    RL 训练建议 ``fx_order='C'``（13906 炮已验证），避免每步 ``infer_fx_reshape_order``。
    打击点用边界扫描，无 matplotlib，单步通常在毫秒级。

    Parameters
    ----------
    obs : 当前步 HFM 观测 dict（经 ``HFMSocketPredictor._parse_observation``）
    reference_rX, reference_zX : reset 初态 ``extract_target_xpoints``
    reference_rS, reference_zS : reset 初态 ``extract_target_strike_points``（8 槽 CCW）
    """
    x_r, x_z, x_fx, x_valid, nX, fb = assign_xpoints_to_slots(
        obs, reference_rX, reference_zX, slots=x_slots
    )
    x_flux_diff = flux_abs_diff(x_fx, fb) * x_valid
    ord_use: Literal["C", "F"]
    if fx_order is None:
        ord_use = infer_fx_reshape_order(obs)
    else:
        ord_use = fx_order

    x_dpsi_dr = np.zeros(x_slots, dtype=np.float64)
    x_dpsi_dz = np.zeros(x_slots, dtype=np.float64)
    if np.any(x_valid > 0.5):
        rx, zx, psi, _ = get_psi_grid(obs, order=ord_use)
        gdr, gdz = gradient_psi_on_grid(rx, zx, psi)
        for s in range(x_slots):
            if x_valid[s] < 0.5:
                continue
            x_dpsi_dr[s] = interp_psi_bilinear(rx, zx, gdr, float(x_r[s]), float(x_z[s]))
            x_dpsi_dz[s] = interp_psi_bilinear(rx, zx, gdz, float(x_r[s]), float(x_z[s]))

    strike_r, strike_z, strike_valid, strike_n_use, strike_n_actual = extract_strike_slots(
        obs,
        reference_rS,
        reference_zS,
        strike_slots=strike_slots,
        z_exclude_half_width=z_exclude_half_width,
        z_ref=z_ref,
        order=ord_use,
    )

    return XptObservationPack(
        nX=nX,
        fb=fb,
        fx_order=ord_use,
        x_r=x_r,
        x_z=x_z,
        x_fx=x_fx,
        x_flux_diff=x_flux_diff,
        x_dpsi_dr=x_dpsi_dr,
        x_dpsi_dz=x_dpsi_dz,
        x_valid=x_valid,
        strike_r=strike_r,
        strike_z=strike_z,
        strike_valid=strike_valid,
        strike_n_use=strike_n_use,
        strike_n_actual=strike_n_actual,
    )


def pack_to_vector(pack: XptObservationPack) -> np.ndarray:
    """将 pack 展平为一维向量；字段顺序见 ``PACK_VECTOR_LAYOUT``（总长 56，8 打击点槽）。"""
    return np.concatenate(
        [
            [float(pack["nX"]), pack["fb"]],
            pack["x_r"],
            pack["x_z"],
            pack["x_fx"],
            pack["x_flux_diff"],
            pack["x_dpsi_dr"],
            pack["x_dpsi_dz"],
            pack["x_valid"],
            pack["strike_r"],
            pack["strike_z"],
            pack["strike_valid"],
            [float(pack["strike_n_use"]), float(pack["strike_n_actual"])],
        ]
    ).astype(np.float64)


def _lcfs_polyline_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """HFM 最外闭合磁面 32 点（``lcfs_points`` 或 ``rB``/``zB``），按索引顺序闭合。"""
    lc = np.asarray(obs.get("lcfs_points", np.zeros((0, 2))), dtype=np.float64)
    if lc.ndim == 2 and lc.shape[0] >= 3 and lc.shape[1] >= 2:
        pts = lc[:, :2]
    else:
        rb = np.asarray(obs.get("rB", []), dtype=np.float64).ravel()
        zb = np.asarray(obs.get("zB", []), dtype=np.float64).ravel()
        n = min(rb.size, zb.size)
        if n < 3:
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.column_stack([rb[:n], zb[:n]])
    return np.vstack([pts, pts[0:1]])


def plot_flux_surfaces_from_obs(
    obs: dict[str, Any],
    *,
    FA: Any = None,
    n_levels: int = 40,
    ax: Any = None,
    fig: Any = None,
    clear: bool = True,
    separatrix_color: str = "#1e293b",
    separatrix_linewidth: float = 2.0,
    draw_lcfs_polyline: bool = False,
) -> tuple[Any, Any, tuple[Any, Any]]:
    """
    与 ``hfm_control_xpt_adjust/inference_runner_xpt.plot_flux_surfaces`` 一致：

    - 背景：``(Fx-FB)/(FA-FB)`` 灰色磁面等高线；
    - Separatrix / 最外闭合磁面：``contour(Fx, levels=[FB])``（含 X 点腿，与 ``flux_step_*.png`` 相同）；
    - 可选 ``draw_lcfs_polyline=True``：叠加 HFM 32 点 ``lcfs_points`` 折线（仅主闭合圈）。
    """
    import matplotlib.pyplot as plt

    rx = np.asarray(obs.get("rx", []), dtype=np.float64)
    zx = np.asarray(obs.get("zx", []), dtype=np.float64)
    fx = np.asarray(obs.get("Fx", []), dtype=np.float64)
    fb = _safe_scalar(obs.get("FB", 0.0))

    if rx.ndim == 1 and zx.ndim == 1:
        rx2d, zx2d = np.meshgrid(rx, zx, indexing="ij")
    elif rx.ndim == 2 and zx.ndim == 2:
        rx2d, zx2d = rx, zx
    else:
        raise ValueError("rx/zx must be both 1D or both 2D")

    if fx.ndim == 1:
        if fx.size == rx2d.size:
            fx = fx.reshape(rx2d.shape)
        else:
            fx = reshape_fx_to_psi(fx, order=infer_fx_reshape_order(obs))
    elif fx.shape != rx2d.shape:
        raise ValueError(f"Fx shape {fx.shape} != grid {rx2d.shape}")

    fa_val = _safe_scalar(obs.get("FA", fb) if FA is None else FA, default=fb)
    denom = fa_val - fb
    if abs(denom) > 1e-12:
        fx_n = (fx - fb) / denom
        levels = np.linspace(-3, 1, n_levels)
    else:
        fx_n = fx
        levels = n_levels

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    if clear:
        ax.cla()

    c_flux = ax.contour(rx2d, zx2d, fx_n, levels=levels, colors="gray", linewidths=0.5)
    c_lcfs = ax.contour(
        rx2d,
        zx2d,
        fx,
        levels=[fb],
        colors=separatrix_color,
        linewidths=separatrix_linewidth,
    )
    if draw_lcfs_polyline:
        lcfs_loop = _lcfs_polyline_from_obs(obs)
        if lcfs_loop.shape[0] >= 4:
            ax.plot(
                lcfs_loop[:, 0],
                lcfs_loop[:, 1],
                color="#b91c1c",
                linewidth=1.6,
                linestyle="--",
                alpha=0.85,
                zorder=5,
                label="LCFS 32-pt",
            )

    ax.set_xlabel("r", fontsize=16)
    ax.set_ylabel("z", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig, ax, (c_flux, c_lcfs)


def plot_xpt_flux_diagnostic(
    obs: dict[str, Any],
    *,
    reference_rX: np.ndarray | list[float] | None = None,
    reference_zX: np.ndarray | list[float] | None = None,
    reference_rS: np.ndarray | list[float] | None = None,
    reference_zS: np.ndarray | list[float] | None = None,
    save_path: str | None = None,
    title: str = "XPT 13906_500",
    show: bool = True,
) -> Any:
    """
    XPT 诊断图：与 inference_runner ``plot_flux_surfaces`` 同款磁面 + separatrix，叠加主/次级 X 与打击点。

    需 ``matplotlib``（``pip install -e '.[xpt]'``）。
    """
    import matplotlib.pyplot as plt

    if reference_rX is None or reference_zX is None:
        reference_rX, reference_zX, _ = extract_target_xpoints(obs, slots=4)
    if reference_rS is None or reference_zS is None:
        reference_rS, reference_zS, _, _ = extract_target_strike_points(obs)

    pack = extract_xpt_observation_pack(
        obs,
        reference_rX,
        reference_zX,
        reference_rS,
        reference_zS,
        fx_order="C",
    )

    fig, ax, _ = plot_flux_surfaces_from_obs(
        obs,
        FA=obs.get("FA"),
        clear=True,
        separatrix_color="#1e293b",
        separatrix_linewidth=2.0,
        draw_lcfs_polyline=False,
    )
    ax.set_title(title, fontsize=14, pad=8)

    for s in range(4):
        if pack["x_valid"][s] < 0.5:
            continue
        rr_x, zz_x = float(pack["x_r"][s]), float(pack["x_z"][s])
        is_pri = s in X_PRIMARY_SLOTS
        ax.scatter(
            rr_x,
            zz_x,
            s=200 if is_pri else 140,
            marker="*",
            c="#2563eb" if is_pri else "#ea580c",
            edgecolors="#1e3a8a" if is_pri else "#9a3412",
            linewidths=0.7,
            zorder=9,
            label="X primary" if s == X_PRIMARY_SLOTS[0] else ("X secondary" if s == X_SECONDARY_SLOTS[0] else None),
        )
        ax.text(rr_x + 0.015, zz_x + 0.04, f"X{s}", fontsize=9, color="#1e293b", zorder=10)

    rx1 = np.asarray(obs.get("rx", []), dtype=np.float64).ravel()
    zx1 = np.asarray(obs.get("zx", []), dtype=np.float64).ravel()
    fx1 = np.asarray(obs.get("Fx", []), dtype=np.float64).ravel()
    fb1 = pack["fb"]
    if rx1.size == _N_RX and zx1.size == _N_ZX and fx1.size == _FX_SIZE:
        psi = reshape_fx_to_psi(fx1, order="C")
        edges = extract_strike_points_edges_fast(rx1, zx1, psi, fb1)
        strike_style = [("top", "^"), ("bottom", "v"), ("left", "<"), ("right", ">")]
        for key, marker in strike_style:
            pts = edges.get(key)
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts, dtype=np.float64)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=90,
                c="#dc2626",
                marker=marker,
                edgecolors="#7f1d1d",
                linewidths=0.5,
                zorder=8,
                label="strike" if key == "top" else None,
            )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


__all__ = [
    "XptObservationPack",
    "PACK_VECTOR_LAYOUT",
    "X_PRIMARY_SLOTS",
    "X_SECONDARY_SLOTS",
    "extract_nx",
    "extract_target_xpoints",
    "extract_target_strike_points",
    "assign_xpoints_to_slots",
    "extract_sorted_xpoints",
    "extract_xpoint_flux",
    "extract_xpoint_psi_gradient",
    "extract_strike_slots",
    "extract_strike_points_ordered",
    "extract_xpt_observation_pack",
    "pack_to_vector",
    "flux_abs_diff",
    "reshape_fx_to_psi",
    "infer_fx_reshape_order",
    "interp_psi_bilinear",
    "get_psi_grid",
    "gradient_psi_on_grid",
    "filter_equatorial_band",
    "sort_strikes_ccw",
    "plot_flux_surfaces_from_obs",
    "plot_xpt_flux_diagnostic",
]
