# Copyright @2025 ENN Energy(enn.cn)
# 复赛基础计算指标：电流、LCFS、X 点、打击点、X 点磁通偏差。

import numpy as np
from typing import List, Sequence, Union

M_TO_CM = 100.0
_EPS = 1e-12


def epsilon_Ip(
    Ip_actual: Union[np.ndarray, Sequence[float]],
    Ip_ref: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """逐步电流误差: |Ip(t_k) - Ip_ref(t_k)| / |Ip_ref(t_k)|"""
    Ip_actual = np.asarray(Ip_actual, dtype=float).flatten()
    Ip_ref = np.asarray(Ip_ref, dtype=float).flatten()
    n = min(len(Ip_actual), len(Ip_ref))
    if n == 0:
        return np.array([], dtype=float)
    ref_abs = np.abs(Ip_ref[:n])
    safe_ref = np.where(ref_abs > _EPS, ref_abs, 1.0)
    return np.abs(Ip_actual[:n] - Ip_ref[:n]) / safe_ref


def epsilon_LCFS(
    target_lcfs_points: np.ndarray,
    actual_lcfs_per_step: List[np.ndarray],
) -> np.ndarray:
    """复赛 LCFS 边界距离（cm）。

    与初赛不同，**不再平移对齐中心**。直接在装置坐标系下按对应极向角采样点
    逐点计算与目标 LCFS 的欧氏距离，再取 RMS。
    """
    target = np.asarray(target_lcfs_points, dtype=float)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_lcfs_points must be (N_b, 2)")
    N_b = target.shape[0]
    if N_b == 0:
        return np.array([], dtype=float)

    step_rms = []
    for contour in actual_lcfs_per_step:
        contour = np.asarray(contour, dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 1:
            # 无效轮廓 → 视为大误差（用 NaN 标记，后续会被零化）
            step_rms.append(np.inf)
            continue
        n_pts = min(N_b, len(contour))
        d_sq = (target[:n_pts, 0] - contour[:n_pts, 0]) ** 2 + \
               (target[:n_pts, 1] - contour[:n_pts, 1]) ** 2
        rms_m = np.sqrt(np.mean(d_sq))
        step_rms.append(rms_m * M_TO_CM)

    return np.array(step_rms, dtype=float) if step_rms else np.array([], dtype=float)


def _epsilon_points_rms(
    actual: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """逐步对 K 个点取 RMS 距离（cm）。

    actual: (N, K, 2)，ref: (K, 2) 单帧目标。返回长度 N 的数组。
    """
    actual = np.asarray(actual, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if actual.ndim != 3 or actual.shape[2] != 2:
        raise ValueError("actual must be (N, K, 2)")
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError("ref must be (K, 2)")
    N, K, _ = actual.shape
    if K != ref.shape[0]:
        # 容错：取较小的 K
        K = min(K, ref.shape[0])
        actual = actual[:, :K, :]
        ref = ref[:K, :]
    if N == 0 or K == 0:
        return np.array([], dtype=float)

    diff = actual - ref[None, :, :]   # (N, K, 2)
    d_sq = np.sum(diff * diff, axis=2)  # (N, K)
    rms_m = np.sqrt(np.mean(d_sq, axis=1))
    return rms_m * M_TO_CM


def epsilon_xpoints(actual: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """主 X 点（或次级 X 点）逐步距离 RMS（cm）。actual=(N,2,2), ref=(2,2)。"""
    return _epsilon_points_rms(actual, ref)


def epsilon_strikes(actual: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """8 个打击点逐步距离 RMS（cm）。actual=(N,8,2), ref=(8,2)。"""
    return _epsilon_points_rms(actual, ref)


def epsilon_psix(
    psiX: np.ndarray,
    psib: np.ndarray,
    psia: np.ndarray,
) -> np.ndarray:
    """X 点磁通偏差（归一化），逐步取 RMS。

    formula: sqrt( mean_m ((psiX^m - psib) / (psib - psia))**2 )
    psiX: (N, M)  M=2（上下）
    psib, psia: (N,)
    """
    psiX = np.asarray(psiX, dtype=float)
    psib = np.asarray(psib, dtype=float).flatten()
    psia = np.asarray(psia, dtype=float).flatten()
    if psiX.ndim == 1:
        psiX = psiX.reshape(-1, 1)
    n = min(len(psiX), len(psib), len(psia))
    if n == 0:
        return np.array([], dtype=float)
    psiX = psiX[:n]
    psib = psib[:n]
    psia = psia[:n]
    denom = psib - psia
    # 分母保护：若 |denom|<eps 视为 1（避免除零；此时输出基本为 inf，但实际不应发生）
    safe_denom = np.where(np.abs(denom) > _EPS, denom, 1.0)
    norm = (psiX - psib[:, None]) / safe_denom[:, None]  # (N, M)
    return np.sqrt(np.mean(norm * norm, axis=1))
