# Copyright @2025 ENN Energy(enn.cn)
# 复赛位形惩罚、电流偏差熔断、线圈电流约束、X 点拓扑约束。

import numpy as np
from typing import Optional

from . import config


def _pad_or_trim(arr: np.ndarray, N: int, fill: float = 0.0) -> np.ndarray:
    """把 arr 拉成长度 N（不足末尾用 fill 填充，超长截断）。"""
    arr = np.asarray(arr).flatten()
    if len(arr) == N:
        return arr.astype(np.float64)
    out = np.full(N, fill, dtype=np.float64)
    n = min(len(arr), N)
    if n > 0:
        out[:n] = arr[:n]
    return out


def compute_eta(lX: np.ndarray, task_id: str, N: int, K_eff: int) -> np.ndarray:
    """位形类型惩罚系数。

    F1 (限制器目标)：lX==0 → 1，否则 0.5
    F2a/F2b (XPT 属于偏滤器目标)：lX==1 → 1，否则 0.5
    超出有效步数 (k >= K_eff) 部分不影响（任意值，因 sigma 已置 0），统一填 1。
    """
    eta = np.ones(N, dtype=np.float64)
    target_topo = config.TASK_TARGET_TOPOLOGY.get(task_id, "limiter")
    lX = _pad_or_trim(lX, N, fill=0.0)
    if K_eff <= 0:
        return eta
    valid = slice(0, K_eff)
    if target_topo == "limiter":
        good = lX[valid] == 0
    else:  # divertor (含 XPT)
        good = lX[valid] == 1
    eta_valid = np.where(good, 1.0, 0.5)
    eta[valid] = eta_valid
    return eta


def compute_mu(
    Ip_actual: np.ndarray,
    Ip_ref: np.ndarray,
    N: int,
    K_eff: int,
) -> np.ndarray:
    """电流偏差熔断系数。|ΔIp|<=50kA 时 1，否则 0。

    超出有效步数部分填 0（与提前终止统一）；不过 scoring 那侧 sigma 已 0，无影响。
    """
    mu = np.zeros(N, dtype=np.float64)
    if K_eff <= 0:
        return mu
    Ia = _pad_or_trim(Ip_actual, N, fill=0.0)
    Ir = _pad_or_trim(Ip_ref, N, fill=0.0)
    diff = np.abs(Ia[:K_eff] - Ir[:K_eff])
    mu[:K_eff] = (diff <= config.CURRENT_FUSE_A).astype(np.float64)
    return mu


def compute_rho(Icoil: np.ndarray, N: int) -> np.ndarray:
    """线圈电流约束系数：首次超限步及之后全部清零。

    Icoil: (M, 12) 时间序列；不足 N 末尾视为不超限继续 1（提前终止本身已通过 K_eff 处理）。
    """
    rho = np.ones(N, dtype=np.float64)
    if Icoil is None:
        return rho
    Ic = np.asarray(Icoil, dtype=float)
    if Ic.ndim != 2 or Ic.shape[1] == 0:
        return rho
    M = min(Ic.shape[0], N)
    if M == 0:
        return rho
    coil_count = Ic.shape[1]
    limits = config.COIL_LIMITS_A[:coil_count]
    over = np.abs(Ic[:M]) > limits[None, :]  # (M, n_coil)
    any_over_step = np.any(over, axis=1)     # (M,)
    if not np.any(any_over_step):
        return rho
    K_coil = int(np.argmax(any_over_step))  # 首个 True 的索引
    rho[K_coil:] = 0.0
    return rho


def compute_topo_zero_mask(nX: np.ndarray, N: int, K_eff: int) -> np.ndarray:
    """XPT 拓扑约束掩码：nX==4 时 1，否则 0。仅对 F2a/F2b 的 XPT 专属指标生效。

    超出有效步数 (k >= K_eff) 部分填 0。
    """
    mask = np.zeros(N, dtype=np.float64)
    if K_eff <= 0:
        return mask
    nX_arr = _pad_or_trim(nX, N, fill=0.0)
    mask[:K_eff] = (np.round(nX_arr[:K_eff]).astype(int) == config.XPT_REQUIRED_NX).astype(np.float64)
    return mask


def compute_gamma(inference_time_ms: Optional[np.ndarray], N: int) -> float:
    """推理时间合规系数。本期不评分 → 默认 1。"""
    if inference_time_ms is None:
        return 1.0
    t = np.asarray(inference_time_ms, dtype=float).flatten()
    if len(t) == 0:
        return 1.0
    over = np.sum(t > config.INFER_TIME_LIMIT_MS)
    r = over / max(N, 1)
    return 1.0 if r <= config.INFER_TIME_OVER_RATIO_MAX else 0.0
