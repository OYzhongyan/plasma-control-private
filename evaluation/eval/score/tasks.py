# Copyright @2025 ENN Energy(enn.cn)
# 复赛子任务：F1, F2a, F2b 指标计算调度。

import numpy as np
from typing import Any, Dict, List, Tuple

from . import config
from . import metrics


def _to_np(x, dtype=float):
    return None if x is None else np.asarray(x, dtype=dtype)


def _pad_eps_to_N(eps: np.ndarray, N: int) -> np.ndarray:
    """把误差数组拉到长度 N。提前终止的步用很大的值（确保得 0 分），
    但 scoring 里我们用 K_eff 严格置 sigma=0，所以这里填 0 也行——
    保险起见用 epsilon_max 之上的大值。
    """
    out = np.full(N, np.inf, dtype=np.float64)
    n = min(len(eps), N)
    if n > 0:
        out[:n] = eps[:n]
    return out


def compute_actual_steps(trajectory: Dict[str, Any]) -> int:
    """估算实际有效步数 K_eff：以关键数组的最小长度为准。"""
    keys = ["Ip", "lcfs_per_step", "lX", "Icoil"]
    lengths = []
    for k in keys:
        v = trajectory.get(k)
        if v is None:
            continue
        try:
            lengths.append(len(v))
        except TypeError:
            continue
    if not lengths:
        return 0
    return int(min(lengths))


def compute_epsilons_for_task(
    trajectory: Dict[str, Any],
    ref: Dict[str, Any],
    task_id: str,
) -> Tuple[Dict[str, np.ndarray], int, np.ndarray]:
    """返回 (eps_dict, K_eff, Ip_ref_array)。

    eps_dict[name] 为长度 N 的数组（超出 K_eff 部分填充 inf）。
    Ip_ref_array 用于熔断系数 mu 的计算。
    """
    N = config.get_total_steps(task_id)
    weights = config.get_task_metrics_and_weights(task_id)
    if not weights:
        return {}, 0, np.zeros(N)

    K_eff = compute_actual_steps(trajectory)
    K_eff = min(K_eff, N)

    Ip_a = _to_np(trajectory.get("Ip"))
    Ip_ref = config.build_Ip_ref(task_id)  # 长度 N
    assert len(Ip_ref) == N

    eps: Dict[str, np.ndarray] = {}

    # 电流
    if config.METRIC_IP in weights:
        if Ip_a is not None and K_eff > 0:
            e = metrics.epsilon_Ip(Ip_a[:K_eff], Ip_ref[:K_eff])
            eps[config.METRIC_IP] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_IP] = np.full(N, np.inf)

    # LCFS
    if config.METRIC_LCFS in weights:
        lcfs_target = ref.get("lcfs_points")
        lcfs_actual = trajectory.get("lcfs_per_step", [])
        if lcfs_target is not None and lcfs_actual and K_eff > 0:
            target = np.asarray(lcfs_target, dtype=float)
            steps_iter = []
            for k in range(K_eff):
                if k < len(lcfs_actual):
                    steps_iter.append(np.asarray(lcfs_actual[k], dtype=float))
            e = metrics.epsilon_LCFS(target, steps_iter)
            eps[config.METRIC_LCFS] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_LCFS] = np.full(N, np.inf)

    # X 点
    if config.METRIC_X in weights:
        x_actual = _to_np(trajectory.get("Xpt_main"))
        x_ref = _to_np(ref.get("Xpt_main"))
        if x_actual is not None and x_ref is not None and K_eff > 0 and x_actual.ndim == 3:
            e = metrics.epsilon_xpoints(x_actual[:K_eff], x_ref)
            eps[config.METRIC_X] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_X] = np.full(N, np.inf)

    # 打击点
    if config.METRIC_STRIKE in weights:
        s_actual = _to_np(trajectory.get("strike"))
        s_ref = _to_np(ref.get("strike"))
        if s_actual is not None and s_ref is not None and K_eff > 0 and s_actual.ndim == 3:
            e = metrics.epsilon_strikes(s_actual[:K_eff], s_ref)
            eps[config.METRIC_STRIKE] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_STRIKE] = np.full(N, np.inf)

    # 主 X 点磁通偏差
    if config.METRIC_PSIX in weights:
        psiX = _to_np(trajectory.get("psiX_main"))
        psib = _to_np(trajectory.get("psib"))
        psia = _to_np(trajectory.get("psia"))
        if (psiX is not None and psib is not None and psia is not None and K_eff > 0):
            e = metrics.epsilon_psix(psiX[:K_eff], psib[:K_eff], psia[:K_eff])
            eps[config.METRIC_PSIX] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_PSIX] = np.full(N, np.inf)

    # 次级 X 点距离
    if config.METRIC_X2 in weights:
        x2_actual = _to_np(trajectory.get("Xpt_sec"))
        x2_ref = _to_np(ref.get("Xpt_sec"))
        if x2_actual is not None and x2_ref is not None and K_eff > 0 and x2_actual.ndim == 3:
            e = metrics.epsilon_xpoints(x2_actual[:K_eff], x2_ref)
            eps[config.METRIC_X2] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_X2] = np.full(N, np.inf)

    # 次级 X 点磁通偏差
    if config.METRIC_PSIX2 in weights:
        psiX2 = _to_np(trajectory.get("psiX_sec"))
        psib = _to_np(trajectory.get("psib"))
        psia = _to_np(trajectory.get("psia"))
        if (psiX2 is not None and psib is not None and psia is not None and K_eff > 0):
            e = metrics.epsilon_psix(psiX2[:K_eff], psib[:K_eff], psia[:K_eff])
            eps[config.METRIC_PSIX2] = _pad_eps_to_N(e, N)
        else:
            eps[config.METRIC_PSIX2] = np.full(N, np.inf)

    return eps, K_eff, Ip_ref
