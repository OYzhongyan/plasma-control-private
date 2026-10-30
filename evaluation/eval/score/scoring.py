# Copyright @2025 ENN Energy(enn.cn)
# 复赛单步得分、子任务总分计算。

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from . import config


@dataclass
class TaskResult:
    task_id: str
    per_step_epsilons: Dict[str, np.ndarray] = field(default_factory=dict)
    n_actual_steps: int = 0
    K_eff: int = 0
    timeout: bool = False
    eta: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    topo_mask: Optional[np.ndarray] = None
    gamma: float = 1.0
    Ip_ref: Optional[np.ndarray] = None
    task_score: float = 0.0
    metric_scores: Dict[str, float] = field(default_factory=dict)
    per_step_score: Optional[np.ndarray] = None


def score_single_step(epsilon: np.ndarray, epsilon_max: float) -> np.ndarray:
    """对逐步误差数组计算逐步得分（未乘权重），返回 [0,1] 数组。"""
    if epsilon_max <= 0:
        return np.zeros_like(epsilon)
    # np.inf 会得到负无穷再 clip 到 0；用 nan_to_num 防 NaN
    raw = 1.0 - np.nan_to_num(epsilon, nan=np.inf, posinf=np.inf) / epsilon_max
    return np.maximum(0.0, raw)


def compute_per_metric_step_scores(
    per_step_epsilons: Dict[str, np.ndarray],
    weights: Dict[str, float],
    epsilon_max_dict: Dict[str, float],
    N: int,
    K_eff: int,
    topo_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """返回 {metric: (N,) 加权后单步得分 W_i * max(0, 1 - eps/eps_max)}，
    并对 XPT 专属指标应用 topo_mask；超出 K_eff 的步置 0。
    """
    out: Dict[str, np.ndarray] = {}
    for name, w in weights.items():
        eps_arr = per_step_epsilons.get(name)
        if eps_arr is None or len(eps_arr) == 0:
            out[name] = np.zeros(N, dtype=np.float64)
            continue
        eps_max = epsilon_max_dict.get(name, config.get_epsilon_max(name))
        s_norm = score_single_step(eps_arr[:N], eps_max)
        scores = w * s_norm
        # 提前终止：超出 K_eff 的步直接置 0
        if K_eff < N:
            scores[K_eff:] = 0.0
        # XPT 专属指标受 topo mask 控制
        if name in config.XPT_TOPO_METRICS and topo_mask is not None:
            scores = scores * topo_mask[:N]
        # 长度对齐
        if len(scores) < N:
            padded = np.zeros(N, dtype=np.float64)
            padded[:len(scores)] = scores
            scores = padded
        out[name] = scores.astype(np.float64)
    return out


def task_score(
    per_step_epsilons: Dict[str, np.ndarray],
    weights: Dict[str, float],
    epsilon_max_dict: Dict[str, float],
    N: int,
    K_eff: int,
    eta: np.ndarray,
    mu: np.ndarray,
    rho: np.ndarray,
    topo_mask: np.ndarray,
    gamma: float = 1.0,
    timeout: bool = False,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """计算子任务总分 + 每指标贡献分 + 每步合成得分（含系数）。

    S_task = gamma * (1/N) * sum_k eta(k) * mu(k) * rho(k) * sum_i sigma_i(k)
    其中 sigma_i 已包含 XPT topo_mask 屏蔽与提前终止零化。
    timeout=True 时整任务 0 分。
    """
    if timeout:
        return 0.0, {name: 0.0 for name in weights}, np.zeros(N)

    per_metric_step = compute_per_metric_step_scores(
        per_step_epsilons, weights, epsilon_max_dict, N, K_eff, topo_mask,
    )

    coef = np.ones(N, dtype=np.float64)
    if eta is not None:
        coef = coef * eta[:N]
    if mu is not None:
        coef = coef * mu[:N]
    if rho is not None:
        coef = coef * rho[:N]

    # 累加各指标，再乘 coef
    sum_metrics = np.zeros(N, dtype=np.float64)
    for s in per_metric_step.values():
        sum_metrics = sum_metrics + s
    step_total = coef * sum_metrics
    s_task = gamma * float(np.sum(step_total)) / max(N, 1)

    # 计算每指标贡献分（也乘系数与 gamma）
    metric_scores: Dict[str, float] = {}
    for name, s_arr in per_metric_step.items():
        metric_scores[name] = float(gamma * np.sum(coef * s_arr) / max(N, 1))

    return s_task, metric_scores, step_total


def total_score(task_scores: Dict[str, float]) -> float:
    return float(sum(task_scores.values()))
