# Copyright @2025 ENN Energy(enn.cn)
# 复赛评估入口：从 trajectory + ref 计算各子任务逐步指标与总分。

import numpy as np
from typing import Any, Dict, List, Optional

from . import config
from . import tasks
from . import scoring
from . import penalties


def _resolve_ref(target_all: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """根据 fallback 配置获得参考目标（F2b 默认 fallback 到 F2a）。"""
    if task_id in target_all:
        return target_all[task_id]
    fallback = config.TARGET_FALLBACK.get(task_id)
    if fallback and fallback in target_all:
        return target_all[fallback]
    return {}


def evaluate(
    trajectories: Dict[str, Any],
    targets: Dict[str, Any],
    *,
    task_ids: Optional[List[str]] = None,
    timeouts: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    trajectories: {task_id: {"trajectory": {...}, "timeout": bool, ...}}
    targets:      {task_id: {...单帧目标 + Ip 时间序列(可选)}}
    返回：包含 total_score、task_scores、task_metric_scores、per_task_results 的字典。
    """
    task_ids = task_ids or config.TASK_IDS
    timeouts = timeouts or {}

    per_task_results: List[scoring.TaskResult] = []
    task_scores: Dict[str, float] = {}
    task_metric_scores: Dict[str, Dict[str, float]] = {}

    for tid in task_ids:
        if tid not in config.TASK_WEIGHTS:
            continue
        weights = config.get_task_metrics_and_weights(tid)
        N = config.get_total_steps(tid)

        entry = trajectories.get(tid, {}) or {}
        traj = entry.get("trajectory", {}) or {}
        timeout = bool(entry.get("timeout", timeouts.get(tid, False)))
        ref = _resolve_ref(targets, tid)

        eps, K_eff, Ip_ref = tasks.compute_epsilons_for_task(traj, ref, tid)

        # 系数
        lX = traj.get("lX", [])
        nX = traj.get("nX", [])
        Ip_a = traj.get("Ip", [])
        Icoil = traj.get("Icoil", [])
        infer_t = traj.get("inference_time_ms")  # 本期不评分

        eta = penalties.compute_eta(np.asarray(lX), tid, N, K_eff)
        mu = penalties.compute_mu(np.asarray(Ip_a, dtype=float),
                                   Ip_ref, N, K_eff)
        rho = penalties.compute_rho(np.asarray(Icoil, dtype=float) if Icoil is not None else None, N)
        topo_mask = penalties.compute_topo_zero_mask(np.asarray(nX), N, K_eff)
        gamma = penalties.compute_gamma(infer_t, N) if infer_t is not None else 1.0

        s_task, metric_scores, step_total = scoring.task_score(
            eps, weights, config.EPSILON_MAX, N, K_eff,
            eta=eta, mu=mu, rho=rho, topo_mask=topo_mask,
            gamma=gamma, timeout=timeout,
        )

        result = scoring.TaskResult(
            task_id=tid,
            per_step_epsilons=eps,
            n_actual_steps=K_eff,
            K_eff=K_eff,
            timeout=timeout,
            eta=eta, mu=mu, rho=rho, topo_mask=topo_mask, gamma=gamma,
            Ip_ref=Ip_ref,
            task_score=s_task,
            metric_scores=metric_scores,
            per_step_score=step_total,
        )
        per_task_results.append(result)
        task_scores[tid] = s_task
        task_metric_scores[tid] = metric_scores

    total = scoring.total_score(task_scores)

    return {
        "total_score": total,
        "task_scores": task_scores,
        "task_metric_scores": task_metric_scores,
        "per_task_results": per_task_results,
    }
