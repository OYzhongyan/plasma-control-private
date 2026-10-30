# Copyright @2025 ENN Energy(enn.cn)
# 复赛评估常量：3 子任务、零分阈值、指标权重、线圈/电流约束。

from typing import Dict, List, Any

import numpy as np


METRIC_IP = "Ip"
METRIC_LCFS = "LCFS"
METRIC_X = "X"
METRIC_STRIKE = "strike"
METRIC_PSIX = "psiX"
METRIC_X2 = "X2"
METRIC_PSIX2 = "psiX2"

ALL_METRICS = [METRIC_IP, METRIC_LCFS, METRIC_X, METRIC_STRIKE,
               METRIC_PSIX, METRIC_X2, METRIC_PSIX2]

XPT_TOPO_METRICS = {METRIC_X, METRIC_STRIKE, METRIC_PSIX, METRIC_X2, METRIC_PSIX2}

EPSILON_MAX: Dict[str, float] = {
    METRIC_IP: 0.05,
    METRIC_LCFS: 3.0,
    METRIC_X: 3.0,
    METRIC_STRIKE: 10.0,
    METRIC_PSIX: 0.003,
    METRIC_X2: 5.0,
    METRIC_PSIX2: 0.005,
}

TASK_IDS: List[str] = ["F1", "F2a", "F2b"]

TASK_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "F1": {
        "total": 40,
        "metrics": {
            METRIC_IP: 12,
            METRIC_LCFS: 28,
        },
    },
    "F2a": {
        "total": 30,
        "metrics": {
            METRIC_IP: 6,
            METRIC_LCFS: 6,
            METRIC_X: 4,
            METRIC_STRIKE: 3,
            METRIC_PSIX: 4,
            METRIC_X2: 3,
            METRIC_PSIX2: 4,
        },
    },
    "F2b": {
        "total": 30,
        "metrics": {
            METRIC_IP: 6,
            METRIC_LCFS: 6,
            METRIC_X: 4,
            METRIC_STRIKE: 3,
            METRIC_PSIX: 4,
            METRIC_X2: 3,
            METRIC_PSIX2: 4,
        },
    },
}

# 总步数 N（评分公式中分母固定）
TOTAL_STEPS: Dict[str, int] = {"F1": 300, "F2a": 500, "F2b": 500}

# 任务演化时长（毫秒），用于生成 Ip_ref 时计算时间戳
TASK_DURATION_MS: Dict[str, float] = {"F1": 300.0, "F2a": 500.0, "F2b": 500.0}

# 配置分组（F2a/F2b 共享 target 位形）
TARGET_FALLBACK: Dict[str, str] = {"F2b": "F2a"}

# 目标位形拓扑类型：限制器 → lX=0，偏滤器 → lX=1
TASK_TARGET_TOPOLOGY: Dict[str, str] = {
    "F1": "limiter",
    "F2a": "divertor",
    "F2b": "divertor",
}

TOTAL_MAX_SCORE: float = 100.0

# 电流偏差熔断阈值（A，文档单位 kA）
CURRENT_FUSE_A: float = 50_000.0

# 线圈电流上限（A）：索引 0=CS, 1..10=PF1..PF10, 11=VS
COIL_LIMITS_A = np.array(
    [45_000.0]                  # CS
    + [14_000.0] * 10           # PF1..PF10
    + [4_000.0],                # VS
    dtype=np.float64,
)

# XPT 拓扑要求的 X 点个数
XPT_REQUIRED_NX: int = 4

# 推理时间约束（本期不评分，留接口）
INFER_TIME_LIMIT_MS: float = 0.5
INFER_TIME_OVER_RATIO_MAX: float = 0.10


def get_epsilon_max(metric: str) -> float:
    return EPSILON_MAX.get(metric, 1.0)


def get_task_metrics_and_weights(task_id: str) -> Dict[str, float]:
    t = TASK_WEIGHTS.get(task_id)
    if not t:
        return {}
    return dict(t["metrics"])


def get_total_steps(task_id: str) -> int:
    return TOTAL_STEPS.get(task_id, 500)


def get_task_duration_ms(task_id: str) -> float:
    return TASK_DURATION_MS.get(task_id, 500.0)


def build_Ip_ref(task_id: str) -> np.ndarray:
    """按文档 2.3 节生成时变电流目标序列，长度 = 该任务 N（单位 A）。

    时间步取 t_k = k * dt（k=0..N-1），dt = 总时长 / N，与仿真器约定一致。
    """
    N = get_total_steps(task_id)
    duration = get_task_duration_ms(task_id)
    dt = duration / N
    t = (np.arange(N) + 1) * dt  # 第 k 步对应时刻（1-based 时间点）

    if task_id == "F1":
        ref = np.where(
            t <= 100.0,
            500_000.0,
            500_000.0 - (t - 100.0) * 1_500.0,
        )
        ref = np.clip(ref, 200_000.0, 500_000.0)
        return ref.astype(np.float64)
    elif task_id in ("F2a", "F2b"):
        ref = np.where(
            t <= 100.0,
            500_000.0 - t * 1_000.0,
            400_000.0,
        )
        ref = np.clip(ref, 400_000.0, 500_000.0)
        return ref.astype(np.float64)
    else:
        return np.full(N, 500_000.0, dtype=np.float64)
