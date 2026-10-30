from typing import Dict, List, Any
import numpy as np

METRIC_IP = "Ip"
METRIC_LCFS = "LCFS"
METRIC_X = "X"
METRIC_STRIKE = "strike"
METRIC_PSIX = "psiX"
METRIC_X2 = "X2"
METRIC_PSIX2 = "psiX2"
ALL_METRICS = [
    METRIC_IP,
    METRIC_LCFS,
    METRIC_X,
    METRIC_STRIKE,
    METRIC_PSIX,
    METRIC_X2,
    METRIC_PSIX2,
]
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
TASK_IDS: List[str] = ["F2b"]
TASK_WEIGHTS: Dict[str, Dict[str, Any]] = {
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
    }
}
TOTAL_STEPS: Dict[str, int] = {"F2b": 500}
TASK_DURATION_MS: Dict[str, float] = {"F2b": 500.0}
TARGET_FALLBACK: Dict[str, str] = {"F2b": "F2a"}
TASK_TARGET_TOPOLOGY: Dict[str, str] = {"F2b": "divertor"}
TOTAL_MAX_SCORE: float = 30.0
CURRENT_FUSE_A: float = 50000.0
COIL_LIMITS_A = np.array([45000.0] + [14000.0] * 10 + [4000.0], dtype=np.float64)
XPT_REQUIRED_NX: int = 4
INFER_TIME_LIMIT_MS: float = 0.5
INFER_TIME_OVER_RATIO_MAX: float = 0.1


def get_epsilon_max(metric: str) -> float:
    return EPSILON_MAX.get(metric, 1.0)


def metric_weights() -> Dict[str, float]:
    t = TASK_WEIGHTS.get("F2b")
    if not t:
        return {}
    return dict(t["metrics"])


def get_total_steps() -> int:
    return TOTAL_STEPS.get("F2b", 500)


def get_task_duration_ms() -> float:
    return TASK_DURATION_MS.get("F2b", 500.0)


def build_Ip_ref() -> np.ndarray:
    """按文档 2.3 节生成时变电流目标序列，长度 = 该任务 N（单位 A）。

    时间步取 t_k = k * dt（k=0..N-1），dt = 总时长 / N，与仿真器约定一致。
    """
    N = get_total_steps()
    duration = get_task_duration_ms()
    dt = duration / N
    t = (np.arange(N) + 1) * dt
    ref = np.where(t <= 100.0, 500000.0 - t * 1000.0, 400000.0)
    ref = np.clip(ref, 400000.0, 500000.0)
    return ref.astype(np.float64)
