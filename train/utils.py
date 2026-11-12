from __future__ import annotations

import math
from typing import Callable


def remaining_learning_timesteps(
    requested_timesteps: int,
    checkpoint_timesteps: int = 0,
    *,
    target_total_timesteps: int | None = None,
    reset_num_timesteps: bool = True,
) -> int:
    """Keep SB3's additional-step default; optionally stop at a global target."""
    if target_total_timesteps is None:
        return int(requested_timesteps)
    if reset_num_timesteps:
        raise ValueError("An absolute resume target requires reset_num_timesteps=false")
    target = int(target_total_timesteps)
    if target <= 0:
        raise ValueError("Resume target_total_timesteps must be positive")
    remaining = target - int(checkpoint_timesteps)
    if remaining <= 0:
        raise ValueError("Checkpoint already reached the absolute resume target")
    return remaining


def warmup_cosine_schedule(
    initial_lr: float,
    timesteps_per_update: int,
    total_timesteps: int,
    warmup_updates: int = 10,
    anneal: bool = True,
) -> Callable[[float], float]:
    """创建带 warmup 的 cosine annealing 学习率调度器。"""
    warmup_timesteps = warmup_updates * timesteps_per_update
    warmup_progress = max(0.0, 1.0 - warmup_timesteps / total_timesteps)
    warmup_lr_start = initial_lr / 10.0

    def schedule(progress_remaining: float) -> float:
        if progress_remaining >= warmup_progress:
            warmup_fraction = (1. - progress_remaining) / max(1.0 - warmup_progress, 1e-6)
            return warmup_lr_start + (initial_lr - warmup_lr_start) * warmup_fraction
        elif anneal:
            decay_progress = (warmup_progress - progress_remaining) / max(warmup_progress, 1e-6)
            return initial_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        else:
            return initial_lr

    return schedule
