#
# Copyright @2025 ENN Energy(enn.cn)
#
# 12-channel power supply: transport delay -> rate limit -> PSM affine map.
#

from __future__ import annotations

from typing import Any

import numpy as np

from .preprocessing import ACTION_7D_TO_12D_INDEX

N_CHANNELS = 12
DT = 0.001  # simulator step: 1 ms

# PSM calibration (final_eval), U_out = slope * U_in + intercept
PSM_SLOPES = np.array(
    [
        0.8580659942, 0.6072767812, 0.6072767812, 0.8035000000,
        0.8035000000, 0.5528314158, 0.5528314158, 0.7901962762,
        0.7901962762, 0.8659211512, 0.8659211512, 1.0,
    ],
    dtype=np.float64,
)
PSM_INTERCEPTS = np.array(
    [
        245.0691566557, -40.2756786822, -44.2045503841, -48.0000000000,
        -48.0000000000, -21.9107201644, -22.5016982305, 72.2624890457,
        73.9466700254, 47.0431299209, 51.8093723495, 0.0,
    ],
    dtype=np.float64,
)

_MAX_CHANGE_7D = np.array([1499, 175, 175, 175, 175, 175, 80], dtype=np.float64) * 0.16
MAX_CHANGE_PER_STEP = _MAX_CHANGE_7D[np.asarray(ACTION_7D_TO_12D_INDEX, dtype=int)]

DELAY_RANGE_PF = (0.002, 0.005)  # channels 0-10, ms-scale latency
DELAY_RANGE_VS = (0.0, 0.001)  # channel 11 (VS), <= 1 ms


def _as_12d(name: str, values: np.ndarray | list[float] | None, default: np.ndarray) -> np.ndarray:
    if values is None:
        return default.copy()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != N_CHANNELS:
        raise ValueError(f"{name} must be length {N_CHANNELS}, got {arr.size}")
    return arr


def _rate_limit(
    u_in: np.ndarray,
    u_prev: np.ndarray | None,
    max_change: np.ndarray,
) -> np.ndarray:
    if u_prev is None:
        return u_in.copy()
    u_out = u_prev.copy()
    delta = u_in - u_prev
    for i in range(N_CHANNELS):
        cap = max_change[i]
        if abs(delta[i]) > cap:
            u_out[i] = u_prev[i] + np.sign(delta[i]) * cap
        else:
            u_out[i] = u_in[i]
    return u_out


def _apply_psm(u_in: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    return slopes * u_in + intercepts


class PowerSupplyModel:
    """Map policy setpoint U_set to coil voltage U_real sent to HFM."""

    def __init__(
        self,
        slopes: np.ndarray | list[float] | None = None,
        intercepts: np.ndarray | list[float] | None = None,
        max_change_per_step: np.ndarray | list[float] | None = None,
        delay_s: np.ndarray | list[float] | None = None,
        dt: float = DT,
        seed: int | None = None,
    ):
        self.dt = float(dt)
        self.slopes = _as_12d("slopes", slopes, PSM_SLOPES)
        self.intercepts = _as_12d("intercepts", intercepts, PSM_INTERCEPTS)
        self.max_change_per_step = _as_12d(
            "max_change_per_step", max_change_per_step, MAX_CHANGE_PER_STEP
        )

        self._rng = np.random.default_rng(seed)
        self._randomize_delay_each_step = delay_s is None
        if delay_s is None:
            self._resample_delay()
        else:
            self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
            self._update_delay_steps()

        self._u_history: list[np.ndarray] = []
        self._step_count = 0
        self._last_rate_limited: np.ndarray | None = None
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)

    def _update_delay_steps(self) -> None:
        if np.any(self.delay_s < 0):
            raise ValueError("delay_s must be non-negative for all channels")
        self.delay_steps = np.round(self.delay_s / self.dt).astype(int)

    def _resample_delay(self, *, delay_s: np.ndarray | list[float] | None = None) -> None:
        if delay_s is None:
            delay_s = np.empty(N_CHANNELS, dtype=np.float64)
            delay_s[:11] = self._rng.uniform(DELAY_RANGE_PF[0], DELAY_RANGE_PF[1], size=11)
            delay_s[11] = self._rng.uniform(DELAY_RANGE_VS[0], DELAY_RANGE_VS[1])
        self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
        self._update_delay_steps()

    def _transport_delay(self, u_set: np.ndarray) -> np.ndarray:
        self._u_history.append(u_set.copy())
        k = self._step_count
        self._step_count += 1
        if self._randomize_delay_each_step:
            self._resample_delay()

        u_delayed = np.empty(N_CHANNELS, dtype=np.float64)
        u_init = self._u_history[0]
        for i in range(N_CHANNELS):
            idx = k - self.delay_steps[i]
            u_delayed[i] = u_init[i] if idx < 0 else self._u_history[idx][i]
        return u_delayed

    def reset(self, *, u_set_init: np.ndarray | None = None) -> None:
        """Clear buffers; random delay is resampled on every step when enabled."""
        if self._randomize_delay_each_step:
            self._resample_delay()
        self._u_history = []
        self._step_count = 0
        self._last_rate_limited = None
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)
        if u_set_init is not None:
            u0 = np.asarray(u_set_init, dtype=np.float64).reshape(N_CHANNELS)
            self._u_history.append(u0.copy())

    def step(self, u_set: np.ndarray) -> np.ndarray:
        """
        U_set[k] -> delay -> rate limit -> PSM -> U_real[k]

            u_d[k]   = U_set[k - d]
            u_r[k]   = rate_limit(u_d[k], u_r[k-1])
            U_real[k]= slope * u_r[k] + intercept
        """
        u_set = np.asarray(u_set, dtype=np.float64).reshape(N_CHANNELS)
        u_delayed = self._transport_delay(u_set)
        u_limited = _rate_limit(u_delayed, self._last_rate_limited, self.max_change_per_step)
        self._last_rate_limited = u_limited.copy()
        self._y = _apply_psm(u_limited, self.slopes, self.intercepts)
        return self._y.copy()

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()

    def get_params(self) -> dict[str, Any]:
        return {
            "slopes": self.slopes.tolist(),
            "intercepts": self.intercepts.tolist(),
            "max_change_per_step": self.max_change_per_step.tolist(),
            "delay_s": self.delay_s.tolist(),
            "delay_steps": self.delay_steps.tolist(),
            "dt": self.dt,
        }


def simulate_power_supply_step(
    u_set: np.ndarray,
    dt: float,
    delay_s: float | np.ndarray,
    slopes: np.ndarray | float = 1.0,
    intercepts: np.ndarray | float = 0.0,
    max_change_per_step: np.ndarray | float | None = None,
) -> np.ndarray:
    """Scalar/broadcast open-loop simulation for plotting / validation."""
    u_set = np.asarray(u_set, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    if np.isscalar(delay_s):
        delay_steps = int(round(float(delay_s) / dt))
    else:
        delay_steps = int(round(np.asarray(delay_s, dtype=np.float64).reshape(-1)[0] / dt))

    if max_change_per_step is None:
        max_change = np.full(N_CHANNELS, np.inf)
    else:
        max_change = np.asarray(max_change_per_step, dtype=np.float64)
        if max_change.size == 1:
            max_change = np.full(N_CHANNELS, float(max_change[0]))

    y = np.empty_like(u_set)
    history: list[float] = []
    last_limited: float | None = None
    for k, u in enumerate(u_set):
        history.append(float(u))
        idx = 0 if k - delay_steps < 0 else k - delay_steps
        u_delayed = history[idx]
        if last_limited is None:
            u_limited = u_delayed
        else:
            delta = u_delayed - last_limited
            cap = float(max_change[0]) if max_change.size == 1 else float(max_change[0])
            if abs(delta) > cap:
                u_limited = last_limited + np.sign(delta) * cap
            else:
                u_limited = u_delayed
        last_limited = u_limited
        y[k] = float(slopes) * u_limited + float(intercepts) if slopes.size == 1 else slopes[0] * u_limited + intercepts[0]
    return y
