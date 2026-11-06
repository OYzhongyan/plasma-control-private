"""Unit tests for power supply model (no socket required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.power_supply import (  # noqa: E402
    DELAY_RANGE_PF,
    DELAY_RANGE_VS,
    MAX_CHANGE_PER_STEP,
    PSM_INTERCEPTS,
    PSM_SLOPES,
    PowerSupplyModel,
    _apply_psm,
    _rate_limit,
)


def test_transport_delay_before_step_change():
    model = PowerSupplyModel(
        slopes=np.ones(12),
        intercepts=np.zeros(12),
        max_change_per_step=np.full(12, np.inf),
        delay_s=np.full(12, 0.003),
        seed=0,
    )
    model.reset()
    u_step = np.full(12, 50.0)
    y0 = model.step(np.zeros(12))
    assert np.allclose(y0, 0.0)
    ch = 0
    d = int(model.delay_steps[ch])
    for _ in range(d):
        y = model.step(u_step)
    assert y[ch] == 0.0
    y_after = model.step(u_step)
    assert y_after[ch] == 50.0


def test_steady_state_psm():
    slopes = np.linspace(0.9, 1.1, 12)
    intercepts = np.linspace(-2, 2, 12)
    model = PowerSupplyModel(
        slopes=slopes,
        intercepts=intercepts,
        max_change_per_step=np.full(12, np.inf),
        delay_s=np.full(12, 0.003),
        seed=1,
    )
    u = np.arange(12, dtype=np.float64)
    model.reset(u_set_init=u)
    for _ in range(10):
        y = model.step(u)
    expected = slopes * u + intercepts
    assert np.allclose(y, expected)


def test_random_delay_ranges():
    model = PowerSupplyModel(seed=42)
    assert model.delay_s.shape == (12,)
    assert np.all(model.delay_s[:11] >= DELAY_RANGE_PF[0] - 1e-9)
    assert np.all(model.delay_s[:11] <= DELAY_RANGE_PF[1] + 1e-9)
    assert model.delay_s[11] >= DELAY_RANGE_VS[0] - 1e-9
    assert model.delay_s[11] <= DELAY_RANGE_VS[1] + 1e-9


def test_rate_limit_caps_change():
    u_prev = np.zeros(12)
    u_in = np.full(12, 100.0)
    max_change = np.full(12, 10.0)
    u_out = _rate_limit(u_in, u_prev, max_change)
    assert np.allclose(u_out, 10.0)


def test_psm_affine():
    u = np.array([10.0, 20.0])
    slopes = np.array([0.5, 1.0])
    intercepts = np.array([1.0, -2.0])
    assert np.allclose(_apply_psm(u, slopes, intercepts), [6.0, 18.0])


def test_default_psm_constants_loaded():
    model = PowerSupplyModel(seed=0)
    assert np.allclose(model.slopes, PSM_SLOPES)
    assert np.allclose(model.intercepts, PSM_INTERCEPTS)
    assert np.allclose(model.max_change_per_step, MAX_CHANGE_PER_STEP)


def test_reset_resamples_delay():
    model = PowerSupplyModel(seed=0)
    delay_before = model.delay_s.copy()
    model.reset()
    assert not np.allclose(model.delay_s, delay_before)


def test_reset_keeps_fixed_delay():
    delay_s = np.full(12, 0.004)
    delay_s[11] = 0.001
    model = PowerSupplyModel(delay_s=delay_s, seed=0)
    model.reset()
    assert np.allclose(model.delay_s, delay_s)


def test_chain_order_delay_then_rate_then_psm():
    model = PowerSupplyModel(
        slopes=np.ones(12),
        intercepts=np.zeros(12),
        max_change_per_step=np.full(12, 5.0),
        delay_s=np.zeros(12),
        seed=0,
    )
    model.reset()
    model.step(np.zeros(12))
    y1 = model.step(np.full(12, 100.0))
    assert np.allclose(y1, 5.0)
    y2 = model.step(np.full(12, 100.0))
    assert np.allclose(y2, 10.0)


if __name__ == "__main__":
    test_transport_delay_before_step_change()
    test_steady_state_psm()
    test_random_delay_ranges()
    test_rate_limit_caps_change()
    test_psm_affine()
    test_default_psm_constants_loaded()
    test_reset_resamples_delay()
    test_reset_keeps_fixed_delay()
    test_chain_order_delay_then_rate_then_psm()
    print("test_power_supply passed.")
