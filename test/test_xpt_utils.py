"""Unit tests for environment/xpt_utils.py (no HFM simulator required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.xpt_utils import (
    _match_to_reference_slots,
    assign_xpoints_to_slots,
    extract_nx,
    extract_sorted_xpoints,
    extract_target_xpoints,
    extract_xpt_observation_pack,
    infer_fx_reshape_order,
    interp_psi_bilinear,
    pack_to_vector,
    reshape_fx_to_psi,
)


def test_extract_nx_and_sort_order():
    obs = {
        "rX": np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
        "zX": np.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
        "FX": np.array([0.1, 0.3, 0.2, 0.0, 0.0, 0.0]),
        "nX": np.array([3]),
        "FB": np.array([0.5]),
    }
    assert extract_nx(obs) == 3
    r, z, fx, valid, nx, fb = extract_sorted_xpoints(obs, slots=4)
    assert nx == 3
    assert fb == 0.5
    assert np.allclose(z[:3], [2.0, 1.0, 0.0])


def test_assign_nx4_sort_only():
    obs = {
        "rX": np.array([0.4, 0.3, 0.2, 0.1, 9.0, 8.0]),
        "zX": np.array([1.0, 0.0, -1.0, -2.0, 3.0, 2.0]),
        "FX": np.arange(6, dtype=float),
        "nX": np.array([4]),
        "FB": np.array([0.0]),
    }
    ref_r = np.array([0.4, 0.3, 0.2, 0.1])
    ref_z = np.array([1.0, 0.0, -1.0, -2.0])
    r, z, _, valid, nx, _ = assign_xpoints_to_slots(obs, ref_r, ref_z, slots=4)
    assert nx == 4
    assert np.allclose(z, ref_z)
    assert np.all(valid == 1.0)


def test_assign_nx6_hungarian_to_reference():
    obs = {
        "rX": np.array([0.1, 0.2, 0.3, 0.4, 9.9, 8.8]),
        "zX": np.array([-2.0, -1.0, 0.0, 1.0, 3.0, 2.0]),
        "FX": np.arange(6, dtype=float),
        "nX": np.array([6]),
        "FB": np.array([0.0]),
    }
    ref_r = np.array([9.9, 8.8, 0.4, 0.3])
    ref_z = np.array([3.0, 2.0, 1.0, 0.0])
    r, z, _, valid, _, _ = assign_xpoints_to_slots(obs, ref_r, ref_z, slots=4)
    assert np.allclose(r, ref_r)
    assert np.allclose(z, ref_z)
    assert np.all(valid == 1.0)


def test_strike_hungarian_when_more_than_8():
    ref_r = np.arange(8, dtype=float)
    ref_z = np.zeros(8)
    cand_r = np.concatenate([ref_r, [9.0, 10.0]])
    cand_z = np.concatenate([ref_z, [1.0, -1.0]])
    r_pad, z_pad, valid, n = _match_to_reference_slots(
        ref_r, ref_z, cand_r, cand_z, slots=8, direct_if_equal=False
    )
    assert n == 10
    assert np.allclose(r_pad, ref_r)
    assert np.all(valid == 1.0)


def test_pack_vector_length():
    obs = {
        "rX": np.ones(6) * 0.5,
        "zX": np.array([1.0, 0.5, -0.5, -1.0, 0.0, 0.0]),
        "FX": np.zeros(6),
        "nX": np.array([4]),
        "FB": np.array([0.0]),
        "Fx": np.zeros(4290),
        "rx": np.linspace(0.4, 1.4, 66),
        "zx": np.linspace(-1.4, 1.4, 65),
    }
    ref_r, ref_z, _ = extract_target_xpoints(obs, slots=4)
    ref_sr = np.linspace(0.5, 1.2, 8)
    ref_sz = np.linspace(-1.0, 1.0, 8)
    pack = extract_xpt_observation_pack(obs, ref_r, ref_z, ref_sr, ref_sz, strike_slots=8)
    assert pack_to_vector(pack).shape == (56,)
    assert pack["nX"] == 4


def test_infer_order_matches_synthetic():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    psi = np.sin(np.outer(np.linspace(0, 1, 66), np.ones(65))) * 0.01
    fx_c = psi.reshape(-1, order="C")
    i, j = 10, 20
    obs = {
        "Fx": fx_c,
        "rx": rx,
        "zx": zx,
        "rX": np.array([rx[i], 0, 0, 0, 0, 0]),
        "zX": np.array([zx[j], 0, 0, 0, 0, 0]),
        "FX": np.array([psi[i, j], 0, 0, 0, 0, 0]),
        "nX": np.array([1]),
        "FB": np.array([0.0]),
    }
    assert infer_fx_reshape_order(obs) == "C"


def run_all():
    test_extract_nx_and_sort_order()
    test_assign_nx4_sort_only()
    test_assign_nx6_hungarian_to_reference()
    test_strike_hungarian_when_more_than_8()
    test_pack_vector_length()
    test_infer_order_matches_synthetic()
    print("test_xpt_utils: all passed")


if __name__ == "__main__":
    run_all()
