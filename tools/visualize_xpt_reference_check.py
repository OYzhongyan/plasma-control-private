from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "eval_files"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from environment.hfm_simulator import HFMSimulator
from environment.xpt_utils import (
    extract_xpt_observation_pack,
    plot_flux_surfaces_from_obs,
    plot_xpt_flux_diagnostic,
)
from trajectory_utils import build_official_target_observation


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _as_vec(value: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} must have at least {length} values, got {arr.size}")
    return arr[:length]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _plot_current_vs_target(
    obs: dict[str, Any],
    pack: dict[str, Any],
    ref_r_x: np.ndarray,
    ref_z_x: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax, _ = plot_flux_surfaces_from_obs(
        obs,
        FA=obs.get("FA"),
        clear=True,
        separatrix_color="#1e293b",
        separatrix_linewidth=2.0,
        draw_lcfs_polyline=False,
    )
    ax.set_title(title, fontsize=13, pad=8)
    ax.scatter(
        ref_r_x,
        ref_z_x,
        s=130,
        marker="x",
        c="#dc2626",
        linewidths=2.2,
        zorder=11,
        label="target X",
    )
    cur_r = np.asarray(pack["x_r"], dtype=np.float64)
    cur_z = np.asarray(pack["x_z"], dtype=np.float64)
    valid = np.asarray(pack["x_valid"], dtype=np.float64) > 0.5
    for idx in range(4):
        ax.text(
            ref_r_x[idx] + 0.015,
            ref_z_x[idx] - 0.045,
            f"T{idx}",
            fontsize=9,
            color="#991b1b",
            zorder=12,
        )
        if not valid[idx]:
            continue
        ax.scatter(
            cur_r[idx],
            cur_z[idx],
            s=170,
            marker="*",
            c="#2563eb",
            edgecolors="#1e3a8a",
            linewidths=0.8,
            zorder=12,
            label="current X" if idx == 0 else None,
        )
        ax.plot(
            [cur_r[idx], ref_r_x[idx]],
            [cur_z[idx], ref_z_x[idx]],
            "--",
            c="#64748b",
            lw=1.0,
            zorder=10,
        )
        ax.text(
            cur_r[idx] + 0.015,
            cur_z[idx] + 0.04,
            f"C{idx}",
            fontsize=9,
            color="#1e3a8a",
            zorder=12,
        )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset one F2 task and plot current X points against official target slots."
    )
    parser.add_argument(
        "--config", default="configs/env_default.yaml", help="Environment YAML."
    )
    parser.add_argument(
        "--reference-config",
        default="eval_files/references.yaml",
        help="Reference YAML.",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Override predictor.port."
    )
    parser.add_argument("--shot-id", default=None, help="Override task shot_id.")
    parser.add_argument(
        "--out",
        default="results/xpt_reference_checks/f2b_reset_xpt_check.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    env_cfg = _load_yaml(_resolve(args.config))
    reference_cfg = _load_yaml(_resolve(args.reference_config))
    reference_spec = dict(reference_cfg)
    if not reference_spec:
        raise KeyError(f"Task 'F2b' not found in {args.reference_config}")
    target_obs = build_official_target_observation(reference_spec)
    if target_obs is None:
        raise ValueError("Task F2b does not provide an official target observation")
    ref_r_x = _as_vec(target_obs["reference_rX"], 4, "reference_rX")
    ref_z_x = _as_vec(target_obs["reference_zX"], 4, "reference_zX")
    ref_r_s = _as_vec(
        target_obs.get("reference_strike_r", np.zeros(8)), 8, "reference_strike_r"
    )
    ref_z_s = _as_vec(
        target_obs.get("reference_strike_z", np.zeros(8)), 8, "reference_strike_z"
    )
    predictor = dict(env_cfg.get("predictor", {}))
    predictor["shot_id"] = args.shot_id or reference_spec.get(
        "shot_id", predictor.get("shot_id")
    )
    if args.port is not None:
        predictor["port"] = args.port
    env_cfg = dict(env_cfg)
    env_cfg["predictor"] = predictor
    env_cfg["max_steps"] = int(
        reference_spec.get(
            "max_steps", reference_cfg.get("max_steps", env_cfg.get("max_steps", 500))
        )
    )
    env = HFMSimulator(env_cfg)
    try:
        obs, _info = env.reset(options={"reference_mode": "hold"})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    pack = extract_xpt_observation_pack(
        obs, ref_r_x, ref_z_x, ref_r_s, ref_z_s, fx_order="C"
    )
    valid = np.asarray(pack["x_valid"], dtype=np.float64) > 0.5
    dist_cm = np.full(4, np.nan, dtype=np.float64)
    dist_cm[valid] = (
        np.hypot(
            np.asarray(pack["x_r"], dtype=np.float64)[valid] - ref_r_x[valid],
            np.asarray(pack["x_z"], dtype=np.float64)[valid] - ref_z_x[valid],
        )
        * 100.0
    )
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"F2b reset XPT check: shot={predictor.get('shot_id')} port={predictor.get('port')}"
    _plot_current_vs_target(obs, pack, ref_r_x, ref_z_x, out_path, title)
    diagnostic_path = out_path.with_name(
        out_path.stem + "_diagnostic" + out_path.suffix
    )
    plot_xpt_flux_diagnostic(
        obs,
        reference_rX=ref_r_x,
        reference_zX=ref_z_x,
        reference_rS=ref_r_s,
        reference_zS=ref_z_s,
        save_path=str(diagnostic_path),
        title=title,
        show=False,
    )
    print(f"Saved: {out_path}")
    print(f"Saved diagnostic: {diagnostic_path}")
    print(f"nX: {int(round(float(pack['nX'])))}")
    print(f"x_valid: {valid.astype(int).tolist()}")
    for idx in range(4):
        cur = (float(pack["x_r"][idx]), float(pack["x_z"][idx]))
        ref = (float(ref_r_x[idx]), float(ref_z_x[idx]))
        dist = "nan" if not np.isfinite(dist_cm[idx]) else f"{dist_cm[idx]:.2f}"
        print(
            f"X{idx}: current=({cur[0]:.4f}, {cur[1]:.4f}) target=({ref[0]:.4f}, {ref[1]:.4f}) dist_cm={dist}"
        )
    if np.any(np.isfinite(dist_cm)):
        print(f"mean_valid_x_dist_cm: {float(np.nanmean(dist_cm)):.2f}")


if __name__ == "__main__":
    main()
