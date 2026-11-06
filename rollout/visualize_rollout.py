"""Plot per-step rollout reward components saved by rollout/eval.py."""

from __future__ import annotations
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_COLUMNS = {
    "episode",
    "env_index",
    "step",
    "env_reward",
    "ip",
    "reference_ip",
    "terminated",
    "truncated",
    "shot_id",
    "port",
    "hfm_timeout",
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_filename(name: str) -> str:
    cleaned = re.sub("[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned or "component"


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _component_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    out: list[str] = []
    for column in columns:
        if column in BASE_COLUMNS or column.startswith("action_"):
            continue
        if any((_safe_float(row.get(column)) is not None for row in rows)):
            out.append(column)
    return out


def _group_rows(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rollout_id = _safe_int(row.get("env_index", row.get("episode", 0)))
        groups[rollout_id].append(row)
    for rollout_rows in groups.values():
        rollout_rows.sort(key=lambda row: _safe_int(row.get("step", 0)))
    return dict(sorted(groups.items()))


def _plot_component(
    component: str, groups: dict[int, list[dict[str, str]]], out_path: Path
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    plotted = False
    for rollout_id, rows in groups.items():
        xs: list[int] = []
        ys: list[float] = []
        label_extra = ""
        if rows:
            shot_id = rows[0].get("shot_id", "")
            port = rows[0].get("port", "")
            label_extra = f" shot={shot_id} port={port}".strip()
        for row in rows:
            value = _safe_float(row.get(component))
            if value is None:
                continue
            xs.append(_safe_int(row.get("step", len(xs) + 1)))
            ys.append(value)
        if not xs:
            continue
        plotted = True
        ax.plot(xs, ys, linewidth=1.5, label=f"rollout {rollout_id}{label_extra}")
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(component)
    ax.set_xlabel("step")
    ax.set_ylabel(component)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _finite_limits(*arrays: np.ndarray) -> tuple[float, float, float, float]:
    points = []
    for array in arrays:
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim < 2 or arr.shape[-1] != 2:
            continue
        flat = arr.reshape(-1, 2)
        flat = flat[np.all(np.isfinite(flat), axis=1)]
        if flat.size:
            points.append(flat)
    if not points:
        return (0.0, 1.0, -0.5, 0.5)
    merged = np.concatenate(points, axis=0)
    x_min, y_min = np.min(merged, axis=0)
    x_max, y_max = np.max(merged, axis=0)
    x_margin = max((x_max - x_min) * 0.08, 0.001)
    y_margin = max((y_max - y_min) * 0.08, 0.001)
    return (x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin)


def _finite_pair_array(
    r: np.ndarray, z: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    """Stack r/z into (..., 2) with invalid slots masked to NaN.

    无效槽位以 (0, 0) 填充，直接进坐标范围计算会把视野拉到原点，这里统一过滤：
    (r, z) 同时为 0 的点视为填充值置 NaN；若另给 valid 掩码则一并生效。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    bad = (r_arr == 0.0) & (z_arr == 0.0)
    if valid is not None:
        bad |= ~(np.asarray(valid, dtype=np.float64) > 0.5)
    r_arr = np.where(bad, np.nan, r_arr)
    z_arr = np.where(bad, np.nan, z_arr)
    return np.stack([r_arr, z_arr], axis=-1)


def _plot_ip_comparison(observations: dict[str, np.ndarray], out_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ip = np.asarray(observations["ip"], dtype=np.float64)
    reference_ip = np.asarray(observations["reference_ip"], dtype=np.float64)
    valid = np.asarray(observations["valid"], dtype=bool)
    if ip.ndim != 2 or reference_ip.shape != ip.shape:
        return False
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    plotted = False
    for env_idx in range(ip.shape[0]):
        mask = (
            valid[env_idx]
            & np.isfinite(ip[env_idx])
            & np.isfinite(reference_ip[env_idx])
        )
        if not np.any(mask):
            continue
        steps = np.arange(1, ip.shape[1] + 1)[mask]
        ax.plot(steps, ip[env_idx, mask], linewidth=1.5, label=f"rollout {env_idx} Ip")
        ax.plot(
            steps,
            reference_ip[env_idx, mask],
            linewidth=1.2,
            linestyle="--",
            label=f"rollout {env_idx} reference",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title("Ip vs reference Ip")
    ax.set_xlabel("step")
    ax.set_ylabel("Ip")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _extract_component_series(
    groups: dict[int, list[dict[str, str]]], component: str
) -> dict[int, np.ndarray]:
    """Extract a component's values for each rollout as a numpy array."""
    series: dict[int, np.ndarray] = {}
    for rollout_id, rows in groups.items():
        values = []
        for row in rows:
            val = _safe_float(row.get(component))
            values.append(val if val is not None else np.nan)
        series[rollout_id] = np.asarray(values, dtype=np.float64)
    return series


F2_SCORE_COLUMNS = (
    "main_x_dist_score",
    "main_x_psi_score",
    "second_x_dist_score",
    "second_x_psi_score",
    "strike_dist_score",
)
PLATFORM_SCORE_CFG = {"F2": (500, 30.0)}


def _detect_task(groups: dict[int, list[dict[str, str]]]) -> str:
    return "F2"


def _remove_contour(contour_set: Any) -> None:
    """Remove a contour set across matplotlib versions (collections removed in 3.10)."""
    if contour_set is None:
        return
    try:
        collections = list(contour_set.collections)
    except AttributeError:
        collections = [contour_set]
    for coll in collections:
        try:
            coll.remove()
        except Exception:
            pass


def _build_psi_series(
    observations: dict[str, np.ndarray], psi_env: int
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract psi grid series from observations.

    Fx (4290,) is C-order flattened with shape (66, 65): axis0 -> rx (R), axis1 -> zx (Z).
    Returns (psi_series [T, 66, 65], rx, zx, fb_series [T]) or (None, None, None, None).
    """
    fx = observations.get("fx")
    rx = observations.get("rx")
    zx = observations.get("zx")
    fb = observations.get("fb")
    if fx is None or rx is None or zx is None:
        return (None, None, None, None)
    fx = np.asarray(fx, dtype=np.float64)
    rx1 = np.asarray(rx, dtype=np.float64).ravel()
    zx1 = np.asarray(zx, dtype=np.float64).ravel()
    if fx.ndim != 3 or psi_env >= fx.shape[0] or fx.shape[-1] != rx1.size * zx1.size:
        return (None, None, None, None)
    psi_series = fx[psi_env].reshape(fx.shape[1], rx1.size, zx1.size)
    fb_series = (
        np.asarray(fb, dtype=np.float64).reshape(fx.shape[:2])[psi_env]
        if fb is not None
        else None
    )
    return (psi_series, rx1, zx1, fb_series)


def _save_lcfs_video(
    observations: dict[str, np.ndarray],
    groups: dict[int, list[dict[str, str]]],
    out_path: Path,
    fps: int,
    psi_env: int = 0,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.lines import Line2D

    lcfs = np.asarray(observations["lcfs_points"], dtype=np.float64)
    reference_lcfs = np.asarray(observations["reference_lcfs_points"], dtype=np.float64)
    ip_values = np.asarray(observations.get("ip", np.zeros((1, 1))), dtype=np.float64)
    reference_ip_values = np.asarray(
        observations.get("reference_ip", np.zeros((1, 1))), dtype=np.float64
    )
    valid = np.asarray(observations["valid"], dtype=bool)
    psi_series, rx_grid, zx_grid, fb_series = _build_psi_series(
        observations, int(psi_env)
    )
    xpoint_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    rX_raw = observations.get("rX")
    zX_raw = observations.get("zX")
    if rX_raw is not None and zX_raw is not None:
        rX_arr = np.asarray(rX_raw, dtype=np.float64)
        zX_arr = np.asarray(zX_raw, dtype=np.float64)
        if (
            rX_arr.ndim == 3
            and zX_arr.shape == rX_arr.shape
            and (rX_arr.shape[-1] == 6)
        ):
            nX_raw = observations.get("nX")
            if nX_raw is not None:
                nX_arr = np.asarray(nX_raw, dtype=np.float64).reshape(rX_arr.shape[:2])
            else:
                nX_arr = np.full(rX_arr.shape[:2], rX_arr.shape[-1], dtype=np.float64)
            xpoint_data = (rX_arr, zX_arr, nX_arr)
    if (
        lcfs.ndim != 4
        or lcfs.shape[-2:] != (32, 2)
        or reference_lcfs.shape != lcfs.shape
    ):
        return False
    if not np.any(valid):
        return False
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "matplotlib ffmpeg writer is not available; install ffmpeg to save mp4 videos"
        )
    ip_score_series = _extract_component_series(groups, "ip_competition_score")
    lcfs_score_series = _extract_component_series(groups, "lcfs_competition_score")
    competition_score_series = _extract_component_series(groups, "competition_score")
    ip_abs_error_series = _extract_component_series(groups, "ip_abs_error")
    coil_margin_series = _extract_component_series(groups, "coil_margin")
    topology_mask_series = _extract_component_series(groups, "topology_mask")
    f2_score_names = list(F2_SCORE_COLUMNS)
    f2_score_series_list = [
        _extract_component_series(groups, name) for name in f2_score_names
    ]
    is_f2 = any(
        (
            s.size and np.any(np.isfinite(s))
            for series in f2_score_series_list
            for s in series.values()
        )
    )
    last_step = int(np.max(np.where(valid)[1])) + 1
    num_envs = lcfs.shape[0]
    slot_data: dict[str, np.ndarray] | None = None
    _slot_keys = (
        "x_slot_r",
        "x_slot_z",
        "x_valid",
        "strike_r",
        "strike_z",
        "strike_valid",
        "ref_rX",
        "ref_zX",
        "ref_strike_r",
        "ref_strike_z",
    )
    if all((observations.get(k) is not None for k in _slot_keys)):
        _slot = {k: np.asarray(observations[k], dtype=np.float64) for k in _slot_keys}
        _t_steps = lcfs.shape[1]
        if _slot["x_slot_r"].shape == (num_envs, _t_steps, 4) and _slot[
            "strike_r"
        ].shape == (num_envs, _t_steps, 8):
            slot_data = _slot
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    n_grid_cols = 5 if is_f2 else 3
    fig = plt.figure(figsize=(26, 12) if is_f2 else (20, 12), dpi=100)
    gs = fig.add_gridspec(3, n_grid_cols, wspace=0.25, hspace=0.35)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08)
    ax_lcfs = fig.add_subplot(gs[0:2, 0])
    limit_arrays: list[np.ndarray] = [lcfs, reference_lcfs]
    if slot_data is not None:
        limit_arrays.append(
            _finite_pair_array(slot_data["ref_rX"], slot_data["ref_zX"])
        )
        limit_arrays.append(
            _finite_pair_array(
                slot_data["x_slot_r"], slot_data["x_slot_z"], slot_data["x_valid"]
            )
        )
        limit_arrays.append(
            _finite_pair_array(slot_data["ref_strike_r"], slot_data["ref_strike_z"])
        )
        limit_arrays.append(
            _finite_pair_array(
                slot_data["strike_r"], slot_data["strike_z"], slot_data["strike_valid"]
            )
        )
    x_min, x_max, y_min, y_max = _finite_limits(*limit_arrays)
    actual_lines = []
    reference_lines = []
    for env_idx in range(num_envs):
        color = colors[env_idx % len(colors)] if colors else None
        (actual_line,) = ax_lcfs.plot(
            [], [], linewidth=2.0, color=color, label=f"rollout {env_idx}"
        )
        (reference_line,) = ax_lcfs.plot(
            [], [], linewidth=1.5, linestyle="--", color=color
        )
        actual_lines.append(actual_line)
        reference_lines.append(reference_line)
    xpoint_lines: list[Any] = []
    if xpoint_data is not None:
        for env_idx in range(num_envs):
            (xline,) = ax_lcfs.plot(
                [],
                [],
                linestyle="None",
                marker="x",
                markersize=7,
                markeredgewidth=1.8,
                color="black",
                label="X points" if env_idx == 0 else None,
            )
            xpoint_lines.append(xline)
    target_x_segments: list[list[Any]] = []
    xslot_line_groups: list[list[Any]] = []
    strike_markers: list[Any] = []
    target_strike_markers: list[Any] = []
    strike_line_groups: list[list[Any]] = []
    if slot_data is not None:
        for env_idx in range(num_envs):
            color = colors[env_idx % len(colors)] if colors else None
            xlines = []
            for slot in range(4):
                (sline,) = ax_lcfs.plot(
                    [],
                    [],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.55,
                    color="black",
                    label="X slot match" if env_idx == 0 and slot == 0 else None,
                )
                xlines.append(sline)
            xslot_line_groups.append(xlines)
            (smarker,) = ax_lcfs.plot(
                [],
                [],
                linestyle="None",
                marker="s",
                markersize=4.5,
                color=color,
                label="strike points" if env_idx == 0 else None,
            )
            strike_markers.append(smarker)
            (tsmarker,) = ax_lcfs.plot(
                [],
                [],
                linestyle="None",
                marker="s",
                markersize=5.5,
                markerfacecolor="none",
                markeredgewidth=1.3,
                color=color,
                label="target strike" if env_idx == 0 else None,
            )
            target_strike_markers.append(tsmarker)
            slines = []
            for slot in range(8):
                (stline,) = ax_lcfs.plot(
                    [],
                    [],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.45,
                    color="black",
                    label="strike slot match" if env_idx == 0 and slot == 0 else None,
                )
                slines.append(stline)
            strike_line_groups.append(slines)
        masked_ref_x = _finite_pair_array(slot_data["ref_rX"], slot_data["ref_zX"])
        masked_ref_strike = _finite_pair_array(
            slot_data["ref_strike_r"], slot_data["ref_strike_z"]
        )
        x_half = 0.03 * (x_max - x_min)
        for env_idx in range(num_envs):
            segs: list[Any] = []
            for r_t, z_t in masked_ref_x[env_idx, 0]:
                if not (np.isfinite(r_t) and np.isfinite(z_t)):
                    continue
                for sgn in (1.0, -1.0):
                    (seg,) = ax_lcfs.plot(
                        [r_t - x_half, r_t + x_half],
                        [z_t - sgn * x_half, z_t + sgn * x_half],
                        linestyle="--",
                        linewidth=1.2,
                        color="black",
                        zorder=5,
                    )
                    segs.append(seg)
            target_x_segments.append(segs)
            target_strike_markers[env_idx].set_data(
                masked_ref_strike[env_idx, 0, :, 0], masked_ref_strike[env_idx, 0, :, 1]
            )
    legend_handles, legend_labels = ax_lcfs.get_legend_handles_labels()
    if target_x_segments:
        legend_handles.append(
            Line2D(
                [], [], linestyle="--", linewidth=1.2, color="black", label="target X"
            )
        )
        legend_labels.append("target X")
    n_legend_rows = 1
    if legend_handles:
        n_legend_cols = min(len(legend_handles), 4)
        n_legend_rows = (len(legend_handles) + n_legend_cols - 1) // n_legend_cols
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=n_legend_cols,
            fontsize=9,
            frameon=False,
        )
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=max(0.08, 0.03 + 0.05 * n_legend_rows)
    )
    ax_lcfs.set_xlim(x_min, x_max)
    ax_lcfs.set_ylim(y_min, y_max)
    ax_lcfs.set_aspect("equal", adjustable="box")
    ax_lcfs.set_xlabel("R (m)", fontsize=10)
    ax_lcfs.set_ylabel("Z (m)", fontsize=10)
    ax_lcfs.grid(True, alpha=0.3)
    ax_lcfs.set_axisbelow(True)
    psi_image = None
    fb_contour: list[Any] = [None]
    if psi_series is not None:
        finite_psi = psi_series[valid[psi_env]]
        finite_psi = finite_psi[np.isfinite(finite_psi)]
        if finite_psi.size > 0:
            vmin, vmax = (float(np.min(finite_psi)), float(np.max(finite_psi)))
            if vmax - vmin < 1e-12:
                vmin, vmax = (vmin - 0.5, vmax + 0.5)
            psi_image = ax_lcfs.imshow(
                np.full((zx_grid.size, rx_grid.size), np.nan),
                extent=[
                    float(rx_grid[0]),
                    float(rx_grid[-1]),
                    float(zx_grid[0]),
                    float(zx_grid[-1]),
                ],
                origin="lower",
                interpolation="bilinear",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                alpha=0.85,
                zorder=0,
                aspect="equal",
            )
            fig.colorbar(psi_image, ax=ax_lcfs, shrink=0.85, pad=0.02, label="psi (Wb)")
    lcfs_title = "LCFS vs reference"
    if psi_image is not None:
        lcfs_title += f" + psi (rollout {psi_env})"
    ax_lcfs.set_title(lcfs_title, fontsize=11, pad=12)
    ax_ip = fig.add_subplot(gs[2, 0])
    ip_lines = []
    ref_ip_lines = []
    for env_idx in range(num_envs):
        color = colors[env_idx % len(colors)] if colors else None
        (ip_line,) = ax_ip.plot(
            [], [], linewidth=1.5, color=color, label=f"Ip {env_idx}"
        )
        (ref_ip_line,) = ax_ip.plot(
            [], [], linewidth=1.2, linestyle="--", color=color, label=f"ref {env_idx}"
        )
        ip_lines.append(ip_line)
        ref_ip_lines.append(ref_ip_line)
    if ip_values.size > 0 and np.any(np.isfinite(ip_values)):
        ip_max = max(np.nanmax(ip_values), np.nanmax(reference_ip_values)) * 1.1
        ip_min = min(np.nanmin(ip_values), np.nanmin(reference_ip_values)) * 0.9
        if ip_max - ip_min < 1e-06:
            ip_min = ip_max - 100000.0
            ip_max = ip_max + 100000.0
    else:
        ip_min, ip_max = (0, 1000000.0)
    ax_ip.set_xlim(1, last_step)
    ax_ip.set_ylim(ip_min, ip_max)
    ax_ip.set_xlabel("step", fontsize=10)
    ax_ip.set_ylabel("Ip (A)", fontsize=10)
    ax_ip.grid(True, alpha=0.3)
    ax_ip.legend(loc="best", fontsize=7, ncol=2)
    ax_ip.set_title("Ip vs reference_Ip", fontsize=11, pad=12)
    ax_ip_score = fig.add_subplot(gs[0, 1])
    ax_lcfs_score = fig.add_subplot(gs[1, 1])
    ax_comp_score = fig.add_subplot(gs[2, 1])
    score_axes = [ax_ip_score, ax_lcfs_score, ax_comp_score]
    score_series_list = [ip_score_series, lcfs_score_series, competition_score_series]
    score_names = ["ip_score", "lcfs_score", "competition_score"]
    score_lines = []
    for ax, series_dict, name in zip(score_axes, score_series_list, score_names):
        lines = []
        for env_idx in range(num_envs):
            color = colors[env_idx % len(colors)] if colors else None
            (line,) = ax.plot(
                [], [], linewidth=1.5, color=color, label=f"rollout {env_idx}"
            )
            lines.append(line)
        score_lines.append(lines)
        ax.set_xlim(1, last_step)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)
        ax.tick_params(axis="both", labelsize=8)
    ax_ip_error = fig.add_subplot(gs[0, 2])
    ax_coil_margin = fig.add_subplot(gs[1, 2])
    ax_topology_mask = fig.add_subplot(gs[2, 2])
    metric_axes = [ax_ip_error, ax_coil_margin, ax_topology_mask]
    metric_series_list = [ip_abs_error_series, coil_margin_series, topology_mask_series]
    metric_names = ["ip_abs_error", "coil_margin", "topology_mask"]
    metric_lines = []
    ip_error_max = max(
        (np.nanmax(s) if len(s) > 0 else 50000 for s in ip_abs_error_series.values())
    )
    coil_margin_max = max(
        (np.nanmax(s) if len(s) > 0 else 1.5 for s in coil_margin_series.values())
    )
    for ax, series_dict, name in zip(metric_axes, metric_series_list, metric_names):
        lines = []
        for env_idx in range(num_envs):
            color = colors[env_idx % len(colors)] if colors else None
            (line,) = ax.plot(
                [], [], linewidth=1.5, color=color, label=f"rollout {env_idx}"
            )
            lines.append(line)
        metric_lines.append(lines)
        ax.set_xlim(1, last_step)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)
        ax.tick_params(axis="both", labelsize=8)
    ax_ip_error.set_ylim(0, max(ip_error_max, 55000) * 1.1)
    ax_ip_error.axhline(
        50000, color="red", linestyle="--", linewidth=1.5, label="50kA threshold"
    )
    ax_coil_margin.set_ylim(0, max(coil_margin_max, 1.5) * 1.1)
    ax_coil_margin.axhline(
        1.0, color="red", linestyle="--", linewidth=1.5, label="limit=1"
    )
    ax_topology_mask.set_ylim(0, 1.1)
    extra_axes: list[Any] = []
    extra_lines_list: list[list[Any]] = []
    if is_f2:
        extra_specs = [gs[0, 3], gs[1, 3], gs[2, 3], gs[0, 4], gs[1, 4]]
        extra_max = max(
            (
                float(np.nanmax(s))
                for series in f2_score_series_list
                for s in series.values()
                if s.size and np.any(np.isfinite(s))
            ),
            default=1.0,
        )
        extra_ymax = max(1.05, extra_max * 1.1)
        for spec, series_dict, name in zip(
            extra_specs, f2_score_series_list, f2_score_names
        ):
            ax = fig.add_subplot(spec)
            lines = []
            for env_idx in range(num_envs):
                color = colors[env_idx % len(colors)] if colors else None
                (line,) = ax.plot(
                    [], [], linewidth=1.5, color=color, label=f"rollout {env_idx}"
                )
                lines.append(line)
            extra_lines_list.append(lines)
            ax.set_xlim(1, last_step)
            ax.set_ylim(0, extra_ymax)
            ax.set_xlabel("step", fontsize=9)
            ax.set_ylabel(name, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=6)
            ax.tick_params(axis="both", labelsize=8)
            extra_axes.append(ax)
    main_title = fig.suptitle("", fontsize=14)

    def _compute_stats(
        series: np.ndarray, current_step: int
    ) -> tuple[float, float, int]:
        if current_step <= 0 or len(series) == 0:
            return (0.0, 0.0, 0)
        valid_slice = series[:current_step]
        valid_vals = valid_slice[np.isfinite(valid_slice)]
        if len(valid_vals) == 0:
            return (0.0, 0.0, 0)
        total = float(np.sum(valid_vals))
        mean = float(np.mean(valid_vals))
        count = len(valid_vals)
        return (total, mean, count)

    def _count_exceed_threshold(
        series: np.ndarray, current_step: int, threshold: float
    ) -> int:
        if current_step <= 0 or len(series) == 0:
            return 0
        valid_slice = series[:current_step]
        valid_vals = valid_slice[np.isfinite(valid_slice)]
        return int(np.sum(valid_vals > threshold))

    def _first_exceed_step(series: np.ndarray, threshold: float) -> int:
        if len(series) == 0:
            return -1
        exceed_mask = np.isfinite(series) & (series > threshold)
        exceed_indices = np.where(exceed_mask)[0]
        if len(exceed_indices) == 0:
            return -1
        return int(exceed_indices[0]) + 1

    def _count_mask_value(
        series: np.ndarray, current_step: int, target_value: float
    ) -> int:
        if current_step <= 0 or len(series) == 0:
            return 0
        valid_slice = series[:current_step]
        valid_vals = valid_slice[np.isfinite(valid_slice)]
        return int(np.sum(np.isclose(valid_vals, target_value)))

    def update(frame_idx: int):
        current_step = frame_idx + 1
        artists = [main_title]
        main_title.set_text(f"Rollout Analysis - Step {current_step}/{last_step}")
        ax_lcfs.set_title(f"{lcfs_title} (step {current_step})", fontsize=11, pad=12)
        _remove_contour(fb_contour[0])
        fb_contour[0] = None
        if psi_image is not None:
            show_psi = frame_idx < valid.shape[1] and bool(valid[psi_env, frame_idx])
            psi_image.set_visible(show_psi)
            if show_psi:
                psi_frame = psi_series[frame_idx]
                psi_image.set_data(psi_frame.T)
                fb_val = (
                    float(fb_series[frame_idx]) if fb_series is not None else np.nan
                )
                if np.isfinite(fb_val):
                    try:
                        fb_contour[0] = ax_lcfs.contour(
                            rx_grid,
                            zx_grid,
                            psi_frame,
                            levels=[fb_val],
                            colors="red",
                            linewidths=1.2,
                        )
                    except Exception:
                        fb_contour[0] = None
        for env_idx, (actual_line, reference_line) in enumerate(
            zip(actual_lines, reference_lines)
        ):
            if frame_idx < valid.shape[1] and valid[env_idx, frame_idx]:
                actual = lcfs[env_idx, frame_idx]
                reference = reference_lcfs[env_idx, frame_idx]
                actual_line.set_data(actual[:, 0], actual[:, 1])
                reference_line.set_data(reference[:, 0], reference[:, 1])
            else:
                actual_line.set_data([], [])
                reference_line.set_data([], [])
            artists.extend([actual_line, reference_line])
        if xpoint_data is not None:
            rX_arr, zX_arr, nX_arr = xpoint_data
            for env_idx, xline in enumerate(xpoint_lines):
                if frame_idx < valid.shape[1] and valid[env_idx, frame_idx]:
                    nx = int(round(float(nX_arr[env_idx, frame_idx])))
                    nx = max(0, min(nx, rX_arr.shape[-1]))
                    if nx > 0:
                        xline.set_data(
                            rX_arr[env_idx, frame_idx, :nx],
                            zX_arr[env_idx, frame_idx, :nx],
                        )
                    else:
                        xline.set_data([], [])
                else:
                    xline.set_data([], [])
                artists.append(xline)
        if slot_data is not None:
            xs_r = slot_data["x_slot_r"]
            xs_z = slot_data["x_slot_z"]
            xvm = slot_data["x_valid"]
            st_r = slot_data["strike_r"]
            st_z = slot_data["strike_z"]
            svm = slot_data["strike_valid"]
            ref_xr = slot_data["ref_rX"]
            ref_xz = slot_data["ref_zX"]
            ref_sr = slot_data["ref_strike_r"]
            ref_sz = slot_data["ref_strike_z"]
            for env_idx in range(num_envs):
                frame_ok = frame_idx < valid.shape[1] and bool(
                    valid[env_idx, frame_idx]
                )
                sv = svm[env_idx, frame_idx] > 0.5 if frame_ok else None
                if frame_ok:
                    strike_markers[env_idx].set_data(
                        np.where(sv, st_r[env_idx, frame_idx], np.nan),
                        np.where(sv, st_z[env_idx, frame_idx], np.nan),
                    )
                else:
                    strike_markers[env_idx].set_data([], [])
                artists.append(strike_markers[env_idx])
                for slot in range(4):
                    line = xslot_line_groups[env_idx][slot]
                    if frame_ok and xvm[env_idx, frame_idx, slot] > 0.5:
                        line.set_data(
                            [
                                xs_r[env_idx, frame_idx, slot],
                                ref_xr[env_idx, frame_idx, slot],
                            ],
                            [
                                xs_z[env_idx, frame_idx, slot],
                                ref_xz[env_idx, frame_idx, slot],
                            ],
                        )
                    else:
                        line.set_data([], [])
                    artists.append(line)
                for slot in range(8):
                    line = strike_line_groups[env_idx][slot]
                    if frame_ok and sv[slot]:
                        line.set_data(
                            [
                                st_r[env_idx, frame_idx, slot],
                                ref_sr[env_idx, frame_idx, slot],
                            ],
                            [
                                st_z[env_idx, frame_idx, slot],
                                ref_sz[env_idx, frame_idx, slot],
                            ],
                        )
                    else:
                        line.set_data([], [])
                    artists.append(line)
        for env_idx, (ip_line, ref_ip_line) in enumerate(zip(ip_lines, ref_ip_lines)):
            if env_idx < ip_values.shape[0]:
                if frame_idx < ip_values.shape[1] and valid[env_idx, frame_idx]:
                    valid_steps = current_step
                else:
                    valid_mask = valid[env_idx, : min(frame_idx + 1, valid.shape[1])]
                    if np.any(valid_mask):
                        valid_steps = int(np.max(np.where(valid_mask)[0])) + 1
                    else:
                        valid_steps = 0
                if valid_steps > 0:
                    steps = np.arange(1, valid_steps + 1)
                    ip_line.set_data(steps, ip_values[env_idx, :valid_steps])
                    ref_ip_line.set_data(
                        steps, reference_ip_values[env_idx, :valid_steps]
                    )
                else:
                    ip_line.set_data([], [])
                    ref_ip_line.set_data([], [])
            else:
                ip_line.set_data([], [])
                ref_ip_line.set_data([], [])
            artists.extend([ip_line, ref_ip_line])
        for ax, lines, series_dict, name in zip(
            score_axes, score_lines, score_series_list, score_names
        ):
            all_totals = []
            all_means = []
            all_counts = []
            for env_idx in range(num_envs):
                series = series_dict.get(env_idx, np.array([]))
                total, mean, count = _compute_stats(series, current_step)
                all_totals.append(total)
                all_means.append(mean)
                all_counts.append(count)
                if len(series) > 0:
                    show_steps = min(current_step, len(series))
                    steps = np.arange(1, show_steps + 1)
                    vals = series[:show_steps]
                    lines[env_idx].set_data(steps, vals)
                else:
                    lines[env_idx].set_data([], [])
            avg_total = np.mean(all_totals) if all_totals else 0.0
            avg_mean = np.mean(all_means) if all_means else 0.0
            avg_count = np.mean(all_counts) if all_counts else 0
            ax.set_title(
                f"{name}\nlen={avg_count:.0f}, total={avg_total:.3f}", fontsize=9, pad=8
            )
            artists.extend(lines)
        ip_error_exceed_count = 0
        for env_idx in range(num_envs):
            series = ip_abs_error_series.get(env_idx, np.array([]))
            ip_error_exceed_count += _count_exceed_threshold(
                series, current_step, 50000
            )
            if len(series) > 0:
                show_steps = min(current_step, len(series))
                steps = np.arange(1, show_steps + 1)
                vals = series[:show_steps]
                metric_lines[0][env_idx].set_data(steps, vals)
            else:
                metric_lines[0][env_idx].set_data([], [])
        ax_ip_error.set_title(
            f"ip_abs_error\nexceed 50kA: {ip_error_exceed_count}", fontsize=9, pad=8
        )
        artists.extend(metric_lines[0])
        first_exceed_steps = []
        for env_idx in range(num_envs):
            series = coil_margin_series.get(env_idx, np.array([]))
            first_exceed = _first_exceed_step(series, 1.0)
            if first_exceed > 0:
                first_exceed_steps.append(first_exceed)
            if len(series) > 0:
                show_steps = min(current_step, len(series))
                steps = np.arange(1, show_steps + 1)
                vals = series[:show_steps]
                metric_lines[1][env_idx].set_data(steps, vals)
            else:
                metric_lines[1][env_idx].set_data([], [])
        first_exceed_info = f"first exceed: {(min(first_exceed_steps) if first_exceed_steps else 'N/A')}"
        ax_coil_margin.set_title(f"coil_margin\n{first_exceed_info}", fontsize=9, pad=8)
        artists.extend(metric_lines[1])
        mask_05_count = 0
        mask_total_count = 0
        for env_idx in range(num_envs):
            series = topology_mask_series.get(env_idx, np.array([]))
            mask_05_count += _count_mask_value(series, current_step, 0.5)
            mask_total_count += current_step
            if len(series) > 0:
                show_steps = min(current_step, len(series))
                steps = np.arange(1, show_steps + 1)
                vals = series[:show_steps]
                metric_lines[2][env_idx].set_data(steps, vals)
            else:
                metric_lines[2][env_idx].set_data([], [])
        mask_05_ratio = (
            mask_05_count / mask_total_count if mask_total_count > 0 else 0.0
        )
        ax_topology_mask.set_title(
            f"topology_mask\nmask=0.5: {mask_05_ratio:.1%}", fontsize=9, pad=8
        )
        artists.extend(metric_lines[2])
        for ax, lines, series_dict, name in zip(
            extra_axes, extra_lines_list, f2_score_series_list, f2_score_names
        ):
            all_means = []
            for env_idx in range(num_envs):
                series = series_dict.get(env_idx, np.array([]))
                _, mean, _ = _compute_stats(series, current_step)
                all_means.append(mean)
                if len(series) > 0:
                    show_steps = min(current_step, len(series))
                    steps = np.arange(1, show_steps + 1)
                    vals = series[:show_steps]
                    lines[env_idx].set_data(steps, vals)
                else:
                    lines[env_idx].set_data([], [])
            avg_mean = np.mean(all_means) if all_means else 0.0
            ax.set_title(f"{name}\nmean={avg_mean:.3f}", fontsize=9, pad=8)
            artists.extend(lines)
        return artists

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=last_step, blit=False)
    writer = animation.FFMpegWriter(fps=int(fps))
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return True


def visualize(args: argparse.Namespace) -> int:
    run_dir = ROOT / "results" / args.run_name
    csv_path = run_dir / "eval" / args.csv_name
    observations_path = run_dir / "eval" / args.observations_name
    figures_dir = run_dir / "eval" / "figures"
    if not csv_path.exists():
        raise FileNotFoundError(f"rollout csv not found: {csv_path}")
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"rollout csv is empty: {csv_path}")
    groups = _group_rows(rows)
    components = _component_columns(rows)
    competition_score_series = _extract_component_series(groups, "competition_score")
    episode_steps, task_total = PLATFORM_SCORE_CFG["F2"]
    print(
        f"\n[competition_score totals per episode] (task=F2, platform total={task_total:.0f})"
    )
    for env_idx, series in sorted(competition_score_series.items()):
        valid_vals = series[np.isfinite(series)]
        total = float(np.sum(valid_vals)) if len(valid_vals) > 0 else 0.0
        mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
        competition_fraction_score = total / episode_steps * task_total
        count = len(valid_vals)
        print(
            f"  episode {env_idx}: total={total:.4f}, mean={mean:.4f}, steps={count}, platform score={competition_fraction_score:.2f}/{task_total:.0f}"
        )
    if args.components:
        wanted = {
            component.strip()
            for component in args.components.split(",")
            if component.strip()
        }
        components = [component for component in components if component in wanted]
    if not components:
        raise ValueError("no reward component columns found to plot")
    saved = 0
    for component in components:
        out_path = figures_dir / f"{_safe_filename(component)}.png"
        if _plot_component(component, groups, out_path):
            saved += 1
    if observations_path.exists():
        with np.load(observations_path, allow_pickle=False) as data:
            observations = {key: data[key] for key in data.files}
        if _plot_ip_comparison(observations, figures_dir / "ip_vs_reference.png"):
            saved += 1
        if _save_lcfs_video(
            observations,
            groups,
            figures_dir / "lcfs_rollout.mp4",
            fps=int(args.fps),
            psi_env=int(args.psi_env),
        ):
            saved += 1
    else:
        print(
            f"warning: rollout observations not found, skipped LCFS/video plots: {observations_path}"
        )
    print(f"loaded: {csv_path}")
    print(f"saved {saved} figures to: {figures_dir}")
    return 0


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--csv-name", type=str, default="step_reward_components.csv")
    parser.add_argument(
        "--observations-name", type=str, default="rollout_observations.npz"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--psi-env",
        type=int,
        default=0,
        help="Rollout index whose psi (Fx) field is shown as background in the LCFS plot.",
    )
    parser.add_argument(
        "--components",
        type=str,
        default=None,
        help="Optional comma-separated component names to plot.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize rollout reward components.")
    add_args(parser)
    return visualize(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
