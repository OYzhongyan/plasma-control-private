#!/usr/bin/env python3
"""Visualize stage0 reward and monitor logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = {"f2b_direct": Path("results/f2b_direct")}


def read_rows(path: Path, comment: str | None = None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        source = (line for line in f if not line.startswith(comment)) if comment else f
        reader = csv.DictReader(source)
        for row in reader:
            out: dict[str, float] = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except Exception:
                    out[key] = np.nan
            rows.append(out)
    return rows


def split_monotonic_runs(
    rows: list[dict[str, float]],
    key: str = "timesteps",
) -> list[list[dict[str, float]]]:
    if not rows:
        return []
    runs = [[rows[0]]]
    for row in rows[1:]:
        if row[key] < runs[-1][-1][key]:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) == 0:
        return y
    window = max(1, min(window, len(y)))
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def safe_col(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


def binned_monitor(
    rows: list[dict[str, float]],
    *,
    max_steps: int,
    bin_size: int,
) -> dict[str, np.ndarray]:
    out = {key: [] for key in ["step", "ep_len_mean", "ep_reward_mean", "full_rate", "episodes"]}
    if not rows:
        return {key: np.asarray(value, dtype=float) for key, value in out.items()}

    cumulative = 0.0
    enriched = []
    for row in rows:
        cumulative += float(row["l"])
        item = dict(row)
        item["cum_steps"] = cumulative
        enriched.append(item)

    previous = 0.0
    for end in np.arange(bin_size, cumulative + bin_size, bin_size):
        bucket = [row for row in enriched if previous < row["cum_steps"] <= end]
        previous = float(end)
        if not bucket:
            continue
        lengths = np.asarray([row["l"] for row in bucket], dtype=float)
        rewards = np.asarray([row["r"] for row in bucket], dtype=float)
        out["step"].append(float(end))
        out["ep_len_mean"].append(float(np.mean(lengths)))
        out["ep_reward_mean"].append(float(np.mean(rewards)))
        out["full_rate"].append(float(np.mean(lengths >= max_steps)))
        out["episodes"].append(float(len(bucket)))
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def infer_max_steps(stage_name: str) -> int:
    return 200 if "stage0b" in stage_name else 100


def plot_stage(
    *,
    stage_name: str,
    stage_dir: Path,
    output_dir: Path,
    bin_size: int,
) -> dict[str, Any]:
    reward_path = stage_dir / "logs" / "reward_components.csv"
    monitor_path = stage_dir / "monitor.csv"
    reward_all = read_rows(reward_path)
    reward_runs = split_monotonic_runs(reward_all)
    reward = reward_runs[-1] if reward_runs else []
    monitor = read_rows(monitor_path, comment="#") if monitor_path.exists() else []
    max_steps = infer_max_steps(stage_name)
    x = safe_col(reward, "timesteps")
    smooth_window = max(5, min(50, len(reward) // 25 if reward else 5))
    monitor_bins = binned_monitor(monitor, max_steps=max_steps, bin_size=bin_size)

    fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=False)
    fig.suptitle(f"{stage_name} training curves", fontsize=15)

    ax = axes[0]
    for key, label in [
        ("reward", "reward"),
        ("failure_penalty", "failure penalty"),
        ("ip_error_boundary_penalty", "Ip boundary penalty"),
    ]:
        y = safe_col(reward, key)
        if len(y):
            ax.plot(x, rolling_mean(y, smooth_window), label=label)
    ax.set_ylabel("reward / penalty")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    for key, label in [
        ("ip_score", "Ip"),
        ("r_score", "R"),
        ("z_score", "Z"),
        ("lcfs_absolute_score", "LCFS"),
        ("topology_score", "topology"),
    ]:
        y = safe_col(reward, key)
        if len(y):
            ax.plot(x, rolling_mean(y, smooth_window), label=label)
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=5)

    ax = axes[2]
    if len(reward):
        ax.plot(
            x,
            rolling_mean(safe_col(reward, "ip_abs_error") / 1000.0, smooth_window),
            label="Ip abs error (kA)",
        )
        ax2 = ax.twinx()
        for key, label in [
            ("coil_penalty", "coil penalty"),
            ("action_l2", "action l2"),
            ("action_smooth", "action smooth"),
        ]:
            ax2.plot(x, rolling_mean(safe_col(reward, key), smooth_window), linestyle="--", label=label)
        ax2.set_ylabel("penalty/action scale")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_ylabel("Ip error kA")
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    if len(monitor_bins["step"]):
        ax.plot(monitor_bins["step"], monitor_bins["ep_len_mean"], label="mean episode length")
        ax.axhline(max_steps, color="gray", linewidth=1, linestyle=":", label=f"max_steps={max_steps}")
        ax.set_ylabel("episode length")
        ax2 = ax.twinx()
        ax2.plot(
            monitor_bins["step"],
            monitor_bins["full_rate"] * 100.0,
            color="tab:green",
            linestyle="--",
            label="full episode rate (%)",
        )
        ax2.set_ylabel("full rate %")
        ax2.set_ylim(-2, 102)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_xlabel("timesteps")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    training_plot = output_dir / f"{stage_name}_training_curves.png"
    fig.savefig(training_plot, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    if len(monitor_bins["step"]):
        ax.plot(monitor_bins["step"], monitor_bins["ep_len_mean"], label="mean episode length")
        ax.fill_between(monitor_bins["step"], monitor_bins["ep_len_mean"], alpha=0.15)
        ax.axhline(max_steps, color="gray", linewidth=1, linestyle=":", label=f"max_steps={max_steps}")
        ax.set_ylim(0, max_steps * 1.08)
    ax.set_title(f"{stage_name} average episode length by {bin_size // 1000}k timesteps")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode length")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    length_plot = output_dir / f"{stage_name}_episode_length.png"
    fig.savefig(length_plot, dpi=160)
    plt.close(fig)

    tail = reward[-max(1, min(50, len(reward))):]
    mon_tail = monitor[-max(1, min(1000, len(monitor))):]
    mon_lengths = np.asarray([row["l"] for row in mon_tail], dtype=float) if mon_tail else np.array([])
    mon_rewards = np.asarray([row["r"] for row in mon_tail], dtype=float) if mon_tail else np.array([])

    def mean_tail(key: str) -> float:
        return float(np.nanmean(safe_col(tail, key))) if tail else float("nan")

    summary: dict[str, Any] = {
        "stage": stage_name,
        "reward_rows_total": len(reward_all),
        "reward_runs_detected": len(reward_runs),
        "reward_rows_used": len(reward),
        "reward_start_timestep": float(reward[0]["timesteps"]) if reward else None,
        "reward_end_timestep": float(reward[-1]["timesteps"]) if reward else None,
        "last50_reward": mean_tail("reward"),
        "last50_ip_abs_error_ka": mean_tail("ip_abs_error") / 1000.0,
        "last50_ip_score": mean_tail("ip_score"),
        "last50_lcfs_score": mean_tail("lcfs_absolute_score"),
        "last50_r_score": mean_tail("r_score"),
        "last50_z_score": mean_tail("z_score"),
        "last50_ip_boundary_penalty": mean_tail("ip_error_boundary_penalty"),
        "last50_coil_penalty": mean_tail("coil_penalty"),
        "last50_action_l2": mean_tail("action_l2"),
        "last50_action_smooth": mean_tail("action_smooth"),
        "monitor_episodes": len(monitor),
        "monitor_total_steps": float(sum(row["l"] for row in monitor)) if monitor else 0.0,
        "last1000_ep_len_mean": float(np.mean(mon_lengths)) if len(mon_lengths) else None,
        "last1000_full_rate": float(np.mean(mon_lengths >= max_steps)) if len(mon_lengths) else None,
        "last1000_ep_reward_mean": float(np.mean(mon_rewards)) if len(mon_rewards) else None,
        "plot_training_curves": str(training_plot),
        "plot_episode_length": str(length_plot),
    }

    if reward:
        for metric in ["reward", "lcfs_absolute_score", "ip_score"]:
            y = safe_col(reward, metric)
            smoothed = rolling_mean(y, smooth_window)
            idx = int(np.nanargmax(smoothed))
            summary[f"best_{metric}_smooth_timestep"] = float(x[idx])
            summary[f"best_{metric}_smooth_value"] = float(smoothed[idx])
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--bin-size", type=int, default=50_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        plot_stage(stage_name=name, stage_dir=stage_dir, output_dir=args.output_dir, bin_size=args.bin_size)
        for name, stage_dir in DEFAULT_RUNS.items()
    ]
    write_summary_csv(args.output_dir / "stage0_reward_visualization_summary.csv", summaries)
    (args.output_dir / "stage0_reward_visualization_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for item in summaries:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
