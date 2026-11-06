#!/usr/bin/env python3
"""Analyze PPO monitor and reward-component logs for one training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_numeric_csv(path: Path, *, comment: str | None = None) -> pd.DataFrame:
    if comment is not None:
        df = pd.read_csv(path, comment=comment)
    else:
        rows: list[dict[str, str | int]] = []
        fieldnames: list[str] = []
        current_header: list[str] | None = None
        section = -1
        row_index = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            for raw in csv.reader(f):
                if not raw:
                    continue
                if raw[0].startswith("#"):
                    continue
                if raw[0] == "timesteps":
                    current_header = raw
                    section += 1
                    for name in raw:
                        if name not in fieldnames:
                            fieldnames.append(name)
                    continue
                if current_header is None:
                    continue
                row = {name: raw[i] for i, name in enumerate(current_header) if i < len(raw)}
                row["_log_section"] = section
                row["_row_index"] = row_index
                row_index += 1
                rows.append(row)
        df = pd.DataFrame(rows, columns=fieldnames + ["_log_section", "_row_index"])
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _rolling_window(length: int, *, minimum: int, maximum: int, divisor: int) -> int:
    return max(minimum, min(maximum, length // divisor or minimum))


def analyze_run(run_dir: Path, out_dir: Path | None = None, *, title: str | None = None) -> dict:
    run_dir = Path(run_dir)
    reward_path = run_dir / "logs" / "reward_components.csv"
    monitor_path = run_dir / "monitor.csv"
    if not reward_path.is_file():
        raise FileNotFoundError(f"reward log not found: {reward_path}")
    if not monitor_path.is_file():
        raise FileNotFoundError(f"monitor log not found: {monitor_path}")

    out_dir = Path(out_dir) if out_dir is not None else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = title or run_dir.name
    max_steps = 500 if "f2" in run_dir.name.lower() else 300

    reward = _read_numeric_csv(reward_path)
    monitor = _read_numeric_csv(monitor_path, comment="#")
    reward = reward.dropna(subset=["timesteps"]).sort_values("_row_index").reset_index(drop=True)
    monitor = monitor.dropna(subset=["l", "r"]).reset_index(drop=True)
    monitor["episode_index"] = np.arange(1, len(monitor) + 1)
    monitor["cumulative_episode_steps"] = monitor["l"].cumsum()

    reward_window = _rolling_window(len(reward), minimum=5, maximum=51, divisor=80)
    monitor_window = _rolling_window(len(monitor), minimum=20, maximum=401, divisor=60)

    reward_cols = [
        "reward",
        "ip_abs_error",
        "ip_score",
        "lcfs_cm",
        "lcfs_initial_cm",
        "lcfs_progress_score",
        "lcfs_precision_score",
        "lcfs_regression_penalty",
        "lcfs_absolute_score",
        "failure",
        "failure_penalty",
        "coil_penalty",
        "action_l2",
        "action_smooth",
        "topology_score",
        "x_score",
        "nx_score",
        "x_count_score",
        "x_position_cm",
        "x_position_initial_cm",
        "x_position_component",
        "x_position_regression_penalty",
        "x_flux_error",
        "x_flux_component",
        "psix_score",
    ]
    for col in reward_cols:
        if col in reward.columns:
            reward[f"{col}_roll"] = reward[col].rolling(reward_window, min_periods=1).mean()
    for col in ["r", "l"]:
        monitor[f"{col}_roll"] = monitor[col].rolling(monitor_window, min_periods=1).mean()

    reward["segment"] = pd.cut(
        reward["timesteps"], bins=10, labels=False, include_lowest=True
    )
    reward_summary = reward.groupby("segment", dropna=True).agg(
        step_start=("timesteps", "min"),
        step_end=("timesteps", "max"),
        n=("timesteps", "size"),
        reward_mean=("reward", "mean"),
        failure_rate=("failure", "mean"),
        ip_abs_error_mean=("ip_abs_error", "mean"),
        ip_score_mean=("ip_score", "mean"),
        lcfs_cm_mean=("lcfs_cm", "mean"),
        lcfs_cm_min=("lcfs_cm", "min"),
        lcfs_cm_last=("lcfs_cm", "last"),
        lcfs_abs_score_mean=("lcfs_absolute_score", "mean"),
        lcfs_progress_mean=("lcfs_progress_score", "mean"),
        lcfs_precision_mean=("lcfs_precision_score", "mean"),
        lcfs_regression_penalty_mean=("lcfs_regression_penalty", "mean"),
        coil_penalty_mean=("coil_penalty", "mean"),
        action_l2_mean=("action_l2", "mean"),
        action_smooth_mean=("action_smooth", "mean"),
    ).reset_index(drop=True)
    reward_summary.to_csv(out_dir / "reward_segment_summary.csv", index=False)

    section_summary = reward.groupby("_log_section", dropna=True).agg(
        row_start=("_row_index", "min"),
        row_end=("_row_index", "max"),
        timestep_start=("timesteps", "first"),
        timestep_end=("timesteps", "last"),
        n=("timesteps", "size"),
        reward_mean=("reward", "mean"),
        reward_last=("reward", "last"),
        lcfs_cm_mean=("lcfs_cm", "mean"),
        lcfs_cm_last=("lcfs_cm", "last"),
        x_position_cm_mean=("x_position_cm", "mean"),
        x_position_cm_last=("x_position_cm", "last"),
        ip_abs_error_mean=("ip_abs_error", "mean"),
        failure_rate=("failure", "mean"),
    ).reset_index()
    section_summary.to_csv(out_dir / "reward_log_section_summary.csv", index=False)

    monitor["segment"] = pd.cut(
        monitor["cumulative_episode_steps"], bins=10, labels=False, include_lowest=True
    )
    monitor_summary = monitor.groupby("segment", dropna=True).agg(
        approx_step_start=("cumulative_episode_steps", "min"),
        approx_step_end=("cumulative_episode_steps", "max"),
        n=("l", "size"),
        ep_len_mean=("l", "mean"),
        ep_len_median=("l", "median"),
        ep_len_max=("l", "max"),
        ep_len_p90=("l", lambda s: float(np.percentile(s, 90))),
        ep_rew_mean=("r", "mean"),
        ep_rew_median=("r", "median"),
        ep_rew_max=("r", "max"),
    ).reset_index(drop=True)
    monitor_summary.to_csv(out_dir / "monitor_segment_summary.csv", index=False)

    first = reward.iloc[: max(1, len(reward) // 10)]
    last = reward.iloc[-max(1, len(reward) // 10) :]
    last_monitor = monitor.iloc[-max(1, len(monitor) // 10) :] if len(monitor) else monitor
    summary = {
        "run_dir": str(run_dir),
        "reward_rows": int(len(reward)),
        "reward_log_sections": int(reward["_log_section"].nunique())
        if "_log_section" in reward.columns
        else 1,
        "monitor_rows": int(len(monitor)),
        "max_reward_timestep": int(reward["timesteps"].max()) if len(reward) else 0,
        "monitor_cumulative_episode_steps": int(monitor["cumulative_episode_steps"].max())
        if len(monitor)
        else 0,
        "first_10pct_lcfs_cm_mean": float(first["lcfs_cm"].mean()),
        "last_10pct_lcfs_cm_mean": float(last["lcfs_cm"].mean()),
        "first_10pct_lcfs_score_mean": float(first["lcfs_absolute_score"].mean()),
        "last_10pct_lcfs_score_mean": float(last["lcfs_absolute_score"].mean()),
        "first_10pct_ip_abs_error_mean": float(first["ip_abs_error"].mean()),
        "last_10pct_ip_abs_error_mean": float(last["ip_abs_error"].mean()),
        "last_10pct_reward_mean": float(last["reward"].mean()),
        "last_10pct_failure_rate": float(last["failure"].mean()),
        "last_10pct_regression_penalty_mean": float(last["lcfs_regression_penalty"].mean()),
        "best_lcfs_cm": float(reward["lcfs_cm"].min()),
        "best_lcfs_timestep": int(reward.loc[reward["lcfs_cm"].idxmin(), "timesteps"]),
        "best_lcfs_score": float(reward["lcfs_absolute_score"].max()),
        "best_lcfs_score_timestep": int(
            reward.loc[reward["lcfs_absolute_score"].idxmax(), "timesteps"]
        ),
        "best_monitor_ep_len": float(monitor["l"].max()) if len(monitor) else 0.0,
        "last_10pct_ep_len_mean": float(last_monitor["l"].mean()) if len(monitor) else 0.0,
        "last_10pct_ep_rew_mean": float(last_monitor["r"].mean()) if len(monitor) else 0.0,
    }
    for key in [
        "x_score",
        "nx_score",
        "x_count_score",
        "x_position_cm",
        "x_position_component",
        "x_flux_error",
        "x_flux_component",
        "psix_score",
    ]:
        if key in reward.columns:
            summary[f"last_10pct_{key}_mean"] = float(last[key].mean())
            summary[f"best_{key}"] = float(
                reward[key].min() if key.endswith("_cm") or key.endswith("_error") else reward[key].max()
            )
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    axes[0].plot(reward["timesteps"], reward["reward_roll"], label="reward rolling mean")
    axes[0].axhline(0, color="#555", linewidth=0.8)
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="best")
    axes[1].plot(reward["timesteps"], reward["lcfs_cm_roll"], label="LCFS RMS cm")
    axes[1].plot(
        reward["timesteps"],
        reward["lcfs_initial_cm_roll"],
        label="initial LCFS cm",
        linestyle="--",
    )
    axes[1].axhline(3.0, color="#2ca02c", linewidth=1.0, linestyle=":", label="3 cm")
    axes[1].set_ylabel("cm")
    axes[1].legend(loc="best")
    axes[2].plot(reward["timesteps"], reward["lcfs_progress_score_roll"], label="progress")
    axes[2].plot(reward["timesteps"], reward["lcfs_precision_score_roll"], label="precision")
    axes[2].plot(
        reward["timesteps"], reward["lcfs_absolute_score_roll"], label="combined LCFS"
    )
    axes[2].plot(
        reward["timesteps"],
        reward["lcfs_regression_penalty_roll"],
        label="regression penalty",
    )
    axes[2].set_ylabel("score / penalty")
    axes[2].legend(loc="best")
    axes[3].plot(
        reward["timesteps"], reward["ip_abs_error_roll"] / 1000.0, label="Ip abs error kA"
    )
    axes[3].plot(reward["timesteps"], reward["ip_score_roll"], label="Ip score")
    axes[3].set_ylabel("kA / score")
    axes[3].set_xlabel("training timesteps")
    axes[3].legend(loc="best")
    fig.suptitle(f"{title}: reward components", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "reward_components_overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    axes[0].plot(monitor["episode_index"], monitor["l"], color="#cccccc", linewidth=0.5)
    axes[0].plot(
        monitor["episode_index"],
        monitor["l_roll"],
        color="#d62728",
        label=f"rolling mean ({monitor_window})",
    )
    axes[0].axhline(max_steps, color="#2ca02c", linestyle=":", label=f"max_steps={max_steps}")
    axes[0].set_ylabel("episode length")
    axes[0].legend(loc="best")
    axes[1].plot(monitor["episode_index"], monitor["r"], color="#dddddd", linewidth=0.5)
    axes[1].plot(
        monitor["episode_index"],
        monitor["r_roll"],
        color="#1f77b4",
        label=f"rolling mean ({monitor_window})",
    )
    axes[1].set_ylabel("episode reward")
    axes[1].legend(loc="best")
    axes[2].scatter(monitor["l"], monitor["r"], s=4, alpha=0.25, color="#9467bd")
    axes[2].set_xlabel("episode length")
    axes[2].set_ylabel("episode reward")
    axes[2].set_title("episode reward vs survival length")
    fig.suptitle(f"{title}: monitor", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "monitor_overview.png", dpi=160)
    plt.close(fig)

    xpt_cols = {"x_position_cm", "x_position_component", "x_flux_error", "x_flux_component"}
    if xpt_cols.issubset(set(reward.columns)):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        axes[0].plot(reward["timesteps"], reward["x_position_cm_roll"], label="X position cm")
        axes[0].plot(
            reward["timesteps"],
            reward["x_position_initial_cm_roll"],
            label="initial X position cm",
            linestyle="--",
        )
        axes[0].axhline(3.0, color="#2ca02c", linewidth=1.0, linestyle=":", label="3 cm")
        axes[0].set_ylabel("cm")
        axes[0].legend(loc="best")
        axes[1].plot(reward["timesteps"], reward["x_position_component_roll"], label="X pos component")
        axes[1].plot(reward["timesteps"], reward["x_flux_component_roll"], label="X flux component")
        if "x_count_score_roll" in reward.columns:
            axes[1].plot(reward["timesteps"], reward["x_count_score_roll"], label="X count")
        if "nx_score_roll" in reward.columns:
            axes[1].plot(reward["timesteps"], reward["nx_score_roll"], label="nX")
        if "x_score_roll" in reward.columns:
            axes[1].plot(reward["timesteps"], reward["x_score_roll"], label="x_score")
        axes[1].set_ylabel("score/component")
        axes[1].legend(loc="best")
        axes[2].plot(reward["timesteps"], reward["x_flux_error_roll"], label="X flux error")
        if "psix_score_roll" in reward.columns:
            axes[2].plot(reward["timesteps"], reward["psix_score_roll"], label="psiX score")
        if "x_position_regression_penalty_roll" in reward.columns:
            axes[2].plot(
                reward["timesteps"],
                reward["x_position_regression_penalty_roll"],
                label="X position regression penalty",
            )
        axes[2].set_ylabel("flux / penalty")
        axes[2].set_xlabel("training timesteps")
        axes[2].legend(loc="best")
        fig.suptitle(f"{title}: F2 XPT components", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_dir / "f2_xpt_components.png", dpi=160)
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(reward["timesteps"], reward["lcfs_cm_roll"], color="#d62728", label="LCFS cm")
    ax1.set_xlabel("training timesteps")
    ax1.set_ylabel("LCFS RMS cm", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.axhline(3.0, color="#2ca02c", linestyle=":", linewidth=1)
    ax2 = ax1.twinx()
    mon_x = monitor["cumulative_episode_steps"].to_numpy(dtype=float)
    mon_y = monitor["l_roll"].to_numpy(dtype=float)
    if len(mon_x) > 1:
        interp_len = np.interp(reward["timesteps"].to_numpy(dtype=float), mon_x, mon_y)
        ax2.plot(
            reward["timesteps"],
            interp_len,
            color="#1f77b4",
            label="episode length rolling",
        )
    ax2.set_ylabel("episode length", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"{title}: LCFS improvement vs survival trend", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "lcfs_vs_survival.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    summary = analyze_run(args.run_dir, args.out_dir, title=args.title)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
