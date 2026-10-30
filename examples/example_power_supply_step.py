"""Plot fused power-supply step response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.power_supply import PowerSupplyModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Power supply step-response demo")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "power_supply_step_response.png",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dt = 0.001
    duration_s = 0.08
    n = int(duration_s / dt)
    t = np.arange(n) * dt

    u_low, u_high = 0.0, 100.0
    step_at = 0.01
    u_set = np.where(t >= step_at, u_high, u_low)

    model = PowerSupplyModel(
        slopes=np.ones(12),
        intercepts=np.zeros(12),
        max_change_per_step=np.full(12, np.inf),
        delay_s=np.full(12, 0.0035),
        seed=0,
    )
    model.reset()
    u_real = np.array([model.step(np.full(12, u))[0] for u in u_set])

    delay_s = 0.0035
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(t * 1000, u_set, "C0--", linewidth=1.5, label=r"$U_{set}$")
    ax0.plot(t * 1000, u_real, "C3-", linewidth=2.0, label=r"$U_{real}$")
    ax0.axvline(step_at * 1000, color="gray", linestyle=":", alpha=0.6)
    ax0.axvline(
        (step_at + delay_s) * 1000,
        color="C2",
        linestyle="--",
        alpha=0.7,
        label=f"delay L={delay_s * 1000:.1f} ms",
    )
    ax0.set_ylabel("Voltage (arb. unit)")
    ax0.set_title(r"Delay $\rightarrow$ rate limit $\rightarrow$ PSM (identity PSM shown)")
    ax0.legend(loc="lower right")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t * 1000, u_real - u_set, "C4-", linewidth=1.5, label=r"$U_{real} - U_{set}$")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Tracking error")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
