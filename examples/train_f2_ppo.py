"""Thin F2 entrypoint for the modular semifinal PPO trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_ppo import main as train_main  # noqa: E402


def main() -> int:
    return train_main(["--task", "F2", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
