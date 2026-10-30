"""Reference loading and trajectory construction for semifinal tasks."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TargetReference:
    """Static target shape loaded from the public reference JSON files."""

    lcfs: np.ndarray
    ip_final: float
    rX: np.ndarray | None = None
    zX: np.ndarray | None = None
    strike_r: np.ndarray | None = None
    strike_z: np.ndarray | None = None
    x_valid: np.ndarray | None = None
    strike_valid: np.ndarray | None = None

    @property
    def has_xpt(self) -> bool:
        return self.rX is not None and self.zX is not None


def default_reference_path() -> Path:
    name = "xpt_reference_targets.json"
    return ROOT / "configs" / name


def _load_public_target_reference(data: dict[str, Any]) -> TargetReference:
    target = next((data[k] for k in ("F2b", "F2a", "F2") if k in data), None)
    if target is None:
        raise ValueError("target JSON has no F2b target")
    lcfs = np.asarray(target["lcfs_points"], dtype=np.float64)
    ip_values = np.asarray(target.get("Ip", []), dtype=np.float64).reshape(-1)
    ip_final = float(ip_values[-1]) if ip_values.size else 400000.0
    x_main = np.asarray(target["Xpt_main"], dtype=np.float64)
    x_sec = np.asarray(target["Xpt_sec"], dtype=np.float64)
    strike = np.asarray(target["strike"], dtype=np.float64)
    return TargetReference(
        lcfs=lcfs,
        ip_final=ip_final,
        rX=np.asarray(
            [x_sec[0, 0], x_main[0, 0], x_main[1, 0], x_sec[1, 0]], dtype=np.float64
        ),
        zX=np.asarray(
            [x_sec[0, 1], x_main[0, 1], x_main[1, 1], x_sec[1, 1]], dtype=np.float64
        ),
        strike_r=strike[:, 0],
        strike_z=strike[:, 1],
        x_valid=np.ones(4, dtype=np.float64),
        strike_valid=np.ones(8, dtype=np.float64),
    )


def load_target_reference(path: str | Path) -> TargetReference:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "reference_lcfs_points" not in data:
        return _load_public_target_reference(data)
    return TargetReference(
        lcfs=np.asarray(data["reference_lcfs_points"], dtype=np.float64),
        ip_final=float(data["reference_Ip_final"]),
        **{
            key: np.asarray(data["reference_" + key], dtype=np.float64)
            for key in ("rX", "zX", "strike_r", "strike_z", "x_valid", "strike_valid")
        },
    )


def _target_reference(target: TargetReference, max_steps: int) -> dict[str, Any]:
    if int(max_steps) <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    reference = {
        "Ip": _ip_reference(int(max_steps), target.ip_final).tolist(),
        "lX": 1.0,
        "lcfs_points": target.lcfs.tolist(),
        "nX": 4,
        "n_strike": 8,
    }
    for key in ("rX", "zX", "strike_r", "strike_z", "x_valid", "strike_valid"):
        value = getattr(target, key)
        if value is not None:
            reference[key] = np.asarray(value).tolist()
    return reference


def build_reference(source: str | Path | TargetReference, max_steps: int | None = None):
    """Build a trajectory from a static target, or a list of trajectories from a JSON path."""
    steps = 500 if max_steps is None else int(max_steps)
    if steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if isinstance(source, TargetReference):
        return _target_reference(source, steps)
    data = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "reference_lcfs_points" not in data:
        return [_target_reference(_load_public_target_reference(data), steps)]
    targets = data if isinstance(data, list) else [data]
    references = []
    for target in targets:
        reference = {
            "Ip": _ip_reference(steps, float(target["reference_Ip_final"])).tolist(),
            "lX": 1.0,
            "lcfs_points": target["reference_lcfs_points"],
        }
        for key in (
            "rX",
            "zX",
            "strike_r",
            "strike_z",
            "x_valid",
            "strike_valid",
            "nX",
            "n_strike",
        ):
            reference[key] = target["reference_" + key]
        references.append(reference)
    return references


def _ip_reference(max_steps: int, ip_final: float) -> np.ndarray:
    """Match the public scoring Ip schedule where possible."""
    t = np.arange(1, max_steps + 1, dtype=np.float64)
    ip = np.where(t <= 100.0, 500000.0 - t * 1000.0, ip_final)
    return np.clip(ip, ip_final, 500000.0).astype(np.float64)
