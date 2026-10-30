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


def normalize_task(task: str) -> str:
    out = str(task).strip().upper()
    if out not in {"F1", "F2"}:
        raise ValueError(f"task must be F1 or F2, got {task!r}")
    return out


def default_reference_path(task: str) -> Path:
    name = "f1_reference_targets.json" if task == "F1" else "xpt_reference_targets.json"
    return ROOT / "configs" / name


def _load_public_target_reference(data: dict[str, Any], task: str | None) -> TargetReference:
    if task is None:
        if "F1" in data and "F2a" not in data:
            task = "F1"
        elif "F2a" in data and "F1" not in data:
            task = "F2"
        else:
            raise ValueError("task is required when loading evaluation target.json")

    task = normalize_task(task)
    target_key = "F1" if task == "F1" else "F2a"
    target = data[target_key]
    lcfs = np.asarray(target["lcfs_points"], dtype=np.float64)
    ip_values = np.asarray(target.get("Ip", []), dtype=np.float64).reshape(-1)
    ip_final = float(ip_values[-1]) if ip_values.size else (200_000.0 if task == "F1" else 400_000.0)
    if task == "F1":
        return TargetReference(lcfs=lcfs, ip_final=ip_final)

    x_main = np.asarray(target["Xpt_main"], dtype=np.float64)
    x_sec = np.asarray(target["Xpt_sec"], dtype=np.float64)
    strike = np.asarray(target["strike"], dtype=np.float64)
    return TargetReference(
        lcfs=lcfs,
        ip_final=ip_final,
        rX=np.asarray([x_sec[0, 0], x_main[0, 0], x_main[1, 0], x_sec[1, 0]], dtype=np.float64),
        zX=np.asarray([x_sec[0, 1], x_main[0, 1], x_main[1, 1], x_sec[1, 1]], dtype=np.float64),
        strike_r=strike[:, 0],
        strike_z=strike[:, 1],
        x_valid=np.ones(4, dtype=np.float64),
        strike_valid=np.ones(8, dtype=np.float64),
    )


def build_reference(
    path: str | Path,
    task: str | None = None,
    *,
    max_steps: int | None = None,
):
    """Build references for the requested horizon without rescaling the Ip schedule.

    Omitted horizons retain the original F1/F2 defaults of 300/500 steps.
    """
    # The references will be a list of references, each reference is a dict[str, Any]
    # When using the default reference, the list contains only one element
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # if "reference_lcfs_points" not in data:
    #     return _load_public_target_reference(data, task)
    if max_steps is None:
        max_steps = 300 if task == "F1" else 500
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    def build_single_reference(target):
        lcfs = target["reference_lcfs_points"]
        ip_final = float(target["reference_Ip_final"])
        ip_trajectory = _ip_reference(task, max_steps, ip_final)
        lx = 0.0 if task == "F1" else 1.0

        reference: dict[str, Any] = {
            "Ip": ip_trajectory.tolist(),
            "lX": lx,
            "lcfs_points": lcfs,
        }

        if task == "F2":
            reference.update(
                {
                    "rX": np.asarray(target["reference_rX"], dtype=np.float64).tolist(),
                    "zX": np.asarray(target["reference_zX"], dtype=np.float64).tolist(),
                    "strike_r": np.asarray(target["reference_strike_r"], dtype=np.float64).tolist(),
                    "strike_z": np.asarray(target["reference_strike_z"], dtype=np.float64).tolist(),
                    "x_valid": np.asarray(target["reference_x_valid"], dtype=np.float64).tolist(),
                    "strike_valid": np.asarray(target["reference_strike_valid"], dtype=np.float64).tolist(),
                    "nX": target["reference_nX"],
                    "n_strike": target["reference_n_strike"],
                }
            )

        return reference
        
    if isinstance(data, dict):
        references = [build_single_reference(data)]
    elif isinstance(data, list):
        references = [build_single_reference(target) for target in data]
    else:
        raise ValueError(f"reference JSON must be a dict or list, got {type(data).__name__}")

    return references


# build the ip reference curve by default
def _ip_reference(task: str, max_steps: int, ip_final: float) -> np.ndarray:
    """Match the public scoring Ip schedule where possible."""
    t = np.arange(1, max_steps + 1, dtype=np.float64)
    if task == "F1":
        ip = np.where(t <= 100.0, 500_000.0, 500_000.0 - (t - 100.0) * 1_500.0)
        return np.clip(ip, ip_final, 500_000.0).astype(np.float64)

    ip = np.where(t <= 100.0, 500_000.0 - t * 1_000.0, ip_final)
    return np.clip(ip, ip_final, 500_000.0).astype(np.float64)


def _scalar_from_obs(observation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = observation.get(key)
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(default)
    out = float(arr[0])
    return out if np.isfinite(out) else float(default)


def _array_from_obs(
    observation: dict[str, Any],
    key: str,
    shape: tuple[int, ...],
    default: float = 0.0,
) -> np.ndarray:
    out = np.full(int(np.prod(shape)), default, dtype=np.float64)
    value = observation.get(key)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        if n > 0:
            out[:n] = flat[:n]
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default).reshape(shape)


def build_hold_initial_reference_from_obs(
    observation: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    """Build a hold trajectory from one reset observation.

    The runtime environment already implements this for real HFM resets. This pure
    helper exists so stage configs and reward assumptions can be tested offline.
    """

    max_steps = int(max_steps)
    lcfs = _array_from_obs(observation, "lcfs_points", (32, 2))
    reference = {
        "Ip": [_scalar_from_obs(observation, "Ip")] * max_steps,
        "R": [_scalar_from_obs(observation, "R")] * max_steps,
        "Z": [_scalar_from_obs(observation, "Z")] * max_steps,
        "lX": [_scalar_from_obs(observation, "lX")] * max_steps,
        "lcfs_points": np.repeat(lcfs[None, :, :], max_steps, axis=0).tolist(),
    }
    return reference
