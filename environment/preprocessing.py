"""Pure preprocessing helpers shared by training and submission."""

from typing import Any, List, Optional, Sequence

import numpy as np

DEFAULT_FLAT_OBSERVATION_KEYS = [
    "Ip",
    "R",
    "Z",
    "lX",
    "reference_Ip",
    "reference_R",
    "reference_Z",
    "reference_lX",
    "lcfs_points",
    "reference_lcfs_points",
    "I_PF",
    "aminor",
    "deltal",
    "deltau",
    "rc",
    "zc",
    "FA",
    "FB",
]

ACTION_7D_TO_12D_INDEX = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]


def action_7d_to_12d(action_7d: np.ndarray) -> np.ndarray:
    """Map 7-dimensional action to 12-dimensional coil voltages."""
    action_7d = np.asarray(action_7d, dtype=np.float32)
    if action_7d.size != 7:
        raise ValueError(f"action_7d must have size 7, got {action_7d.size}")
    return action_7d[ACTION_7D_TO_12D_INDEX]


def flatten_dict_observation(
    observation: dict[str, Any],
    keys: Optional[Sequence[str]] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Flatten selected observation keys into one vector."""
    selected_keys = keys or DEFAULT_FLAT_OBSERVATION_KEYS
    parts: List[np.ndarray] = []
    for key in selected_keys:
        val = observation.get(key)
        if val is None:
            parts.append(np.zeros(1, dtype=dtype))
        else:
            parts.append(np.asarray(val, dtype=dtype).reshape(-1))
    return np.concatenate(parts, axis=0)
