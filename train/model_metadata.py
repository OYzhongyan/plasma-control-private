"""Training metadata; never imported by the grading runtime."""

from __future__ import annotations

from typing import Any


from train.wrappers import (
    ReferenceObservationWrapper,
)

REFERENCE_FEATURE_MODE = "reference_only_v1"


def reference_metadata(action_mode: str, *, norm_obs: bool = True) -> dict[str, Any]:
    """The current 90 target values, with no observation or action history."""
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"Invalid action_mode: {action_mode!r}")
    return {
        "version": 3,
        "feature_mode": REFERENCE_FEATURE_MODE,
        "history_steps": 0,
        "include_xpt": True,
        "action_history_steps": 0,
        "action_history_dim": 0,
        "reference_keys": list(ReferenceObservationWrapper.REFERENCE_SHAPES),
        "reference_shapes": {key: list(shape) for key, shape in ReferenceObservationWrapper.REFERENCE_SHAPES.items()},
        "obs_dim": 90,
        "action_dim": 7,
        "action_mode": action_mode,
        "action_units": "normalized",
        "observation_normalization": bool(norm_obs),
        "requires_vecnormalize": bool(norm_obs),
    }


def validate_reference_metadata(metadata: dict[str, Any]) -> None:
    """Validate the complete input layout before training artifacts are consumed."""
    expected = reference_metadata(
        metadata["action_mode"], norm_obs=metadata.get("observation_normalization", True))
    for key, value in expected.items():
        actual = metadata.get(key)
        if isinstance(value, bool):
            valid = isinstance(actual, bool) and actual == value
        elif isinstance(value, int):
            valid = type(actual) is int and actual == value
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"Reference metadata {key} does not match training layout")


