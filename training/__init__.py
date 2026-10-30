"""Training utilities for semifinal F1/F2 baselines."""

from .features import (
    F1_FEATURE_KEYS,
    F2_FEATURE_KEYS,
    FeatureMetadata,
    PhysicalFeatureHistoryWrapper,
    feature_keys_for_task,
    flatten_observation,
    physical_feature_vector,
)
from .references import (
    TargetReference,
    build_hold_initial_reference_from_obs,
    build_reference,
    default_reference_path,
    # load_target_reference,
)
from .rewards import SemifinalDenseReward

__all__ = [
    "F1_FEATURE_KEYS",
    "F2_FEATURE_KEYS",
    "FeatureMetadata",
    "PhysicalFeatureHistoryWrapper",
    "feature_keys_for_task",
    "flatten_observation",
    "physical_feature_vector",
    "TargetReference",
    "build_hold_initial_reference_from_obs",
    "build_reference",
    "default_reference_path",
    "load_target_reference",
    "SemifinalDenseReward",
]
