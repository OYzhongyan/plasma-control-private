"""Current-reference inference using the exact training observation/action wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from train.wrappers import (
    ReferenceObservationWrapper,
    NormalizedAction7DTo12DWrapper,
)

REFERENCE_FEATURE_MODE = "reference_only_v1"
OBSERVATION_FEATURE_MODE = "historical_observation_v1"
FLUX_FEATURE_MODE = "cnn_flux_critic_v1"


def observation_metadata(action_mode: str, *, history_steps: int = 4,
                         norm_obs: bool = True, critic_type: str = "mlp") -> dict[str, Any]:
    """SII-585-THU's physical observation/action-history layout."""
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"Invalid action_mode: {action_mode!r}")
    if type(history_steps) is not int or history_steps < 1:
        raise ValueError("history_steps must be a positive integer")
    if critic_type not in {"mlp", "flux_cnn"}:
        raise ValueError(f"Invalid critic_type: {critic_type!r}")
    if critic_type == "flux_cnn" and (action_mode != "delta" or history_steps != 4):
        raise ValueError("flux_cnn requires four-frame delta-action history")
    action_dim = 12 if action_mode == "delta" else 7
    metadata = dict(version=2, feature_mode=OBSERVATION_FEATURE_MODE,
                history_steps=history_steps, include_xpt=True,
                action_history_steps=5, action_history_dim=action_dim,
                obs_dim=188 * history_steps + 5 * action_dim,
                action_dim=7, action_mode=action_mode, action_units="normalized",
                observation_normalization=bool(norm_obs), requires_vecnormalize=bool(norm_obs))
    if critic_type == "flux_cnn":
        metadata.update(feature_mode=FLUX_FEATURE_MODE,
            actor_feature_mode=OBSERVATION_FEATURE_MODE,
            observation_shapes={"actor_obs": [812], "flux_history": [4, 66, 65], "flux_mask": [4]},
            observation_dtype="float32", flux_order="C")
    return metadata


def validate_observation_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("feature_mode") == REFERENCE_FEATURE_MODE:
        return validate_reference_metadata(metadata)
    expected = observation_metadata(metadata.get("action_mode"),
        history_steps=metadata.get("history_steps"),
        norm_obs=metadata.get("observation_normalization", True),
        critic_type="flux_cnn" if metadata.get("feature_mode") == FLUX_FEATURE_MODE else "mlp")
    for key, value in expected.items():
        actual = metadata.get(key)
        same_type = (isinstance(actual, bool) if isinstance(value, bool)
                     else type(actual) is int if isinstance(value, int) else True)
        if not same_type or actual != value:
            raise ValueError(f"Observation metadata {key} does not match training layout")


def build_policy_adapter(metadata):
    if metadata.get("feature_mode") == REFERENCE_FEATURE_MODE:
        return ReferencePolicyAdapter(metadata)
    return ObservationPolicyAdapter(metadata)


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


class _ActionSink(gym.Env):
    """Supply action-space/reset plumbing without simulating an environment."""
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return {}, {}


class ReferencePolicyAdapter:
    """Encode the current target only; retain delta-action accumulation."""

    def __init__(self, metadata: dict[str, Any]):
        validate_reference_metadata(metadata)
        sink = _ActionSink()
        self.actions = NormalizedAction7DTo12DWrapper(sink, action_mode=metadata["action_mode"])
        self.observations = ReferenceObservationWrapper(sink, action_mode=metadata["action_mode"])
        self.reset()

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        return self.observations.convert_obs_dict_to_vector(observation)

    def reset(self) -> None:
        self.actions.reset()

    def decode(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise ValueError("The policy must produce 7 finite actions")
        # SB3 predict() clips Box actions; ONNX emits the raw actor output.
        return self.actions.action(np.clip(action, -1., 1.)).astype(np.float32)


class _RemoteObservation(gym.Env):
    """A buffer for C's observations; it does not run a simulator on A."""
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def __init__(self):
        self.observation = {}
        self.physical_action = np.zeros(12, dtype=np.float32)

    def reset(self, **kwargs):
        return self.observation, {}

    def step(self, action):
        return self.observation, 0., False, False, {"physical_action": self.physical_action.copy()}


class CheckpointPolicyAdapter:
    """Use the real training wrappers on A, including their reset/step state."""

    def __init__(self, model, metadata, config):
        # Import actual participant classes at construction, not a second copy
        # of their feature/action implementation or an ONNX-specific encoder.
        from train import wrappers
        self.buffer = _RemoteObservation()
        mode = metadata.get("action_mode", config.get("actions", {}).get("mode", "absolute"))
        self.actions = wrappers.NormalizedAction7DTo12DWrapper(self.buffer, action_mode=mode)
        features = config.get("features", {})
        feature_mode = metadata.get("feature_mode", features.get("observation_mode"))
        if feature_mode == FLUX_FEATURE_MODE:
            validate_observation_metadata(metadata)
            from train.flux_observation import FluxObservationWrapper
            wrapper = FluxObservationWrapper
        else:
            wrapper = (wrappers.ObservationWrapper if feature_mode == OBSERVATION_FEATURE_MODE
                       else wrappers.ReferenceObservationWrapper)
        self.observations = wrapper(
            self.buffer, obs_history_step=int(metadata.get("history_steps", features.get("history_steps", 0))),
            include_xpt=True, action_mode=mode)
        if self.observations.observation_space != model.observation_space:
            raise ValueError("Actual observation wrapper does not match checkpoint observation space")
        if self.actions.action_space != model.action_space:
            raise ValueError("Actual action wrapper does not match checkpoint action space")
        self.reset()

    def reset(self):
        self.started = False
        self.pending_action = None
        self.buffer.physical_action = np.zeros(12, np.float32)
        self.actions.reset()

    def encode(self, observation):
        self.buffer.observation = observation
        if not self.started:
            vector, _ = self.observations.reset()
            self.started = True
        else:
            if self.pending_action is None:
                raise RuntimeError("Missing previous action; start a new remote episode")
            vector, _, _, _, _ = self.observations.step(self.pending_action)
        self.pending_action = None
        return vector

    def decode(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise ValueError("The policy must produce 7 finite actions")
        action = np.clip(action, -1., 1.)
        physical = np.asarray(self.actions.action(action), dtype=np.float32)
        self.pending_action = action.copy()
        self.buffer.physical_action = physical.copy()
        return physical


def build_checkpoint_adapter(model, metadata, config):
    return CheckpointPolicyAdapter(model, metadata, config)


class ObservationPolicyAdapter:
    """Use the actual history wrapper for ONNX's external preprocessing too."""

    def __init__(self, metadata):
        from types import SimpleNamespace
        validate_observation_metadata(metadata)
        model = SimpleNamespace(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (metadata["obs_dim"],), dtype=np.float32),
            action_space=gym.spaces.Box(-1., 1., (7,), dtype=np.float32),
        )
        self.adapter = CheckpointPolicyAdapter(model, metadata, {})

    def reset(self):
        self.adapter.reset()

    def encode(self, observation):
        return self.adapter.encode(observation)

    def decode(self, action):
        return self.adapter.decode(action)
