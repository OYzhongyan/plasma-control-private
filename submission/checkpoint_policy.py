"""A-only checkpoint loading. Never imported by the C remote evaluation path."""
from __future__ import annotations
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import re

import numpy as np

from submission.remote_protocol import checked_action


def resolve_checkpoint(model_path, vecnormalize_path=None):
    model = Path(model_path).expanduser().resolve()
    if model.is_dir():
        final = next((model / name for name in ("ppo_f2_latest.zip", "model.zip")
                      if (model / name).is_file()), None)
        if final is None:
            checkpoints = [*model.glob("ppo_f2_*_steps.zip"), *model.glob("checkpoints/ppo_f2_*_steps.zip")]
            numbered = [(int(match.group(1)), path) for path in checkpoints
                        if (match := re.fullmatch(r"ppo_f2_(\d+)_steps.zip", path.name))]
            if not numbered:
                raise FileNotFoundError(f"No checkpoint in {model}")
            final = max(numbered, key=lambda item: item[0])[1]
        model = final
    if not model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model}")
    if vecnormalize_path is not None:
        stats = Path(vecnormalize_path).expanduser().resolve()
        if not stats.is_file():
            raise FileNotFoundError(f"VecNormalize not found: {stats}")
    else:
        candidates = [model.with_name(model.stem + "_vecnormalize.pkl")]
        # A checkpoint must use statistics from its own saved training step.
        if not re.fullmatch(r"ppo_f2_\d+_steps", model.stem):
            candidates.append(model.parent / "vecnormalize.pkl")
        stats = next((path for path in candidates if path.is_file()), None)
    return model, stats


def _training_settings(model):
    directories = [model.parent, model.parent.parent]
    metadata, config = {}, {}
    for directory in directories:
        path = directory / "f2_features.metadata.json"
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8-sig"))
            break
    for directory in directories:
        path = directory / "configs.pkl"
        if path.is_file():
            with path.open("rb") as stream:
                config = pickle.load(stream)
            break
    return metadata, config


def _identity(paths):
    digest = hashlib.sha256()
    for path in paths:
        if path is not None and Path(path).is_file():
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class CheckpointPolicy:
    def __init__(self, model_path, *, vecnormalize_path=None, adapter_factory=None, device="cpu"):
        self.ready = False
        self.load_error = None
        self.model = self.normalization = self.adapter = None
        self.model_id = None
        try:
            model_path, stats = resolve_checkpoint(model_path, vecnormalize_path)
            self.metadata, config = _training_settings(model_path)
            normalize = self.metadata.get("observation_normalization",
                        bool(config.get("vecnormalize", {}).get("enabled", True)
                             and config.get("vecnormalize", {}).get("norm_obs", True)))
            if normalize and stats is None:
                raise FileNotFoundError("Matching VecNormalize.pkl is required for this checkpoint")
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path), device=device)
            if stats is not None:
                # These files belong to A, never to or loaded by C.
                with stats.open("rb") as stream:
                    self.normalization = pickle.load(stream)
                self.normalization.training = False
                self.normalization.norm_reward = False
                if bool(self.normalization.norm_obs) != bool(normalize):
                    raise ValueError("VecNormalize normalization mode disagrees with training settings")
                if normalize:
                    rms = self.normalization.obs_rms
                    if isinstance(rms, dict):
                        spaces = self.model.observation_space.spaces
                        items = [(item, spaces[key].shape) for key, item in rms.items()]
                        if "actor_obs" in spaces:
                            if set(rms) != {"actor_obs"}:
                                raise ValueError("Flux VecNormalize requires actor_obs statistics")
                            flux_rms = vars(self.normalization).get("flux_rms")
                            if (flux_rms is None
                                    or np.asarray(flux_rms.mean).shape != ()
                                    or np.asarray(flux_rms.var).shape != ()
                                    or not np.isfinite(flux_rms.count) or flux_rms.count <= 0
                                    or not np.isfinite(flux_rms.mean)
                                    or not np.isfinite(flux_rms.var) or flux_rms.var < 0):
                                raise ValueError("VecNormalize contains invalid flux statistics")
                    else:
                        items = [(rms, self.model.observation_space.shape)]
                    for item, shape in items:
                        if item.mean.shape != shape or item.var.shape != shape:
                            raise ValueError("VecNormalize statistics shape disagrees with checkpoint")
                        if (not np.all(np.isfinite(item.mean)) or not np.all(np.isfinite(item.var))
                                or np.any(item.var < 0)):
                            raise ValueError("VecNormalize contains invalid statistics")
            factory_name = adapter_factory or "submission.history:build_checkpoint_adapter"
            module, separator, name = factory_name.partition(":")
            if not separator:
                raise ValueError("adapter factory must be module:function")
            factory = getattr(importlib.import_module(module), name)
            self.adapter = factory(self.model, self.metadata, config)
            if not all(callable(getattr(self.adapter, name, None)) for name in ("reset", "encode", "decode")):
                raise ValueError("Adapter must implement reset, encode and decode")
            sources = [model_path, stats, inspect.getsourcefile(factory), inspect.getsourcefile(type(self.adapter))]
            for name in ("observations", "actions"):
                wrapper = getattr(self.adapter, name, None)
                if wrapper is not None:
                    sources.append(inspect.getsourcefile(type(wrapper)))
            self.model_id = _identity(sources)
            self.ready = True
        except Exception as exc:
            self.load_error = str(exc)

    def reset(self, **kwargs):
        if self.adapter is not None:
            self.adapter.reset()

    def act(self, observation):
        if not self.ready:
            raise RuntimeError(self.load_error or "Checkpoint is not ready")
        vector = self.adapter.encode(observation)
        if self.normalization is not None:
            vector = self.normalization.normalize_obs(vector)
        action, _ = self.model.predict(vector, deterministic=True)
        return np.asarray(checked_action(self.adapter.decode(action)), dtype=np.float32)

    def status(self):
        shape = self.model.observation_space.shape if self.model is not None else None
        if self.model is not None and hasattr(self.model.observation_space, "spaces"):
            actor_space = self.model.observation_space.spaces.get("actor_obs")
            shape = actor_space.shape if actor_space is not None else None
        return dict(ok=self.ready, model_loaded=self.ready, backend="checkpoint", model_id=self.model_id,
                    normalization_loaded=self.normalization is not None,
                    obs_dim=int(np.prod(shape)) if shape is not None else None,
                    error=None if self.ready else "Checkpoint not ready; inspect A startup logs")
