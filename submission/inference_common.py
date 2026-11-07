"""Generic ONNX control: field transport and optional state, no training imports.

Feature extraction, normalization and action conversion are embedded in ONNX.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from submission.controller_contract import MANIFEST_NAME, MODEL_NAME, observation_inputs, read_manifest
from submission.onnx_validation import validate_graph


class BaseOnnxPolicy:
    def __init__(self, *, model_dir, preferred_name=MODEL_NAME, session_options=None, raise_on_error=False):
        self.model_dir = Path(model_dir)
        self.session = None
        self.metadata = {}
        self.state = None
        self.load_error = None
        try:
            import onnxruntime as ort
            self.metadata = read_manifest(self.model_dir / MANIFEST_NAME)
            model_bytes = (self.model_dir / preferred_name).read_bytes()
            validate_graph(model_bytes, self.metadata)
            try:
                kwargs = {"providers": ["CPUExecutionProvider"]}
                if session_options is not None:
                    kwargs["sess_options"] = session_options
                self.session = ort.InferenceSession(model_bytes, **kwargs)
                self.reset()
                values = {f["name"]: np.zeros((1, *f["shape"]), np.float32)
                          for f in self.metadata["observations"]}
                if self.state is not None:
                    values["state"] = self.state.copy()
                self._check_outputs(self.session.run(self.output_names, values))
            except Exception as exc:
                raise ValueError(f"Controller load/preflight failed: {exc}") from exc
        except Exception as exc:
            self.load_error = str(exc)
            self.session = None
            if raise_on_error:
                raise

    @property
    def ready(self):
        return self.session is not None

    @property
    def output_names(self):
        return ["action", "next_state"] if self.metadata["state_dim"] else ["action"]

    def reset(self):
        size = self.metadata.get("state_dim", 0)
        self.state = (np.zeros((1, size), dtype=self.metadata["state_dtype"]) if size else None)

    def _check_outputs(self, values):
        action = np.asarray(values[0])
        if action.shape != (1, 12) or action.dtype != np.float32 or not np.all(np.isfinite(action)):
            raise ValueError("Controller action must contain 12 finite float32 physical voltages")
        state = None
        if self.metadata["state_dim"]:
            state = np.asarray(values[1])
            if (state.shape != self.state.shape or state.dtype != self.state.dtype
                    or not np.all(np.isfinite(state))):
                raise ValueError("Controller next_state must match state shape/dtype and be finite")
        return action[0].copy(), None if state is None else state.copy()

    def act(self, observation):
        if not self.ready:
            raise RuntimeError(self.load_error or "Controller is not ready")
        values = observation_inputs(observation, self.metadata)
        if self.state is not None:
            values["state"] = self.state.copy()
        action, next_state = self._check_outputs(self.session.run(self.output_names, values))
        self.state = next_state
        return action

    def status(self):
        return dict(ok=self.ready, model_loaded=self.ready, error=self.load_error,
                    interface=self.metadata.get("interface"),
                    observations=[f["name"] for f in self.metadata.get("observations", [])],
                    obs_dim=sum(math.prod(f["shape"]) for f in self.metadata.get("observations", [])),
                    state_dim=self.metadata.get("state_dim"), action_dim=12,
                    normalization_embedded=True)
