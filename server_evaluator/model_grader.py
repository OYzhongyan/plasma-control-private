"""Supervise one isolated CPU model evaluation and preserve its audit files."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
import time

import yaml

from submission.model_bundle import BundleError, unpack_bundle
from .grader import Grader, GradingUnavailable, InvalidSubmission, METRIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


class ModelGrader:
    download_suffix = ".zip"
    metric_names = METRIC_NAMES

    def __init__(self, target_path, data_dir, *, env_config=None, reference_config=None,
                 obs_perturb_config=None, hfm_host="127.0.0.1", hfm_port=5558,
                 socket_timeout=300.0, evaluation_timeout=1800.0, seed=0,
                 onnx_threads=1, memory_limit_mb=4096, inference_timeout=1.0, stop_event=None):
        self.shots_config = ROOT / "configs/shots.yaml"
        # Importing environment also loads its shot catalogue. Pin that import
        # to the same trusted file as the child, without changing the caller's
        # environment variables.
        previous_catalogue = os.environ.get("SAISDATA_SHOTS_CONFIG")
        os.environ["SAISDATA_SHOTS_CONFIG"] = str(self.shots_config)
        try:
            from environment.power_supply import DT, DELAY_RANGE_PF, DELAY_RANGE_VS
        finally:
            if previous_catalogue is None:
                os.environ.pop("SAISDATA_SHOTS_CONFIG", None)
            else:
                os.environ["SAISDATA_SHOTS_CONFIG"] = previous_catalogue
        self.data_dir = Path(data_dir).resolve()
        self.stop_event = stop_event if stop_event is not None else Event()
        self.evaluation_timeout = float(evaluation_timeout)
        self.inference_timeout = float(inference_timeout)
        self.config = {
            "target_path": str(Path(target_path).resolve()),
            "env_config": str(Path(env_config or ROOT / "configs/env_default.yaml").resolve()),
            "reference_config": str(Path(reference_config or ROOT / "eval_files/references.yaml").resolve()),
            "obs_perturb_config": str(Path(obs_perturb_config).resolve()) if obs_perturb_config else None,
            "hfm_host": hfm_host, "hfm_port": int(hfm_port),
            "socket_timeout": float(socket_timeout), "seed": int(seed), "onnx_threads": int(onnx_threads),
            "memory_limit_mb": int(memory_limit_mb),
            "model_interface": {"interface": "hfm_controller_v1", "observations": "requested_simulator_fields", "action_dim": 12},
            "power_supply": {"dt_s": DT, "pf_delay_range_s": list(DELAY_RANGE_PF),
                             "vs_delay_range_s": list(DELAY_RANGE_VS), "delay_sampling": "each_step"},
        }
        if not hfm_host or not 1 <= self.config["hfm_port"] <= 65535:
            raise ValueError("HFM host/port is invalid")
        if self.config["onnx_threads"] < 1 or self.config["seed"] < 0:
            raise ValueError("onnx_threads must be positive and seed nonnegative")
        if self.config["memory_limit_mb"] < 256:
            raise ValueError("memory_limit_mb must be at least 256")
        for value in (self.evaluation_timeout, self.inference_timeout, self.config["socket_timeout"]):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Evaluation and socket timeouts must be positive and finite")
        scorer = Grader(self.config["target_path"])
        for key in ("env_config", "reference_config", "obs_perturb_config"):
            if self.config[key]:
                value = yaml.safe_load(Path(self.config[key]).read_text(encoding="utf-8-sig"))
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must contain a YAML mapping")
                if key == "reference_config" and not value.get("shot_id"):
                    raise ValueError("reference_config must define shot_id")
        # Pin the shot catalogue instead of inheriting a shell's unrelated override.
        files = {Path(self.config[k]) for k in ("target_path", "env_config", "reference_config", "obs_perturb_config") if self.config[k]}
        files.add(self.shots_config)
        for folder in ("server_evaluator", "submission", "environment", "evaluation/eval/score"):
            files.update((ROOT / folder).glob("*.py"))
        for name in ("eval_files/trajectory_utils.py", "eval_files/obs_perturb.py"):
            files.add(ROOT / name)
        self._snapshots = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
        versions = {}
        for name in ("numpy", "gymnasium", "onnx", "onnxruntime"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "missing"
        self.fingerprint = hashlib.sha256(json.dumps({
            "format": "onnx-controller-zip-v1", "rules": scorer.fingerprint,
            "config": self.config, "evaluation_timeout": self.evaluation_timeout,
            "inference_timeout": self.inference_timeout,
            "files": {str(p): digest for p, digest in self._snapshots.items()},
            "versions": versions, "python": sys.version,
        }, sort_keys=True).encode()).hexdigest()

    def _check_configuration(self):
        if self.stop_event.is_set():
            raise GradingUnavailable("Worker is stopping; submission remains pending")
        for path, digest in self._snapshots.items():
            try:
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise GradingUnavailable(f"Server configuration is unavailable: {path}") from exc
            if current != digest:
                raise GradingUnavailable("Server code/configuration changed; restart with a new data directory")

    def _command(self, model_dir, job_dir, config_path):
        return [sys.executable, "-m", "server_evaluator.model_rollout",
                "--model-dir", str(model_dir), "--config", str(config_path),
                "--trajectory", str(job_dir / "trajectory.json"),
                "--result", str(job_dir / "result.json")]

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    @staticmethod
    def _read_phase(job_dir):
        try:
            with (job_dir / "phase.json").open(encoding="utf-8") as stream:
                state = json.loads(stream.read(1024))
                return state.get("phase"), state.get("step")
        except (OSError, ValueError, AttributeError):
            return None, None

    @staticmethod
    def _process_failure(job_dir, message):
        phase, _ = ModelGrader._read_phase(job_dir)
        if phase in {"model_load", "inference"}:
            raise InvalidSubmission(f"Model {phase}: {message}")
        raise GradingUnavailable(message)

    def _run_child(self, model_dir, job_dir, config_path):
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith("PLATFORM_")}
        environment.update(OMP_NUM_THREADS=str(self.config["onnx_threads"]),
                           MKL_NUM_THREADS=str(self.config["onnx_threads"]),
                           OPENBLAS_NUM_THREADS=str(self.config["onnx_threads"]),
                           SAISDATA_SHOTS_CONFIG=str(self.shots_config), PYTHONUNBUFFERED="1")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        log_path = job_dir / "evaluation.log"
        with log_path.open("wb") as log:
            try:
                process = subprocess.Popen(self._command(model_dir, job_dir, config_path), cwd=ROOT,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT, **options)
            except OSError as exc:
                raise GradingUnavailable(f"Cannot start evaluation process: {exc}") from exc
            deadline = time.monotonic() + self.evaluation_timeout
            last_phase = None
            phase_started = time.monotonic()
            try:
                while process.poll() is None:
                    if self.stop_event.is_set():
                        raise GradingUnavailable("Worker is stopping; submission remains pending")
                    now = time.monotonic()
                    phase = self._read_phase(job_dir)
                    if phase != last_phase:
                        last_phase, phase_started = phase, now
                    if phase[0] == "inference" and now - phase_started >= self.inference_timeout:
                        raise InvalidSubmission(f"Model inference exceeded its {self.inference_timeout:g}s time limit at step {phase[1]}")
                    if now >= deadline:
                        message = f"Evaluation exceeded the {self.evaluation_timeout:g}s time limit; inspect {log_path}"
                        if phase[0] == "model_load":
                            raise InvalidSubmission("Model loading: " + message)
                        # A total-budget expiry can be caused by slow HFM steps,
                        # even when a normal inference just happened to start.
                        raise GradingUnavailable(message)
                    try:
                        process.wait(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                self._terminate(process)
        try:
            raw = (job_dir / "result.json").read_bytes()
            if len(raw) > 1024 * 1024:
                raise ValueError("Result envelope is too large")
            envelope = json.loads(raw)
            if not isinstance(envelope, dict):
                raise ValueError("Result envelope must be an object")
            return envelope
        except (OSError, ValueError) as exc:
            self._process_failure(job_dir, f"Evaluation process returned no valid result (exit={process.returncode}); inspect {log_path}")

    def grade_file(self, path):
        self._check_configuration()
        path = Path(path).resolve()
        parent = self.data_dir / "evaluations" / hashlib.sha256(str(path).encode()).hexdigest()
        # Download names already identify a submission; preserve that name for operators.
        if len(path.stem) == 64 and all(c in "0123456789abcdef" for c in path.stem):
            parent = self.data_dir / "evaluations" / path.stem
        parent.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=parent))
        try:
            model_dir = unpack_bundle(path, job_dir / "model")
        except BundleError as exc:
            raise InvalidSubmission(f"Invalid model ZIP: {exc}") from exc
        except OSError as exc:
            raise GradingUnavailable(f"Cannot stage model ZIP: {exc}") from exc
        try:
            return self._grade_staged(model_dir, job_dir)
        finally:
            # Only remove the three files created in this attempt, never a
            # user-provided source directory. Keep downloaded ZIP and audit data.
            if model_dir.resolve().parent != job_dir.resolve() or model_dir.name != "model":
                raise RuntimeError("Unexpected model staging path")
            try:
                from submission.model_bundle import MODEL_FILES
                for name in MODEL_FILES:
                    (model_dir / name).unlink(missing_ok=True)
                model_dir.rmdir()
            except OSError as exc:
                LOG.warning("Cannot clean model staging directory %s: %s", model_dir, exc)

    def _grade_staged(self, model_dir, job_dir):
        (job_dir / "controller.json").write_bytes((model_dir / "controller.json").read_bytes())
        config_path = job_dir / "config.json"
        config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        envelope = self._run_child(model_dir, job_dir, config_path)
        self._check_configuration()
        if envelope.get("kind") == "invalid":
            raise InvalidSubmission(str(envelope.get("error", "Invalid model")))
        if envelope.get("kind") == "unavailable":
            raise GradingUnavailable(str(envelope.get("error", "HFM unavailable")))
        result = envelope.get("result")
        if envelope.get("kind") != "completed" or not isinstance(result, dict):
            raise GradingUnavailable("Evaluation returned an invalid result envelope")
        scores = result.get("metrics_scores", {})
        if (result.get("status") != "completed" or set(scores) != set(self.metric_names)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values())):
            raise GradingUnavailable("Evaluation returned invalid metric contributions")
        return result
