"""Run a submitted ONNX bundle using organizer-owned inference and HFM code.

This module never imports code from a submission. A supervising process owns
the wall-clock limit and the dedicated HFM port; this child owns one rollout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from server_evaluator.grader import Grader, GradingUnavailable, InvalidSubmission

LOG = logging.getLogger(__name__)
MODEL_NAME = "controller.onnx"
METADATA_NAME = "controller.json"


def _dependency(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise GradingUnavailable(f"Server dependency {name} is unavailable: {exc}") from exc


def _read_json(path: Path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def _validate_metadata(model_dir: Path) -> dict[str, Any]:
    contract = _dependency("submission.controller_contract")
    try:
        return contract.read_manifest(model_dir / METADATA_NAME)
    except OSError as exc:
        raise GradingUnavailable(f"Cannot read controller manifest: {exc}") from exc
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise InvalidSubmission(f"Invalid controller manifest: {exc}") from exc


def _load_policy(model_dir: Path, config: dict):
    _dependency("onnx")
    ort = _dependency("onnxruntime")
    _validate_metadata(model_dir)
    try:
        threads = int(config["onnx_threads"])
        if threads < 1:
            raise ValueError("onnx_threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    except Exception as exc:
        raise GradingUnavailable(f"Invalid server ONNX configuration: {exc}") from exc
    inference = _dependency("submission.inference_common")
    try:
        # Generic loader checks the graph, transport shapes and finite outputs;
        # it never imports the contestant's training wrapper or export factory.
        return inference.BaseOnnxPolicy(model_dir=model_dir, session_options=options,
                                         raise_on_error=True)
    except (OSError, ImportError) as exc:
        raise GradingUnavailable(f"Cannot load controller: {exc}") from exc
    except Exception as exc:
        raise InvalidSubmission(f"Invalid ONNX controller: {exc}") from exc


def _read_yaml(path: str | Path) -> dict:
    yaml = _dependency("yaml")
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected a YAML mapping")
        return value
    except Exception as exc:
        raise GradingUnavailable(f"Cannot read server configuration {path}: {exc}") from exc


def _reject_observation_delay(config: dict) -> None:
    """The shared perturbation utility supports noise, not observation delay."""
    if not config.get("enabled", False):
        return
    def active(value):
        if isinstance(value, dict):
            if value.get("enabled") is False:
                return False
            settings = [v for k, v in value.items() if k != "enabled"]
            return any(active(v) for v in settings) if settings else bool(value.get("enabled"))
        if isinstance(value, (list, tuple)):
            return any(active(v) for v in value)
        if value is None or value is False:
            return False
        if isinstance(value, str) and value.lower() in {"none", "off", "disabled"}:
            return False
        try:
            return float(value) != 0.
        except (TypeError, ValueError):
            return True
    for name, value in config.items():
        if "delay" in str(name).lower() and active(value):
            raise GradingUnavailable(f"Observation delay setting {name!r} is unsupported; only observation noise is supported")


def _audit_jsonable(value):
    """Represent unavailable numbers as JSON null, only after original scoring."""
    if isinstance(value, dict):
        return {key: _audit_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


@contextmanager
def _environment(config: dict, seed: int):
    environment = _dependency("environment")
    power = _dependency("environment.power_supply")
    env = None
    failed = False
    try:
        try:
            env = environment.HFMSimulator(copy.deepcopy(config))
            # Gym's reset(seed) does not seed the independent actuator RNG.
            env.engine._power_supply = power.PowerSupplyModel(seed=seed)
        except Exception as exc:
            raise GradingUnavailable(f"Cannot create HFM environment: {exc}") from exc
        yield env
    except BaseException:
        failed = True
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if not failed:
                    raise GradingUnavailable(f"Cannot close HFM connection: {exc}") from exc
                LOG.warning("HFM close failed while handling another error: %s", exc)


def _reset(env, seed: int, options: dict):
    try:
        return env.reset(seed=seed, options=options)[0]
    except Exception as exc:
        raise GradingUnavailable(f"HFM reset failed: {exc}") from exc


def _atomic_json(path: Path, data: Any, *, sync: bool = True) -> bytes:
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
            file.flush()
            if sync:
                os.fsync(file.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                # A concurrent Windows reader can briefly deny replacement of
                # phase.json. Never retry indefinitely or mask other I/O errors.
                if getattr(exc, "winerror", None) not in {5, 32} or attempt == 4:
                    raise
                time.sleep(.02)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return raw


def _phase(config: dict, phase: str, step: int = 0) -> None:
    path = config.get("_progress_path")
    if path is None:
        return
    try:
        _atomic_json(Path(path), {"phase": phase, "step": step}, sync=False)
    except Exception as exc:
        raise GradingUnavailable(f"Cannot write rollout phase: {exc}") from exc


def _apply_memory_limit(config: dict) -> None:
    """Apply a per-child Linux address-space limit before loading model code."""
    try:
        megabytes = int(config.get("memory_limit_mb", 4096))
        if megabytes < 256:
            raise ValueError("memory_limit_mb must be at least 256")
        if not sys.platform.startswith("linux"):
            LOG.warning("RLIMIT_AS is unavailable on %s; memory limit is not applied for this local test", sys.platform)
            return
        resource = _dependency("resource")
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limits = [megabytes * 1024 * 1024]
        limits.extend(value for value in (soft, hard) if value != resource.RLIM_INFINITY)
        limit = min(limits)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception as exc:
        raise GradingUnavailable(f"Cannot apply child memory limit: {exc}") from exc


def _memory_failure(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _run_model(model_dir: Path, config: dict, trajectory_path: Path) -> dict:
    """Validate artifacts, run one deterministic episode, then score its trajectory."""
    _phase(config, "model_load")
    policy = _load_policy(Path(model_dir), config)
    utils = _dependency("eval_files.trajectory_utils")
    perturb = _dependency("eval_files.obs_perturb")
    score_config = _dependency("evaluation.eval.score.config")
    try:
        grader = Grader(config["target_path"])
        seed = int(config["seed"])
        env_config = _read_yaml(config["env_config"])
        reference_spec = _read_yaml(config["reference_config"])
        reference_spec.update(target_source="official_target", target_json=str(config["target_path"]),
                              max_steps=500, reference_mode="trajectory", target_lX=1)
        env_config["max_steps"] = 500
        env_config.setdefault("predictor", {}).update(
            host=str(config["hfm_host"]), port=int(config["hfm_port"]),
            timeout=float(config["socket_timeout"]), shot_id=reference_spec["shot_id"])
        perturb_config = _read_yaml(config["obs_perturb_config"]) if config.get("obs_perturb_config") else {"enabled": False}
        perturb_config["seed"] = seed
        _reject_observation_delay(perturb_config)
        target_obs = utils.build_official_target_observation(reference_spec)
        if target_obs is None:
            raise ValueError("Official target does not contain F2b or its supported fallback")
    except GradingUnavailable:
        raise
    except Exception as exc:
        raise GradingUnavailable(f"Invalid server rollout configuration or target: {exc}") from exc

    _phase(config, "hfm_reset")
    with _environment(env_config, seed) as warmup:
        initial_obs = _reset(warmup, seed, {"reference_mode": "hold"})
    try:
        reference = utils.build_reference_trajectory(reference_spec, initial_obs, 500, target_obs=target_obs)
        # Scoring defines this curve independently of an optional target JSON Ip.
        reference["Ip"] = score_config.build_Ip_ref()
    except Exception as exc:
        raise GradingUnavailable(f"Cannot build official reference trajectory: {exc}") from exc
    options = {"reference_mode": "trajectory", "reference": reference}
    observations = []
    inference_ms = []
    rng = perturb.make_observation_rng(seed)
    _phase(config, "hfm_reset")
    with _environment(env_config, seed) as env:
        obs = _reset(env, seed, options)
        policy.reset()
        for step in range(500):
            try:
                if perturb_config.get("enabled", False):
                    sent = perturb.build_policy_observation(obs, rng, perturb_config)
                else:
                    sent = obs
            except Exception as exc:
                raise GradingUnavailable(f"Cannot prepare observation at step {step + 1}: {exc}") from exc
            _phase(config, "inference", step + 1)
            try:
                started = time.perf_counter()
                action = np.asarray(policy.act(sent), dtype=np.float32)
                elapsed = (time.perf_counter() - started) * 1000.
                if action.shape != (12,) or not np.all(np.isfinite(action)):
                    raise ValueError("policy must return 12 finite physical voltages")
            except Exception as exc:
                raise InvalidSubmission(f"Model inference failed at step {step + 1}: {exc}") from exc
            _phase(config, "hfm_step", step + 1)
            try:
                obs, _, terminated, truncated, _ = env.step(action)
            except Exception as exc:
                raise GradingUnavailable(f"HFM STEP {step + 1} failed: {exc}") from exc
            # Record only true, post-step observations, including the terminal one.
            observations.append(copy.deepcopy(obs))
            inference_ms.append(elapsed)
            if terminated or truncated:
                break
    _phase(config, "scoring", len(observations))
    try:
        trajectory = utils.extract_semifinal_trajectory_from_obs_list(observations, target_obs=target_obs)
        data = {"F2b": {"trajectory": utils.to_jsonable(trajectory),
                         "f_surv": len(observations) / 500., "timeout": False},
                "audit": {"seed": seed, "inference_time_ms": inference_ms}}
        # Timing is audit-only, matching run_test: no new 0.5 ms scoring penalty.
        result = grader.grade_data(data)
        _atomic_json(Path(trajectory_path), _audit_jsonable(data))
        return result
    except Exception as exc:
        raise GradingUnavailable(f"Cannot serialize or score HFM trajectory: {exc}") from exc


def run_model(model_dir: Path, config: dict, trajectory_path: Path) -> dict:
    try:
        return _run_model(model_dir, config, trajectory_path)
    except Exception as exc:
        if _memory_failure(exc):
            raise InvalidSubmission("Model evaluation exceeded the child memory limit") from exc
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = _read_json(args.config)
        if not isinstance(config, dict):
            raise GradingUnavailable("Server config must be a JSON object")
        config["_progress_path"] = str(args.result.parent / "phase.json")
        _apply_memory_limit(config)
        result = run_model(args.model_dir, config, args.trajectory)
        envelope = {"kind": "completed", "result": result}
    except InvalidSubmission as exc:
        envelope = {"kind": "invalid", "error": str(exc)}
    except MemoryError:
        envelope = {"kind": "invalid", "error": "Model evaluation exceeded the child memory limit"}
    except Exception as exc:
        envelope = {"kind": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    _atomic_json(args.result, envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
