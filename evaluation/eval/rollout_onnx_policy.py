#!/usr/bin/env python3
"""Roll out semifinal ONNX policies in HFM and score them locally."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import HFMSimulator  # noqa: E402
from environment.shot_registry import get_shot_spec  # noqa: E402
from environment.xpt_utils import X_PRIMARY_SLOTS, X_SECONDARY_SLOTS, extract_xpt_observation_pack  # noqa: E402
from submission.inference_common import BaseOnnxPolicy  # noqa: E402
from score import config as score_config  # noqa: E402
from training.train_ppo import ResetPerturbationSampler  # noqa: E402


TASK_IDS = ["F1", "F2a", "F2b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out trained semifinal ONNX policies and run local scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy1-path",
        default=str(PROJECT_ROOT / "submission" / "model" / "policy1.onnx"),
        help="F1 ONNX policy path.",
    )
    parser.add_argument(
        "--policy2-path",
        default=str(PROJECT_ROOT / "submission" / "model" / "policy2.onnx"),
        help="F2a/F2b ONNX policy path.",
    )
    parser.add_argument(
        "--target-path",
        default=str(PROJECT_ROOT / "evaluation" / "results" / "target.json"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "evaluation" / "results" / "infer_result_local.json"),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "env_default.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2223)
    parser.add_argument("--f1-shot-id", default="13844_500")
    parser.add_argument("--f2a-shot-id", default="13844_500")
    parser.add_argument("--f2b-shot-id", default="15892_500")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_IDS,
        default=TASK_IDS,
        help="Tasks to roll out. Omitted tasks are written as empty zero-score entries.",
    )
    parser.add_argument(
        "--reset-perturbation",
        choices=["none", "mild", "custom"],
        default="none",
        help="Apply initial equilibrium reset perturbation before rollout.",
    )
    parser.add_argument(
        "--perturb-signeo-scale",
        nargs=2,
        type=float,
        default=[0.75, 1.25],
        metavar=("LOW", "HIGH"),
        help="signeo multiplicative range for reset perturbation.",
    )
    parser.add_argument(
        "--perturb-bp-scale",
        nargs=2,
        type=float,
        default=[0.95, 1.05],
        metavar=("LOW", "HIGH"),
        help="bp multiplicative range for reset perturbation.",
    )
    parser.add_argument(
        "--perturb-q0-scale",
        nargs=2,
        type=float,
        default=[0.95, 1.05],
        metavar=("LOW", "HIGH"),
        help="q0 multiplicative range for reset perturbation.",
    )
    parser.add_argument(
        "--perturb-apply",
        choices=["one", "all"],
        default="one",
        help="Apply one randomly selected perturbation or all enabled perturbations at reset.",
    )
    parser.add_argument(
        "--perturb-strengths",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional sweep strengths that scale the configured scale ranges around 1.0. "
            "For example, with signeo [0.75, 1.25], strength 0.5 becomes [0.875, 1.125]."
        ),
    )
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=0,
        help="Seed for reset perturbation sampling.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files and target schema without connecting to HFM.",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="Use BaseOnnxPolicy zero-action fallback when a model file is missing.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Only write infer_result JSON; do not invoke evaluate.py.",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Print rollout diagnostics for initial steps and early stops.",
    )
    parser.add_argument(
        "--zero-reference-rz-for-policy",
        action="store_true",
        help=(
            "Set reference_R/reference_Z to zero only in the observation passed to the policy. "
            "The environment reference and scored trajectory remain unchanged."
        ),
    )
    parser.add_argument("--debug-steps", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for each task rollout. Defaults to scoring task length.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _as_array(value: Any, shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    out = np.full(int(np.prod(shape)), default, dtype=np.float64)
    if value is not None:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        n = min(out.size, flat.size)
        if n > 0:
            out[:n] = flat[:n]
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default).reshape(shape)


def _scalar(source: dict[str, Any], key: str, default: float = 0.0) -> float:
    arr = _as_array(source.get(key), (1,), default=default).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def load_target(path: Path) -> dict[str, dict[str, Any]]:
    _require_file(path, "Target file")
    target = json.loads(path.read_text(encoding="utf-8"))
    if "F1" not in target:
        raise ValueError("Target missing task: F1")
    if "F2a" not in target:
        raise ValueError("Target missing task: F2a")

    f1 = target["F1"]
    if np.asarray(f1.get("lcfs_points"), dtype=np.float64).shape != (32, 2):
        raise ValueError("F1 lcfs_points must have shape (32, 2)")
    f2 = target["F2a"]
    for key, shape in {
        "lcfs_points": (32, 2),
        "Xpt_main": (2, 2),
        "Xpt_sec": (2, 2),
        "strike": (8, 2),
    }.items():
        if np.asarray(f2.get(key), dtype=np.float64).shape != shape:
            raise ValueError(f"F2a {key} must have shape {shape}")
    return target


def task_length(task_id: str, target_task: dict[str, Any], cap: int | None) -> int:
    length = int(score_config.get_total_steps(task_id))
    if "Ip" in target_task:
        length = min(length, int(np.asarray(target_task["Ip"]).reshape(-1).size))
    if cap is not None:
        length = min(length, int(cap))
    if length <= 0:
        raise ValueError(f"{task_id} has invalid rollout length: {length}")
    return length


def _f2_target_reference(target_task: dict[str, Any]) -> dict[str, Any]:
    x_main = np.asarray(target_task["Xpt_main"], dtype=np.float64)
    x_sec = np.asarray(target_task["Xpt_sec"], dtype=np.float64)
    strike = np.asarray(target_task["strike"], dtype=np.float64)
    r_x = np.array([x_sec[0, 0], x_main[0, 0], x_main[1, 0], x_sec[1, 0]], dtype=np.float64)
    z_x = np.array([x_sec[0, 1], x_main[0, 1], x_main[1, 1], x_sec[1, 1]], dtype=np.float64)
    return {
        "rX": r_x.tolist(),
        "zX": z_x.tolist(),
        "x_valid": [1.0, 1.0, 1.0, 1.0],
        "strike_r": strike[:, 0].tolist(),
        "strike_z": strike[:, 1].tolist(),
        "strike_valid": [1.0] * 8,
        "nX": [4.0],
        "n_strike": [8.0],
    }


def reference_options(task_id: str, target_task: dict[str, Any], max_steps: int) -> dict[str, Any]:
    ip_ref = score_config.build_Ip_ref(task_id)[:max_steps].astype(np.float64)
    target_lx = 0.0 if task_id == "F1" else 1.0
    reference: dict[str, Any] = {
        "Ip": ip_ref.tolist(),
        "lX": [target_lx] * max_steps,
        "lcfs_points": np.asarray(target_task["lcfs_points"], dtype=np.float64).tolist(),
    }
    if task_id != "F1":
        reference.update(_f2_target_reference(target_task))
    return {
        "reference_mode": "trajectory",
        "reference": reference,
    }


def _scale_range_around_one(values: list[float], strength: float) -> list[float]:
    lo, hi = float(values[0]), float(values[1])
    strength = float(strength)
    return [1.0 + (lo - 1.0) * strength, 1.0 + (hi - 1.0) * strength]


def reset_perturbation_config(args: argparse.Namespace, *, strength: float = 1.0) -> dict[str, Any]:
    if args.reset_perturbation == "none":
        return {"enabled": False}
    return {
        "enabled": True,
        "signeo_scale_range": _scale_range_around_one(args.perturb_signeo_scale, strength),
        "bp_scale_range": _scale_range_around_one(args.perturb_bp_scale, strength),
        "q0_scale_range": _scale_range_around_one(args.perturb_q0_scale, strength),
    }


def empty_task_result() -> dict[str, Any]:
    return {
        "trajectory": empty_trajectory(),
        "f_surv": 0.0,
        "timeout": False,
        "terminated": False,
        "truncated": False,
        "n_steps": 0,
    }


def sample_reset_params(
    *,
    shot_id: str,
    reset_perturbation: dict[str, Any],
    seed: int,
    apply_mode: str,
) -> dict[str, float]:
    sampler = ResetPerturbationSampler(
        shot_id=shot_id,
        reset_perturbation=reset_perturbation,
        seed=int(seed),
    )
    if apply_mode == "one":
        return sampler.sample()

    if not bool(reset_perturbation.get("enabled", False)):
        return {}
    defaults = dict(get_shot_spec(shot_id).get("reset_defaults", {}))
    params: dict[str, float] = {}
    for name in sampler._perturbation_candidates(defaults):
        value = sampler._sample_value(name, defaults)
        if value is not None:
            params[name] = float(value)
    return params


def output_path_for_strength(path: Path, *, strength: float, total: int) -> Path:
    if total <= 1:
        return path
    label = f"{float(strength):g}".replace("-", "m").replace(".", "p")
    return path.with_name(f"{path.stem}_strength{label}{path.suffix}")


def load_env_config(
    config_path: Path,
    *,
    host: str,
    port: int,
    shot_id: str,
    max_steps: int,
) -> dict[str, Any]:
    _require_file(config_path, "Environment config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["max_steps"] = int(max_steps)
    config.setdefault("predictor", {})
    config["predictor"]["host"] = host
    config["predictor"]["port"] = int(port)
    config["predictor"]["shot_id"] = shot_id
    config["reference"] = {"mode": "trajectory"}
    return config


def make_policy(model_path: Path, *, task: str, allow_missing: bool) -> BaseOnnxPolicy:
    if not allow_missing:
        _require_file(model_path, f"{task} ONNX model")
    return BaseOnnxPolicy(
        task=task,
        model_dir=model_path.parent,
        preferred_name=model_path.name,
    )


def _xpt_from_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    ref_rx = _as_array(observation.get("reference_rX"), (4,))
    ref_zx = _as_array(observation.get("reference_zX"), (4,))
    ref_sr = _as_array(observation.get("reference_strike_r"), (8,))
    ref_sz = _as_array(observation.get("reference_strike_z"), (8,))
    pack = extract_xpt_observation_pack(
        observation,
        ref_rx,
        ref_zx,
        ref_sr,
        ref_sz,
        fx_order="C",
    )

    x_r = np.asarray(pack["x_r"], dtype=np.float64)
    x_z = np.asarray(pack["x_z"], dtype=np.float64)
    x_fx = np.asarray(pack["x_fx"], dtype=np.float64)
    strike = np.stack(
        [
            np.asarray(pack["strike_r"], dtype=np.float64),
            np.asarray(pack["strike_z"], dtype=np.float64),
        ],
        axis=-1,
    )
    return {
        "Xpt_main": np.stack([x_r[list(X_PRIMARY_SLOTS)], x_z[list(X_PRIMARY_SLOTS)]], axis=-1),
        "Xpt_sec": np.stack([x_r[list(X_SECONDARY_SLOTS)], x_z[list(X_SECONDARY_SLOTS)]], axis=-1),
        "strike": strike,
        "psiX_main": x_fx[list(X_PRIMARY_SLOTS)],
        "psiX_sec": x_fx[list(X_SECONDARY_SLOTS)],
    }


def _empty_xpt() -> dict[str, np.ndarray]:
    return {
        "Xpt_main": np.zeros((2, 2), dtype=np.float64),
        "Xpt_sec": np.zeros((2, 2), dtype=np.float64),
        "strike": np.zeros((8, 2), dtype=np.float64),
        "psiX_main": np.zeros((2,), dtype=np.float64),
        "psiX_sec": np.zeros((2,), dtype=np.float64),
    }


def append_step(
    trajectory: dict[str, list[Any]],
    observation: dict[str, Any],
    info: dict[str, Any],
    action: np.ndarray,
) -> None:
    raw = info.get("raw_observation")
    source = raw if isinstance(raw, dict) else observation
    trajectory["Ip"].append(_scalar(observation, "Ip"))
    trajectory["lX"].append(_scalar(observation, "lX"))
    trajectory["nX"].append(_scalar(observation, "nX"))
    trajectory["Icoil"].append(_as_array(observation.get("I_PF"), (12,)).tolist())
    trajectory["psia"].append(_scalar(source, "psia", _scalar(observation, "FA")))
    trajectory["psib"].append(_scalar(source, "psib", _scalar(observation, "FB")))
    trajectory["lcfs_per_step"].append(_as_array(observation.get("lcfs_points"), (32, 2)).tolist())
    trajectory["actions"].append(np.asarray(action, dtype=np.float64).reshape(-1).tolist())

    try:
        xpt = _xpt_from_observation(observation)
    except Exception:
        xpt = _empty_xpt()
    for key in ("Xpt_main", "Xpt_sec", "strike", "psiX_main", "psiX_sec"):
        trajectory[key].append(xpt[key].tolist())


def empty_trajectory() -> dict[str, list[Any]]:
    return {
        "Ip": [],
        "lX": [],
        "nX": [],
        "Icoil": [],
        "psia": [],
        "psib": [],
        "lcfs_per_step": [],
        "Xpt_main": [],
        "Xpt_sec": [],
        "strike": [],
        "psiX_main": [],
        "psiX_sec": [],
        "actions": [],
    }


def print_debug(
    task_id: str,
    step: int,
    action: np.ndarray,
    observation: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
    ip = _scalar(observation, "Ip")
    ref_ip = _scalar(observation, "reference_Ip", ip)
    lcfs = _as_array(observation.get("lcfs_points"), (32, 2))
    ref_lcfs = _as_array(observation.get("reference_lcfs_points"), (32, 2))
    lcfs_rms = float(np.sqrt(np.mean((lcfs - ref_lcfs) ** 2)))
    print(
        f"[debug] {task_id} step={step} "
        f"action_abs_max={float(np.max(np.abs(action_arr))):.3f} "
        f"ip_err_ka={(ip - ref_ip) / 1000.0:.3f} "
        f"lcfs_rms_m={lcfs_rms:.5f} "
        f"lX={_scalar(observation, 'lX'):.0f} nX={_scalar(observation, 'nX'):.0f} "
        f"reward={reward:.6f} terminated={terminated} truncated={truncated} "
        f"failure={info.get('failure')}"
    )


def policy_observation(
    observation: dict[str, Any],
    *,
    zero_reference_rz: bool,
) -> dict[str, Any]:
    if not zero_reference_rz:
        return observation
    out = dict(observation)
    out["reference_R"] = np.zeros((1,), dtype=np.float64)
    out["reference_Z"] = np.zeros((1,), dtype=np.float64)
    return out


def rollout_task(
    *,
    task_id: str,
    target_task: dict[str, Any],
    env: HFMSimulator,
    policy: BaseOnnxPolicy,
    max_steps: int,
    debug_actions: bool,
    debug_steps: int,
    reset_params: dict[str, float] | None = None,
    zero_reference_rz_for_policy: bool = False,
) -> dict[str, Any]:
    policy.reset()
    options = reference_options(task_id, target_task, max_steps)
    if reset_params:
        options["reset_params"] = dict(reset_params)
    obs, _ = env.reset(options=options)
    trajectory = empty_trajectory()
    terminated = False
    truncated = False
    rollout_error: str | None = None

    for _ in range(max_steps):
        policy_obs = policy_observation(
            obs,
            zero_reference_rz=zero_reference_rz_for_policy,
        )
        action = policy.act(policy_obs)
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:
            rollout_error = f"{type(exc).__name__}: {exc}"
            if debug_actions:
                action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
                print(
                    f"[debug] {task_id} step_failed={len(trajectory['Ip']) + 1} "
                    f"action_abs_max={float(np.max(np.abs(action_arr))):.3f} "
                    f"error={rollout_error}"
                )
            break
        append_step(trajectory, obs, info, action)
        step = len(trajectory["Ip"])
        if debug_actions and (step <= debug_steps or terminated or truncated):
            print_debug(task_id, step, action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break

    return {
        "trajectory": trajectory,
        "f_surv": len(trajectory["Ip"]) / max(float(max_steps), 1.0),
        "timeout": False,
        "terminated": bool(terminated),
        "truncated": bool(truncated or rollout_error is not None),
        "n_steps": len(trajectory["Ip"]),
        "error": rollout_error,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evaluate(target_path: Path, output_path: Path) -> None:
    evaluate_path = EVAL_DIR / "evaluate.py"
    completed = subprocess.run(
        [sys.executable, str(evaluate_path), str(target_path), str(output_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)


def main() -> int:
    args = parse_args()
    selected_tasks = list(dict.fromkeys(args.tasks))
    policy1_path = Path(args.policy1_path).resolve()
    policy2_path = Path(args.policy2_path).resolve()
    target_path = Path(args.target_path).resolve()
    output_path = Path(args.output_path).resolve()
    config_path = Path(args.config).resolve()

    target = load_target(target_path)
    _require_file(config_path, "Environment config")
    if not args.allow_missing_model and "F1" in selected_tasks:
        _require_file(policy1_path, "F1 ONNX model")
    if not args.allow_missing_model and any(task_id in selected_tasks for task_id in ("F2a", "F2b")):
        _require_file(policy2_path, "F2 ONNX model")

    lengths = {
        "F1": task_length("F1", target["F1"], args.max_steps),
        "F2a": task_length("F2a", target["F2a"], args.max_steps),
        "F2b": task_length("F2b", target.get("F2b", target["F2a"]), args.max_steps),
    }
    print(f"Loaded target: {target_path}")
    print(f"Task lengths: {lengths}")
    print(f"Policy1: {policy1_path}")
    print(f"Policy2: {policy2_path}")
    print(f"Selected tasks: {selected_tasks}")
    print(f"Reset perturbation: {args.reset_perturbation}")
    print(f"Perturb apply mode: {args.perturb_apply}")
    print(f"Zero reference R/Z for policy: {args.zero_reference_rz_for_policy}")

    if args.validate_only:
        print("Validation passed; no HFM rollout requested.")
        return 0

    policies: dict[str, BaseOnnxPolicy] = {}
    if "F1" in selected_tasks:
        policies["F1"] = make_policy(policy1_path, task="F1", allow_missing=args.allow_missing_model)
    if "F2a" in selected_tasks or "F2b" in selected_tasks:
        policy2 = make_policy(policy2_path, task="F2", allow_missing=args.allow_missing_model)
        policies["F2a"] = policy2
        policies["F2b"] = policy2
    shots = {
        "F1": args.f1_shot_id,
        "F2a": args.f2a_shot_id,
        "F2b": args.f2b_shot_id,
    }
    strengths = args.perturb_strengths if args.perturb_strengths else [1.0]
    if args.reset_perturbation == "none":
        strengths = [0.0]

    for sweep_index, strength in enumerate(strengths):
        current_output_path = output_path_for_strength(
            output_path,
            strength=float(strength),
            total=len(strengths),
        )
        perturb_cfg = reset_perturbation_config(args, strength=float(strength))
        print(
            f"Sweep {sweep_index + 1}/{len(strengths)}: "
            f"strength={float(strength):g}, output={current_output_path}"
        )
        print(f"Perturb config: {perturb_cfg}")

        results: dict[str, Any] = {}
        for task_id in TASK_IDS:
            if task_id not in selected_tasks:
                results[task_id] = empty_task_result()
                continue
            task_target = target.get(task_id, target["F2a"])
            max_steps = lengths[task_id]
            env_config = load_env_config(
                config_path,
                host=args.host,
                port=args.port,
                shot_id=shots[task_id],
                max_steps=max_steps,
            )
            env = HFMSimulator(env_config)
            try:
                print(f"Rolling out {task_id}: shot={shots[task_id]}, max_steps={max_steps}")
                reset_params = sample_reset_params(
                    shot_id=shots[task_id],
                    reset_perturbation=perturb_cfg,
                    seed=int(args.perturb_seed) + sweep_index * 1000 + TASK_IDS.index(task_id),
                    apply_mode=args.perturb_apply,
                )
                if reset_params:
                    print(f"{task_id}: reset_params={reset_params}")
                results[task_id] = rollout_task(
                    task_id=task_id,
                    target_task=task_target,
                    env=env,
                    policy=policies[task_id],
                    max_steps=max_steps,
                    debug_actions=args.debug_actions,
                    debug_steps=args.debug_steps,
                    reset_params=reset_params,
                    zero_reference_rz_for_policy=bool(args.zero_reference_rz_for_policy),
                )
                print(
                    f"{task_id}: steps={results[task_id]['n_steps']}, "
                    f"terminated={results[task_id]['terminated']}, "
                    f"truncated={results[task_id]['truncated']}, "
                    f"error={results[task_id].get('error')}"
                )
            finally:
                env.close()

        write_json(current_output_path, results)
        print(f"Saved infer result to {current_output_path}")

        if not args.no_evaluate:
            run_evaluate(target_path, current_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
