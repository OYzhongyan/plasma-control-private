#!/usr/bin/env python3
"""Integrated F1/F2 robustness rollout for reset perturbation, observation noise, and target perturbation.

The script evaluates scenarios in this order:
1. baseline: official target, no reset perturb, no observation noise
2. reset_only: reset perturb only
3. noise_only: normalized observation-feature noise only
4. target_only: perturbed target LCFS only
5. combined: reset perturb + observation noise + perturbed target LCFS
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "evaluation" / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from environment import HFMSimulator  # noqa: E402
from environment.xpt_utils import extract_target_strike_points, extract_target_xpoints  # noqa: E402
from eval_files.trajectory_utils import extract_semifinal_trajectory_from_obs_list  # noqa: E402
from score import config as score_config  # noqa: E402
from score.evaluate_core import evaluate  # noqa: E402
from training.distill_noise_student import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_VECNORMALIZE,
    FeatureSequenceBuilder,
    build_raw_env,
    load_vecnormalize_stats,
    noise_feature_mask,
    normalize_obs,
    parse_float_list,
    parse_int_spec,
)
from training.features import (  # noqa: E402
    ACTION_SCALE_7D_V,
    COIL_LIMITS_A,
    FeatureMetadata,
    _array,
    _clip_finite,
    _scalar,
    physical_f2_xpt_feature_vector,
    physical_feature_vector,
)
from training.references import TargetReference, build_reference, load_target_reference, normalize_task  # noqa: E402
from training.train_ppo import (  # noqa: E402
    ResetPerturbationSampler,
    _rooted_path,
    load_train_config,
    load_yaml,
    make_env_config,
    parse_ports,
)

DEFAULT_TARGET = PROJECT_ROOT / "evaluation/results/target.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/integrated_robustness_ablation"



class GenericFeatureSequenceBuilder:
    """Offline feature builder matching PhysicalFeatureHistoryWrapper for F1/F2 physical modes."""

    def __init__(self, metadata: FeatureMetadata):
        self.metadata = metadata
        self.history_steps = max(0, int(metadata.history_steps))
        self._last_observation: dict[str, Any] | None = None
        self._reset_history()

    @staticmethod
    def _push(buffer: np.ndarray, value: np.ndarray) -> np.ndarray:
        if buffer.shape[0] == 0:
            return buffer
        out = np.roll(buffer, shift=-1, axis=0)
        out[-1] = value
        return out

    def _reset_history(self) -> None:
        self._history_action = np.zeros((self.history_steps, 7), dtype=np.float32)
        self._history_delta_ip = np.zeros((self.history_steps, 1), dtype=np.float32)
        self._history_delta_rz = np.zeros((self.history_steps, 2), dtype=np.float32)
        self._history_delta_i_pf = np.zeros((self.history_steps, 12), dtype=np.float32)

    def _history_vector(self) -> np.ndarray:
        if self.history_steps <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [
                self._history_action.reshape(-1),
                self._history_delta_ip.reshape(-1),
                self._history_delta_rz.reshape(-1),
                self._history_delta_i_pf.reshape(-1),
            ]
        ).astype(np.float32)

    def _base_feature(self, observation: dict[str, Any]) -> np.ndarray:
        if self.metadata.feature_mode in {"physical_f2_xpt_v1", "physical_f2_xpt_no_ref_rz_v1"}:
            return physical_f2_xpt_feature_vector(
                observation,
                no_ref_rz=self.metadata.feature_mode == "physical_f2_xpt_no_ref_rz_v1",
                dtype=np.float32,
            )
        return physical_feature_vector(
            observation,
            self.metadata.task,
            mode=self.metadata.feature_mode,
            dtype=np.float32,
        )

    def _build_observation(self, observation: dict[str, Any]) -> np.ndarray:
        out = np.concatenate([self._base_feature(observation), self._history_vector()]).astype(np.float32)
        if out.shape != (self.metadata.obs_dim,):
            raise ValueError(f"feature dim mismatch: got {out.shape}, expected {(self.metadata.obs_dim,)}")
        return out

    def reset(self, observation: dict[str, Any]) -> np.ndarray:
        self._last_observation = observation
        self._reset_history()
        return self._build_observation(observation)

    def step(self, action: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        prev = self._last_observation
        if self.history_steps > 0:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            action7 = np.zeros(7, dtype=np.float32)
            n_action = min(7, action_arr.size)
            if n_action > 0:
                action7[:n_action] = action_arr[:n_action]
            if self.metadata.action_units == "normalized":
                action_norm = np.clip(action7, -1.0, 1.0)
            else:
                action_norm = action7 / ACTION_SCALE_7D_V

            if prev is None:
                delta_ip = np.zeros(1, dtype=np.float32)
                delta_rz = np.zeros(2, dtype=np.float32)
                delta_i_pf = np.zeros(12, dtype=np.float32)
            else:
                delta_ip = np.asarray([(_scalar(observation, "Ip") - _scalar(prev, "Ip")) / 50_000.0], dtype=np.float32)
                delta_rz = np.asarray(
                    [
                        (_scalar(observation, "R") - _scalar(prev, "R")) / 0.03,
                        (_scalar(observation, "Z") - _scalar(prev, "Z")) / 0.03,
                    ],
                    dtype=np.float32,
                )
                delta_i_pf = (_array(observation, "I_PF", (12,)) - _array(prev, "I_PF", (12,))) / COIL_LIMITS_A

            self._history_action = self._push(self._history_action, _clip_finite(action_norm))
            self._history_delta_ip = self._push(self._history_delta_ip, _clip_finite(delta_ip))
            self._history_delta_rz = self._push(self._history_delta_rz, _clip_finite(delta_rz))
            self._history_delta_i_pf = self._push(self._history_delta_i_pf, _clip_finite(delta_i_pf))

        self._last_observation = observation
        return self._build_observation(observation)

class OnnxPolicyAdapter:
    """Small predict() adapter matching the SB3 PPO interface used by this script."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs: np.ndarray, deterministic: bool = True):  # noqa: ARG002 - SB3-compatible signature.
        obs = np.asarray(obs, dtype=np.float32)
        action = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if action.shape[-1] == 12:
            action = action[:, [0, 1, 3, 5, 7, 9, 11]]
        return action, None


@dataclass(frozen=True)
class TargetVariant:
    variant_id: str
    source_shot: str
    source_seed: int
    reset_params: dict[str, float]
    lcfs: np.ndarray
    target_lcfs_delta_cm: float
    rX: np.ndarray | None = None
    zX: np.ndarray | None = None
    strike_r: np.ndarray | None = None
    strike_z: np.ndarray | None = None
    x_valid: np.ndarray | None = None
    strike_valid: np.ndarray | None = None


def infer_model_format(model_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "onnx" if model_path.suffix.lower() == ".onnx" else "sb3"


def same_path(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return Path(a) == Path(b)


def onnx_metadata_path(model_path: Path, current: Path) -> Path:
    candidate = model_path.with_suffix(".metadata.json")
    if same_path(current, DEFAULT_METADATA) and candidate.is_file():
        return candidate
    return current


def onnx_vecnormalize_path(model_path: Path, current: Path) -> Path:
    stem = model_path.stem
    candidates = [
        model_path.with_name(f"{stem}_vecnormalize_stats.npz"),
        model_path.with_name(f"{stem}.vecnormalize_stats.npz"),
        model_path.with_name(f"{stem}_vecnormalize.npz"),
        model_path.with_name(f"{stem}.vecnormalize.npz"),
        model_path.parent / "vecnormalize_stats.npz",
    ]
    if same_path(current, DEFAULT_VECNORMALIZE):
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return current


def load_policy_model(model_path: Path, *, model_format: str, device: str):
    if model_format == "onnx":
        return OnnxPolicyAdapter(model_path)
    from stable_baselines3 import PPO

    return PPO.load(str(model_path), device=device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated F1/F2 robustness rollout ablation.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--model-format", choices=["auto", "sb3", "onnx"], default="auto", help="Load --model as SB3 PPO zip or exported ONNX policy. auto uses the file suffix.")
    parser.add_argument("--model-label", type=str, default="policy")
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ports", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel rollout workers. Requires at least this many HFM ports.")
    parser.add_argument("--rollout-shots", type=str, default="13844_500,13844_600")
    parser.add_argument("--rollout-seeds", type=str, default="0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--strict-rollout", action="store_true")

    parser.add_argument("--noise-stds", type=str, default="0.03", help="Normalized feature noise stds for noise_only/combined.")
    parser.add_argument("--noise-feature-groups", type=str, default="ip,rz,lcfs,xpt,history_ip,history_rz")

    parser.add_argument("--reset-perturb-mode", choices=["manual", "config"], default="manual")
    parser.add_argument("--reset-perturb-apply-mode", choices=["all", "one"], default="all")
    parser.add_argument("--reset-signeo-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--reset-bp-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--reset-q0-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--reset-bp-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--reset-q0-delta-range", type=str, default="0.0,0.0")

    parser.add_argument("--target-variant-source", choices=["sample", "file", "train_config"], default="sample", help="Where perturbed targets come from. sample generates fresh HFM reset targets for evaluation; file/train_config can reproduce a fixed target set.")
    parser.add_argument("--target-variants-file", type=Path, default=None, help="Precomputed target variants JSON used only with --target-variant-source=file.")
    parser.add_argument("--target-source-shots", type=str, default="15892_400,15892_500")
    parser.add_argument("--target-seeds", type=str, default="0-3")
    parser.add_argument("--target-perturb-apply-mode", choices=["all", "one"], default="all")
    parser.add_argument("--target-signeo-scale-range", type=str, default="0.9,1.1")
    parser.add_argument("--target-bp-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--target-q0-scale-range", type=str, default="0.95,1.05")
    parser.add_argument("--target-bp-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--target-q0-delta-range", type=str, default="0.0,0.0")
    parser.add_argument("--target-variant-indices", type=str, default="", help="Subset of perturbed target variants after sampling, e.g. 0,2-3.")
    parser.add_argument("--scenarios", type=str, default="baseline,reset_only,noise_only,target_only,combined")
    parser.add_argument("--all-once", action="store_true", help="Run exactly one combined rollout with one target variant, one shot, one seed, and one noise std.")
    parser.add_argument("--all-once-target-index", type=int, default=None, help="Target variant index for --all-once. Default samples randomly.")
    parser.add_argument("--all-once-random-seed", type=int, default=0, help="Random seed used to choose target/shot/seed when --all-once values are not fixed.")
    parser.add_argument("--save-target-variants", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _split_csv(spec: str) -> list[str]:
    values = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty csv spec: {spec!r}")
    return values


def _range_pair(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError(f"range must contain exactly two numbers, got {spec!r}")
    lo, hi = values
    if lo > hi:
        raise ValueError(f"range low must be <= high, got {spec!r}")
    return [lo, hi]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def lcfs_rms_delta_cm(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float64).reshape(32, 2)
    arr_b = np.asarray(b, dtype=np.float64).reshape(32, 2)
    return float(np.sqrt(np.mean(np.sum((arr_a - arr_b) ** 2, axis=1))) * 100.0)


def perturb_config(prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "apply_mode": str(getattr(args, f"{prefix}_perturb_apply_mode")),
        "signeo_scale_range": _range_pair(getattr(args, f"{prefix}_signeo_scale_range")),
        "bp_scale_range": _range_pair(getattr(args, f"{prefix}_bp_scale_range")),
        "q0_scale_range": _range_pair(getattr(args, f"{prefix}_q0_scale_range")),
        "bp_delta_range": _range_pair(getattr(args, f"{prefix}_bp_delta_range")),
        "q0_delta_range": _range_pair(getattr(args, f"{prefix}_q0_delta_range")),
    }


def infer_task_id(stage: str, shots: list[str] | None = None) -> str:
    stage_lower = str(stage).lower()
    if "f2b" in stage_lower:
        return "F2b"
    if "f2a" in stage_lower:
        return "F2a"
    if "f1" in stage_lower:
        return "F1"
    if shots and all(str(shot).startswith("15892") for shot in shots):
        return "F2b"
    return "F2a"



def integrated_noise_feature_mask(metadata: FeatureMetadata, groups_spec: str) -> np.ndarray:
    groups = [part.strip() for part in str(groups_spec or "").split(",") if part.strip()]
    expanded: list[str] = []
    xpt_keys = [
        "xpt_nx_norm",
        "xpt_nx_error",
        "xpt_x_valid",
        "xpt_r",
        "xpt_z",
        "xpt_position_error_3cm",
        "xpt_flux_error_0p003",
    ]
    available = set(metadata.keys)
    for group in groups:
        if group == "xpt":
            expanded.extend([key for key in xpt_keys if key in available])
        else:
            expanded.append(group)
    return noise_feature_mask(metadata, ",".join(expanded)).astype(np.float32)

def build_context(args: argparse.Namespace, metadata: FeatureMetadata) -> dict[str, Any]:
    train_config = load_train_config(_rooted_path(args.train_config))
    task = normalize_task(train_config.get("task", "F1"))
    max_steps = int(args.max_steps or train_config.get("max_steps") or 300)
    env_config_path = _rooted_path(train_config.get("env_config"), PROJECT_ROOT / "configs/env_default.yaml")
    target_path = _rooted_path(args.target)
    target_all = json.loads(target_path.read_text(encoding="utf-8"))
    rollout_shots = _split_csv(args.rollout_shots)
    task_id = "F1" if task == "F1" else infer_task_id(str(train_config.get("stage", "")), rollout_shots)
    official_ref = load_target_reference(target_path, task=task)
    return {
        "train_config": train_config,
        "task": task,
        "task_id": task_id,
        "base_config": load_yaml(env_config_path),
        "max_steps": max_steps,
        "metadata": metadata,
        "target_all": target_all,
        "official_ref": official_ref,
        "strict_rollout": bool(args.strict_rollout),
    }



def _optional_array(record: dict[str, Any], names: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray | None:
    for name in names:
        if name in record:
            return np.asarray(record[name], dtype=np.float64).reshape(shape)
    return None


def load_target_variants_file(path: Path, official_ref: TargetReference) -> list[TargetVariant]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("variants") or data.get("target_lcfs_variants") or []
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"unsupported target variants JSON format: {path}")
    variants: list[TargetVariant] = []
    official_lcfs = np.asarray(official_ref.lcfs, dtype=np.float64).reshape(32, 2)
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        variant_id = str(record.get("variant_id", record.get("id", f"variant_{idx}")))
        if variant_id == "official":
            continue
        lcfs_value = record.get("lcfs_points", record.get("lcfs", record.get("reference_lcfs_points")))
        if lcfs_value is None:
            continue
        lcfs = np.asarray(lcfs_value, dtype=np.float64).reshape(32, 2)
        variants.append(
            TargetVariant(
                variant_id=variant_id,
                source_shot=str(record.get("source_shot", "file")),
                source_seed=int(record.get("source_seed", -1)),
                reset_params={k: float(v) for k, v in dict(record.get("reset_params", {})).items()},
                lcfs=lcfs,
                target_lcfs_delta_cm=float(record.get("target_lcfs_delta_cm", lcfs_rms_delta_cm(lcfs, official_lcfs))),
                rX=_optional_array(record, ("reference_rX", "rX"), (4,)),
                zX=_optional_array(record, ("reference_zX", "zX"), (4,)),
                strike_r=_optional_array(record, ("reference_strike_r", "strike_r"), (8,)),
                strike_z=_optional_array(record, ("reference_strike_z", "strike_z"), (8,)),
                x_valid=_optional_array(record, ("reference_x_valid", "x_valid"), (4,)),
                strike_valid=_optional_array(record, ("reference_strike_valid", "strike_valid"), (8,)),
            )
        )
    if not variants:
        raise ValueError(f"no usable perturbed target variants in {path}")
    return variants

def sample_target_variants(args: argparse.Namespace, ctx: dict[str, Any], port: int) -> tuple[TargetVariant, list[TargetVariant]]:
    official_ref: TargetReference = ctx["official_ref"]
    official_lcfs = np.asarray(official_ref.lcfs, dtype=np.float64).reshape(32, 2)
    official = TargetVariant(
        variant_id="official",
        source_shot="official",
        source_seed=-1,
        reset_params={},
        lcfs=official_lcfs.copy(),
        target_lcfs_delta_cm=0.0,
        rX=official_ref.rX,
        zX=official_ref.zX,
        strike_r=official_ref.strike_r,
        strike_z=official_ref.strike_z,
        x_valid=official_ref.x_valid,
        strike_valid=official_ref.strike_valid,
    )
    target_source = str(args.target_variant_source)
    if target_source == "file":
        if args.target_variants_file is None:
            raise ValueError("--target-variant-source=file requires --target-variants-file")
        variants = load_target_variants_file(_rooted_path(args.target_variants_file), official_ref)
    elif target_source == "train_config":
        config_path = ctx["train_config"].get("reference_variants")
        if not config_path:
            raise ValueError("--target-variant-source=train_config requires train_config.reference_variants")
        variants = load_target_variants_file(_rooted_path(Path(config_path)), official_ref)
    else:
        if args.target_variants_file is not None:
            print(
                "warning: --target-variants-file is ignored because --target-variant-source=sample; "
                "fresh target variants will be sampled for out-of-sample evaluation",
                flush=True,
            )
        variants = []
        perturb = perturb_config("target", args)
        for source_shot in _split_csv(args.target_source_shots):
            for source_seed in parse_int_spec(args.target_seeds):
                sampler = ResetPerturbationSampler(
                    shot_id=source_shot,
                    reset_perturbation=perturb,
                    seed=int(source_seed),
                )
                reset_params = sampler.sample()
                cfg = make_env_config(
                    ctx["base_config"],
                    port=port,
                    shot_id=source_shot,
                    max_steps=1,
                    reference_mode="hold_initial",
                )
                env = HFMSimulator(cfg, reward_fn=None)
                try:
                    obs, _info = env.reset(
                        seed=int(source_seed),
                        options={"reference_mode": "hold", "reset_params": reset_params},
                    )
                    lcfs = np.asarray(obs["lcfs_points"], dtype=np.float64).reshape(32, 2)
                    rX = zX = x_valid = strike_r = strike_z = strike_valid = None
                    if ctx["task"] == "F2":
                        rX, zX, x_valid = extract_target_xpoints(obs, slots=4)
                        strike_r, strike_z, strike_valid, _n_strike = extract_target_strike_points(obs, strike_slots=8)
                except Exception as exc:
                    if ctx.get("strict_rollout"):
                        raise
                    print(
                        f"warning: failed to sample target variant source={source_shot} seed={source_seed}: {exc}",
                        flush=True,
                    )
                    continue
                finally:
                    env.close()
                variant = TargetVariant(
                    variant_id=f"{source_shot}_seed{source_seed}",
                    source_shot=source_shot,
                    source_seed=int(source_seed),
                    reset_params={k: float(v) for k, v in reset_params.items()},
                    lcfs=lcfs,
                    target_lcfs_delta_cm=lcfs_rms_delta_cm(lcfs, official_lcfs),
                    rX=None if rX is None else np.asarray(rX, dtype=np.float64),
                    zX=None if zX is None else np.asarray(zX, dtype=np.float64),
                    strike_r=None if strike_r is None else np.asarray(strike_r, dtype=np.float64),
                    strike_z=None if strike_z is None else np.asarray(strike_z, dtype=np.float64),
                    x_valid=None if x_valid is None else np.asarray(x_valid, dtype=np.float64),
                    strike_valid=None if strike_valid is None else np.asarray(strike_valid, dtype=np.float64),
                )
                variants.append(variant)
                print(
                    f"sampled target variant={variant.variant_id} source={target_source} "
                    f"delta_cm={variant.target_lcfs_delta_cm:.3f}",
                    flush=True,
                )
    if args.target_variant_indices.strip():
        selected = parse_int_spec(args.target_variant_indices)
        bad = [i for i in selected if i < 0 or i >= len(variants)]
        if bad:
            raise IndexError(f"target variant indices out of range {bad}; sampled/loaded variants={len(variants)}")
        variants = [variants[i] for i in selected]
    if not variants:
        raise RuntimeError("no perturbed target variants sampled")
    return official, variants

def target_and_reference(ctx: dict[str, Any], variant: TargetVariant) -> tuple[dict[str, Any], TargetReference, dict[str, Any]]:
    target_all = copy.deepcopy(ctx["target_all"])
    task = str(ctx["task"])
    task_id = str(ctx["task_id"])
    target_key = "F1" if task == "F1" else task_id
    if target_key not in target_all and task == "F2":
        target_key = "F2a"
    if target_key in target_all:
        target_all[target_key]["lcfs_points"] = np.asarray(variant.lcfs, dtype=np.float64).tolist()
        if task == "F2" and variant.rX is not None and variant.zX is not None:
            rx = np.asarray(variant.rX, dtype=np.float64).reshape(4)
            zx = np.asarray(variant.zX, dtype=np.float64).reshape(4)
            target_all[target_key]["Xpt_sec"] = [[float(rx[0]), float(zx[0])], [float(rx[3]), float(zx[3])]]
            target_all[target_key]["Xpt_main"] = [[float(rx[1]), float(zx[1])], [float(rx[2]), float(zx[2])]]
        if task == "F2" and variant.strike_r is not None and variant.strike_z is not None:
            sr = np.asarray(variant.strike_r, dtype=np.float64).reshape(8)
            sz = np.asarray(variant.strike_z, dtype=np.float64).reshape(8)
            target_all[target_key]["strike"] = np.stack([sr, sz], axis=1).tolist()
    official_ref: TargetReference = ctx["official_ref"]
    target_ref = TargetReference(
        lcfs=np.asarray(variant.lcfs, dtype=np.float64),
        ip_final=official_ref.ip_final,
        rX=variant.rX if variant.rX is not None else official_ref.rX,
        zX=variant.zX if variant.zX is not None else official_ref.zX,
        strike_r=variant.strike_r if variant.strike_r is not None else official_ref.strike_r,
        strike_z=variant.strike_z if variant.strike_z is not None else official_ref.strike_z,
        x_valid=variant.x_valid if variant.x_valid is not None else official_ref.x_valid,
        strike_valid=variant.strike_valid if variant.strike_valid is not None else official_ref.strike_valid,
    )
    reference = build_reference(task, target_ref, int(ctx["max_steps"]))
    return target_all, target_ref, reference


def coil_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    icoil = np.asarray(trajectory.get("Icoil", []), dtype=np.float64)
    if icoil.ndim != 2 or icoil.shape[0] == 0:
        return {"max_coil_margin": float("nan"), "mean_coil_margin": float("nan"), "first_overlimit_step": -1, "overlimit_steps": 0}
    limits = score_config.COIL_LIMITS_A[: icoil.shape[1]]
    margins = np.max(np.abs(icoil) / limits[None, :], axis=1)
    over = margins > 1.0
    return {
        "max_coil_margin": float(np.max(margins)),
        "mean_coil_margin": float(np.mean(margins)),
        "first_overlimit_step": int(np.argmax(over) + 1) if np.any(over) else -1,
        "overlimit_steps": int(np.sum(over)),
    }


def mean_component(rows: list[dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def rollout_one(
    *,
    model,
    model_label: str,
    scenario: str,
    ctx: dict[str, Any],
    variant: TargetVariant,
    shot_id: str,
    seed: int,
    port: int,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    clip_obs: float,
    eps: float,
    noise_std: float,
    feature_noise_mask: np.ndarray,
    reset_perturbation: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: FeatureMetadata = ctx["metadata"]
    target_all, target_ref, reference = target_and_reference(ctx, variant)
    env = build_raw_env(
        train_config=ctx["train_config"],
        base_config=ctx["base_config"],
        reference=reference,
        target=target_ref,
        shot_id=shot_id,
        port=port,
        max_steps=int(ctx["max_steps"]),
        seed=seed,
        action_units=metadata.action_units,
        reset_perturbation=reset_perturbation,
    )
    builder = GenericFeatureSequenceBuilder(metadata)
    rng = np.random.default_rng(seed + int(round(noise_std * 1_000_000)) + abs(hash((scenario, variant.variant_id))) % 100_000)
    obs_list: list[dict[str, Any]] = []
    rewards: list[float] = []
    component_rows: list[dict[str, float]] = []
    error = ""
    terminated_flag = False
    truncated_flag = False
    last_failure = float("nan")
    try:
        raw_obs, _ = env.reset(seed=seed)
        feature = builder.reset(dict(raw_obs))
        for _step in range(1, int(ctx["max_steps"]) + 1):
            clean_norm = normalize_obs(feature, obs_mean, obs_var, clip_obs, eps)
            if float(noise_std) > 0.0:
                norm = clean_norm + rng.normal(0.0, float(noise_std), size=clean_norm.shape).astype(np.float32) * feature_noise_mask
            else:
                norm = clean_norm
            action, _ = model.predict(norm.reshape(1, -1), deterministic=True)
            action7 = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            raw_next, reward, terminated, truncated, info = env.step(action7)
            obs_list.append(dict(raw_next))
            rewards.append(float(reward))
            components = dict((info or {}).get("reward_components", {}) or {})
            component_rows.append({k: float(v) for k, v in components.items() if isinstance(v, (int, float, np.number))})
            last_failure = float(np.asarray(raw_next.get("failure", np.nan)).reshape(-1)[0]) if "failure" in raw_next else float("nan")
            terminated_flag = bool(terminated)
            truncated_flag = bool(truncated)
            feature = builder.step(action7, dict(raw_next))
            if terminated or truncated:
                break
    except Exception as exc:
        if ctx.get("strict_rollout"):
            raise
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    task_id = str(ctx["task_id"])
    target_obs = target_all.get(task_id, target_all.get("F2a", target_all.get("F1")))
    trajectory = extract_semifinal_trajectory_from_obs_list(obs_list, task_id=task_id, target_obs=target_obs)
    result = evaluate({task_id: {"trajectory": trajectory, "timeout": False}}, target_all, task_ids=[task_id])
    metric_scores = result["task_metric_scores"].get(task_id, {})
    row = {
        "scenario": scenario,
        "model": model_label,
        "variant_id": variant.variant_id,
        "target_source_shot": variant.source_shot,
        "target_source_seed": variant.source_seed,
        "target_lcfs_delta_cm": variant.target_lcfs_delta_cm,
        "target_reset_params": json.dumps(variant.reset_params, sort_keys=True),
        "rollout_shot": shot_id,
        "rollout_seed": int(seed),
        "noise_std": float(noise_std),
        "reset_perturb_enabled": bool(reset_perturbation and reset_perturbation.get("enabled", False)),
        "n_steps": len(obs_list),
        "f_surv": len(obs_list) / max(float(ctx["max_steps"]), 1.0),
        "early_stop": int(len(obs_list) < int(ctx["max_steps"])),
        "terminated": int(terminated_flag),
        "truncated": int(truncated_flag),
        "last_failure": last_failure,
        "score_task": float(result["task_scores"].get(task_id, 0.0)),
        "score_f1": float(result["task_scores"].get(task_id, 0.0)),
        "score_ip": float(metric_scores.get("Ip", 0.0)),
        "score_lcfs": float(metric_scores.get("LCFS", metric_scores.get("LCFS_points", 0.0))),
        "score_xpt": float(metric_scores.get("Xpt", metric_scores.get("X-point", metric_scores.get("xpt", 0.0)))),
        "score_strike": float(metric_scores.get("strike", metric_scores.get("Strike", 0.0))),
        "score_psix": float(metric_scores.get("Psix", metric_scores.get("psi_x", 0.0))),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "mean_ip_abs_error_ka": mean_component(component_rows, "ip_abs_error") / 1000.0,
        "mean_lcfs_cm": mean_component(component_rows, "lcfs_cm"),
        "mean_topology_score": mean_component(component_rows, "topology_score"),
        "error": error,
    }
    row.update(coil_metrics(trajectory))
    return row



def nanmin_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def nanmax_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(str(r["scenario"]), float(r["noise_std"])) for r in rows}, key=lambda x: (x[0], x[1]))
    for scenario, noise_std in groups:
        group = [r for r in rows if str(r["scenario"]) == scenario and float(r["noise_std"]) == noise_std]
        def mean(key: str) -> float:
            vals = [float(r[key]) for r in group if key in r and np.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        out.append({
            "scenario": scenario,
            "noise_std": noise_std,
            "episodes": len(group),
            "error_episodes": sum(1 for r in group if r.get("error")),
            "early_stop_episode_count": sum(1 for r in group if int(r.get("early_stop", 0)) > 0),
            "overlimit_episode_count": sum(1 for r in group if int(r.get("first_overlimit_step", -1)) > 0),
            "score_task_mean": mean("score_task"),
            "score_task_min": nanmin_or_nan([float(r.get("score_task", r.get("score_f1", 0.0))) for r in group]),
            "score_f1_mean": mean("score_f1"),
            "score_f1_min": nanmin_or_nan([float(r["score_f1"]) for r in group]),
            "score_ip_mean": mean("score_ip"),
            "score_lcfs_mean": mean("score_lcfs"),
            "f_surv_mean": mean("f_surv"),
            "n_steps_mean": mean("n_steps"),
            "max_coil_margin_mean": mean("max_coil_margin"),
            "max_coil_margin_max": nanmax_or_nan([float(r["max_coil_margin"]) for r in group]),
            "ip_abs_error_ka_mean": mean("mean_ip_abs_error_ka"),
            "lcfs_cm_mean": mean("mean_lcfs_cm"),
            "target_lcfs_delta_cm_mean": mean("target_lcfs_delta_cm"),
        })
    return out


def save_target_variants(path: Path, official: TargetVariant, variants: list[TargetVariant]) -> None:
    payload = [official, *variants]
    path.write_text(json.dumps([
        {
            "variant_id": v.variant_id,
            "source_shot": v.source_shot,
            "source_seed": v.source_seed,
            "reset_params": v.reset_params,
            "target_lcfs_delta_cm": v.target_lcfs_delta_cm,
            "lcfs_points": v.lcfs,
            "reference_rX": v.rX,
            "reference_zX": v.zX,
            "reference_strike_r": v.strike_r,
            "reference_strike_z": v.strike_z,
            "reference_x_valid": v.x_valid,
            "reference_strike_valid": v.strike_valid,
        }
        for v in payload
    ], ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def build_rollout_tasks(
    *,
    scenarios: list[str],
    official_variant: TargetVariant,
    perturbed_variants: list[TargetVariant],
    reset_perturbation: dict[str, Any],
    noise_stds: list[float],
    rollout_shots: list[str],
    rollout_seeds: list[int],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario not in {"baseline", "reset_only", "noise_only", "target_only", "combined"}:
            raise ValueError(f"unknown scenario: {scenario}")
        scenario_variants = [official_variant] if scenario in {"baseline", "reset_only", "noise_only"} else perturbed_variants
        scenario_reset = reset_perturbation if scenario in {"reset_only", "combined"} else {"enabled": False}
        scenario_noise_stds = noise_stds if scenario in {"noise_only", "combined"} else [0.0]
        for noise_std in scenario_noise_stds:
            for variant in scenario_variants:
                for shot_id in rollout_shots:
                    for seed in rollout_seeds:
                        tasks.append(
                            {
                                "scenario": scenario,
                                "variant": variant,
                                "shot_id": shot_id,
                                "seed": int(seed),
                                "noise_std": float(noise_std),
                                "reset_perturbation": copy.deepcopy(scenario_reset),
                            }
                        )
    return tasks


def rollout_worker(
    *,
    worker_id: int,
    port: int,
    tasks: list[dict[str, Any]],
    model_path: Path,
    model_format: str,
    device: str,
    model_label: str,
    ctx: dict[str, Any],
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    clip_obs: float,
    eps: float,
    feature_noise_mask: np.ndarray,
) -> list[dict[str, Any]]:
    model = load_policy_model(model_path, model_format=model_format, device=device)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        print(
            f"worker={worker_id} port={port} rollout scenario={task['scenario']} "
            f"noise={task['noise_std']} variant={task['variant'].variant_id} "
            f"shot={task['shot_id']} seed={task['seed']}",
            flush=True,
        )
        row = rollout_one(
            model=model,
            model_label=model_label,
            scenario=str(task["scenario"]),
            ctx=ctx,
            variant=task["variant"],
            shot_id=str(task["shot_id"]),
            seed=int(task["seed"]),
            port=int(port),
            obs_mean=obs_mean,
            obs_var=obs_var,
            clip_obs=clip_obs,
            eps=eps,
            noise_std=float(task["noise_std"]),
            feature_noise_mask=feature_noise_mask,
            reset_perturbation=task["reset_perturbation"],
        )
        row["worker_id"] = int(worker_id)
        row["port"] = int(port)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
    return rows
def main() -> int:
    args = parse_args()
    args.output_dir = _rooted_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = _rooted_path(args.model)
    model_format = infer_model_format(model_path, str(args.model_format))
    metadata_path = _rooted_path(args.metadata)
    vecnormalize_path = _rooted_path(args.vecnormalize)
    if model_format == "onnx":
        metadata_path = onnx_metadata_path(model_path, metadata_path)
        vecnormalize_path = onnx_vecnormalize_path(model_path, vecnormalize_path)

    metadata = FeatureMetadata.load(metadata_path)
    ctx = build_context(args, metadata)
    ports = parse_ports(args.ports or ctx["train_config"].get("ports") or "5558")
    obs_mean, obs_var, clip_obs, eps = load_vecnormalize_stats(vecnormalize_path)
    feature_noise_mask = integrated_noise_feature_mask(metadata, args.noise_feature_groups)
    noise_stds = parse_float_list(args.noise_stds)
    rollout_shots = _split_csv(args.rollout_shots)
    rollout_seeds = parse_int_spec(args.rollout_seeds)
    scenarios = _split_csv(args.scenarios)

    official_variant, perturbed_variants = sample_target_variants(args, ctx, ports[0])
    if args.all_once:
        rng = np.random.default_rng(int(args.all_once_random_seed))
        if args.all_once_target_index is None:
            chosen_target_index = int(rng.integers(0, len(perturbed_variants)))
        else:
            chosen_target_index = int(args.all_once_target_index)
        if chosen_target_index < 0 or chosen_target_index >= len(perturbed_variants):
            raise IndexError(
                f"--all-once-target-index {chosen_target_index} out of range 0..{len(perturbed_variants) - 1}"
            )
        perturbed_variants = [perturbed_variants[chosen_target_index]]
        rollout_shots = [rollout_shots[int(rng.integers(0, len(rollout_shots)))] if len(rollout_shots) > 1 else rollout_shots[0]]
        rollout_seeds = [rollout_seeds[int(rng.integers(0, len(rollout_seeds)))] if len(rollout_seeds) > 1 else rollout_seeds[0]]
        noise_stds = [float(noise_stds[0])]
        scenarios = ["combined"]
        print(
            f"all_once selected target_index={chosen_target_index} "
            f"variant={perturbed_variants[0].variant_id} shot={rollout_shots[0]} "
            f"seed={rollout_seeds[0]} noise_std={noise_stds[0]}",
            flush=True,
        )
    if args.save_target_variants:
        save_target_variants(args.output_dir / "target_lcfs_variants.json", official_variant, perturbed_variants)

    if str(args.reset_perturb_mode) == "config":
        reset_perturbation = ctx["train_config"].get("reset_perturbation") or {"enabled": False}
    else:
        reset_perturbation = perturb_config("reset", args)

    tasks = build_rollout_tasks(
        scenarios=scenarios,
        official_variant=official_variant,
        perturbed_variants=perturbed_variants,
        reset_perturbation=reset_perturbation,
        noise_stds=noise_stds,
        rollout_shots=rollout_shots,
        rollout_seeds=rollout_seeds,
    )
    requested_workers = max(1, int(args.num_workers))
    if requested_workers > len(ports):
        raise ValueError(f"--num-workers={requested_workers} requires at least that many ports, got {ports}")
    worker_count = min(requested_workers, max(1, len(tasks)))
    worker_ports = ports[:worker_count]
    task_shards = [tasks[i::worker_count] for i in range(worker_count)]
    print(
        f"integrated rollout: scenarios={scenarios} shots={rollout_shots} seeds={rollout_seeds} "
        f"noise_stds={noise_stds} target_variants={len(perturbed_variants)} reset={reset_perturbation} "
        f"tasks={len(tasks)} workers={worker_count} ports={worker_ports}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    if worker_count == 1:
        rows.extend(
            rollout_worker(
                worker_id=0,
                port=int(worker_ports[0]),
                tasks=task_shards[0],
                model_path=model_path,
                model_format=model_format,
                device=args.device,
                model_label=str(args.model_label),
                ctx=ctx,
                obs_mean=obs_mean,
                obs_var=obs_var,
                clip_obs=clip_obs,
                eps=eps,
                feature_noise_mask=feature_noise_mask,
            )
        )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    rollout_worker,
                    worker_id=i,
                    port=int(worker_ports[i]),
                    tasks=task_shards[i],
                    model_path=model_path,
                    model_format=model_format,
                    device=args.device,
                    model_label=str(args.model_label),
                    ctx=ctx,
                    obs_mean=obs_mean,
                    obs_var=obs_var,
                    clip_obs=clip_obs,
                    eps=eps,
                    feature_noise_mask=feature_noise_mask,
                )
                for i in range(worker_count)
                if task_shards[i]
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())
    rows.sort(key=lambda r: (str(r.get("scenario", "")), float(r.get("noise_std", 0.0)), str(r.get("variant_id", "")), str(r.get("rollout_shot", "")), int(r.get("rollout_seed", 0))))
    summary = aggregate(rows)
    _write_csv(args.output_dir / "rollout_rows.csv", rows)
    _write_csv(args.output_dir / "rollout_summary.csv", summary)
    (args.output_dir / "rollout_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (args.output_dir / "rollout_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    manifest = {
        "task": str(ctx["task"]),
        "task_id": str(ctx["task_id"]),
        "model": str(model_path),
        "model_format": model_format,
        "vecnormalize": str(vecnormalize_path),
        "metadata": str(metadata_path),
        "train_config": str(_rooted_path(args.train_config)),
        "noise_stds": noise_stds,
        "noise_feature_groups": str(args.noise_feature_groups),
        "reset_perturbation": reset_perturbation,
        "scenarios": scenarios,
        "all_once": bool(args.all_once),
        "all_once_target_index": args.all_once_target_index,
        "all_once_random_seed": int(args.all_once_random_seed),
        "rollout_shots": rollout_shots,
        "rollout_seeds": rollout_seeds,
        "target_variant_source": str(args.target_variant_source),
        "target_variants_file": str(_rooted_path(args.target_variants_file)) if args.target_variants_file else "",
        "target_source_shots": _split_csv(args.target_source_shots),
        "target_seeds": str(args.target_seeds),
        "num_workers": int(worker_count),
        "ports": worker_ports,
        "tasks": len(tasks),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"rows: {args.output_dir / 'rollout_rows.csv'}")
    print(f"summary: {args.output_dir / 'rollout_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
