"""Browser adapter for the unchanged project scorer. JSON is never Python code."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scorer_bundle import load_scorer, SCORER_SHA256


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _array(value, name, ndim, tail=None):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if arr.size == 0 and isinstance(value, list) and not value:
        return
    if arr.ndim != ndim or (tail is not None and arr.shape[-len(tail):] != tail):
        raise ValueError(f"{name}: invalid shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Infinity")


def validate(infer, target):
    _object(infer, "infer")
    _object(target, "target")
    if "F2b" not in infer:
        raise ValueError("infer JSON must contain F2b.trajectory (not boundary JSONL)")
    entry = _object(infer["F2b"], "F2b")
    if "timeout" in entry and not isinstance(entry["timeout"], bool):
        raise ValueError("F2b.timeout must be a JSON boolean")
    traj = _object(entry.get("trajectory"), "F2b.trajectory")
    ref = target.get("F2b", target.get("F2a"))
    if not isinstance(ref, dict):
        raise ValueError("target JSON must contain F2b (or original F2a fallback); A1/B1 targets are incompatible")
    for name in ("lcfs_points", "Xpt_main", "Xpt_sec", "strike"):
        if name not in ref:
            raise ValueError(f"target.F2b.{name} is required")
        _array(ref[name], f"target.{name}", 2, (2,))
        if not len(ref[name]):
            raise ValueError(f"target.{name} cannot be empty")
    for name in ("Ip", "lX", "Icoil", "lcfs_per_step"):
        if name not in traj:
            raise ValueError(f"F2b.trajectory.{name} is required")
    for name in ("Ip", "lX", "nX", "psia", "psib", "inference_time_ms"):
        if name in traj:
            _array(traj[name], name, 1)
    _array(traj["Icoil"], "Icoil", 2, (12,))
    if not isinstance(traj["lcfs_per_step"], list):
        raise ValueError("lcfs_per_step must be a JSON array of contours")
    for contour in traj["lcfs_per_step"]:
        _array(contour, "lcfs_per_step", 2, (2,))
    for name in ("Xpt_main", "Xpt_sec", "strike"):
        if name in traj:
            _array(traj[name], name, 3, (2,))
    for name in ("psiX_main", "psiX_sec"):
        if name in traj:
            _array(traj[name], name, 2)


def compute(infer, target):
    validate(infer, target)
    scorer = load_scorer()
    result = scorer.evaluate(infer, target)
    detail = result["per_task_results"][0]
    total = round(float(result["total_score"]), 6)
    metric_scores = result["task_metric_scores"]
    return {
        "success": True, "errorMsg": "", "score": total,
        "score1": round(float(result["task_scores"]["F2b"]), 6),
        "scoreJson": {
            "score": total, "score1": round(float(result["task_scores"]["F2b"]), 6),
            **{f"F2b_{name}": round(float(value), 6) for name, value in metric_scores["F2b"].items()},
        },
        "task_scores": result["task_scores"], "task_metric_scores": metric_scores,
        "K_eff": int(detail.K_eff), "gamma": float(detail.gamma),
        "timeout": bool(detail.timeout), "max_score": float(scorer.config.TOTAL_MAX_SCORE),
        "scorer_sha256": SCORER_SHA256,
        "warnings": [f"{key} missing: corresponding original metric/constraint may not apply"
                     for key in ("nX", "Xpt_main", "Xpt_sec", "strike", "psiX_main", "psiX_sec", "psia", "psib", "inference_time_ms")
                     if key not in infer["F2b"]["trajectory"]],
    }


def compute_json(infer_json_text, target_json_text):
    result = compute(json.loads(infer_json_text), json.loads(target_json_text))
    result["infer_sha256"] = hashlib.sha256(infer_json_text.encode("utf-8")).hexdigest()
    result["target_sha256"] = hashlib.sha256(target_json_text.encode("utf-8")).hexdigest()
    return json.dumps(result, ensure_ascii=False, allow_nan=False)
