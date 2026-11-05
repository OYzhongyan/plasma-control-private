"""Compare uploaded F2b trajectories against one operator-controlled target."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluation.eval.score import evaluate

METRIC_NAMES = ("Ip", "LCFS", "X", "strike", "psiX", "X2", "psiX2")


class InvalidSubmission(ValueError):
    pass


class GradingUnavailable(RuntimeError):
    """Organizer infrastructure is unavailable; keep the submission pending."""


def decode_json(raw):
    def reject(value):
        raise InvalidSubmission(f"Nonstandard JSON number: {value}")
    try:
        return json.loads(raw.decode("utf-8-sig"), parse_constant=reject)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InvalidSubmission("Expected a UTF-8 JSON trajectory document, not ZIP or JSONL") from exc


def numeric(value, name, ndim, tail=None):
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidSubmission(f"{name} must be a rectangular numeric array") from exc
    if isinstance(value, list) and not value:
        return
    if arr.ndim != ndim or (tail and arr.shape[-len(tail):] != tail):
        raise InvalidSubmission(f"{name} has invalid shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise InvalidSubmission(f"{name} contains non-finite values")


class Grader:
    def __init__(self, target_path):
        self.metric_names = METRIC_NAMES
        raw = Path(target_path).read_bytes()
        self.target = decode_json(raw)
        if not isinstance(self.target, dict):
            raise ValueError("Official target must be an object containing F2b")
        ref = self.target.get("F2b", self.target.get("F2a"))
        if not isinstance(ref, dict):
            raise ValueError("Official target must contain F2b (or original F2a fallback); A1/B1 targets are incompatible")
        for name in ("lcfs_points", "Xpt_main", "Xpt_sec", "strike"):
            if name not in ref or not isinstance(ref[name], list) or not ref[name]:
                raise ValueError(f"Official target is missing {name}")
            numeric(ref[name], f"target.{name}", 2, (2,))
        digest = hashlib.sha256(raw + json.dumps(self.metric_names).encode())
        for path in sorted((Path(__file__).resolve().parents[1] / "evaluation/eval/score").glob("*.py")):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
        # Changes to this adapter also invalidate cached results.
        digest.update(Path(__file__).read_bytes())
        self.fingerprint = digest.hexdigest()

    def grade_bytes(self, raw):
        data = decode_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("F2b"), dict):
            raise InvalidSubmission("Trajectory JSON must contain F2b.trajectory")
        entry = data["F2b"]
        if "timeout" in entry and not isinstance(entry["timeout"], bool):
            raise InvalidSubmission("F2b.timeout must be a boolean")
        traj = entry.get("trajectory")
        if not isinstance(traj, dict):
            raise InvalidSubmission("F2b.trajectory must be an object")
        for name in ("Ip", "lX", "Icoil", "lcfs_per_step"):
            if name not in traj:
                raise InvalidSubmission(f"Trajectory is missing {name}")
        for name in ("Ip", "lX", "nX", "psia", "psib", "inference_time_ms"):
            if name in traj: numeric(traj[name], name, 1)
        numeric(traj["Icoil"], "Icoil", 2, (12,))
        if not isinstance(traj["lcfs_per_step"], list):
            raise InvalidSubmission("lcfs_per_step must be an array")
        for contour in traj["lcfs_per_step"]: numeric(contour, "lcfs_per_step", 2, (2,))
        for name in ("Xpt_main", "Xpt_sec", "strike"):
            if name in traj: numeric(traj[name], name, 3, (2,))
        for name in ("psiX_main", "psiX_sec"):
            if name in traj: numeric(traj[name], name, 2)
        return self.grade_data(data)

    def grade_data(self, data):
        """Score a trusted, server-generated trajectory, including missing geometry.

        The official scorer treats non-finite geometric errors as zero credit.
        Uploaded trajectory documents must still use grade_bytes validation.
        """
        try:
            result = evaluate(data, self.target)
        except (TypeError, ValueError, IndexError, AssertionError) as exc:
            raise InvalidSubmission(f"Invalid trajectory structure: {exc}") from exc
        score = round(float(result["total_score"]), 6)
        # These contributions already include the original metric weights,
        # time averaging and every penalty. The platform should sum them once.
        metrics = {name: float(result["task_metric_scores"]["F2b"][name]) for name in self.metric_names}
        if not np.isfinite(score) or not all(np.isfinite(value) for value in metrics.values()):
            raise InvalidSubmission("Scorer returned a non-finite score")
        detail = result["per_task_results"][0]
        return {"metrics_scores": metrics, "status": "completed",
                "evaluator_message": f"F2b score={score:.6f}/30; K_eff={detail.K_eff}/500; gamma={detail.gamma:g}"}
