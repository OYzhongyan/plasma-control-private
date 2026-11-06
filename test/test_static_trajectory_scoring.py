"""The free browser evaluator must preserve the project's F2b scoring rules."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "platformtohf-main"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SITE))


def compute_module():
    spec = importlib.util.spec_from_file_location("trajectory_compute", SITE / "compute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(steps=500):
    from evaluation.eval.score import config
    target = {
        "lcfs_points": [[0.5, -0.2], [0.7, 0.2]],
        "Xpt_main": [[0.5, -0.3], [0.5, 0.3]],
        "Xpt_sec": [[0.6, -0.3], [0.6, 0.3]],
        "strike": [[0.6, 0.4]] * 8,
    }
    traj = {
        "Ip": config.build_Ip_ref()[:steps].tolist(),
        "lcfs_per_step": [copy.deepcopy(target["lcfs_points"]) for _ in range(steps)],
        "lX": [1] * steps, "nX": [4] * steps,
        "Icoil": [[0.0] * 12 for _ in range(steps)],
        "psia": [0.0] * steps, "psib": [1.0] * steps,
        "psiX_main": [[1.0, 1.0] for _ in range(steps)],
        "psiX_sec": [[1.0, 1.0] for _ in range(steps)],
        "Xpt_main": [copy.deepcopy(target["Xpt_main"]) for _ in range(steps)],
        "Xpt_sec": [copy.deepcopy(target["Xpt_sec"]) for _ in range(steps)],
        "strike": [copy.deepcopy(target["strike"]) for _ in range(steps)],
    }
    return {"F2b": {"trajectory": traj, "timeout": False}}, {"F2b": target}


@pytest.mark.parametrize("case", ["ideal", "offset", "short", "empty", "timeout", "coil", "current", "topology", "timing", "optional"])
def test_matches_existing_evaluator(case):
    from evaluation.eval.score import evaluate
    infer, target = inputs(0 if case == "empty" else 7 if case == "short" else 500)
    trajectory = infer["F2b"]["trajectory"]
    if case == "offset":
        trajectory["lcfs_per_step"][0][0][0] += 0.01
        trajectory["Xpt_main"][1][0][1] += 0.015
    elif case == "timeout":
        infer["F2b"]["timeout"] = True
    elif case == "coil":
        trajectory["Icoil"][8][0] = 50000.0
    elif case == "current":
        trajectory["Ip"][0] += 60000.0
    elif case == "topology":
        trajectory["nX"][0] = 2
        trajectory["lX"][1] = 0
    elif case == "timing":
        trajectory["inference_time_ms"] = [1.0] * 500
    elif case == "optional":
        del trajectory["psiX_sec"]
    expected = evaluate(infer, target)
    actual = compute_module().compute(infer, target)
    assert actual["success"] is True
    assert actual["score"] == round(expected["total_score"], 6)
    assert actual["task_metric_scores"] == expected["task_metric_scores"]
    assert actual["K_eff"] == expected["per_task_results"][0].K_eff
    assert actual["scoreJson"]["score"] == actual["score"]
    if case == "ideal":
        assert actual["score"] == 30.0


def test_json_bridge_and_original_fallback():
    infer, target = inputs(2)
    target = {"F2a": target["F2b"]}
    result = json.loads(compute_module().compute_json(json.dumps(infer), json.dumps(target)))
    assert result["success"] is True
    assert result["score"] == pytest.approx(30 * 2 / 500)


@pytest.mark.parametrize("case, message", [
    ("wrong_target", "F2b"), ("wrong_infer", "F2b"),
    ("coil_shape", "Icoil"), ("ip_shape", "Ip"),
    ("nan", "Ip"), ("target_shape", "lcfs_points"),
    ("timeout_type", "timeout"), ("empty_target", "lcfs_points"),
])
def test_invalid_inputs_fail_explicitly(case, message):
    infer, target = inputs(3)
    if case == "wrong_target": target = {"A1": {}}
    elif case == "wrong_infer": infer = {"B3": {}}
    elif case == "coil_shape": infer["F2b"]["trajectory"]["Icoil"][0] = [0] * 13
    elif case == "ip_shape": infer["F2b"]["trajectory"]["Ip"] = [[1], [2]]
    elif case == "nan": infer["F2b"]["trajectory"]["Ip"][0] = float("nan")
    elif case == "target_shape": target["F2b"]["lcfs_points"] = [1, 2]
    elif case == "timeout_type": infer["F2b"]["timeout"] = "false"
    elif case == "empty_target": target = {"F2b": {}}
    with pytest.raises(ValueError, match=message):
        compute_module().compute(infer, target)


def test_bundle_matches_project_sources():
    check = subprocess.run([sys.executable, str(SITE / "build_scorer.py"), "--check"], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr


def test_space_metadata_short_description_fits_hf_limit():
    readme = (SITE / "README.md").read_text(encoding="utf-8")
    metadata = readme.split("---", 2)[1]
    description = next(line.split(":", 1)[1].strip() for line in metadata.splitlines() if line.startswith("short_description:"))
    assert 1 <= len(description) <= 60


def test_browser_runner_reports_missing_files_before_browser_start(tmp_path):
    proc = subprocess.run([sys.executable, str(SITE / "submit.py"), str(tmp_path / "missing infer.json"), str(tmp_path / "target.json")], capture_output=True, text=True)
    assert proc.returncode != 0
    result = json.loads(proc.stdout)
    assert result["success"] is False
    assert "missing infer.json" in result["errorMsg"]


@pytest.mark.parametrize("case", ["jsonl", "nan", "overwrite", "timeout", "old_url"])
def test_cli_rejects_invalid_inputs_without_starting_browser(tmp_path, case):
    infer, target = inputs(2)
    ipath, tpath = tmp_path / "infer.json", tmp_path / "target.json"
    text = json.dumps(infer)
    if case == "jsonl": text += "\n" + text
    if case == "nan": text = '{"x":NaN}'
    ipath.write_text(text)
    tpath.write_text(json.dumps(target))
    args = {"overwrite": ["--output", str(ipath)], "timeout": ["--timeout", "nan"], "old_url": ["--space-url", "https://example.com/evaluate"]}.get(case, [])
    proc = subprocess.run([sys.executable, str(SITE / "submit.py"), str(ipath), str(tpath), *args], capture_output=True, text=True)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["success"] is False
    assert ipath.read_text() == text
