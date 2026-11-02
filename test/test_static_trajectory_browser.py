"""Opt-in real Chromium/Pyodide checks: RUN_PYODIDE_BROWSER=1 pytest this file."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_static_trajectory_scoring import ROOT, SITE, inputs

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PYODIDE_BROWSER") != "1", reason="downloads Pyodide and requires Chromium")


def runner_module():
    spec = importlib.util.spec_from_file_location("static_submit", SITE / "submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pyodide_ui_parity_errors_download_and_cli(tmp_path):
    from playwright.sync_api import sync_playwright
    from evaluation.eval.score import evaluate

    infer, target = inputs()
    runner = runner_module()
    with runner.local_page() as url, sync_playwright() as playwright:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.goto(url)
            page.wait_for_function("typeof window.scoreTrajectoryFiles === 'function'")
            for case in ("ideal", "offset", "coil", "current", "topology", "timing", "short", "timeout"):
                data = copy.deepcopy(infer)
                traj = data["F2b"]["trajectory"]
                if case == "offset": traj["lcfs_per_step"][0][0][0] += 0.02
                elif case == "coil": traj["Icoil"][5][0] = 50000
                elif case == "current": traj["Ip"][0] += 60000
                elif case == "topology": traj["nX"] = [2] * 500
                elif case == "timing": traj["inference_time_ms"] = [1] * 500
                elif case == "short": data, _ = inputs(8)
                elif case == "timeout": data["F2b"]["timeout"] = True
                expected = evaluate(data, target)
                actual = page.evaluate("async ([i,t]) => window.scoreTrajectoryFiles(i,t,'test')", [json.dumps(data), json.dumps(target)])
                assert actual["success"] is True
                assert actual["score"] == round(expected["total_score"], 6), case
                for key, value in expected["task_metric_scores"]["F2b"].items():
                    assert actual["task_metric_scores"]["F2b"][key] == pytest.approx(value, abs=1e-10)

            infer_path = tmp_path / "trajectory with spaces.json"
            target_path = tmp_path / "目标 target.json"
            infer_path.write_text(json.dumps(infer), encoding="utf-8")
            target_path.write_text(json.dumps(target), encoding="utf-8-sig")
            page.locator("#inferFile").set_input_files(str(infer_path))
            page.locator("#targetFile").set_input_files(str(target_path))
            page.locator("#runButton").click()
            page.wait_for_function("document.getElementById('totalScore').textContent === '30.000000'", timeout=120000)
            with page.expect_download() as download_info:
                page.locator("#downloadButton").click()
            result = json.loads(Path(download_info.value.path()).read_text(encoding="utf-8"))
            assert result["score"] == 30
            page.screenshot(path=str(tmp_path / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path / "mobile.png"), full_page=True)

            error = page.evaluate("async ([i,t]) => { try { await window.scoreTrajectoryFiles(i,t); } catch(e) { return e.message; } }", [json.dumps(infer), '{"A1":{}}'])
            assert "F2b" in error
            assert page.locator("#downloadButton").is_disabled()
            assert page.locator("#totalScore").inner_text() == "—"
            # Retry after failure using the same Worker/runtime.
            result = page.evaluate("async ([i,t]) => window.scoreTrajectoryFiles(i,t)", [json.dumps(infer), json.dumps(target)])
            assert result["score"] == 30

            page.evaluate("([i,t]) => { window.cancelOutcome = null; window.scoreTrajectoryFiles(i,t).catch(e => window.cancelOutcome = e.message); document.getElementById('cancelButton').click(); }", [json.dumps(infer), json.dumps(target)])
            page.wait_for_function("window.cancelOutcome !== null")
            assert "取消" in page.evaluate("window.cancelOutcome")
            assert page.locator("#downloadButton").is_disabled()

            # Actual saved simulator rollout, not only synthetic ideal inputs.
            real = ROOT / "results/flow_check_20261030/train_hydra/eval/infer_result.json"
            if real.is_file():
                real_data = json.loads(real.read_text())
                expected = evaluate(real_data, target)
                actual = page.evaluate("async ([i,t]) => window.scoreTrajectoryFiles(i,t)", [json.dumps(real_data), json.dumps(target)])
                assert actual["score"] == round(expected["total_score"], 6)

            # CLI uses the same public page API, starts/closes its own Chromium.
            cli = subprocess.run([sys.executable, str(SITE / "submit.py"), str(infer_path), str(target_path), "--space-url", url, "--output", str(tmp_path / "cli-score.json"), "--tag", "CLI"], capture_output=True, text=True, encoding="utf-8", timeout=330)
            assert cli.returncode == 0, cli.stdout + cli.stderr
            cli_result = json.loads(cli.stdout)
            assert cli_result["score"] == 30
            assert cli_result["tag"] == "CLI"
            assert json.loads((tmp_path / "cli-score.json").read_text())["score"] == 30

            # Also exercise the default ephemeral-local-page CLI mode.
            env = dict(os.environ)
            env.pop("HF_SPACE_URL", None)
            cli = subprocess.run([sys.executable, str(SITE / "submit.py"), str(infer_path), str(target_path)], capture_output=True, text=True, encoding="utf-8", timeout=330, env=env)
            assert cli.returncode == 0, cli.stdout + cli.stderr
            assert json.loads(cli.stdout)["score"] == 30
