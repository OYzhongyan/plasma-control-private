"""Offline worker tests use a local fake platform, never real camp accounts."""
import copy
import json
from pathlib import Path
import sys
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample():
    from evaluation.eval.score.config import build_Ip_ref
    ref = {"lcfs_points": [[.5, -.2], [.7, .2]], "Xpt_main": [[.5, -.3], [.5, .3]],
           "Xpt_sec": [[.6, -.3], [.6, .3]], "strike": [[.6, .4]] * 8}
    n = 500
    trajectory = {"Ip": build_Ip_ref().tolist(), "lX": [1] * n, "nX": [4] * n,
                  "Icoil": [[0.] * 12 for _ in range(n)], "lcfs_per_step": [ref["lcfs_points"]] * n,
                  "Xpt_main": [ref["Xpt_main"]] * n, "Xpt_sec": [ref["Xpt_sec"]] * n,
                  "strike": [ref["strike"]] * n, "psia": [0.] * n, "psib": [1.] * n,
                  "psiX_main": [[1., 1.]] * n, "psiX_sec": [[1., 1.]] * n}
    return {"F2b": {"trajectory": trajectory, "timeout": False}}, {"F2b": ref}


def test_grader_uses_original_scoring_and_fixed_target(tmp_path, sample):
    from server_evaluator.grader import Grader, InvalidSubmission
    infer, target = sample
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target))
    grader = Grader(target_path, metric_name="得分")
    assert grader.grade_bytes(json.dumps(infer).encode())["metrics_scores"] == {"得分": 30.0}
    changed = copy.deepcopy(infer)
    changed["F2b"]["trajectory"]["Icoil"][4][0] = 50000
    from evaluation.eval.score import evaluate
    expected = round(evaluate(changed, target)["total_score"], 6)
    assert grader.grade_bytes(json.dumps(changed).encode())["metrics_scores"]["得分"] == expected
    for payload in (b'not json', b'{"A1":{}}', b'{"F2b":{"trajectory":{"Ip":[NaN]}}}'):
        with pytest.raises(InvalidSubmission): grader.grade_bytes(payload)


def test_store_persists_scores_and_rejects_changed_rules(tmp_path):
    from server_evaluator.store import Store, instance_lock
    task = {"id": "s1", "problem_id": "p1", "source": "offline", "status": "pending", "file_url": "https://example.org/a.json"}
    with Store(tmp_path, "settings-v1") as store:
        store.register(task)
        store.save_result("s1", {"submission_id": "s1", "metrics_scores": {"score": 1}, "status": "completed"})
    with Store(tmp_path, "settings-v1") as store:
        assert store.get("s1")["result"]["status"] == "completed"
        assert store.outstanding_count() == 1
        store.mark_reported("s1")
        assert store.outstanding_count() == 0
    with pytest.raises(ValueError, match="fingerprint"):
        Store(tmp_path, "changed-target")
    with instance_lock(tmp_path):
        with pytest.raises(RuntimeError, match="running"):
            with instance_lock(tmp_path): pass


@pytest.fixture
def platform(sample):
    infer, _ = sample
    state = {"records": {}, "files": {}, "download_headers": [], "downloads": 0,
             "login_count": 0, "batches": [], "fail_reports": 0, "ambiguous": False,
             "expire_once": False, "partial": False, "hide_all": False, "fail_after_write": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            path = urlsplit(self.path)
            if path.path.startswith("/files/"):
                state["downloads"] += 1
                state["download_headers"].append(self.headers.get("Authorization"))
                key = path.path.rsplit("/", 1)[-1]
                if key not in state["files"]: self.reply(404, {}); return
                data = state["files"][key]
                self.send_response(200); self.end_headers(); self.wfile.write(data); return
            if state["expire_once"]:
                state["expire_once"] = False; self.reply(401, {}); return
            if self.headers.get("Authorization") != "Bearer test-platform-token":
                self.reply(401, {}); return
            qs = parse_qs(path.query)
            rows = list(state["records"].values())
            if state["hide_all"] and path.path == "/api/submissions" and not qs.get("status"):
                rows = []
            if path.path.endswith("offline-pending"):
                rows = [r for r in rows if r["source"] == "offline" and r["status"] == "pending"]
            for key in ("problem_id", "source", "status"):
                if key in qs: rows = [r for r in rows if r[key] == qs[key][0]]
            self.reply(200, rows)
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/login/teacher"):
                state["login_count"] += 1; self.reply(200, {"token": "test-platform-token"}); return
            if self.headers.get("Authorization") != "Bearer test-platform-token":
                self.reply(401, {}); return
            state["batches"].append(data["items"])
            if state["fail_reports"]:
                state["fail_reports"] -= 1; self.reply(503, {}); return
            results, errors = [], []
            for i, item in enumerate(data["items"]):
                sid = item["submission_id"]
                if state["partial"] and i > 0:
                    errors.append({"submission_id": sid, "message": "temporary test error"}); continue
                state["records"][sid].update(status=item["status"], metrics_scores=item["metrics_scores"])
                results.append({"submission_id": sid, "success": True})
            if state["ambiguous"]: self.reply(200, {"unexpected": "response lost shape"}); return
            if state["fail_after_write"]: self.reply(503, {}); return
            self.reply(200, {"updated": len(results), "failed": len(errors), "results": results, "errors": errors})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    for sid, source in (("s1", "offline"), ("s2", "manual")):
        state["records"][sid] = {"id": sid, "problem_id": "p1", "team_name": "Team A", "source": source,
                                 "status": "pending", "file_url": url + "/files/" + sid}
        state["files"][sid] = json.dumps(infer).encode()
    state["url"] = url
    yield state
    server.shutdown(); server.server_close(); thread.join()


def make_worker(tmp_path, sample, platform):
    from server_evaluator.client import PlatformClient
    from server_evaluator.grader import Grader
    from server_evaluator.store import Store
    from server_evaluator.worker import Worker
    _, target = sample
    p = tmp_path / "target.json"; p.write_text(json.dumps(target))
    grader = Grader(p)
    client = PlatformClient(platform["url"], "p1", username="test-only", password="not-a-real-password")
    store = Store(tmp_path / "data", grader.fingerprint)
    return Worker(client, store, grader, tmp_path / "downloads"), store


def test_poll_score_report_and_never_send_token_to_download(tmp_path, sample, platform):
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert platform["downloads"] == 2
        assert platform["download_headers"] == [None, None]
        assert all(r["status"] == "completed" for r in platform["records"].values())
        assert all(r["metrics_scores"] == {"得分": 30.0} for r in platform["records"].values())
        assert store.outstanding_count() == 0
        worker.cycle()
        assert platform["downloads"] == 2


def test_report_retry_survives_restart_without_rescoring(tmp_path, sample, platform):
    platform["fail_reports"] = 1
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert store.outstanding_count() == 2
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert store.outstanding_count() == 0
        assert platform["downloads"] == 2


def test_partial_result_and_uncertain_ack_reconciliation(tmp_path, sample, platform):
    platform["partial"] = True
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert store.outstanding_count() == 1
        assert store.get("s2")["last_error"] == "temporary test error"
        platform["partial"] = False
        platform["ambiguous"] = True
        worker.cycle()
        assert store.outstanding_count() == 0
        assert platform["downloads"] == 2


def test_bad_file_becomes_failed_and_auth_refreshes(tmp_path, sample, platform):
    platform["files"]["s1"] = b'{"A1":{}}'
    platform["expire_once"] = True
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert platform["login_count"] == 2
        assert platform["records"]["s1"]["status"] == "failed"
        assert platform["records"]["s1"]["metrics_scores"] == {}
        assert store.outstanding_count() == 0


def test_missing_records_page_does_not_block_fresh_pending(tmp_path, sample, platform):
    platform["hide_all"] = True
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert store.outstanding_count() == 0
        assert platform["downloads"] == 2


def test_response_lost_after_write_is_not_reported_twice(tmp_path, sample, platform):
    platform["fail_after_write"] = True
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert store.outstanding_count() == 2
        worker.cycle()
        assert store.outstanding_count() == 0
        assert len(platform["batches"]) == 1
        assert platform["downloads"] == 2


def test_oversized_file_and_download_retry_limit(tmp_path, sample, platform):
    worker, store = make_worker(tmp_path, sample, platform)
    platform["files"].pop("s1")
    with store:
        worker.max_bytes = 16  # s2 exceeds the bound; s1 is missing.
        for _ in range(3): worker.cycle()
        assert platform["records"]["s1"]["status"] == "failed"
        assert platform["records"]["s2"]["status"] == "failed"
        assert store.outstanding_count() == 0
        assert store.get("s1")["attempts"] == 3


def test_non_owned_or_auto_tasks_are_never_processed(tmp_path, sample, platform):
    platform["records"]["s1"]["problem_id"] = "another-problem"
    platform["records"]["s2"]["source"] = "auto"
    worker, store = make_worker(tmp_path, sample, platform)
    with store:
        worker.cycle()
        assert platform["downloads"] == 0
        assert platform["batches"] == []


def test_no_credentials_are_persisted(tmp_path, sample, platform):
    worker, store = make_worker(tmp_path, sample, platform)
    with store: worker.cycle()
    for path in (tmp_path / "data").glob("*"):
        if not path.is_file(): continue
        raw = path.read_bytes()
        assert b"test-platform-token" not in raw
        assert b"not-a-real-password" not in raw


def test_cli_check_config_is_offline(tmp_path, sample):
    import subprocess
    _, target = sample
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target))
    env = dict(__import__("os").environ)
    for name in ("PLATFORM_USERNAME", "PLATFORM_PASSWORD", "PLATFORM_TOKEN"):
        env.pop(name, None)
    proc = subprocess.run([sys.executable, str(ROOT / "server_evaluator/run.py"), "--problem-id", "p1", "--target", str(path), "--check-config"], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["max_score"] == 30


def test_cli_once_full_protocol(tmp_path, sample, platform):
    import os
    import subprocess
    _, target = sample
    path = tmp_path / "target.json"; path.write_text(json.dumps(target))
    env = dict(os.environ)
    env.pop("PLATFORM_TOKEN", None)
    env.update(PLATFORM_USERNAME="test-only", PLATFORM_PASSWORD="test-password")
    command = [sys.executable, str(ROOT / "server_evaluator/run.py"), "--problem-id", "p1",
               "--target", str(path), "--data-dir", str(tmp_path / "data"), "--once", "--platform-url", platform["url"]]
    for _ in range(2):
        proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert "test-password" not in proc.stderr
    assert platform["downloads"] == 2
    assert len(platform["batches"]) == 1


def test_oversized_numeric_and_deep_json_are_invalid_submissions(tmp_path, sample):
    from server_evaluator.grader import Grader, InvalidSubmission
    infer, target = sample
    path = tmp_path / "target.json"; path.write_text(json.dumps(target))
    grader = Grader(path)
    infer["F2b"]["trajectory"]["Ip"][0] = 10**1000
    with pytest.raises(InvalidSubmission): grader.grade_bytes(json.dumps(infer).encode())
    with pytest.raises(InvalidSubmission): grader.grade_bytes(b'[' * 2000 + b']' * 2000)
