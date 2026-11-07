"""HTTP protocol retries must not advance a participant policy twice."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = "hfm-remote-policy-v1"


class CounterPolicy:
    ready = True
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.total = 0.
        self.bad_output = False
    def status(self):
        return {"ok": True, "model_loaded": True, "model_id": "test-model"}
    def reset(self, **kwargs):
        self.resets += 1
        self.total = 0.
    def act(self, observation):
        self.calls += 1
        self.total += float(observation["Ip"][0])
        return np.full(12, np.nan if self.bad_output else self.total)


def make_app(policy=None, token=""):
    from submission.service2 import create_app
    return create_app(policy=policy or CounterPolicy(), access_token=token)


def reset(episode="episode-one"):
    return dict(protocol=PROTOCOL, episode_id=episode)


def step(index=0, value=2., episode="episode-one"):
    return dict(protocol=PROTOCOL, episode_id=episode, step_id=index, observation={"Ip": [value]})


def test_retry_reset_and_act_do_not_repeat_policy_state_changes():
    policy = CounterPolicy()
    client = make_app(policy).test_client()
    assert client.get("/health").json["protocol"] == PROTOCOL
    assert client.post("/reset", json=reset()).status_code == 200
    first = client.post("/act", json=step())
    assert first.json["action"] == [2.] * 12
    assert client.post("/reset", json=reset()).status_code == 200
    assert client.post("/act", json=step()).json == first.json
    assert policy.calls == policy.resets == 1
    second = client.post("/act", json=step(1))
    assert second.json["action"] == [4.] * 12 and second.json["step_id"] == 1
    assert client.post("/act", json=step(1)).json == second.json
    assert policy.calls == 2


@pytest.mark.parametrize("payload", [step(2), step(0, 3.), step(1, episode="other"),
                                      {**step(1), "step_id": True}])
def test_step_conflicts_do_not_call_model(payload):
    policy = CounterPolicy()
    client = make_app(policy).test_client()
    client.post("/reset", json=reset())
    client.post("/act", json=step())
    assert client.post("/act", json=payload).status_code in (400, 409)
    assert policy.calls == 1


def test_new_episode_rejects_old_requests_and_resets_accumulator():
    client = make_app().test_client()
    client.post("/reset", json=reset())
    client.post("/act", json=step())
    client.post("/reset", json=reset("new"))
    assert client.post("/act", json=step()).status_code == 409
    assert client.post("/act", json=step(episode="new")).json["action"] == [2.] * 12


def test_model_failure_poisoned_episode_requires_a_new_reset():
    policy = CounterPolicy()
    policy.bad_output = True
    client = make_app(policy).test_client()
    client.post("/reset", json=reset())
    assert client.post("/act", json=step()).status_code == 500
    policy.bad_output = False
    assert client.post("/act", json=step()).status_code == 409
    assert policy.calls == 1
    client.post("/reset", json=reset("new"))
    assert client.post("/act", json=step(episode="new")).status_code == 200


def test_bearer_auth_applies_to_all_routes():
    client = make_app(token="private-test-token").test_client()
    for route in ("/health", "/reset", "/act"):
        response = client.get(route) if route == "/health" else client.post(route, json={})
        assert response.status_code == 401
        assert "private-test-token" not in response.get_data(as_text=True)
    assert client.get("/health", headers={"Authorization": "Bearer private-test-token"}).status_code == 200


def test_endpoint_file_and_strict_json(tmp_path):
    from submission.remote_protocol import make_endpoint, read_endpoint
    value = make_endpoint("https://Example.com:443/", "token-value")
    assert value["service_url"] == "https://example.com"
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(value))
    assert read_endpoint(path) == value
    for data in ('{"version":1,"version":1}', '{"value":NaN}', '[]'):
        path.write_text(data)
        with pytest.raises(ValueError):
            read_endpoint(path)


@pytest.mark.parametrize("url", ["file:///tmp/a", "http://user:pass@example.com", "https://example.com/a",
                                     "https://example.com?x=1", "http://example.com:0", "http://example.com/#x"])
def test_endpoint_cannot_select_paths_or_non_http_protocols(url):
    from submission.remote_protocol import make_endpoint
    with pytest.raises(ValueError):
        make_endpoint(url)


def test_launcher_imports_from_any_directory_without_pythonpath(tmp_path):
    script = "import runpy,sys; runpy.run_path(sys.argv[1], run_name='smoke_import')"
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT / "submission/service2.py")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_connection_file_generator_never_prints_access_token(tmp_path):
    target = tmp_path / "endpoint.json"
    env = dict(os.environ, SUBMISSION_ACCESS_TOKEN="secret-agent-key", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", str(ROOT / "submission/export_endpoint.py"),
                             "--service-url", "https://example.com", "--output", str(target)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "secret-agent-key" not in result.stdout + result.stderr
    assert json.loads(target.read_text())["access_token"] == "secret-agent-key"


def test_failed_new_reset_invalidates_previous_episode():
    policy = CounterPolicy()
    client = make_app(policy).test_client()
    client.post("/reset", json=reset())
    client.post("/act", json=step())
    def broken_reset(**kwargs):
        raise RuntimeError("reset partially changed model state")
    policy.reset = broken_reset
    assert client.post("/reset", json=reset("new")).status_code == 500
    assert client.post("/act", json=step(1)).status_code == 409
    assert policy.calls == 1


def test_concurrent_duplicate_steps_only_call_policy_once():
    from concurrent.futures import ThreadPoolExecutor
    import time
    policy = CounterPolicy()
    original = policy.act
    def slow_act(obs):
        time.sleep(.02)
        return original(obs)
    policy.act = slow_act
    app = make_app(policy)
    app.test_client().post("/reset", json=reset())
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: app.test_client().post("/act", json=step()).json, range(4)))
    assert all(reply == replies[0] for reply in replies)
    assert policy.calls == 1
