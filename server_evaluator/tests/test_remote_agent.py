"""Real HTTP exchanges between C and a loopback participant A."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from threading import Thread

import numpy as np
import pytest

from submission.remote_protocol import make_endpoint


@pytest.fixture
def agent_server():
    from submission.service2 import create_app
    class Policy:
        ready = True
        calls = resets = 0
        model_id = "test-model"
        def status(self):
            return dict(ok=True, model_loaded=True, model_id=self.model_id)
        def reset(self, **kwargs):
            self.resets += 1
            self.total = 0.
        def act(self, obs):
            self.calls += 1
            self.total += 1.
            return np.full(12, self.total, np.float32)
    policy = Policy()
    token = "private-agent-test-key"
    app = create_app(policy=policy, access_token=token)
    state = dict(requests=[], dropped=set(), drop_once=set(), truncate_once=set(), redirect=False, bad_action=False)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def handle_request(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state["requests"].append((self.command, self.path, dict(self.headers), data))
            if state["redirect"]:
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/")
                self.end_headers()
                return
            response = app.test_client().open(self.path, method=self.command, data=data,
                                               headers={"Authorization": self.headers.get("Authorization", ""),
                                                        "Content-Type": "application/json"})
            if self.path in state["drop_once"] and self.path not in state["dropped"]:
                state["dropped"].add(self.path)
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            raw = response.get_data()
            if state["bad_action"] and self.path == "/act":
                value = json.loads(raw)
                value["action"] = [False] * 12
                raw = json.dumps(value).encode()
            self.send_response(response.status_code)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path in state['truncate_once'] and self.path not in state['dropped']:
                state['dropped'].add(self.path)
                self.wfile.write(raw[:20])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            self.wfile.write(raw)
        do_GET = do_POST = handle_request
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    yield dict(endpoint=make_endpoint(url, token), url=url, policy=policy, state=state)
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def client(server, **kwargs):
    from server_evaluator.remote_agent import RemoteAgent
    return RemoteAgent(server["endpoint"], allowed_origins=[server["url"]], timeout=1, **kwargs)


def test_real_http_roundtrip_and_idempotent_retry_after_lost_response(agent_server):
    agent_server["state"]["drop_once"] = {"/reset", "/act"}
    remote = client(agent_server, retries=2)
    remote.reset()
    np.testing.assert_array_equal(remote.act({"Ip": np.array([123.])}), np.ones(12))
    np.testing.assert_array_equal(remote.act({"Ip": [124.]}), np.full(12, 2.))
    assert agent_server["policy"].resets == 1
    assert agent_server["policy"].calls == 2
    attempts = [row[3] for row in agent_server["state"]["requests"] if row[1] == "/act"]
    assert attempts[0] == attempts[1]
    assert remote.audit["retry_count"] == 2


def test_private_origin_requires_explicit_operator_allowance(agent_server):
    from server_evaluator.remote_agent import RemoteAgent, RemoteAgentError
    with pytest.raises(RemoteAgentError, match="private|allowed"):
        RemoteAgent(agent_server["endpoint"], timeout=1)
    assert not agent_server["state"]["requests"]


def test_redirect_is_not_followed_and_credentials_are_not_exposed(agent_server):
    from server_evaluator.remote_agent import RemoteAgentError
    agent_server["state"]["redirect"] = True
    with pytest.raises(RemoteAgentError, match="302") as error:
        client(agent_server)
    assert agent_server["endpoint"]["access_token"] not in str(error.value)
    assert len(agent_server["state"]["requests"]) == 1


def test_auth_error_does_not_retry_or_echo_token(agent_server):
    from server_evaluator.remote_agent import RemoteAgent, RemoteAgentError
    value = {**agent_server["endpoint"], "access_token": "wrong-secret"}
    with pytest.raises(RemoteAgentError, match="401") as error:
        RemoteAgent(value, allowed_origins=[agent_server["url"]], retries=2, timeout=1)
    assert "wrong-secret" not in str(error.value)
    assert len(agent_server["state"]["requests"]) == 1


def test_invalid_actions_fail_without_retry_or_step_advance(agent_server):
    from server_evaluator.remote_agent import RemoteAgentError
    remote = client(agent_server)
    remote.reset()
    agent_server["state"]["bad_action"] = True
    with pytest.raises(RemoteAgentError, match="12 finite"):
        remote.act({"Ip": [1.]})
    assert remote.step_id == 0 and agent_server["policy"].calls == 1


def test_environment_proxies_are_ignored(agent_server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("PLATFORM_TOKEN", "platform-secret-not-for-A")
    remote = client(agent_server)
    remote.reset()
    remote.act({"Ip": [1.]})
    assert "platform-secret-not-for-A" not in repr(agent_server["state"]["requests"])


def test_dns_rebinding_to_private_address_is_rejected(monkeypatch):
    from server_evaluator.remote_agent import RemoteAgent, RemoteAgentError
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8001))])
    with pytest.raises(RemoteAgentError, match="private|allowed"):
        RemoteAgent(make_endpoint("http://agent.example.com:8001"), timeout=1)


def test_agent_restart_is_detected_during_episode(agent_server):
    from server_evaluator.remote_agent import RemoteAgentError
    remote = client(agent_server)
    remote.reset()
    agent_server["policy"].model_id = "different-model"
    with pytest.raises(RemoteAgentError, match="changed"):
        remote.act({"Ip": [1.]})


def test_truncated_content_length_retries_identical_act_without_advancing_twice(agent_server):
    remote = client(agent_server, retries=2)
    remote.reset()
    agent_server['state']['truncate_once'] = {'/act'}
    np.testing.assert_array_equal(remote.act({'Ip': [1.]}), np.ones(12))
    assert remote.audit['retry_count'] == 1 and agent_server['policy'].calls == 1
    bodies = [r[3] for r in agent_server['state']['requests'] if r[1] == '/act']
    assert len(bodies) == 2 and bodies[0] == bodies[1]
