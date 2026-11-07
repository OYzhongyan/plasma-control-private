from pathlib import Path
from threading import Event, Thread
import time

import numpy as np
import pytest

from submission.shared_channel import make_descriptor
from submission.shared_service import SharedWorker


@pytest.fixture
def shared_service(tmp_path):
    class Policy:
        ready = True
        calls = 0
        def status(self): return dict(ok=True, model_id='test-model')
        def reset(self, **kwargs): self.total = 0.
        def act(self, observation):
            self.calls += 1; self.total += 1.
            return np.full(12, self.total)
    root = tmp_path/'shared'; endpoint = make_descriptor()
    worker = SharedWorker(root, endpoint, Policy())
    stop = Event(); errors = []
    def run():
        try:
            while not stop.is_set(): worker.poll_once(); stop.wait(.005)
        except BaseException as exc: errors.append(exc)
    thread = Thread(target=run, daemon=True); thread.start()
    yield root, endpoint, worker, stop
    stop.set(); thread.join(timeout=5); worker.close()
    assert not errors


def client(service, **kwargs):
    from server_evaluator.shared_agent import SharedAgent
    root, endpoint, _, _ = service
    return SharedAgent(endpoint, shared_root=root, timeout=.3, poll_interval=.005, **kwargs)


def test_roundtrip_actions_cleanup_and_single_client(shared_service):
    from server_evaluator.grader import GradingUnavailable
    with client(shared_service) as agent:
        agent.reset()
        np.testing.assert_array_equal(agent.act({'Ip': [1.]}), np.ones(12))
        np.testing.assert_array_equal(agent.act({'Ip': [1.]}), np.full(12, 2.))
        assert shared_service[2].policy.calls == 2
        assert not shared_service[2].mailbox.requests.names()
        assert not shared_service[2].mailbox.responses.names()
        with pytest.raises(GradingUnavailable, match='client|use'): client(shared_service)


def test_timeout_retry_does_not_advance_model_twice(shared_service, monkeypatch):
    worker = shared_service[2]
    original = worker.mailbox.responses.write
    dropped = []
    def lost_once(name, raw):
        if b'"action"' in raw and not dropped: dropped.append(name); return
        return original(name, raw)
    monkeypatch.setattr(worker.mailbox.responses, 'write', lost_once)
    with client(shared_service, retries=1) as agent:
        agent.reset()
        np.testing.assert_array_equal(agent.act({'Ip': [1.]}), np.ones(12))
        assert worker.policy.calls == 1


def test_stopped_agent_times_out_and_cleans_request(shared_service):
    from server_evaluator.shared_agent import SharedAgentError
    with client(shared_service, retries=0) as agent:
        shared_service[3].set(); time.sleep(.02)
        with pytest.raises(SharedAgentError, match='timed out'): agent.reset()
        assert not shared_service[2].mailbox.requests.names()


def test_missing_shared_root_is_infrastructure_but_missing_agent_is_invalid(tmp_path):
    from server_evaluator.shared_agent import SharedAgent, SharedAgentError
    from server_evaluator.grader import GradingUnavailable
    with pytest.raises(GradingUnavailable): SharedAgent(make_descriptor(), shared_root=tmp_path/'missing')
    with pytest.raises(SharedAgentError): SharedAgent(make_descriptor(), shared_root=tmp_path)


def test_restarted_worker_during_episode_is_detected(shared_service):
    from server_evaluator.shared_agent import SharedAgentError
    with client(shared_service) as agent:
        agent.reset()
        shared_service[2].service_id = 'different-service'
        with pytest.raises(SharedAgentError, match='changed'): agent.act({'Ip': [1.]})


def test_unsafe_agent_lock_is_submission_failure(shared_service, monkeypatch):
    from submission.shared_files import Mailbox
    from server_evaluator.shared_agent import SharedAgentError
    def unsafe(*args): raise ValueError('Mailbox files must not be symlinks')
    monkeypatch.setattr(Mailbox,'lock',unsafe)
    with pytest.raises(SharedAgentError,match='unsafe'): client(shared_service)
