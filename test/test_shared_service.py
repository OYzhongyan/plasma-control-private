import json
from pathlib import Path
import time
import uuid
import os
import subprocess
import sys

import numpy as np
import pytest

from test_checkpoint_policy import source

from submission.remote_protocol import PROTOCOL


class CounterPolicy:
    ready = True
    calls = resets = 0
    def status(self): return dict(model_id='test-model', ok=True)
    def reset(self, **kwargs): self.resets += 1; self.total = 0.
    def act(self, observation):
        self.calls += 1; self.total += float(observation['Ip'][0])
        return np.full(12, self.total)


def reset(): return dict(protocol=PROTOCOL, episode_id='test-episode')
def step(index=0): return dict(protocol=PROTOCOL, episode_id='test-episode', step_id=index, observation={'Ip': [2.]})


def exchange(worker, endpoint, method, body, *, request_id=None, expires_at=None):
    from submission.shared_channel import encode_packet, decode_packet, digest
    request_id = request_id or uuid.uuid4().hex
    payload = dict(method=method, body=body)
    if method != 'health': payload['service_id'] = worker.service_id
    raw = encode_packet(endpoint, request_id, 'request', payload,
                        expires_at=expires_at or time.time()+30)
    name = request_id+'.json'
    worker.mailbox.requests.write(name, raw)
    worker.poll_once()
    response = worker.mailbox.responses.read(name, 65536)
    return decode_packet(response, endpoint, 'response', request_id=request_id, request_digest=digest(raw))['payload']


@pytest.fixture
def worker(tmp_path):
    from submission.shared_channel import make_descriptor
    from submission.shared_service import SharedWorker
    endpoint = make_descriptor()
    with SharedWorker(tmp_path/'shared', endpoint, CounterPolicy()) as worker:
        yield worker, endpoint


def test_signed_files_drive_same_policy_and_idempotent_actions(worker):
    service, endpoint = worker
    health = exchange(service, endpoint, 'health', {})
    assert health['ok'] and health['service_id'] == service.service_id
    assert exchange(service, endpoint, 'reset', reset())['ok']
    first = exchange(service, endpoint, 'act', step())
    second = exchange(service, endpoint, 'act', step())
    assert first == second and first['action'] == [2.] * 12
    assert service.policy.calls == 1
    assert exchange(service, endpoint, 'act', step(2))['ok'] is False
    assert service.policy.calls == 1


def test_unsigned_or_expired_requests_never_reach_policy(worker):
    from submission.shared_channel import encode_packet
    service, endpoint = worker
    for expired in (False, True):
        rid = uuid.uuid4().hex
        raw = encode_packet(endpoint, rid, 'request', dict(method='reset', body=reset()),
                            expires_at=time.time()-1 if expired else time.time()+30)
        if not expired:
            value=json.loads(raw); value['signature']='0'*64; raw=json.dumps(value).encode()
        service.mailbox.requests.write(rid+'.json', raw)
        service.poll_once()
        assert rid+'.json' not in service.mailbox.responses.names()
    assert service.policy.resets == 0


def test_one_worker_per_mailbox_and_secret_never_in_shared_files(worker):
    from submission.shared_service import SharedWorker
    service, endpoint = worker
    with pytest.raises(RuntimeError, match='already in use'):
        SharedWorker(service.root, endpoint, CounterPolicy())
    exchange(service, endpoint, 'health', {})
    for directory in (service.mailbox.requests, service.mailbox.responses):
        for name in directory.names():
            assert endpoint['access_token'].encode() not in directory.read(name, 4*1024*1024)


def test_worker_restart_changes_service_id(tmp_path):
    from submission.shared_channel import make_descriptor
    from submission.shared_service import SharedWorker
    endpoint = make_descriptor()
    with SharedWorker(tmp_path, endpoint, CounterPolicy()) as first: before = first.service_id
    with SharedWorker(tmp_path, endpoint, CounterPolicy()) as second: assert second.service_id != before


def test_pre_restart_requests_cannot_displace_new_episode(worker):
    from submission.agent_session import AgentSession
    from submission.shared_channel import encode_packet
    service, endpoint = worker
    old_request_id=uuid.uuid4().hex
    # This is the old wire request, missing an incarnation binding.
    old=encode_packet(endpoint,old_request_id,'request',dict(method='reset',body=reset()),expires_at=time.time()+30)
    service.service_id=uuid.uuid4().hex
    service.session=AgentSession(service.policy)
    current=dict(protocol=PROTOCOL,episode_id='new-episode')
    assert exchange(service,endpoint,'reset',current)['ok']
    before=service.policy.resets
    service.mailbox.requests.write(old_request_id+'.json',old)
    service.poll_once()
    assert service.policy.resets==before
    assert service.session.episode=='new-episode'


def test_startup_migrates_existing_http_descriptor_without_manual_address_choice(tmp_path, monkeypatch):
    from submission import shared_service, inference2
    from submission.remote_protocol import make_endpoint
    from submission.shared_channel import read_descriptor
    path = tmp_path/'endpoint.json'
    path.write_text(json.dumps(make_endpoint('http://example.com:8001', 'old-token')))
    monkeypatch.setattr(inference2, 'Policy', lambda **kw: CounterPolicy())
    handlers = {}
    monkeypatch.setattr(shared_service.signal, 'signal', lambda sig, fn: handlers.setdefault(sig, fn))
    monkeypatch.setattr(shared_service.SharedWorker, 'poll_once', lambda self: handlers[shared_service.signal.SIGINT]())
    assert shared_service.main(['--model', 'local-model', '--shared-root', str(tmp_path/'shared'), '--output', str(path)]) == 0
    assert read_descriptor(path)['transport'] == 'shared_directory'


def test_one_command_loads_checkpoint_creates_descriptor_and_serves_health(source, tmp_path):
    from submission.shared_channel import read_descriptor, encode_packet, decode_packet, digest
    from submission.shared_files import Mailbox
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'private/endpoint.json'
    shared = tmp_path/'shared'
    with (tmp_path/'agent.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-B', str(root/'submission/shared_service.py'),
            '--model', str(source[0]), '--shared-root', str(shared), '--output', str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        try:
            deadline = time.monotonic()+60
            while not output.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
            assert output.exists(), (tmp_path/'agent.log').read_text(errors='replace')
            endpoint = read_descriptor(output)
            with Mailbox(shared, endpoint['agent_id']) as box:
                rid = uuid.uuid4().hex
                raw = encode_packet(endpoint, rid, 'request', dict(method='health', body={}), expires_at=time.time()+30)
                box.requests.write(rid+'.json', raw)
                while rid+'.json' not in box.responses.names() and time.monotonic()<deadline: time.sleep(.02)
                result = decode_packet(box.responses.read(rid+'.json',65536), endpoint, 'response', request_digest=digest(raw))
                assert result['payload']['ok'] and result['payload']['model_loaded']
            assert endpoint['access_token'] not in (tmp_path/'agent.log').read_text(errors='replace')
        finally:
            process.terminate(); process.wait(timeout=10)
