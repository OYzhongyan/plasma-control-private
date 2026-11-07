from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest


def test_managed_hfm_uses_c_runtime_and_cleans_up(tmp_path, monkeypatch):
    import server_evaluator.hfm_runtime as runtime
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    root = tmp_path / 'runtime'
    root.mkdir()
    (root / 'run_hfm_server.sh').write_text('# organizer script')
    calls = []
    class Process:
        pid = 123456
        def poll(self): return None
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write(b'Waiting for a client to connect...\n')
        kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    stopped = []
    monkeypatch.setattr(runtime.ManagedHFM, '_stop_process', staticmethod(lambda proc: stopped.append(proc.pid)))
    monkeypatch.setattr(runtime, '_ensure_free_port', lambda *args: None)
    service = runtime.ManagedHFM(root, tmp_path / 'data', port=15558, startup_timeout=1)
    service.start()
    service.close()
    assert stopped == [123456]
    command, options = calls[0]
    assert command == ['bash', str(root / 'run_hfm_server.sh')]
    assert options['start_new_session'] is True
    assert options['env']['LISTEN_PORT'] == '15558'
    assert not any(k.startswith('PLATFORM_') for k in options['env'])


def test_managed_hfm_rejects_remote_host(tmp_path):
    from server_evaluator.hfm_runtime import ManagedHFM
    with pytest.raises(ValueError, match='local'):
        ManagedHFM(tmp_path, tmp_path / 'data', host='10.0.0.2')
