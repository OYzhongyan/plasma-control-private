"""Real worker subprocesses, no platform credentials and no real HFM."""
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

import pytest
from server_evaluator.tests.test_offline_worker import platform, sample


@pytest.fixture
def fake_worker(tmp_path):
    path=tmp_path/'worker.py'
    path.write_text('''
import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--case');p.add_argument('--marker')
a,extra=p.parse_known_args();root=Path(a.data_dir);root.mkdir(parents=True,exist_ok=True)
(root/'observed.json').write_text(json.dumps({'extra':extra,'supervised':os.environ.get('PLASMA_GRADER_SUPERVISED'),'cwd':str(Path.cwd())}))
marker=Path(a.marker) if a.marker else root/'unused'
if a.case in {'change','mismatch'} and not marker.exists():
    marker.write_text('first');(root/'old-result.json').write_text('preserve')
    if a.case=='change':print('2026-11-07 14:13:42,120 WARNING submission=test evaluation deferred: Server code/configuration changed; restart with a new data directory',flush=True)
    else:
        print('2026-11-07 14:13:42,120 ERROR Worker stopped: Data directory fingerprint differs (target/rules/metric/platform/problem changed); use a new data directory',flush=True)
        raise SystemExit(1)
    time.sleep(60)
elif a.case=='ordinary':
    print('WARNING submission=test evaluation deferred: HFM STEP timed out',flush=True)
elif a.case=='failure':
    print('ERROR Worker stopped: target is missing',flush=True);raise SystemExit(7)
elif a.case=='wait':
    marker.write_text(str(os.getpid()));time.sleep(60)
else: print('stable worker completed',flush=True)
''',encoding='utf-8')
    return [sys.executable,'-B','-u',str(path)]


@pytest.mark.parametrize('case',['change','mismatch'])
def test_configuration_warning_rotates_cache_and_preserves_old_results(tmp_path,fake_worker,case):
    from server_evaluator.watchdog import Supervisor
    base=tmp_path/'supervisor'
    service=Supervisor(base,['--case',case,'--marker',str(tmp_path/'first'),
        '--target','official.json','--poll-interval','15'],worker_command=fake_worker,
        restart_delay=.01,shutdown_timeout=.5)
    assert service.run()==0
    runs=list((base/'runs').iterdir())
    assert len(runs)==2
    assert sum((p/'old-result.json').exists() for p in runs)==1
    state=json.loads((base/'active.json').read_text())
    current=base/'runs'/state['active_run']
    observed=json.loads((current/'observed.json').read_text())
    assert observed['extra']==['--target','official.json','--poll-interval','15']
    assert observed['supervised']=='1'
    assert 'stable worker completed' in (current/'worker.log').read_text()
    assert 'restart' in (base/'supervisor.log').read_text().lower()


def test_ordinary_deferral_does_not_restart_and_relaunch_resumes_cache(tmp_path,fake_worker):
    from server_evaluator.watchdog import Supervisor
    base=tmp_path/'supervisor'
    for _ in range(2):
        assert Supervisor(base,['--case','ordinary'],worker_command=fake_worker,restart_delay=.01).run()==0
    assert len(list((base/'runs').iterdir()))==1


def test_unrelated_fatal_exit_is_exposed_without_restart_loop(tmp_path,fake_worker):
    from server_evaluator.watchdog import Supervisor
    assert Supervisor(tmp_path/'supervisor',['--case','failure'],worker_command=fake_worker).run()==7
    assert len(list((tmp_path/'supervisor/runs').iterdir()))==1


def test_forwarded_relative_paths_keep_invocation_directory(tmp_path,fake_worker,monkeypatch):
    from server_evaluator.watchdog import Supervisor
    invocation=tmp_path/'caller';invocation.mkdir()
    (invocation/'target.json').write_text('{}')
    monkeypatch.chdir(invocation)
    base=tmp_path/'state'
    assert Supervisor(base,['--case','ordinary','--target','target.json'],worker_command=fake_worker).run()==0
    current=next((base/'runs').iterdir())
    assert json.loads((current/'observed.json').read_text())['cwd']==str(invocation)


def test_systemd_does_not_restart_explicit_fatal_exit_forever():
    path=Path(__file__).resolve().parents[1]/'plasma-grader.service'
    assert 'Restart=no' in path.read_text()


def test_restart_warning_must_be_a_complete_trusted_log_event():
    from server_evaluator.watchdog import restart_reason
    warning='2026-11-07 14:13:42,120 WARNING submission=test evaluation deferred: Server code/configuration changed; restart with a new data directory'
    assert restart_reason(warning)
    assert restart_reason('WARNING submission=x report rejected: evaluation deferred: Server code/configuration changed; restart with a new data directory') is None
    assert restart_reason('quoted message: '+warning) is None
    assert restart_reason(warning+' mentioned in another error') is None


def test_stop_does_not_restart_or_touch_external_process(tmp_path,fake_worker):
    from server_evaluator.watchdog import Supervisor
    external=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'])
    stop=Event();marker=tmp_path/'ready';result=[]
    service=Supervisor(tmp_path/'supervisor',['--case','wait','--marker',str(marker)],
        worker_command=fake_worker,stop_event=stop,shutdown_timeout=.5)
    thread=Thread(target=lambda:result.append(service.run()));thread.start()
    try:
        deadline=time.monotonic()+10
        while not marker.exists() and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists()
        stop.set();thread.join(timeout=10)
        assert result==[0] and not thread.is_alive()
        assert external.poll() is None
        assert len(list((tmp_path/'supervisor/runs').iterdir()))==1
    finally:
        stop.set();thread.join(timeout=10)
        external.terminate();external.wait(timeout=5)


@pytest.mark.parametrize('flag',['--hfm-runtime-dir','--hfm-runtime-d'])
def test_cli_rejects_managed_hfm_even_abbreviated(flag,tmp_path):
    from server_evaluator.watchdog import main
    with pytest.raises(SystemExit) as error:
        main(['--data-dir',str(tmp_path/'supervisor'),flag,'/runtime','--target','x.json'])
    assert error.value.code==2
    assert not (tmp_path/'supervisor/runs').exists()


def test_corrupt_active_state_does_not_start_or_discard_history(tmp_path,fake_worker):
    from server_evaluator.watchdog import Supervisor
    base=tmp_path/'supervisor';base.mkdir()
    (base/'active.json').write_text('{"active_run":"../../outside"}')
    with pytest.raises(ValueError,match='state'):
        Supervisor(base,[],worker_command=fake_worker).run()
    assert (base/'active.json').exists()


def test_real_run_config_change_restarts_and_processes_pending_platform_task(tmp_path,platform,sample,monkeypatch):
    from server_evaluator.watchdog import Supervisor
    platform['records'].clear();platform['files'].clear()
    monkeypatch.setenv('PLATFORM_USERNAME','test-only')
    monkeypatch.setenv('PLATFORM_PASSWORD','test-only')
    monkeypatch.delenv('PLATFORM_TOKEN',raising=False)
    target=tmp_path/'target.json';target.write_text(json.dumps(sample[1]))
    base=tmp_path/'watchdog';stop=Event();results=[]
    service=Supervisor(base,['--target',str(target),'--platform-url',platform['url'],
        '--problem-id','p1','--poll-interval','.1'],restart_delay=.05,shutdown_timeout=2,stop_event=stop)
    thread=Thread(target=lambda:results.append(service.run()));thread.start()
    try:
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            logs=list((base/'runs').glob('*/worker.log'))
            if logs and 'Cycle finished' in logs[0].read_text(errors='replace'):break
            time.sleep(.05)
        else:pytest.fail('Real worker did not begin polling')
        target.write_text(json.dumps(sample[1],indent=2))
        platform['files']['new']=b'{"invalid":"descriptor"}'
        platform['records']['new']=dict(id='new',problem_id='p1',source='offline',status='pending',file_url=platform['url']+'/files/new')
        while time.monotonic()<deadline:
            if platform['records']['new']['status']=='failed':break
            time.sleep(.05)
        else:pytest.fail('Replacement worker did not process pending task')
        assert len(list((base/'runs').iterdir()))==2
        assert 'Restart requested' in (base/'supervisor.log').read_text()
    finally:
        stop.set();thread.join(timeout=15)
    assert not thread.is_alive() and results==[0]
