import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from server_evaluator.tests.test_shared_agent import shared_service
from server_evaluator.tests.test_model_rollout import artifacts, fake_hfm
from server_evaluator.tests.test_offline_worker import platform, sample


def settings(service):
    return dict(shared_root=str(service[0]), timeout=1., retries=0, poll_interval=.005)


def test_signed_shared_rollout_keeps_official_hfm_score_and_delays(artifacts, fake_hfm, shared_service, tmp_path):
    from server_evaluator.shared_rollout import run_shared
    _, config, _ = artifacts
    config['remote_agent'] = settings(shared_service)
    endpoint = tmp_path/'endpoint.json'; endpoint.write_text(json.dumps(shared_service[1]))
    output=tmp_path/'trajectory.json'
    result=run_shared(endpoint, config, output)
    assert result['status']=='completed'
    assert set(result['metrics_scores'])=={'Ip','LCFS','X','strike','psiX','X2','psiX2'}
    assert fake_hfm.instances[-1].config['predictor']['shot_id']=='15892_400'
    np.testing.assert_array_equal(fake_hfm.instances[-1].actions, [[1.]*12,[2.]*12])
    assert all(instance.closed for instance in fake_hfm.instances)
    data=json.loads(output.read_text())
    assert data['audit']['remote_agent']['protocol']=='hfm-shared-policy-v1'
    assert shared_service[1]['access_token'] not in output.read_text()


def test_shared_platform_queue_scores_and_reports_without_loading_model(artifacts, fake_hfm, shared_service, platform, tmp_path, monkeypatch):
    from server_evaluator.shared_grader import SharedGrader
    from server_evaluator.shared_rollout import run_shared
    from server_evaluator.client import PlatformClient
    from server_evaluator.store import Store
    from server_evaluator.worker import Worker
    _, cfg, _ = artifacts
    grader=SharedGrader(cfg['target_path'],tmp_path/'runtime',shared_root=shared_service[0],
        env_config=cfg['env_config'],reference_config=cfg['reference_config'],agent_timeout=1,agent_retries=0)
    def child(endpoint,job,config):
        return dict(kind='completed',result=run_shared(endpoint,json.loads(config.read_text()),job/'trajectory.json'))
    monkeypatch.setattr(grader,'_run_child',child)
    for sid in platform['files']: platform['files'][sid]=json.dumps(shared_service[1]).encode()
    client=PlatformClient(platform['url'],'p1',username='test',password='test')
    try:
        with Store(tmp_path/'queue',grader.fingerprint) as store:
            worker=Worker(client,store,grader,tmp_path/'downloads',max_bytes=65536)
            assert worker.cycle()==0
            assert worker.cycle()==0
            assert shared_service[2].policy.calls==4 and platform['downloads']==2
            assert all(row['status']=='completed' for row in platform['records'].values())
        assert not list((tmp_path/'runtime').glob('evaluations/*/attempt-*/endpoint.json'))
        assert grader.config['power_supply']['pf_delay_range_s']==[.002,.005]
    finally: client.close()


def test_shared_cli_default_and_no_model_dependency():
    from server_evaluator.run import build_parser
    args=build_parser().parse_args(['--shared-root','/shared/agents'])
    assert args.submission_mode=='remote' and args.shared_root==Path('/shared/agents')
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'train','torch','stable_baselines3','onnx','onnxruntime'} or name=='submission.shared_service':
            raise RuntimeError('Forbidden import '+name)
sys.meta_path.insert(0,Block())
import server_evaluator.shared_rollout,server_evaluator.shared_grader
'''
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_explicit_legacy_http_mode_accepts_local_endpoint_file(tmp_path,monkeypatch):
    from server_evaluator import run
    class Legacy:
        config={'model_interface':{},'power_supply':{}}
        def __init__(self,*args,**kwargs): pass
        def grade_file(self,path): return dict(status='completed',metrics_scores={})
    monkeypatch.setattr(run,'RemoteGrader',Legacy)
    monkeypatch.setattr(run.signal,'signal',lambda *args:None)
    assert run.main(['--submission-mode','http','--endpoint-file','legacy.json',
                     '--data-dir',str(tmp_path/'runtime')])==0


def test_blocked_shared_write_watchdog_defers_instead_of_failing_submission(artifacts,shared_service,tmp_path,monkeypatch):
    from server_evaluator.shared_grader import SharedGrader
    from server_evaluator.grader import GradingUnavailable
    _,cfg,_=artifacts
    grader=SharedGrader(cfg['target_path'],tmp_path/'runtime',shared_root=shared_service[0],
        env_config=cfg['env_config'],reference_config=cfg['reference_config'],agent_timeout=.1,agent_retries=0)
    grader.inference_timeout=.3
    endpoint=tmp_path/'endpoint.json'; endpoint.write_text(json.dumps(shared_service[1]))
    job=tmp_path/'job';job.mkdir()
    config=job/'config.json';config.write_text(json.dumps({'_progress_path':str(job/'phase.json')}))
    def command(*args):
        script=("import json,time; from pathlib import Path; "
            "from submission.shared_files import Directory; "
            "from server_evaluator.shared_agent import SharedAgent; "
            "from server_evaluator.model_rollout import _phase; "
            "Directory.write=lambda *args:time.sleep(60); "
            f"cfg=json.loads(Path({str(config)!r}).read_text()); "
            f"SharedAgent(json.loads(Path({str(endpoint)!r}).read_text()),shared_root={str(shared_service[0])!r},"
            "timeout=.1,retries=0,on_request=lambda method,step:_phase(cfg,'shared_io' if method=='storage' else 'inference',step))")
        return [sys.executable,'-B','-c',script]
    monkeypatch.setattr(grader,'_command',command)
    with pytest.raises(GradingUnavailable,match='storage|Storage'):
        grader._run_child(endpoint,job,config)
