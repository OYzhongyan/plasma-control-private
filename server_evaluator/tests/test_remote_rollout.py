"""A's real HTTP interface with C-owned fake simulator and official scorer."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from server_evaluator.tests.test_remote_agent import agent_server
from server_evaluator.tests.test_model_rollout import artifacts, fake_hfm
from server_evaluator.tests.test_offline_worker import platform, sample


def test_remote_rollout_owns_shot_actions_and_score(artifacts, fake_hfm, agent_server, tmp_path):
    from server_evaluator.remote_rollout import run_remote
    _, config, _ = artifacts
    config['remote_agent'] = dict(timeout=1., retries=1, allowed_origins=[agent_server['url']])
    path = tmp_path / 'endpoint.json'
    path.write_text(json.dumps(agent_server['endpoint']))
    output = tmp_path / 'trajectory.json'
    result = run_remote(path, config, output)
    assert result['status'] == 'completed'
    assert set(result['metrics_scores']) == {'Ip', 'LCFS', 'X', 'strike', 'psiX', 'X2', 'psiX2'}
    assert len(fake_hfm.instances) == 2
    assert fake_hfm.instances[-1].config['predictor']['shot_id'] == '15892_400'
    np.testing.assert_array_equal(fake_hfm.instances[-1].actions, np.array([[1.] * 12, [2.] * 12]))
    assert all(env.closed for env in fake_hfm.instances)
    requests = agent_server['state']['requests']
    assert len([r for r in requests if r[1] == '/act']) == 2
    reset = json.loads(next(r[3] for r in requests if r[1] == '/reset'))
    assert set(reset) == {'protocol', 'episode_id'}
    assert 'target_path' not in repr(requests) and 'shot_id' not in repr(requests)
    data = json.loads(output.read_text())
    assert data['audit']['remote_agent']['model_id'] == 'test-model'
    assert agent_server['endpoint']['access_token'] not in output.read_text()


def test_remote_failure_is_submission_error_and_hfm_failure_is_deferred(artifacts, fake_hfm, agent_server, tmp_path):
    from server_evaluator.remote_rollout import run_remote
    from server_evaluator.grader import InvalidSubmission, GradingUnavailable
    _, config, _ = artifacts
    config['remote_agent'] = dict(timeout=1., retries=0, allowed_origins=[agent_server['url']])
    path = tmp_path / 'endpoint.json'
    path.write_text(json.dumps(agent_server['endpoint']))
    agent_server['state']['bad_action'] = True
    with pytest.raises(InvalidSubmission, match='12 finite'):
        run_remote(path, config, tmp_path / 'trajectory.json')
    agent_server['state']['bad_action'] = False
    fake_hfm.fail_step = True
    with pytest.raises(GradingUnavailable, match='HFM STEP'):
        run_remote(path, config, tmp_path / 'trajectory.json')


def test_remote_imports_do_not_load_training_or_inference_code():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'train', 'torch', 'stable_baselines3', 'onnx', 'onnxruntime'} or name in {'submission.history', 'submission.inference_common', 'submission.checkpoint_policy'}:
            raise RuntimeError('Forbidden import: ' + name)
sys.meta_path.insert(0, Block())
import server_evaluator.remote_rollout, server_evaluator.remote_grader
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_remote_grader_stages_only_endpoint_without_secret_in_config(artifacts, agent_server, tmp_path, monkeypatch):
    from server_evaluator.remote_grader import RemoteGrader
    from server_evaluator.grader import InvalidSubmission
    _, config, _ = artifacts
    grader = RemoteGrader(config['target_path'], tmp_path / 'runtime', env_config=config['env_config'],
        reference_config=config['reference_config'], allowed_origins=[agent_server['url']], agent_timeout=1)
    expected = dict(status='completed', metrics_scores=dict.fromkeys(grader.metric_names, 0.))
    def child(endpoint, job, configuration):
        assert json.loads(endpoint.read_text()) == agent_server['endpoint']
        assert 'private-agent-test-key' not in configuration.read_text()
        assert not any('/train/' in str(path).replace('\\', '/') for path in grader._snapshots)
        return dict(kind='completed', result=expected)
    monkeypatch.setattr(grader, '_run_child', child)
    path = tmp_path / 'endpoint.json'
    path.write_text(json.dumps(agent_server['endpoint']))
    assert grader.grade_file(path) == expected
    assert not list((tmp_path / 'runtime').glob('evaluations/*/attempt-*/endpoint.json'))
    path.write_text('{"protocol":"bad"}')
    with pytest.raises(InvalidSubmission, match='endpoint'):
        grader.grade_file(path)


def test_platform_download_remote_evaluate_and_report_once(artifacts, fake_hfm, agent_server, platform, tmp_path, monkeypatch):
    from server_evaluator.remote_grader import RemoteGrader
    from server_evaluator.remote_rollout import run_remote
    from server_evaluator.client import PlatformClient
    from server_evaluator.store import Store
    from server_evaluator.worker import Worker
    _, config, _ = artifacts
    grader = RemoteGrader(config['target_path'], tmp_path / 'runtime', env_config=config['env_config'],
        reference_config=config['reference_config'], allowed_origins=[agent_server['url']], agent_timeout=1)
    def child(endpoint, job, configuration):
        return dict(kind='completed', result=run_remote(endpoint, json.loads(configuration.read_text()), job / 'trajectory.json'))
    monkeypatch.setattr(grader, '_run_child', child)
    for sid in platform['files']:
        platform['files'][sid] = json.dumps(agent_server['endpoint']).encode()
    client = PlatformClient(platform['url'], 'p1', username='test-only', password='test-only')
    try:
        with Store(tmp_path / 'queue', grader.fingerprint) as store:
            worker = Worker(client, store, grader, tmp_path / 'downloads', max_bytes=65536)
            assert worker.cycle() == 0
            calls, downloads = agent_server['policy'].calls, platform['downloads']
            assert calls == 4 and downloads == 2
            assert worker.cycle() == 0
            assert agent_server['policy'].calls == calls and platform['downloads'] == downloads
            assert all(row['status'] == 'completed' for row in platform['records'].values())
            assert all(set(row['metrics_scores']) == set(grader.metric_names) for row in platform['records'].values())
    finally:
        client.close()


def test_cli_remote_mode_and_endpoint_options():
    from server_evaluator.run import build_parser
    args = build_parser().parse_args(['--endpoint-file', 'a.json', '--allow-agent-origin', 'http://10.1.2.3:8001'])
    assert args.submission_mode == 'remote'
    assert args.endpoint_file == Path('a.json')
    assert args.allow_agent_origin == ['http://10.1.2.3:8001']


@pytest.mark.parametrize('failed_step', [-1, 0])
def test_c_phase_write_failure_during_remote_rpc_remains_pending(artifacts, fake_hfm, agent_server, tmp_path, monkeypatch, failed_step):
    from server_evaluator import remote_rollout
    from server_evaluator.grader import GradingUnavailable
    _, config, _ = artifacts
    config['remote_agent'] = dict(timeout=1., retries=0, allowed_origins=[agent_server['url']])
    path = tmp_path / 'endpoint.json'
    path.write_text(json.dumps(agent_server['endpoint']))
    def phase(config, name, step):
        if step == failed_step:
            raise GradingUnavailable('Cannot write rollout phase: disk full')
    monkeypatch.setattr(remote_rollout, '_phase', phase)
    with pytest.raises(GradingUnavailable, match='disk full'):
        remote_rollout.run_remote(path, config, tmp_path / 'trajectory.json')
