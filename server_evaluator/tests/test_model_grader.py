"""Model supervision and retry semantics, without a production HFM or platform."""
import json
from pathlib import Path
import sys
from threading import Event

import numpy as np
import pytest

from server_evaluator.tests.test_offline_worker import sample, platform, METRIC_MAXIMA


def make_grader(tmp_path, sample, **kwargs):
    from server_evaluator.model_grader import ModelGrader
    target = tmp_path / "target.json"
    target.write_text(json.dumps(sample[1]))
    return ModelGrader(target, tmp_path / "runtime", **kwargs)


def make_bundle(tmp_path):
    from submission.model_bundle import create_bundle
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "policy2.onnx").write_bytes(b"validated by child")
    (directory / "policy2.metadata.json").write_text('{"obs_dim": 2}')
    np.savez(directory / "policy2_vecnormalize_stats.npz", obs_mean=np.zeros(2), obs_var=np.ones(2))
    return create_bundle(directory, tmp_path / "model.zip")


def test_grading_audit_declares_reference_interface_and_physical_delays(tmp_path, sample):
    grader = make_grader(tmp_path, sample)
    assert grader.config["model_interface"] == {
        "feature_mode": "reference_only_v1", "obs_dim": 90, "action_dim": 7}
    assert grader.config["power_supply"] == {
        "dt_s": .001, "pf_delay_range_s": [.002, .005],
        "vs_delay_range_s": [0., .001], "delay_sampling": "each_step"}


def test_model_grader_unpacks_and_preserves_seven_scores(tmp_path, sample, monkeypatch):
    grader = make_grader(tmp_path, sample)
    result = {"metrics_scores": METRIC_MAXIMA, "status": "completed", "evaluator_message": "F2b score=30"}
    def child(model_dir, job_dir, config_path):
        assert (model_dir / "policy2.onnx").is_file()
        config = json.loads(config_path.read_text())
        assert config["target_path"] == str((tmp_path / "target.json").resolve())
        assert config["onnx_threads"] == 1
        return {"kind": "completed", "result": result}
    monkeypatch.setattr(grader, "_run_child", child)
    assert grader.grade_file(make_bundle(tmp_path)) == result
    assert not list((tmp_path / "runtime/evaluations").glob("*/attempt-*/model"))


def test_model_grader_rejects_trajectory_instead_of_executing_it(tmp_path, sample):
    from server_evaluator.grader import InvalidSubmission
    grader = make_grader(tmp_path, sample)
    upload = tmp_path / "infer.json"
    upload.write_text(json.dumps(sample[0]))
    with pytest.raises(InvalidSubmission, match="ZIP|zip"):
        grader.grade_file(upload)


def test_model_config_changes_invalidate_cache_and_active_worker(tmp_path, sample):
    from server_evaluator.grader import GradingUnavailable
    one = make_grader(tmp_path, sample, seed=1)
    two = make_grader(tmp_path, sample, seed=2)
    assert one.fingerprint != two.fingerprint
    (tmp_path / "target.json").write_text('{}')
    with pytest.raises(GradingUnavailable, match="changed"):
        one.grade_file(make_bundle(tmp_path))


@pytest.mark.parametrize('kind,exception', [('invalid','InvalidSubmission'), ('unavailable','GradingUnavailable')])
def test_child_failure_classification(tmp_path, sample, monkeypatch, kind, exception):
    from server_evaluator import grader as errors
    grader = make_grader(tmp_path, sample)
    monkeypatch.setattr(grader, '_run_child', lambda *args: {'kind': kind, 'error': 'reason'})
    with pytest.raises(getattr(errors, exception), match='reason'):
        grader.grade_file(make_bundle(tmp_path))
    assert not list((tmp_path / "runtime/evaluations").glob("*/attempt-*/model"))


def test_hard_timeout_terminates_child_and_defers_submission(tmp_path, sample, monkeypatch):
    from server_evaluator.grader import GradingUnavailable
    grader = make_grader(tmp_path, sample, evaluation_timeout=0.1)
    monkeypatch.setattr(grader, '_command', lambda *args: [sys.executable, '-c', 'import time; time.sleep(60)'])
    with pytest.raises(GradingUnavailable, match='time limit'):
        grader.grade_file(make_bundle(tmp_path))


@pytest.mark.parametrize("phase", ["model_load", "inference"])
def test_model_execution_timeout_is_failed_not_infrastructure_retry(tmp_path, sample, monkeypatch, phase):
    from server_evaluator.grader import InvalidSubmission
    grader = make_grader(tmp_path, sample, evaluation_timeout=2.0, inference_timeout=0.1)
    def command(model_dir, job_dir, config_path):
        script = ("from pathlib import Path; import time; "
                  f"Path({str(job_dir / 'phase.json')!r}).write_text({json.dumps({'phase': phase})!r}); "
                  "time.sleep(60)")
        return [sys.executable, "-c", script]
    monkeypatch.setattr(grader, "_command", command)
    with pytest.raises(InvalidSubmission, match="time limit"):
        grader.grade_file(make_bundle(tmp_path))


def test_total_timeout_during_new_inference_remains_infrastructure_error(tmp_path, sample, monkeypatch):
    from server_evaluator.grader import GradingUnavailable
    grader = make_grader(tmp_path, sample, evaluation_timeout=0.3, inference_timeout=10)
    def command(model_dir, job_dir, config_path):
        state = json.dumps({"phase": "inference", "step": 1})
        return [sys.executable, '-c', "from pathlib import Path; import time; "
                f"Path({str(job_dir / 'phase.json')!r}).write_text({state!r}); time.sleep(60)"]
    monkeypatch.setattr(grader, '_command', command)
    with pytest.raises(GradingUnavailable, match='time limit'):
        grader.grade_file(make_bundle(tmp_path))


def test_stop_event_prevents_starting_another_simulation(tmp_path, sample):
    from server_evaluator.grader import GradingUnavailable
    stop = Event(); stop.set()
    grader = make_grader(tmp_path, sample, stop_event=stop)
    with pytest.raises(GradingUnavailable, match='stopping'):
        grader.grade_file(make_bundle(tmp_path))


def test_worker_retries_infrastructure_without_failing_participant(tmp_path, sample, platform):
    from server_evaluator.client import PlatformClient
    from server_evaluator.grader import GradingUnavailable
    from server_evaluator.store import Store
    from server_evaluator.worker import Worker
    class FakeModelGrader:
        download_suffix = '.zip'
        calls = 0
        def grade_file(self, path):
            assert path.suffix == '.zip'
            self.calls += 1
            if self.calls == 1:
                raise GradingUnavailable('HFM is offline')
            return {'status':'completed', 'metrics_scores':METRIC_MAXIMA, 'evaluator_message':'ok'}
    grader = FakeModelGrader()
    client = PlatformClient(platform['url'], 'p1', token='test-platform-token')
    with Store(tmp_path / 'store', 'model-test') as store:
        worker = Worker(client, store, grader, tmp_path / 'downloads')
        worker.cycle()
        assert store.get('s1')['result'] is None
        assert store.get('s1')['attempts'] == 0
        assert platform['batches'] == []
        worker.cycle()
        assert all(row['status'] == 'completed' for row in platform['records'].values())
        assert len(platform['batches']) == 2  # return each slow evaluation promptly
        count = grader.calls
        worker.cycle()
        assert grader.calls == count
    client.close()
