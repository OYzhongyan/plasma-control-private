"""Experiment records and failure handling without HFM or real training."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_baseline_experiment.py"
    assert path.is_file(), "The experiment runner has not been implemented"
    spec = importlib.util.spec_from_file_location("baseline_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    config = tmp_path / "train/configs/base.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("ppo:\n  n_steps: 256\ntarget:\n  default_reference_path: configs/target.json\n")
    target = tmp_path / "configs/target.json"
    target.parent.mkdir()
    target.write_text('{"geometry": 1}')
    return tmp_path


def read(path):
    return json.loads(path.read_text())


def test_dry_run_records_reviewable_commands_without_launch(project):
    module = runner()
    exp = project / "experiments/review"
    exp.mkdir(parents=True)
    (exp / "DESIGN.md").write_text("approved design")
    def forbidden(*args, **kwargs):
        pytest.fail("dry run launched a child")
    assert module.main(["--experiment-id", "review", "--dry-run"], root=project, executor=forbidden) == 0
    manifest = read(exp / "manifest.json")
    assert manifest["requested_timesteps"] == 10000000
    assert manifest["expected_rounded_timesteps"] == 10002432
    assert manifest["target_hashes"]["configs/target.json"]
    assert read(exp / "requested_config.json")["features"]["observation_mode"] == "historical_observation_v1"
    commands = read(exp / "commands.json")
    train = commands["training"]
    assert "actions.mode=delta" in train
    assert "features.history_steps=4" in train
    assert "features.delay_step=1" in train
    assert "reset_perturbation.fix_reset_params=false" in train
    assert "ppo.device=cpu" in train
    assert "shot_groups.limiter_start=['15892_400','15892_500']" in train
    assert commands["evaluation"]["validation"]["command"][-1] == "eval.seed=200"
    assert read(exp / "status.json")["phase"] == "DRY_RUN"
    assert (exp / "DESIGN.md").read_text() == "approved design"
    assert "dense reward" in (exp / "README.md").read_text()


@pytest.mark.parametrize("args", [["--experiment-id", "../escape"], ["--experiment-id", "x", "--num-envs", "0"], ["--experiment-id", "x", "--base-port", "65535", "--num-envs", "2"], ["--experiment-id", "x", "--total-timesteps", "-1"]])
def test_invalid_arguments_rejected_before_records(project, args):
    with pytest.raises(SystemExit):
        runner().main(args, root=project)
    assert not (project / "experiments").exists()


@pytest.mark.parametrize("phase", ["RUNNING", "TRAINING", "COMPLETED"])
def test_existing_active_or_complete_experiment_is_preserved(project, phase):
    exp = project / "experiments/old"
    exp.mkdir(parents=True)
    (exp / "status.json").write_text(json.dumps({"phase": phase}))
    assert runner().main(["--experiment-id", "old", "--dry-run"], root=project) != 0
    assert read(exp / "status.json") == {"phase": phase}
    assert not (exp / "manifest.json").exists()


@pytest.mark.parametrize("failure", [17, RuntimeError("child launch failed")])
def test_training_failure_is_recorded_and_never_evaluated(project, failure):
    exp = project / "experiments/broken"
    def execute(command, *, cwd, env, log_path, on_start):
        assert command[1] == "train/train_ppo.py"
        assert env["WANDB_MODE"] == "offline"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        on_start(4242)
        log_path.write_text("training stopped")
        if isinstance(failure, Exception):
            raise failure
        return failure
    assert runner().main(["--experiment-id", "broken"], root=project, executor=execute) != 0
    status = read(exp / "status.json")
    assert status["phase"] == "FAILED"
    assert status["returncode"] != 0
    assert status["child_pid"] == 4242
    assert (exp / "train.log").read_text() == "training stopped"


def test_success_preserves_each_split_and_formal_scores(project):
    exp = project / "experiments/full"
    def execute(command, *, cwd, env, log_path, on_start):
        on_start(42)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("child output")
        if command[1] == "train/train_ppo.py":
            training = exp / "training"
            training.mkdir()
            (training / "progress.csv").write_text("time/total_timesteps,rollout/ep_rew_mean\n8192,99\n16384,100\n")
        else:
            output = exp / "training/eval"
            output.mkdir()
            seed = int(command[-1].split("=")[1])
            score = {100: 18., 200: 12., 300: 6.}[seed]
            (output / "score_summary.json").write_text(json.dumps({"task_id": "F2b", "mean_task_score": score, "min_task_score": score-1, "max_task_score": score+1, "rollouts": [{"task_score": score-1, "total_score": score-1}, {"task_score": score, "total_score": score}, {"task_score": score+1, "total_score": score+1}]}))
        return 0
    assert runner().main(["--experiment-id", "full", "--total-timesteps", "10000"], root=project, executor=execute) == 0
    metrics = read(exp / "metrics.json")
    assert metrics["actual_timesteps"] == 16384
    assert metrics["evaluation"]["validation"]["mean_task_score"] == 12.
    assert metrics["evaluation"]["test"]["mean_total_score"] == 6.
    assert len(list(csv.DictReader((exp / "metrics.csv").open()))) == 3
    assert all((exp / "evaluation" / split / "score_summary.json").exists() for split in ("seen", "validation", "test"))
    status = read(exp / "status.json")
    assert status["phase"] == "COMPLETED"
    assert status["returncode"] == 0
    assert (exp / "README.md").read_text().count("12.000000") >= 1


def test_successful_child_without_step_evidence_fails(project):
    def execute(command, **kwargs):
        return 0
    assert runner().main(["--experiment-id", "missing", "--skip-evaluation"], root=project, executor=execute) != 0
    assert read(project / "experiments/missing/status.json")["phase"] == "FAILED"


def test_evaluation_failure_retains_prior_split_and_nonzero_status(project):
    exp = project / "experiments/eval-failed"
    def execute(command, *, cwd, env, log_path, on_start):
        on_start(77)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("evaluation error")
        if command[1] == "train/train_ppo.py":
            training = exp / "training"
            training.mkdir()
            (training / "progress.csv").write_text("time/total_timesteps\n8192\n")
            return 0
        seed = int(command[-1].split("=")[1])
        if seed == 200:
            return 9
        output = exp / "training/eval"
        output.mkdir()
        (output / "score_summary.json").write_text(json.dumps({"rollouts": [{"task_score": 1., "total_score": 1.}] * 3}))
        return 0
    assert runner().main(["--experiment-id", "eval-failed", "--total-timesteps", "1"], root=project, executor=execute) == 9
    assert (exp / "evaluation/seen/score_summary.json").is_file()
    assert read(exp / "metrics.json")["evaluation"]["seen"]["mean_task_score"] == 1.
    assert read(exp / "status.json")["phase"] == "FAILED"
    assert read(exp / "status.json")["returncode"] == 9


def test_final_model_step_evidence_precedes_older_csv(project):
    import zipfile
    training = project / "training"
    training.mkdir()
    (training / "progress.csv").write_text("time/total_timesteps\n8192\n")
    with zipfile.ZipFile(training / "ppo_f2_latest.zip", "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 16384}))
    assert runner().actual_timesteps(training) == (16384, "ppo_f2_latest.zip:data.num_timesteps")


def test_local_ppo_log_fallback_completes_when_evaluation_skipped(project):
    exp = project / "experiments/local-logs"
    def execute(command, *, cwd, env, log_path, on_start):
        progress = exp / "training/logs/ppo/progress.csv"
        progress.parent.mkdir(parents=True)
        progress.write_text("time/total_timesteps,rollout/ep_rew_mean\n8192,10\n16384,20\n")
        return 0
    assert runner().main(["--experiment-id", "local-logs", "--total-timesteps", "10000", "--skip-evaluation"], root=project, executor=execute) == 0
    metrics = read(exp / "metrics.json")
    assert metrics["actual_timesteps"] == 16384
    assert metrics["actual_timesteps_evidence"] == "logs/ppo/progress.csv:time/total_timesteps"
    assert metrics["evaluation"] == {}
    status = read(exp / "status.json")
    assert status["phase"] == "COMPLETED"
    assert status["evaluation_skipped"] is True
    if "matplotlib unavailable" not in metrics["plot_status"]:
        assert (exp / "picture/training_reward.png").is_file()


def test_dense_reward_source_hash_is_recorded(project):
    source = project / "train/dense_rewards.py"
    source.write_text("# restored upstream reward\n")
    assert runner().main(["--experiment-id", "source", "--dry-run"], root=project) == 0
    hashes = read(project / "experiments/source/manifest.json")["source_hashes"]
    assert len(hashes["train/dense_rewards.py"]) == 64
