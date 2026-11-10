"""Run a recorded F2b baseline against an already running local HFM pool.

This command owns its training/evaluation subprocesses and experiment records;
it never creates, stops or reconfigures HFM services. Use --dry-run to review
the exact commands and provenance before starting a long experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as packages
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "ffda09ab4621ede477ff71247547f8cbc43d133a"
SPLITS = {"seen": ("15892_400", 100), "validation": ("14795_400", 200), "test": ("13705_500", 300)}
SOURCE_FILES = ["train/train_ppo.py", "train/wrappers.py", "train/rewards.py", "train/dense_rewards.py", "train/callbacks.py",
                "train/flux_critic_policy.py", "train/flux_observation.py",
                "train/references.py", "train/evaluate.py", "train/export_onnx.py", "train/configs/base.yaml",
                "train/configs/eval.yaml", "submission/history.py", "submission/checkpoint_policy.py",
                "environment/hfm_simulator.py", "environment/hfm_predictor.py", "environment/power_supply.py",
                "environment/xpt_utils.py", "configs/env_default.yaml",
                "tools/run_baseline_experiment.py"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_output(root, *args):
    if not (root / ".git").exists():
        return ""
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Unable to capture Git provenance")
    return result.stdout.decode("utf-8", errors="replace")


def provenance(root):
    files = sorted(set(SOURCE_FILES + [p.relative_to(root).as_posix() for p in (root / "evaluation/eval/score").glob("*.py")]))
    # Only explicitly listed participant/scorer code is captured, never runtime,
    # credential directories, private target files or generated results.
    patch = git_output(root, "diff", "HEAD", "--", *files)
    patch = re.sub(r"(?im)^([+-].*(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*).+$", r"\1<redacted>", patch)
    return {"initial_git_sha": git_output(root, "rev-parse", "HEAD").strip() or None,
            "dirty_diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "source_hashes": {name: sha256(root / name) for name in files if (root / name).is_file()},
            "upstream_commit": UPSTREAM_COMMIT}, patch


def environment_record():
    versions = {}
    for name in ("torch", "stable-baselines3", "gymnasium", "numpy", "hydra-core", "omegaconf", "tensorboard", "wandb", "matplotlib"):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    cgroup = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            cgroup[name] = path.read_text().strip()
    installed = {dist.metadata.get("Name", "unknown"): dist.version for dist in packages.distributions()}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "installed_packages": dict(sorted(installed.items())),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": versions, "cgroup": cgroup}


def config_record(root, args):
    import yaml
    path = root / "train/configs/base.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base.update(num_envs=args.num_envs, ports=str(args.base_port), max_steps=500, shot_id="15892_400",
                shot_groups={"limiter_start": ["15892_400", "15892_500"]})
    overrides = {"ppo": {"total_timesteps": args.total_timesteps, "device": "cpu", "seed": args.seed, "batch_size": 1024},
                 "features": {"observation_mode": "historical_observation_v1", "history_steps": 4, "delay_step": 1}, "actions": {"mode": "delta"},
                 "reset_perturbation": {"enabled": True, "fix_reset_params": False, "apply_mode": "all",
                                        "signeo_scale_range": [.95, 1.05], "bp_scale_range": [.95, 1.05], "q0_scale_range": [.95, 1.05]},
                 "wandb": {"enabled": args.wandb_mode != "disabled"}}
    for key, values in overrides.items():
        base.setdefault(key, {}).update(values)
    base.setdefault("critic", {})["type"] = getattr(args, "critic_type", "mlp")
    targets = ["configs/xpt_reference_targets.json"]
    target = base.get("target", {})
    for key in ("default_reference_path", "reference_variants_path"):
        if target.get(key):
            targets.append(str(target[key]))
    return base, {name: sha256(root / name) for name in sorted(set(targets)) if (root / name).is_file()}


def compare_baseline_config(baseline, candidate):
    """Permit critic settings only; reject silent changes to the comparison."""
    differences = []
    def walk(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                if not prefix and key == "critic":
                    continue
                walk(a.get(key), b.get(key), f"{prefix}.{key}".strip("."))
        elif a != b:
            differences.append({"key": prefix, "baseline": a, "candidate": b})
    walk(baseline, candidate)
    if differences:
        raise ValueError("Non-critic baseline settings changed: " + ", ".join(row["key"] for row in differences))
    return {"non_critic_settings_identical": True,
            "baseline_critic": baseline.get("critic", {"type": "mlp"}),
            "candidate_critic": candidate.get("critic", {"type": "mlp"}),
            "other_differences": []}


def hydra_value(value):
    # Hydra's override parser retains these quotes as a string, including spaces
    # and Windows drive letters; subprocess receives a list without a shell.
    return json.dumps(str(value).replace("\\", "/"))


def commands_for(root, exp, args):
    training = exp / "training"
    train = [sys.executable, "train/train_ppo.py", f"ppo.total_timesteps={args.total_timesteps}", "ppo.device=cpu",
             f"ppo.seed={args.seed}", "ppo.batch_size=1024", f"num_envs={args.num_envs}", f"ports='{args.base_port}'",
             "max_steps=500", "shot_id='15892_400'", "shot_groups.limiter_start=['15892_400','15892_500']",
             "actions.mode=delta", "features.observation_mode=historical_observation_v1", "features.history_steps=4", "features.delay_step=1",
             "reset_perturbation.enabled=true", "reset_perturbation.fix_reset_params=false", "reset_perturbation.apply_mode=all",
             "reset_perturbation.signeo_scale_range=[0.95,1.05]", "reset_perturbation.bp_scale_range=[0.95,1.05]",
             "reset_perturbation.q0_scale_range=[0.95,1.05]", f"wandb.enabled={'false' if args.wandb_mode == 'disabled' else 'true'}",
             "hydra.run.dir=" + hydra_value(training)]
    if getattr(args, "critic_type", "mlp") != "mlp":
        train.append("critic.type=" + args.critic_type)
    evaluation = {}
    for split, (shot, seed) in SPLITS.items():
        command = [sys.executable, "train/evaluate.py", "eval.model_path=" + hydra_value(training), f"eval.shots=['{shot}']",
                   "eval.episodes_per_shot=3", "eval.obs_perturb.enabled=true", "eval.obs_perturb.delay_steps=1", "eval.policy_mode=model",
                   "eval.score=true", "eval.deterministic=true", f"ports='{args.eval_port}'", "reset_perturbation.enabled=true",
                   "reset_perturbation.fix_reset_params=false", "reset_perturbation.apply_mode=all",
                   "reset_perturbation.signeo_scale_range=[0.95,1.05]", "reset_perturbation.bp_scale_range=[0.95,1.05]",
                   "reset_perturbation.q0_scale_range=[0.95,1.05]", "wandb.enabled=false",
                   "hydra.run.dir=" + hydra_value(exp / "evaluation" / split / "hydra"), f"eval.seed={seed}"]
        evaluation[split] = {"shot_id": shot, "seed": seed, "episodes": 3, "command": command}
    return {"training": train, "evaluation": evaluation}


def execute_child(command, *, cwd, env, log_path, on_start):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            on_start(child.pid)
            return child.wait()
        except BaseException:
            # Only this runner's own child; HFM services are never touched.
            if child.poll() is None:
                child.terminate()
                child.wait()
            raise


def actual_timesteps(training):
    model = training / "ppo_f2_latest.zip"
    if model.is_file():
        # SB3 stores num_timesteps as plain JSON in the ZIP's data member.
        with zipfile.ZipFile(model) as archive:
            value = json.loads(archive.read("data"))["num_timesteps"]
        if type(value) is int and value > 0:
            return value, "ppo_f2_latest.zip:data.num_timesteps"
    for progress in (training / "logs/ppo/progress.csv", training / "progress.csv"):
        if progress.is_file():
            with progress.open(encoding="utf-8-sig", newline="") as stream:
                values = [int(float(row["time/total_timesteps"])) for row in csv.DictReader(stream) if row.get("time/total_timesteps")]
            if values:
                return max(values), progress.relative_to(training).as_posix() + ":time/total_timesteps"
    steps = [int(match.group(1)) for p in (training / "checkpoints").glob("ppo_f2_*_steps.zip")
             if (match := re.fullmatch(r"ppo_f2_(\d+)_steps.zip", p.name))]
    if steps:
        return max(steps), "checkpoint_filename"
    raise RuntimeError("Training returned success without evidence of actual model timesteps")


def summarize_score(path, specification):
    result = json.loads(path.read_text(encoding="utf-8"))
    rollouts = result.get("rollouts", [])
    if len(rollouts) != specification["episodes"]:
        raise ValueError("Evaluation did not record all three episodes")
    task = [float(row["task_score"]) for row in rollouts]
    total = [float(row["total_score"]) for row in rollouts]
    if not all(math.isfinite(value) and 0 <= value <= 30.000001 for value in task + total):
        raise ValueError("Invalid formal competition score")
    return {"shot_id": specification["shot_id"], "seed": specification["seed"], "episodes": len(rollouts),
            "mean_task_score": sum(task) / len(task), "min_task_score": min(task), "max_task_score": max(task),
            "mean_total_score": sum(total) / len(total), "official_max_score": 30.0}


def save_metrics(exp, metrics):
    write_json(exp / "metrics.json", metrics)
    with (exp / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["split", "shot_id", "seed", "episodes", "mean_task_score", "min_task_score", "max_task_score", "mean_total_score", "official_max_score"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"split": split, **row} for split, row in metrics["evaluation"].items())


def write_readme(exp, args, metrics=None):
    rows = []
    for split, (shot, seed) in SPLITS.items():
        score = (metrics or {}).get("evaluation", {}).get(split)
        values = [f"{score[key]:.6f}" for key in ("mean_task_score", "min_task_score", "max_task_score")] if score else ["pending"] * 3
        rows.append(f"| {split} | {shot} | {seed} | " + " | ".join(values) + " |")
    text = (f"# {args.experiment_id}\n\nRestore SII-585-THU F2b history inputs and upstream dense reward; "
            f"run a recorded seed-{args.seed} baseline. Source commit: `{UPSTREAM_COMMIT}`.\n\n"
            "Changes: four observation frames, five physical-action slots in delta mode (812 inputs), one-step sensor delay, "
            "fresh +/-5% reset perturbations, CPU PPO, local scalar records and W&B " + args.wandb_mode + ".\n\n"
            f"Requested timesteps: {args.total_timesteps}; actual: {(metrics or {}).get('actual_timesteps', 'pending')}. "
            "The training dense reward is not the final score. Formal competition scores below use the unchanged F2b scorer, maximum 30. "
            "Seen, validation and test results are reported separately; one seed does not establish broad generalization.\n\n"
            "| Split | Shot | Seed | Mean task score /30 | Min | Max |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            "Reproduction commands are in `commands.json`; exact requested config and provenance are in `requested_config.json`, "
            "`manifest.json`, `code.patch` and `environment.json`. Hydra saves the resolved training config under "
            "`training/.hydra/config.yaml`, copied to `resolved_config.yaml` after training. Raw logs/checkpoints are under "
            "`training/`, split scoring artifacts under `evaluation/`, and figures under `picture/`. The runner reuses an existing HFM pool.\n")
    if getattr(args, "critic_type", "mlp") == "flux_cnn":
        text = text.replace("run a recorded seed-", "replace only the critic with four-frame flux CNN and run a matched seed-")
        text += "\nCritic: shared per-frame CNN, ordered four-frame fusion, original 812 auxiliary inputs and validity masks. Actor architecture/initialization unchanged. See DESIGN.md and baseline_comparison.json. This treatment adds critic information as well as changing its architecture.\n"
    (exp / "README.md").write_text(text, encoding="utf-8")


def make_plots(exp):
    """Optional derived views; missing plotting packages never discard raw data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; raw CSV retained"
    picture = exp / "picture"
    picture.mkdir(exist_ok=True)
    progress = exp / "training/logs/ppo/progress.csv"
    if not progress.is_file():
        progress = exp / "training/progress.csv"
    sources = [(progress, "time/total_timesteps", "rollout/ep_rew_mean", "training_reward.png"),
               (exp / "training/monitor.csv", "t", "r", "episode_reward.png")]
    sources.extend((exp / "evaluation" / split / "step_reward_components.csv", "step", "ip", f"{split}_ip.png") for split in SPLITS)
    for path, xkey, ykey, name in sources:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
        valid = [row for row in rows if row.get(xkey) and row.get(ykey)]
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            if ykey == "ip":
                episodes = sorted({(row.get("shot_id", ""), row.get("episode", "")) for row in valid})
                for shot, episode in episodes:
                    group = [row for row in valid if (row.get("shot_id", ""), row.get("episode", "")) == (shot, episode)]
                    ax.plot([float(row[xkey]) for row in group], [float(row[ykey]) for row in group], label=f"{shot} ep{episode}")
                group = valid[:501]
                if all(row.get("reference_ip") for row in group):
                    ax.plot([float(row[xkey]) for row in group], [float(row["reference_ip"]) for row in group], "k--", label="reference")
                ax.legend()
            else:
                ax.plot([float(row[xkey]) for row in valid], [float(row[ykey]) for row in valid])
            ax.set(xlabel=xkey, ylabel=ykey)
            fig.tight_layout()
            fig.savefig(picture / name, dpi=140)
        finally:
            plt.close(fig)
    return "generated from recorded CSV"


def parser_and_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--base-port", type=int, default=2223)
    parser.add_argument("--eval-port", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-type", choices=("mlp", "flux_cnn"), default="mlp")
    parser.add_argument("--baseline-config", type=Path, help="Require exact non-critic equality to an archived requested_config.json")
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.experiment_id):
        parser.error("experiment-id must be a single simple directory name")
    if args.total_timesteps <= 0 or args.num_envs <= 0 or args.seed < 0:
        parser.error("timesteps and num-envs must be positive; seed must be nonnegative")
    args.eval_port = args.base_port if args.eval_port is None else args.eval_port
    if not (1 <= args.base_port <= 65535 and args.base_port + args.num_envs - 1 <= 65535 and 1 <= args.eval_port <= 65535):
        parser.error("training and evaluation ports must fit within 1..65535")
    return args


def main(argv=None, *, root=ROOT, executor=None):
    args = parser_and_args(argv)
    root = Path(root).resolve()
    exp = root / "experiments" / args.experiment_id
    if exp.is_symlink() or not exp.resolve().is_relative_to(root / "experiments"):
        print("Experiment directory must stay within experiments/", file=sys.stderr)
        return 1
    exp.mkdir(parents=True, exist_ok=True)
    status_path = exp / "status.json"
    lock = exp / ".runner.lock"
    try:
        if status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if str(previous.get("phase", "")).upper() not in {"DRY_RUN"}:
                raise RuntimeError("Experiment already has a status record; use a new experiment-id")
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    status = {"pid": os.getpid(), "child_pid": None, "phase": "RUNNING", "returncode": None}
    def update(phase, **values):
        status.update(phase=phase, updated_utc=utc_now(), **values)
        write_json(status_path, status)
    try:
        update("RUNNING")
        config, target_hashes = config_record(root, args)
        if args.baseline_config:
            comparison = compare_baseline_config(json.loads(args.baseline_config.read_text(encoding="utf-8")), config)
            comparison["baseline_config_sha256"] = sha256(args.baseline_config)
            write_json(exp / "baseline_comparison.json", comparison)
        commands = commands_for(root, exp, args)
        source, patch = provenance(root)
        n_steps = int(config.get("ppo", {}).get("n_steps", 256))
        rollout_steps = n_steps * args.num_envs
        manifest = {**source, "experiment_id": args.experiment_id, "created_utc": utc_now(),
                    "requested_timesteps": args.total_timesteps, "expected_rounded_timesteps": math.ceil(args.total_timesteps / rollout_steps) * rollout_steps,
                    "n_envs": args.num_envs, "n_steps": n_steps, "seed": args.seed, "shots": ["15892_400", "15892_500"],
                    "target_hashes": target_hashes, "command": commands["training"], "wandb_mode": args.wandb_mode,
                    "official_score_max": 30, "training_reward": "upstream F2 dense reward",
                    "power_supply_seed_rule": "environment seed + 30000; continuous RNG stream across resets",
                    "evaluation_sensor_noise_std": {"Ip": 2300.0, "R": 0.008},
                    "inference_kernel_latency_measured": False}
        manifest["critic_type"] = args.critic_type
        write_json(exp / "manifest.json", manifest)
        write_json(exp / "environment.json", environment_record())
        write_json(exp / "requested_config.json", config)
        write_json(exp / "commands.json", commands)
        (exp / "code.patch").write_text(patch, encoding="utf-8")
        write_readme(exp, args)
        if args.dry_run:
            update("DRY_RUN", returncode=0)
            print(f"Review experiment records at {exp}")
            return 0
        env = os.environ.copy()
        env.update(WANDB_MODE=args.wandb_mode, WANDB_DIR=str(exp / "wandb"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        (exp / "wandb").mkdir(exist_ok=True)
        execute = executor or execute_child
        update("TRAINING")
        code = execute(commands["training"], cwd=root, env=env, log_path=exp / "train.log", on_start=lambda pid: update("TRAINING", child_pid=pid))
        if code:
            update("FAILED", returncode=code, error="Training child failed")
            return code if code > 0 else 1
        training = exp / "training"
        actual, evidence = actual_timesteps(training)
        if actual < args.total_timesteps:
            raise RuntimeError(f"Training stopped at {actual}, below requested {args.total_timesteps}")
        manifest.update(actual_timesteps=actual, actual_timesteps_evidence=evidence)
        write_json(exp / "manifest.json", manifest)
        resolved = training / ".hydra/config.yaml"
        if resolved.is_file():
            shutil.copyfile(resolved, exp / "resolved_config.yaml")
        metrics = {"requested_timesteps": args.total_timesteps, "actual_timesteps": actual,
                   "actual_timesteps_evidence": evidence, "training_reward": "upstream F2 dense reward", "evaluation": {}}
        save_metrics(exp, metrics)
        if not args.skip_evaluation:
            for split, specification in commands["evaluation"].items():
                update("EVALUATING", split=split)
                destination = exp / "evaluation" / split
                destination.mkdir(parents=True, exist_ok=True)
                output = training / "eval"
                if output.exists():
                    raise RuntimeError("Unarchived training/eval directory exists; refusing to overwrite evaluation")
                code = execute(specification["command"], cwd=root, env=env, log_path=destination / "evaluate.log",
                               on_start=lambda pid: update("EVALUATING", child_pid=pid))
                if code:
                    update("FAILED", returncode=code, error=f"{split} evaluation failed")
                    return code if code > 0 else 1
                metrics["evaluation"][split] = summarize_score(output / "score_summary.json", specification)
                for path in output.iterdir():
                    if (destination / path.name).exists():
                        raise RuntimeError(f"Evaluation artifact already exists: {path.name}")
                    shutil.move(str(path), str(destination / path.name))
                output.rmdir()
                save_metrics(exp, metrics)
                write_readme(exp, args, metrics)
        write_readme(exp, args, metrics)
        try:
            metrics["plot_status"] = make_plots(exp)
        except Exception as exc:
            metrics["plot_status"] = f"Plot generation failed: {exc}"
        save_metrics(exp, metrics)
        update("COMPLETED", returncode=0, evaluation_skipped=args.skip_evaluation,
               artifacts=sorted(path.relative_to(exp).as_posix() for path in exp.rglob("*") if path.is_file() and path != lock))
        print(f"Completed experiment: {exp}")
        return 0
    except BaseException as exc:
        update("FAILED", returncode=1, error=f"{type(exc).__name__}: {exc}")
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
