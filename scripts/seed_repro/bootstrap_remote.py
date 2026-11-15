"""Prepare an isolated, version-matched baseline replay on plasma-control3."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

PARENT = Path('/inspire/qb-ilm/project/ai-for-fusion/czxs25220173')
ROOT = PARENT/'plasma-control-seed-repro'
EXP = ROOT/'experiments/20261115_f2b_history_baseline_10m_seed0_repro'
RUNTIME = PARENT/'hfm-runtime-release-sealed-latest'
PYTHON = Path('/root/miniforge3/envs/competition_repro_20261115/bin/python')
BASELINE = PARENT/'plasma-control/experiments/20261109_f2b_history_baseline_10m_seed0'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, data):
    path = EXP/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def ports():
    listening, connected = set(), set()
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(name).read_text().splitlines()[1:]:
            fields = line.split()
            port = int(fields[1].split(':')[1], 16)
            if fields[3] == '0A':
                listening.add(port)
            if fields[3] == '01':
                connected.add(port)
    return listening, connected


def main():
    assert os.uname().nodename.startswith('plasma-control3--')
    write('setup/launch_status.json', {'phase': 'IMPORTING_ENVIRONMENT', 'pid': os.getpid(), 'started_utc': now()})
    export = json.loads((EXP/'setup/environment_export.json').read_text())
    assert export['phase'] == 'EXPORTED', export
    destination = PYTHON.parent.parent
    assert not destination.exists(), 'Do not overwrite an existing environment'
    destination.mkdir()
    result = subprocess.run(['tar', '-xf', str(EXP/'setup/competition_env.tar'), '-C', str(destination)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key)
    env.update(NO_PROXY='*', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               PATH=str(PYTHON.parent)+':'+env.get('PATH', ''),
               REPRO_PROJECT_ROOT=str(ROOT), REPRO_EXPERIMENT_DIR=str(EXP), REPRO_BASELINE_DIR=str(BASELINE))
    for key in ('WANDB_RUN_ID', 'WANDB_RESUME'):
        env.pop(key, None)
    check = "import json,torch,numpy,stable_baselines3,scipy;print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,'stable-baselines3':stable_baselines3.__version__,'scipy':scipy.__version__,'torch_file':torch.__file__}))"
    result = subprocess.run([str(PYTHON), '-B', '-c', check], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    imported = json.loads(result.stdout.strip())
    assert imported['torch'] == '2.10.0+cu128', imported
    assert imported['stable-baselines3'] == '2.8.0', imported
    assert imported['numpy'] == '2.4.4', imported
    write('setup/environment_import.json', {'phase': 'IMPORTED', 'finished_utc': now(), 'environment': str(destination), 'imports': imported})
    manifest = json.loads((EXP/'reference_manifest.json').read_text())
    actual_hashes = {}
    for name, expected in {**manifest['source_hashes'], **manifest['target_hashes']}.items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        assert actual == expected, 'Historical source mismatch: '+name
        actual_hashes[name] = actual
    write('source_parity.json', {'historical_commit': manifest['initial_git_sha'], 'all_expected_hashes_match': True, 'hashes': actual_hashes})
    runtime_reference = json.loads((EXP/'reference_runtime.json').read_text())
    runtime_hashes = {}
    for name, item in runtime_reference['files'].items():
        actual = hashlib.sha256((RUNTIME/name).read_bytes()).hexdigest()
        assert actual == item['sha256'], 'HFM runtime changed: '+name
        runtime_hashes[name] = actual
    write('runtime_parity.json', {'all_recorded_hashes_match': True, 'hashes': runtime_hashes})
    args = ['--experiment-id', EXP.name, '--total-timesteps', '10000000', '--num-envs', '32',
            '--base-port', '2223', '--seed', '0', '--wandb-mode', 'offline', '--dry-run']
    result = subprocess.run([str(PYTHON), '-B', 'tools/run_baseline_experiment.py', *args], cwd=ROOT,
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout[-1000:]+result.stderr[-2000:]
    candidate = json.loads((EXP/'requested_config.json').read_text())
    reference = json.loads((EXP/'reference_config.json').read_text())
    assert candidate == reference, 'Training hyperparameters differ from original baseline'
    write('config_parity.json', {'all_training_settings_identical': True, 'seed': 0, 'num_envs': 32,
                                'only_execution_differences': ['experiment/output path', 'initial-state observer', 'fresh HFM pool and isolated cache', 'independent Python environment prefix'],
                                'original_cpu_quota': '54 cores', 'current_cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                                'original_uninterrupted_end_step': 2211840})
    assert not set(range(2223, 2255)) & ports()[0], 'Instance HFM ports are occupied'
    hfm_output = EXP/'training/hfm'
    hfm_output.mkdir()
    cpus = sorted(os.sched_getaffinity(0))
    per = len(cpus)//32
    assert per >= 1
    servers = []
    children = []
    write('setup/launch_status.json', {'phase': 'STARTING_HFM', 'pid': os.getpid(), 'updated_utc': now()})
    for index, port in enumerate(range(2223, 2255)):
        cache = Path('/tmp/plasma-seed-repro-20261115')/str(port)
        assert not cache.exists(), 'Refusing stale HFM cache'
        cache.mkdir(parents=True)
        selected = cpus[index*per:] if index == 31 else cpus[index*per:(index+1)*per]
        server_env = env.copy()
        server_env.update(LISTEN_PORT=str(port), MATLAB_SOCKET_TIMEOUT='600', MATLAB_IDLE_TIMEOUT='600', MCR_CACHE_ROOT=str(cache))
        command = ['taskset', '-c', ','.join(map(str, selected)), str(RUNTIME/'run_hfm_server.sh')]
        with (hfm_output/f'hfm_{port}.log').open('w') as output:
            child = subprocess.Popen(command, cwd=RUNTIME, env=server_env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        servers.append({'port': port, 'pid': child.pid, 'cpus': selected, 'cache': str(cache)})
        write('hfm_launch.json', {'hostname': os.uname().nodename, 'started_utc': now(), 'servers': servers})
        time.sleep(2)
    deadline = time.monotonic()+600
    while not set(range(2223, 2255)) <= ports()[0]:
        assert all(child.poll() is None for child in children), 'A new HFM service exited; inspect its log'
        assert time.monotonic() < deadline, 'Timed out waiting for the new HFM pool'
        time.sleep(5)
    assert not set(range(2223, 2255)) & ports()[1], 'An unexpected client has taken the HFM pool'
    with (EXP/'launcher.log').open('w') as output:
        runner = subprocess.Popen([str(PYTHON), '-B', str(EXP/'scripts/launch_recorded.py')], cwd=ROOT,
                                  env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    with (EXP/'comparison.log').open('w') as output:
        observer = subprocess.Popen([str(PYTHON), '-B', str(EXP/'scripts/compare_training.py')], cwd=ROOT,
                                    env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    write('launch.json', {'hostname': os.uname().nodename, 'started_utc': now(), 'runner_pid': runner.pid,
                         'comparison_pid': observer.pid, 'project_root': str(ROOT), 'experiment_directory': str(EXP),
                         'transport': 'Jupyter; no SSH or network proxy', 'python': str(PYTHON)})
    write('setup/launch_status.json', {'phase': 'LAUNCHED', 'runner_pid': runner.pid, 'updated_utc': now()})
    # The copied runtime is now independently installed; remove the disposable transfer archive only.
    archive = EXP/'setup/competition_env.tar'
    assert archive.parent == EXP/'setup' and archive.name == 'competition_env.tar'
    archive.unlink()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write('setup/launch_status.json', {'phase': 'FAILED', 'pid': os.getpid(), 'error': type(exc).__name__+': '+str(exc), 'updated_utc': now()})
        raise
