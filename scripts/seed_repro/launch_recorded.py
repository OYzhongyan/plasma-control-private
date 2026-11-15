"""Use the original experiment runner with a read-only initial-state recorder."""
from pathlib import Path
import json
import os
import sys


def main():
    root = Path(os.environ['REPRO_PROJECT_ROOT'])
    exp = Path(os.environ['REPRO_EXPERIMENT_DIR'])
    sys.path.insert(0, str(root))
    from tools.run_baseline_experiment import main as run, execute_child

    def execute(command, **kwargs):
        actual = list(command)
        if command[1] == 'train/train_ppo.py':
            actual[1] = str(exp/'scripts/capture_initial_train.py')
            (exp/'actual_training_command.json').write_text(json.dumps({
                'original_command': command, 'executed_command': actual,
                'change': 'Initial state recording only; historical trainer source unchanged',
            }, indent=2)+'\n')
        return execute_child(actual, **kwargs)

    return run(['--experiment-id', exp.name, '--total-timesteps', '10000000',
                '--num-envs', '32', '--base-port', '2223', '--seed', '0',
                '--wandb-mode', 'offline'], root=root, executor=execute)


if __name__ == '__main__':
    raise SystemExit(main())
