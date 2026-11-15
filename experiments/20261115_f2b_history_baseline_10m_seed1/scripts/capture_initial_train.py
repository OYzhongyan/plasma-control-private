"""Observe initial SB3 state, then execute an unchanged historical trainer."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import random
import runpy
import sys


def main():
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    root = Path(os.environ['REPRO_PROJECT_ROOT'])
    exp = Path(os.environ['REPRO_EXPERIMENT_DIR'])
    original_learn = PPO.learn

    def rng_state():
        return (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()),
                pickle.dumps(random.getstate()))

    def capture_then_learn(model, *args, **kwargs):
        before = rng_state()
        actor = hashlib.sha256()
        all_parameters = hashlib.sha256()
        actor_count = 0
        for name, tensor in sorted(model.policy.named_parameters()):
            data = tensor.detach().cpu().contiguous().numpy().tobytes()
            all_parameters.update(name.encode())
            all_parameters.update(data)
            if name.startswith(('mlp_extractor.policy_net.', 'action_net.')) or name == 'log_std':
                actor.update(name.encode())
                actor.update(data)
                actor_count += tensor.numel()
        directory = exp/'training/initial'
        directory.mkdir(parents=True, exist_ok=False)
        model.save(str(directory/'model.zip'))
        normalizer = model.get_vec_normalize_env()
        if normalizer is not None:
            normalizer.save(str(directory/'vecnormalize.pkl'))
        after = rng_state()
        unchanged = torch.equal(before[0], after[0]) and before[1:] == after[1:]
        if not unchanged:
            raise RuntimeError('Initial-state recorder consumed training RNG')
        (directory/'rng_state.pkl').write_bytes(pickle.dumps(before))
        record = {
            'num_timesteps': model.num_timesteps,
            'actor_parameters': actor_count,
            'initial_actor_sha256': actor.hexdigest(),
            'all_parameter_sha256': all_parameters.hexdigest(),
            'sampling_rng_sha256': hashlib.sha256(before[0].numpy().tobytes()).hexdigest(),
            'capture_preserved_rng': unchanged,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        }
        (exp/'initial_state.json').write_text(json.dumps(record, indent=2)+'\n')
        return original_learn(model, *args, **kwargs)

    PPO.learn = capture_then_learn
    sys.path.insert(0, str(root))
    train = root/'train/train_ppo.py'
    sys.argv = [str(train), *sys.argv[1:]]
    runpy.run_path(str(train), run_name='__main__')


if __name__ == '__main__':
    main()
