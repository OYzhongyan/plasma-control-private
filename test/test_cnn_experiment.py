"""Opt-in CNN training and matched experiment settings."""
import importlib.util
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


def config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'train/configs')):
        return compose(config_name='base')


def runner():
    spec = importlib.util.spec_from_file_location('cnn_runner', ROOT / 'tools/run_baseline_experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_critic_remains_original_mlp():
    from train.train_ppo import critic_components
    from train.wrappers import ObservationWrapper
    from stable_baselines3.common.vec_env import VecNormalize
    cfg = config()
    assert cfg.critic.type == 'mlp'
    assert critic_components('mlp') == (ObservationWrapper, VecNormalize, None)
    with pytest.raises(ValueError, match='critic'):
        critic_components('unknown')


def test_flux_factory_selects_only_critic_specific_components():
    from train.train_ppo import critic_components
    from train.flux_observation import FluxObservationWrapper, FluxVecNormalize
    from train.flux_critic_policy import FluxCriticPolicy
    assert critic_components('flux_cnn') == (FluxObservationWrapper, FluxVecNormalize, FluxCriticPolicy)


def test_architecture_record_tracks_unchanged_actor_and_new_critic():
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from train.flux_critic_policy import FluxCriticPolicy
    from train.train_ppo import model_architecture_record
    actor_space = gym.spaces.Box(-np.inf, np.inf, (812,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
    spaces = gym.spaces.Dict({'actor_obs': actor_space,
        'flux_history': gym.spaces.Box(-np.inf,np.inf,(4,66,65),dtype=np.float32),
        'flux_mask': gym.spaces.Box(0,1,(4,),dtype=np.float32)})
    kwargs = dict(net_arch={'pi':[256,256],'vf':[256,256]},activation_fn=torch.nn.Tanh,log_std_init=-1.)
    torch.manual_seed(0)
    original=ActorCriticPolicy(actor_space,action_space,lambda _:1e-4,**kwargs)
    a=model_architecture_record(original,'mlp')
    torch.manual_seed(0)
    changed=FluxCriticPolicy(spaces,action_space,lambda _:1e-4,**kwargs)
    b=model_architecture_record(changed,'flux_cnn')
    assert a['initial_actor_sha256']==b['initial_actor_sha256']
    assert a['sampling_rng_sha256']==b['sampling_rng_sha256']
    assert b['actor_parameters']==275726
    assert b['critic_parameters']==989185


def test_cnn_requested_settings_equal_original_except_critic():
    module = runner()
    args = module.parser_and_args(['--experiment-id', 'cnn', '--critic-type', 'flux_cnn'])
    candidate, _ = module.config_record(ROOT, args)
    baseline = json.loads((ROOT / 'experiments/20261109_f2b_history_baseline_10m_seed0/remote_snapshot/requested_config.json').read_text())
    result = module.compare_baseline_config(baseline, candidate)
    assert result['non_critic_settings_identical'] is True
    assert candidate['critic']['type'] == 'flux_cnn'
    assert candidate['features'] == baseline['features']
    commands = module.commands_for(ROOT, ROOT / 'experiments/cnn', args)
    assert 'critic.type=flux_cnn' in commands['training']
    assert commands['evaluation']['test']['seed'] == 300


@pytest.mark.parametrize('key,value', [('gamma', .99), ('batch_size', 512), ('total_timesteps', 1000)])
def test_baseline_comparison_rejects_other_changes(key, value):
    module = runner()
    a = {'ppo': {'gamma': .995, 'batch_size': 1024, 'total_timesteps': 10000000}, 'critic': {'type': 'mlp'}}
    b = json.loads(json.dumps(a))
    b['ppo'][key] = value
    b['critic']['type'] = 'flux_cnn'
    with pytest.raises(ValueError, match=key):
        module.compare_baseline_config(a, b)
