"""Baseline history, dense reward, and local logging contracts."""
import csv
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest

from test_submission_history import Replay, observation


def test_history_reset_layout_and_physical_action_slots():
    from train.wrappers import ObservationWrapper, NormalizedAction7DTo12DWrapper
    env = Replay()
    actions = NormalizedAction7DTo12DWrapper(env, action_mode='delta')
    wrapper = ObservationWrapper(actions, 4, include_xpt=True, action_mode='delta')
    vector, _ = wrapper.reset()
    assert vector.shape == (812,)
    np.testing.assert_array_equal(vector[:3*188], 0)
    np.testing.assert_array_equal(vector[4*188:], 0)
    assert vector[3*188] == observation()['Ip'][0]
    first = np.linspace(-.7,.7,7,dtype=np.float32)
    vector, *_ = wrapper.step(first)
    expected = actions.current_abs_action[[0,1,1,2,2,3,3,4,4,5,5,6]].astype(np.float32)
    np.testing.assert_array_equal(vector[-60:].reshape(5,12), np.tile(expected,(5,1)))
    wrapper.step(-first)
    again, _ = wrapper.reset()
    np.testing.assert_array_equal(again[-60:],0)


@pytest.mark.parametrize('mode', ['absolute','delta'])
@pytest.mark.parametrize('history_steps',[1,4])
def test_history_adapter_matches_actual_training_sequence(mode, history_steps):
    from train.wrappers import ObservationWrapper, NormalizedAction7DTo12DWrapper
    from submission.history import observation_metadata, ObservationPolicyAdapter, validate_observation_metadata
    metadata = observation_metadata(mode, history_steps=history_steps)
    assert metadata['obs_dim'] == 188*history_steps + (60 if mode=='delta' else 35)
    validate_observation_metadata(metadata)
    teacher = ObservationWrapper(NormalizedAction7DTo12DWrapper(Replay(),action_mode=mode),history_steps,include_xpt=True,action_mode=mode)
    adapter = ObservationPolicyAdapter(metadata)
    for _ in range(2):
        expected, _ = teacher.reset()
        adapter.reset()
        for i in range(5):
            np.testing.assert_array_equal(adapter.encode(observation(i)),expected)
            action=np.linspace(-1.3,1.3,7,dtype=np.float32)*(-1 if i%2 else 1)
            voltage=adapter.decode(action)
            expected,_,_,_,info=teacher.step(np.clip(action,-1,1))
            np.testing.assert_array_equal(voltage,info['physical_action'])


def test_history_metadata_rejects_wrong_dimension():
    from submission.history import observation_metadata, validate_observation_metadata
    metadata = observation_metadata('delta', history_steps=4)
    metadata['obs_dim']=90
    with pytest.raises(ValueError,match='obs_dim'):
        validate_observation_metadata(metadata)


def test_training_restores_upstream_reward_and_history_defaults():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from train.train_ppo import build_reward
    from train.dense_rewards import SemifinalDenseReward
    with initialize_config_dir(version_base=None,config_dir=str(Path('train/configs').resolve())):
        cfg=compose(config_name='base')
    assert cfg.features.history_steps==4
    assert cfg.actions.mode=='delta'
    assert cfg.features.delay_step==1
    assert cfg.reward_weights.score_lower_bound is False
    assert cfg.reward_weights.use_competition_score is False
    assert isinstance(build_reward(dict(cfg.reward_weights)), SemifinalDenseReward)


def test_delay_noise_keeps_scoring_truth_and_repeats_initial_frame():
    from train.wrappers import ObservationDelayNoiseWrapper
    wrapper=ObservationDelayNoiseWrapper(Replay(),delay_steps=1,noise_std={},seed=0)
    first,_=wrapper.reset()
    delayed,*_=wrapper.step(np.zeros(12))
    np.testing.assert_array_equal(delayed['Ip'],first['Ip'])
    delayed,*_=wrapper.step(np.zeros(12))
    np.testing.assert_array_equal(delayed['Ip'],observation(1)['Ip'])


def test_reward_component_csv_has_real_data(tmp_path):
    from train.callbacks import RewardComponentsCallback
    callback=RewardComponentsCallback(tmp_path,log_interval=1)
    callback.model=SimpleNamespace(logger=SimpleNamespace(record=lambda *args:None))
    callback._on_training_start()
    callback.num_timesteps=128
    callback._buffer=[{'reward':.75,'ip_score':1.25}]
    callback._info_count=1
    callback._flush()
    with callback.csv_path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==1
    assert float(rows[0]['reward'])==.75
    assert float(rows[0]['timesteps'])==128
    assert rows[0]['lcfs_score']==''


def test_local_logger_records_ppo_without_wandb(tmp_path):
    from stable_baselines3 import PPO
    from train.train_ppo import configure_training_logger
    class Flat(gym.Env):
        observation_space=gym.spaces.Box(-1.,1.,(3,),np.float32)
        action_space=gym.spaces.Box(-1.,1.,(1,),np.float32)
        def reset(self,**kwargs): return np.zeros(3,np.float32),{}
        def step(self,action): return np.zeros(3,np.float32),1.,False,False,{}
    model=PPO('MlpPolicy',Flat(),n_steps=4,batch_size=4,n_epochs=1,policy_kwargs={'net_arch':[4]},device='cpu')
    model.set_logger(configure_training_logger(tmp_path))
    model.learn(total_timesteps=8)
    assert (tmp_path/'progress.csv').is_file()
    with (tmp_path/'progress.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)>=2 and float(rows[-1]['time/total_timesteps'])>=8
    assert list(tmp_path.glob('events.out.tfevents.*'))


def test_checkpoint_frequency_counts_completed_rollouts(tmp_path):
    from train.callbacks import PlasmaCheckpointCallback
    callback=PlasmaCheckpointCallback(save_freq=2,save_path=tmp_path,name_prefix='policy',save_vecnormalize=False)
    calls=[]
    callback._save_checkpoint=lambda: calls.append(callback.num_timesteps)
    callback.n_calls=256;callback.num_timesteps=8192;callback._on_rollout_end()
    assert calls==[]
    callback.n_calls=512;callback.num_timesteps=16384;callback._on_rollout_end()
    assert calls==[16384]

def test_evaluation_restores_training_contract_but_preserves_eval_overrides(tmp_path):
    import pickle
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from train.evaluate import restore_training_configuration
    with initialize_config_dir(version_base=None,config_dir=str(Path('train/configs').resolve())):
        cfg=compose(config_name='eval')
    training={'features':{'observation_mode':'historical_observation_v1','history_steps':4},'actions':{'mode':'delta'},'reward_weights':{'ip_score':6.,'score_lower_bound':False},'target':{'default_reference_path':'saved-target.json'},'max_steps':500}
    with (tmp_path/'configs.pkl').open('wb') as stream: pickle.dump(training,stream)
    restored=restore_training_configuration(cfg,tmp_path/'ppo_f2_latest.zip',overrides=['ports=3333','eval.seed=123'])
    assert restored.target.default_reference_path=='saved-target.json'
    assert restored.reward_weights.ip_score==6.
    assert restored.features.history_steps==4
    assert restored.ports==3333
    assert restored.eval.seed==123
    assert restored.reset_perturbation.enabled is False


def test_evaluation_accepts_real_history_model_dimensions(tmp_path):
    import json
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from train.evaluate import configure_model_interface
    from submission.history import observation_metadata
    with initialize_config_dir(version_base=None,config_dir=str(Path('train/configs').resolve())):
        cfg=compose(config_name='eval')
    (tmp_path/'f2_features.metadata.json').write_text(json.dumps(observation_metadata('delta')))
    model=SimpleNamespace(observation_space=gym.spaces.Box(-np.inf,np.inf,(812,),np.float32),action_space=gym.spaces.Box(-1.,1.,(7,),np.float32))
    configure_model_interface(cfg,model,tmp_path/'ppo_f2_latest.zip')
    assert cfg.features.history_steps==4 and cfg.actions.mode=='delta'

def test_evaluation_records_timeout_and_true_truncation():
    from train.evaluate import record_row, summarize_rows
    row=record_row(step=1,shot_id='15892_400',episode=0,port=2223,
        observation=observation(),reward=-5.,done=True,
        info={'hfm_timeout':True,'TimeLimit.truncated':True},reward_components={})
    assert row['hfm_timeout']==1
    assert row['truncated']==1 and row['terminated']==0
    assert summarize_rows([row],max_steps=500)['hfm_timeout'] is True

def test_history_checkpoint_and_onnx_use_frozen_matching_stats(tmp_path):
    import json
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from submission.checkpoint_policy import CheckpointPolicy
    from submission.history import observation_metadata
    from train.wrappers import ObservationWrapper, NormalizedAction7DTo12DWrapper
    metadata=observation_metadata('delta')
    class Flat(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(812,),np.float32)
        action_space=gym.spaces.Box(-1.,1.,(7,),np.float32)
    norm=VecNormalize(DummyVecEnv([Flat]))
    norm.obs_rms.mean=np.linspace(-.5,.5,812)
    norm.obs_rms.var=np.linspace(.5,2.,812)
    norm.training=False
    model=PPO('MlpPolicy',norm,n_steps=2,batch_size=2,n_epochs=1,policy_kwargs={'net_arch':[8]},seed=11,device='cpu')
    model.save(tmp_path/'model.zip');norm.save(str(tmp_path/'vecnormalize.pkl'))
    (tmp_path/'f2_features.metadata.json').write_text(json.dumps(metadata))
    checkpoint=CheckpointPolicy(tmp_path/'model.zip')
    assert checkpoint.ready,checkpoint.load_error
    teacher=ObservationWrapper(NormalizedAction7DTo12DWrapper(Replay(),action_mode='delta'),4,include_xpt=True,action_mode='delta')
    vector,_=teacher.reset()
    before=checkpoint.normalization.obs_rms.mean.copy()
    for index in range(5):
        action,_=model.predict(norm.normalize_obs(vector),deterministic=True)
        voltage=checkpoint.act(observation(index))
        vector,_,_,_,info=teacher.step(action)
        np.testing.assert_allclose(voltage,info['physical_action'],atol=1e-4,rtol=1e-5)
    np.testing.assert_array_equal(checkpoint.normalization.obs_rms.mean,before)
    pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    from train.export_onnx import main
    from submission.inference2 import Policy
    main(['--model',str(tmp_path/'model.zip'),'--out',str(tmp_path/'policy2.onnx')])
    onnx=Policy(tmp_path,backend='onnx')
    assert onnx.ready,onnx.load_error
    checkpoint.reset();onnx.reset()
    for index in range(5):
        np.testing.assert_allclose(onnx.act(observation(index)),checkpoint.act(observation(index)),atol=1e-3,rtol=1e-4)

def test_predictor_power_supply_delay_has_explicit_reproducible_seed(monkeypatch):
    from environment.hfm_predictor import HFMSocketPredictor
    from environment.docker_socket_predictor import DockerSocketPredictor
    monkeypatch.setattr(DockerSocketPredictor,'__init__',lambda *args,**kwargs:None)
    first=HFMSocketPredictor(config={'power_supply_seed':30007})
    second=HFMSocketPredictor(config={'power_supply_seed':30007})
    for step in range(10):
        command=np.linspace(-100,100,12)*(step+1)
        np.testing.assert_array_equal(first._power_supply.step(command),second._power_supply.step(command))
        np.testing.assert_array_equal(first._power_supply.delay_s,second._power_supply.delay_s)
