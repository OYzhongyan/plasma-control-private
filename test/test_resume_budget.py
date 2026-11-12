import pytest

from train.utils import remaining_learning_timesteps, warmup_cosine_schedule


def test_absolute_target_subtracts_saved_progress():
    assert remaining_learning_timesteps(10_000_000, 2_211_840,
        target_total_timesteps=10_000_000, reset_num_timesteps=False) == 7_788_160
    assert remaining_learning_timesteps(10_000_000, 901_120,
        target_total_timesteps=10_000_000, reset_num_timesteps=False) == 9_098_880


def test_existing_additional_budget_semantics_remain_default():
    assert remaining_learning_timesteps(1024, 8192, reset_num_timesteps=False) == 1024
    assert remaining_learning_timesteps(1024, 8192, reset_num_timesteps=True) == 1024


@pytest.mark.parametrize('target,start,reset', [(10,5,True),(0,0,False),(10,10,False),(10,11,False)])
def test_invalid_absolute_resume_budget_is_rejected(target,start,reset):
    with pytest.raises(ValueError):
        remaining_learning_timesteps(10,start,target_total_timesteps=target,reset_num_timesteps=reset)


def test_real_ppo_save_load_retains_global_step_and_schedule(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO
    env = gym.make('CartPole-v1')
    schedule = warmup_cosine_schedule(1e-4, 8, 64, warmup_updates=1)
    model = PPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=1,
                learning_rate=schedule,seed=0,policy_kwargs={'net_arch':[8]})
    model.learn(16)
    model.save(tmp_path/'checkpoint')
    restored = PPO.load(tmp_path/'checkpoint',env=env)
    budget = remaining_learning_timesteps(64,restored.num_timesteps,
        target_total_timesteps=64,reset_num_timesteps=False)
    assert restored.lr_schedule(.5) == pytest.approx(schedule(.5))
    restored.learn(budget,reset_num_timesteps=False)
    assert restored.num_timesteps == 64
    assert restored._total_timesteps == 64
    assert restored.lr_schedule(0) == pytest.approx(0)
    env.close()


def test_invalid_target_is_rejected_before_creating_environment():
    from omegaconf import OmegaConf
    from train.train_ppo import run_training
    cfg = OmegaConf.create({'ppo':{'total_timesteps':64},
        'resume':{'enabled':True,'target_total_timesteps':0,'reset_num_timesteps':False}})
    with pytest.raises(ValueError,match='must be positive'):
        run_training(cfg)
