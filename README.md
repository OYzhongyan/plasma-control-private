# Train
The best version trained so far for F1:
```
python train_v2/train_ppo.py \
    task=f1 \
    output_dir=output_dir \
    reset_perturbation.fix_reset_params=true \
    actions.mode=delta \
    num_envs=32
```
For F2a:
```
python train_v2/train_ppo.py \
    task=f2 \
    output_dir=f2a_fix_reset_lr_1e-4_obs_delay_1_delta_action \
    reset_perturbation.fix_reset_params=true \
    num_envs=32 \
    ~shot_groups.limiter_start \
    features.delay_step=1 \
    actions.mode=delta
```
python train_v2/train_ppo.py \
    task=f2 \
    output_dir=f2a_fix_reset_and_shot_lr_1e-4_obs_delay_0_delta_action \
    reset_perturbation.enabled=false \
    num_envs=32 \
    ~shot_groups.limiter_start \
    features.delay_step=0 \
    actions.mode=delta \
    ppo.total_timesteps=10000000 \
    'shot_groups={}'    

python train_v2/train_ppo.py \
    task=f2 \
    output_dir=f2a_fix_reset_lr_1e-4_obs_delay_0_delta_action_train_episode_1k \
    max_steps=1000 \
    reset_perturbation.fix_reset_params=true \
    num_envs=32 \
    ~shot_groups.limiter_start \
    features.delay_step=0 \
    actions.mode=delta

For F2b:
```
python train_v2/train_ppo.py \
    task=f2 \
    output_dir=f2b_fix_reset_lr_1e-4_obs_delay_1_delta_action \
    reset_perturbation.fix_reset_params=true \
    num_envs=32 \
    ~shot_groups.divertor_start \
    features.delay_step=1 \
    actions.mode=delta
```
Here ~ represents deleting the configuration term.

# Evaluation
```
python train_v2/evaluate.py \
    task=f1 \
    eval.model_path=`output_dir` \
    eval.cp_num=`checkpoint_index` \
    target.default_reference_path=configs/f1_reference_targets.json \
    actions.mode=delta

python train_v2/evaluate.py \
    task=f2 \
    eval.model_path=`output_dir` \
    eval.cp_num=`checkpoint_index` \
    target.default_reference_path=configs/xpt_reference_targets.json 'eval.shots="13705_500"' \
    eval.obs_perturb.enabled=true \
    actions.mode=delta
```

```
python train_v2/evaluate.py \
    task=f2 \
    eval.model_path=results/20261011_090245_f2_continue_20261007_153309_retry \
    target.default_reference_path=configs/xpt_reference_targets.json 'eval.shots="13705_500"' \
    eval.obs_perturb.enabled=true \
    actions.mode=delta
```

# Visualization
```
python rollout/visualize_rollout.py --run_name `output_dir`
```
