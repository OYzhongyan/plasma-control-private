# 策略训练思路
团队主要通过强化学习(RL)方式针对F1、F2a、F2b三个任务分别训练控制策略。为降低RL训练难度，我们采用了分阶段的训练方式，逐步增强控制策略对于不同类型微扰的泛化能力。具体步骤如下：
1. **单环境训练**: 针对默认初始化参数与给定的单一目标位形进行RL训练，指令如下
```
python training/train_ppo_x.py --train-config training/configs/f1_direct_wo_DR.json
```
2. **初始化参数泛化**: 继续在存在初始化参数微扰的情况下进行RL训练，指令如下
```
python training/train_ppo_x.py --train-config training/configs/f1_direct_reward_v2_origin_finetune_reset_params.json
```
3. **位形目标泛化**：继续在同时存在初始化参数微扰和位形目标微扰的情况下进行RL训练，指令如下：
```
python training/train_ppo_x.py --train-config training/configs/f1_direct_reward_v2_finetune_lcfs_targets_512_reset_params.json
```
4. **观测噪声鲁棒性**：为提升控制的噪声鲁棒性、并尽量减少模型训练时间，对观测噪声的鲁棒性提升不再采用RL训练，而是通过策略蒸馏学习一个神经网络降噪层，添加在观测输入与policy之间进行降噪。以步骤3学习到的policy作为teacher policy，在不同扰动状态但无噪声的环境中采集rollouts，然后在rollouts的每步观测上添加随机噪声，训练降噪层将加噪后的观测重建为无噪声的观测，并在降噪层和teacher policy组合为student policy进行最终部署。指令如下：
```
```