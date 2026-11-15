# seed 1 对照实验执行记录

- [x] 确认 plasma-control4 空闲，32 个 HFM 端口均未占用。
- [x] 创建原始 baseline 提交的独立 checkout。
- [x] 将原 baseline Python 环境复制到本实例独立前缀，核验全部依赖。
- [x] 对齐源码、目标文件、HFM 初始化文件哈希。
- [x] 生成并核验配置仅 `ppo.seed` 不同，评估种子保持不变。
- [x] 启动 32 路 HFM 和 10m 正式训练；确认真实 STEP 推进。
- [x] 保存初始权重及 RNG 哈希，核验与 seed 0 的初始化不同。
- [ ] 生成训练曲线，比较不同训练种子。
- [ ] 完成 seen / validation / test 三组正式评估。

实际使用的外部编排副本保存在 `scripts/`，来自当前仓库 `scripts/seed_variation/prepare.py` 的确定性生成；核心训练文件不做修改。
