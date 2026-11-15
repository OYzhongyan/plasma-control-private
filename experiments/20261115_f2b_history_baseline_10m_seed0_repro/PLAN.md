# 执行与验证记录

- [x] 检查 plasma-control3 空闲，训练端口 2223–2254 未被占用。
- [x] 独立 checkout 原始 baseline 提交，不更新或停止原有 baseline / CNN 训练。
- [x] 发现实例 PyTorch 2.6.0 与 baseline 不匹配；从 baseline 复制环境到新实例独立前缀，验证核心依赖版本一致。
- [x] 逐项核对原 manifest 的源码、目标文件与 HFM 初始化文件哈希。
- [x] 用原始 runner 生成配置，确认与原始 requested_config 的全部设置相等。
- [x] 启动本实例独立 HFM 池、原始训练入口及后台比较程序。
- [x] 保存初始模型、归一化、参数哈希及全局 RNG；确认记录动作不改变随机流。
- [x] 本地对比较器运行三个检查：排除耗时差异、检测微小数值差异、拒绝缺列时的完整一致结论。
- [ ] 核验首轮 PPO 日志与原 baseline 的公共步数。
- [ ] 对比原 baseline 中断前的完整日志前缀及首个公共 checkpoint。
- [ ] 达到 10m 后完成原有 seen / validation / test 三组正式评估。

“已启动”不等于“已完成复现验证”。实时进度以远端 CSV、comparison.json 和状态文件为准，提交到 Git 的 live_check.json 仅为带时间戳快照。
