# 实验记录约定

这里只保留正式训练、诊断及其验证记录。每次正式实验使用独立子文件夹，保存可复现配置、代码版本、曲线、结果表及 `picture/` 图片。尚未运行的方法分析须标记为 `ANALYSIS_ONLY`，不能作为实验结果。

Smoke 测试使用项目外的临时目录；无论测试成功还是失败，结束后都删除模型、日志、曲线、图片及临时归档。不在本地或启智的 `experiments/` 下保留 smoke 实验，也不将其提交到 Git。

后续实验优先保持第 500 步 bootstrap：第 500 步视为人工截断，使用 reset 前末状态的 value；真正的物理失败或任务终止才将 continuation value 置零。验证 critic 的 MC 参考回报必须包含第 500 步之后的 continuation，不能把 500 步有限时域回报直接作为同一目标下的真值。

当前启智项目目录：`/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control`。

历史运行快照中的路径是采集时的原始值，不做追溯替换。新操作使用上述当前目录。
