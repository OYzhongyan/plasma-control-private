# 复赛评测脚本说明

本脚本用于智能控制挑战赛复赛阶段（赛题 F1 / F2a / F2b）的自动评分。详细评分规则见 `智能控制挑战赛——复赛评估标准v1.md`。

## 1. 文件结构

```
eval/
├── evaluate.py             顶层 CLI 入口（不可改名）
├── py_entrance.sh          固定入口脚本（不可修改）
├── requirements.txt
└── score/                  评分核心包
    ├── __init__.py
    ├── config.py           常量：任务、权重、阈值、Ip_ref 生成
    ├── metrics.py          各类指标计算（电流、LCFS、X 点、打击点、磁通偏差）
    ├── penalties.py        位形惩罚 η、电流熔断 μ、线圈约束 ρ、X 点拓扑 mask
    ├── tasks.py            每子任务的指标计算调度
    ├── scoring.py          单步得分合成、子任务总分
    └── evaluate_core.py    顶层评估入口（被 evaluate.py 调用）
```

## 2. 用法

```bash
python evaluate.py <target.json> <infer_result.json>
```

退出码恒为 0，结果按比赛规范打印到 stdout（JSON）。

## 3. 数据格式

### target.json

```json
{
  "F1":  { "Ip": [N=300], "lcfs_points": [[r, z], ...] },
  "F2a": {
    "Ip": [N=500], "lcfs_points": [[r, z], ...],
    "Xpt_main": [[r, z], [r, z]],
    "Xpt_sec":  [[r, z], [r, z]],
    "strike":   [[r, z] × 8]
  }
  // F2b 默认 fallback 到 F2a（位形目标完全一致）
}
```

> 注：脚本内部按文档 2.3 节公式生成 `Ip_ref`，`target.Ip` 字段当前仅作为信息冗余，不参与误差计算。

### infer_result.json

每个子任务下：

```json
"F1": {
  "trajectory": {
    "Ip":            [N],            // A 为单位
    "lcfs_per_step": [N × (N_b, 2)],
    "Xpt_main":      [N × (2, 2)],
    "Xpt_sec":       [N × (2, 2)],
    "psiX_main":     [N × 2],
    "psiX_sec":      [N × 2],
    "strike":        [N × (8, 2)],
    "nX":            [N],            // X 点个数（整数）
    "lX":            [N],            // 1=偏滤器位形, 0=限制器位形
    "Icoil":         [N × 12],       // CS, PF1..PF10, VS
    "psia":          [N],            // 磁轴磁通
    "psib":          [N]             // 边界磁通
  },
  "timeout": false
}
```

提前终止以数组长度 < N 体现，超出部分按 0 分计入分母为 N 的平均。

## 4. 评分规则速查

每个子任务：

```
S_task = γ · (1/N) · Σ_k  η(k)·μ(k)·ρ(k) · Σ_i σ_i(k)
σ_i(k) = W_i · max(0, 1 − ε_i(k) / ε_max_i)
```

- `η(k)`：位形类型惩罚。F1 要求限制器 (`lX==0`)，F2a/F2b 要求偏滤器 (`lX==1`)；否则 `η=0.5`。
- `μ(k)`：电流熔断。`|Ip-Ip_ref| > 50 kA` 时 `μ=0`，该步全部清零。
- `ρ(k)`：线圈约束。CS>45 kA / PF1-10>14 kA / VS>4 kA 任一发生，**该步及之后全部清零**。
- 拓扑屏蔽：F2a/F2b 中 `nX != 4` 时，**XPT 专属指标** (X / strike / psiX / X2 / psiX2) 当步 σ 清零；电流与 LCFS 不受影响。
- `γ`：推理时间合规系数（本期固定为 1）。

## 5. 输出 JSON 字段

```json
{
  "score":      <总分>,
  "score1":     <F1 子任务得分>,
  "score2":     <F2a 子任务得分>,
  "score3":     <F2b 子任务得分>,
  "scoreJson":  {
    "score":   <总分>,
    "score1":  ..., "score2": ..., "score3": ...,
    "F1_Ip":  ..., "F1_LCFS": ...,
    "F2a_Ip": ..., "F2a_LCFS": ..., "F2a_X": ..., "F2a_strike": ...,
    "F2a_psiX": ..., "F2a_X2": ..., "F2a_psiX2": ...,
    "F2b_Ip": ..., "F2b_LCFS": ..., "F2b_X": ..., "F2b_strike": ...,
    "F2b_psiX": ..., "F2b_X2": ..., "F2b_psiX2": ...
  },
  "errorMsg":   "",
  "success":    true
}
```

异常情况：`success=false`，`scoreJson={}`，`errorMsg` 描述原因。

## 6. 测试

`evaluation/tests/` 提供：

- `gen_mock_data.py`：生成 8 个典型场景的 target / infer_result mock。
- `test_evaluate.py`：直接调用评分函数验证期望分数。

```bash
cd evaluation/tests
python test_evaluate.py
```

预期所有 8 个场景全通过。
