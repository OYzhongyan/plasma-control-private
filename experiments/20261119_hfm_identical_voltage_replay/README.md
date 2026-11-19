# HFM 相同输入回放

比较 shot `15892_500` 在独立 HFM 进程中的输出。每个进程按同一顺序执行两个回合，每回合从 RESET 起记录，直到 HFM 返回失败或满 500 步。

| 输入 | RESET | 每步发送到 STEP 的 12 维电压 |
|---|---|---|
| zero | shot 默认参数 | 全部为 0 V |
| varying | 从 NumPy seed 15 的 ±5% 均匀扰动中取第四组三元组，对应训练中 env15 的第四次 reset 采样 | 第 j 路电压为 `((step*(j+7)+31*j)%161)-80` V，j 从 0 开始，step 从 1 开始 |

进程 A、C、D 位于启智 plasma-control，B 位于 plasma-control2，E 位于原复现实验实例 plasma-control3。A、B、D、E 各使用本实例的逻辑核 60–63，C 使用 64–67。D 额外先执行一次 varying 回合，然后在同一进程执行两个测量回合。所有进程直接使用 HFM socket 协议，分别创建缓存和监听端口。此前训练没有保存实际送入 HFM 的电压，varying 使用新构造的序列。

在 Linux 机器上使用含 NumPy 的 Python，所有实例共享同一输出目录。将 `DIR` 设为本实验脚本目录，`HFM` 设为 sealed runtime 的绝对路径，`OUT` 设为新建的共享结果目录。按上述实例分配执行相应命令。

```bash
python "$DIR/replay.py" "$HFM" "$OUT/A" 60-63
python "$DIR/replay.py" "$HFM" "$OUT/B" 60-63
python "$DIR/replay.py" "$HFM" "$OUT/C" 64-67
python "$DIR/replay.py" "$HFM" "$OUT/D" 60-63 --warmup
python "$DIR/replay.py" "$HFM" "$OUT/E" 60-63
python "$DIR/compare.py" "$OUT" > "$OUT/comparison.json"
```

各进程保存压缩 JSONL、HFM stdout 和输入字节的 SHA-256。比较时逐帧核对实际请求、全部返回字段以及求解器输出表的数值行。数值使用解析后的值精确比较，相同位置的 NaN 视为一致。结果文件中的第 0 帧是 reset。启智原始记录保存在共享项目的 `experiments/20261119_hfm_identical_voltage_replay/{A,B,C,D,E}/`。

## 结果

五个进程在全零电压下都于第 123 步终止，在变化电压下都于第 76 步终止。计入 reset 后，各进程分别返回 124 帧和 77 帧，各帧的全部返回字段在数值上精确一致；对齐 STEP 后的求解器日志行也全部一致（以打印精度计）。同机预热一个完整回合，以及在原 seed=0 复现实例上运行，都得到相同结果。完整数值见 [comparison.json](comparison.json)。

这项测试覆盖两条开环电压序列。历史第 102 步分叉的闭环轨迹没有逐步电压记录，因此这里没有验证那一次具体输入。
