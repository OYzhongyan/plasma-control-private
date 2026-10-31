# platformtohf

---

**李文洋**

[**上海创智学院**](http://www.sii.edu.cn) | 博士研究生

[AI 驱动受控核聚变仿真、控制与设计实验室](https://github.com/SII-AI4Fusion)

* **电话**：[+86-156-2003-5216](tel:+86-156-2003-5216)
* **邮箱**：[lwydsg@mail.nankai.edu.cn](mailto:lwydsg@mail.nankai.edu.cn)

---

>把本地 JSONL 提交格式的评测样本 POST 到远程 HF Space (`SII-LWY/sii-ste-eva`) 的 `/evaluate`。

## 文件

```
platformtohf/
├── submit.sh          # 读 jsonl → 拼 inline payload → POST 到 /evaluate
├── submission.jsonl   # 样例提交(1 行, tag="example", NFP=3)
└── README.md
```

## 快速上手

```bash
cd platformtohf
./submit.sh
```

默认行为:读同目录的 `submission.jsonl` → 打到 `https://sii-lwy-sii-ste-eva.hf.space/evaluate`。

## submission.jsonl 格式


```jsonc
{"tag": "example",          // 必选,提交标识符,服务端会回显到响应
 "nfp": 3,
 "boundary": {"r_cos": [[...5x9...]], "z_sin": [[...5x9...]]}
```

- `tag`/`nfp` / `boundary.r_cos` / `boundary.z_sin`:**必填**


## submit.sh 的环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HF_SPACE_URL` | `https://sii-lwy-sii-ste-eva.hf.space` | Space 服务根 URL |
| `HF_TOKEN` | 脚本里预填了一个 | 我的私有 Space token，供访问提交评测 |
| `TAG` | 空 | 覆盖本次提交的 tag(优先级高于 jsonl 里 row 带的 tag) |

## 用法示例

```bash
./submit.sh                              # 默认 jsonl + jsonl 里的 tag='example'
./submit.sh /path/to/other.jsonl         # 指定别的 jsonl 文件
TAG=run-20260513 ./submit.sh             # 临时覆盖 tag
```

## 典型输出

终端会先打一行提交摘要,然后跟上服务端返回的 JSON:
```
[submit] /path/submission.jsonl  →  https://sii-lwy-sii-ste-eva.hf.space/evaluate   (rows=1 tag='example')
{
  "success": true,
  "tag": "example",
  "metrics_scores": {
    "得分": 47.266,
    "边界数量": 1.0,
    "最小可行性": 0.00894,
    "平均可行性": 0.00894,
    "通过": 1.0, "达到目标": 1.0, "优于基线": 1.0
  },
  "metrics_per_boundary": [{...12 个物理指标...}],
  "timings_seconds": {...},
  "message": "评测完成: score=47.2655, pass=True, num_boundaries=1, total=59.43s",
  "detail_url": null
}
```

## 关于 tag

- 优先级:**`TAG` 环境变量 > JSONL row 里的 `tag` > 不带**
- 服务端不强制,不带也能跑,响应里 `"tag": null`,但是强烈建议带上,以方便后期查找
- 并发提交时**务必带上**,否则 HF Space `logs/run` 里多个请求的日志会交织,很难对账
- 在日志里找某次提交:`grep "tag='example'" <log-stream>`

## 依赖

- `curl`(系统自带)
- `python3`(标准库就够,用来把 jsonl 拼成 JSON payload,避开 `jq` 对浮点的精度坑)

## 关于题目评测平台的一些设想

- 在平台上实现每次接收选手上传的jsonl文件，建立对应的文件夹，标注时间等相关信息，内部存放其提交的文件

- 使用sh文件提交到云端评测服务，返回的所有信息中提取得分作为排行，并将所有信息返回给选手，实时更新

- 由于云端评测能力限制，每支队伍每天最多提交三个样本

- 经过测试，评测服务可以同时评测多个样本，但是并行后单个样本的评估时间均会变长