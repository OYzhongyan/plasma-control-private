---
title: F2b Trajectory Scoring
emoji: 📊
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
license: apache-2.0
pinned: false
short_description: Score F2b trajectory and target JSON in your browser using Pyodide.
---

# F2b 轨迹 JSON 评分

使用免费的 **Hugging Face Static Space + Pyodide + NumPy**，在访问者自己的浏览器里运行 Python，对比仿真轨迹与目标并评分。不需要 Docker、Gradio、HF PRO、远程 CPU 服务器或 HF Token。参考所提供的 `hf-static-pyodide-json.zip`，将数值演示替换成项目 F2b 评分。

旧版边界系数 JSONL、预填 token 和远程 `/evaluate` 已移除。此目录既是独立静态应用，也是命令行评分工具。

## 命令行

在 `plasma-control` 项目根目录执行：

```bash
python -m pip install -r platformtohf-main/platformtohf-main/requirements.txt
python -m playwright install chromium
PYTHON_BIN=python bash platformtohf-main/platformtohf-main/submit.sh \
  results/submission_infer_result.json \
  evaluation/results/target.json \
  --tag team-a-run-01 \
  --output results/submission_score.json
```

Linux 缺少 Chromium 系统库时使用 `python -m playwright install --with-deps chromium`，可能需要系统安装权限。Windows 可直接使用 `python platformtohf-main/platformtohf-main/submit.py`，参数相同。

- 必填路径依次为**轨迹 JSON、目标 JSON**，相对当前工作目录解析，也支持绝对路径。
- 读取完整 JSON 文档，不是逐行 JSONL，支持多行缩进和 UTF-8 BOM。
- 默认临时开启 `127.0.0.1` 静态文件服务，在本机无界面 Chromium 的 Pyodide 中评分，结束后关闭。不需要显示器、HFM 或推理服务。
- `--output` 保存结果；`--tag` 或 `TAG` 标记本次计算；`--timeout` 默认 300 秒。
- stdout 为评分 JSON；失败时 `success: false`、`score: null`，退出码非零。
- `PYTHON_BIN` 指定 Bash 调用的 Python，默认 `python3`。

加载已经部署的公开 Static 页面：

```bash
PYTHON_BIN=python bash platformtohf-main/platformtohf-main/submit.sh \
  results/submission_infer_result.json evaluation/results/target.json \
  --space-url https://YOUR-STATIC-SPACE.hf.space
```

也可通过 `HF_SPACE_URL` 指定页面地址；必须部署本目录的新版页面，不是旧评测接口，不加 `/evaluate`。如果旧变量影响本地模式，先 `unset HF_SPACE_URL`。命令行将文件内容传给本机浏览器里的页面函数，不发送评分 POST 请求。只加载信任的页面代码；私有 Space 登录不在本工具范围内。

## 网页部署

1. 创建或打开自己的 HF Space，选择 **Static → Blank**。
2. 将 `README.md`、`index.html`、`app.js`、`worker.js`、`compute.py`、`scorer_bundle.py`、`LICENSE` 放在 Space 根目录；解压后上传文件，不上传 ZIP 本身。
3. 等待发布并打开 App。选择两个 JSON 文件，输入可选标签，点击“开始评分”。可取消计算、查看各指标贡献分并下载 `f2b_score.json`。

网页用户无需安装本地依赖；首次需联网下载 Pyodide 314.0.7 与 NumPy。使用 module Web Worker 避免阻塞界面。两个输入上限分别为 128 MiB，浏览器内存可能限制实际可处理大小。

本地预览（项目根目录）：

```bash
python -m http.server 8000 --bind 127.0.0.1 \
  --directory platformtohf-main/platformtohf-main
```

打开 `http://127.0.0.1:8000`，不要双击 HTML 通过 `file://` 访问。

## JSON 格式与评分

轨迹结构（数组内为逐步实际观测）：

```json
{"F2b": {"trajectory": {"Ip": [], "lX": [], "Icoil": [], "lcfs_per_step": []}, "timeout": false}}
```

上面仅演示结构，空轨迹不产生有效得分；真实文件由项目 rollout 输出。

| 字段 | 形状与含义 |
| --- | --- |
| `Ip`, `lX`, `nX` | 电流、位形、X 点数，`(T,)` |
| `Icoil` | 线圈电流，`(T,12)` |
| `lcfs_per_step` | 边界采样点，`(T,P,2)` |
| `Xpt_main`, `Xpt_sec` | 主/次 X 点坐标，`(T,2,2)` |
| `strike` | 打击点坐标，`(T,8,2)` |
| `psiX_main`, `psiX_sec` | X 点磁通，`(T,2)` |
| `psia`, `psib` | 轴/边界磁通，`(T,)` |
| `inference_time_ms` | 可选推理时间，影响原规则中的 gamma |

目标顶层为 `F2b`，兼容原评分器的 `F2a` 回退；目标块须有 `lcfs_points`、`Xpt_main`、`Xpt_sec`、`strike`。A1/A2/B1/B2/B3、边界系数 JSONL 和训练参考配置不能直接替代官方目标。数组应为有限数值；NaN/Infinity 或错误形状明确报错。

`scorer_bundle.py` 是 `evaluation/eval/score` 的未修改快照，按原规则以 500 步、30 分为基准。电流参考来自 `config.build_Ip_ref()`，目标 JSON 提供几何参考。保留提前终止、timeout、拓扑、电流熔断、线圈电流和推理时间惩罚。缺少非核心指标字段时沿用原规则，在 `warnings` 中提示；不会补造观测。

输出兼容原 `evaluate.py` 的 `success`、`errorMsg`、`score`、`score1`、`scoreJson`，另有各指标详细得分、有效步数、规则及输入哈希。哈希用于识别文件和版本，不证明轨迹真实或分数未被篡改。

## 更新规则

在完整项目中运行，之后重新上传快照：

```bash
python platformtohf-main/platformtohf-main/build_scorer.py
python platformtohf-main/platformtohf-main/build_scorer.py --check
```

独立 Space 无需训练包、Torch、SB3 或 MATLAB；运行评分仅需 `compute.py` 和生成的 `scorer_bundle.py`。发布包应保留 `LICENSE` 与原版权声明。

## 范围

文件不发送到原远程服务，结果仅存在当前页面或下载文件中。本版不自动维护跨设备共享排行榜，不能保密浏览器内的目标或防止用户修改评分；不把本地评分当作有身份验证的竞赛裁决。

官方资料：[Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)、[Pyodide Workers](https://pyodide.org/en/stable/usage/webworker.html)。
