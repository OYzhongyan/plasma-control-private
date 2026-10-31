#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# submit.sh — 读本地 JSONL, 拼成 inline payload, POST 到远程 HF Space /evaluate
#
# 用法:
#     ./submit.sh                                  # 用脚本同目录的 submission.jsonl
#     ./submit.sh /path/to/other.jsonl             # 指定别的 jsonl
#     TAG=my-run ./submit.sh                       # 覆盖 payload.tag (否则用 jsonl 里第一条带 tag 的 row)
#     HF_TOKEN=hf_xxx ./submit.sh                  # Space 是私有的才需要 token
#     HF_SPACE_URL=https://xxx.hf.space ./submit.sh  # 覆盖默认 Space URL
#
# 依赖: curl + python3 (用来把 jsonl 拼成 JSON payload, 省掉 jq 的浮点精度坑)
# -----------------------------------------------------------------------------
set -euo pipefail

# ── 配置 ────────────────────────────────────────────────────────────────────
# Space: SII-LWY/sii-ste-eva
# HF 的 Space 服务 URL 规则:  https://<username-lowercase>-<spacename>.hf.space
HF_SPACE_URL="${HF_SPACE_URL:-https://sii-lwy-sii-ste-eva.hf.space}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
JSONL_PATH="${1:-$SCRIPT_DIR/submission.jsonl}"

# 可选: Space 若为 private, 从环境变量读 token;public 留空即可
HF_TOKEN="${HF_TOKEN:-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}"

# 可选: 提交的唯一标识 tag。优先级:
#   环境变量 TAG  >  jsonl 里第一条带 tag 的 row  >  (服务端兜底 None)
TAG="${TAG:-}"

# ── 预检 ────────────────────────────────────────────────────────────────────
if [[ ! -f "$JSONL_PATH" ]]; then
  echo "ERROR: jsonl file not found: $JSONL_PATH" >&2
  exit 1
fi
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 is required (to stringify rows into JSON)" >&2
  exit 1
fi

# ── 读 jsonl → 拼 payload ──────────────────────────────────────────────────
# 把每一行解析成 dict, 打包成 {"points": [...], "tag": ...} 走 /evaluate 的 inline 分支。
# 顶层 tag:TAG 环境变量优先;否则扫 rows, 取第一条遇到的 row["tag"]。
# 用 python 而不是 jq, 避免浮点数被 jq 转成字符串或截精度。
PAYLOAD=$(TAG_ENV="$TAG" python3 - <<PY
import json, os
rows = []
with open("$JSONL_PATH") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {i} 不是合法 JSON: {exc}")

tag = os.environ.get("TAG_ENV") or None
if not tag:
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("tag"), str):
            tag = r["tag"]
            break

payload = {"points": rows}
if tag:
    payload["tag"] = tag
print(json.dumps(payload))
PY
)

ROW_COUNT=$(printf '%s' "$PAYLOAD" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["points"]))')
PAYLOAD_TAG=$(printf '%s' "$PAYLOAD" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tag") or "")')
echo "[submit] $JSONL_PATH  →  $HF_SPACE_URL/evaluate   (rows=$ROW_COUNT tag='${PAYLOAD_TAG}')"

# ── 打请求 ──────────────────────────────────────────────────────────────────
# -N: 关闭输出缓冲, 让服务端的渐进式响应能实时流回来(评测慢, 有大段静默期)
# --fail-with-body: 非 2xx 时仍打印返回体, 方便看服务端返回的错误 JSON
# -X POST + -d @- : payload 走 stdin, 避免命令行长度上限
AUTH_ARGS=()
if [[ -n "$HF_TOKEN" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer $HF_TOKEN")
fi

printf '%s' "$PAYLOAD" | curl -sS -N --fail-with-body -X POST \
  "$HF_SPACE_URL/evaluate" \
  "${AUTH_ARGS[@]}" \
  -H 'Content-Type: application/json' \
  --data-binary @-

# 末尾补个换行, 让终端显示干净些
echo
