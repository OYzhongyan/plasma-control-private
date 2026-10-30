#!/usr/bin/env bash
# 推理评测启动脚本（赛事方固定，选手请勿修改）
# 顺序：启动仿真服务 → 启动 submission 服务 → 运行评测脚本，结果写入 /saisresult
set -euo pipefail

# --- 1. 环境变量：工作目录、端口、结果与日志路径（由评测机挂载决定，请勿修改）---
SAISDATA_ROOT="${SAISDATA_ROOT:-/saisdata/40}"
MCR_ROOT="${MCR_ROOT:-/opt/matlabruntime/R2025a}"
SUBMISSION_HOST="${SUBMISSION_HOST:-0.0.0.0}"
SUBMISSION_PORT="${SUBMISSION_PORT:-8000}"
SUBMISSION_PORT2="${SUBMISSION_PORT2:-8001}"
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:${SUBMISSION_PORT}}"
SERVICE_URL2="${SERVICE_URL2:-http://127.0.0.1:${SUBMISSION_PORT2}}"
INFER_RESULT_PATH="${INFER_RESULT_PATH:-/saisresult/infer_result.json}"
ENTRYPOINT_LOG="${ENTRYPOINT_LOG:-/saisresult/entrypoint.log}"
HFM_LOG="${HFM_LOG:-/saisresult/hfm_socket_server.log}"
SUBMISSION_LOG="${SUBMISSION_LOG:-/saisresult/submission_service.log}"
RUN_TEST_LOG="${RUN_TEST_LOG:-/saisresult/run_test.log}"

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

export AGREE_TO_MATLAB_RUNTIME_LICENSE="${AGREE_TO_MATLAB_RUNTIME_LICENSE:-yes}"
export SAISDATA_SHOTS_CONFIG="${SAISDATA_SHOTS_CONFIG:-${SAISDATA_ROOT}/inference/shots.yaml}"
export SUBMISSION_HOST
export SUBMISSION_PORT
export SUBMISSION_PORT2

mkdir -p /saisresult "$(dirname "${INFER_RESULT_PATH}")" "$(dirname "${HFM_LOG}")" "$(dirname "${SUBMISSION_LOG}")" "$(dirname "${RUN_TEST_LOG}")"
touch "${ENTRYPOINT_LOG}" "${HFM_LOG}" "${SUBMISSION_LOG}" "${RUN_TEST_LOG}"
exec > >(tee -a "${ENTRYPOINT_LOG}") 2>&1

# --- 2. 校验挂载：run_hfm_socket_server.sh、run_test.py 为评测机提供，路径不可改 ---
if [[ ! -d "${SAISDATA_ROOT}" ]]; then
  echo "Missing mounted saisdata directory: ${SAISDATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SAISDATA_ROOT}/env/run_hfm_socket_server.sh" ]]; then
  echo "Missing HFM startup script under ${SAISDATA_ROOT}/env" >&2
  exit 1
fi
if [[ ! -f "${SAISDATA_ROOT}/inference/run_test.py" ]]; then
  echo "Missing inference runner under ${SAISDATA_ROOT}/inference" >&2
  exit 1
fi

# --- 2.5 将 OSS 挂载数据一次性 copy 到容器本地，避免后续频繁读取导致 OSS 连接中断 ---
LOCAL_SAISDATA="/saisdata_local/40"
echo "[start_infer] copying ${SAISDATA_ROOT} -> ${LOCAL_SAISDATA} ..." >&2
mkdir -p "${LOCAL_SAISDATA}"
cp -a "${SAISDATA_ROOT}/." "${LOCAL_SAISDATA}/"
echo "[start_infer] copy done, switching SAISDATA_ROOT to local." >&2
SAISDATA_ROOT="${LOCAL_SAISDATA}"
export SAISDATA_ROOT
export SAISDATA_SHOTS_CONFIG="${SAISDATA_ROOT}/inference/shots.yaml"
export PYTHONPATH="/app:${SAISDATA_ROOT}/standalone-env:${SAISDATA_ROOT}/standalone-env/environment"

# --- 3. 退出时清理子进程 ---
cleanup() {
  if [[ -n "${SUBMISSION_PID:-}" ]]; then
    kill "${SUBMISSION_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SUBMISSION_PID2:-}" ]]; then
    kill "${SUBMISSION_PID2}" 2>/dev/null || true
  fi
  if [[ -n "${HFM_PID:-}" ]]; then
    kill "${HFM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- 4. 等待端口 / 健康检查 ---
wait_for_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  python3 - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
deadline = time.time() + timeout_s
last_error = None

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            sys.exit(0)
    except OSError as exc:
        last_error = exc
        time.sleep(1.0)

raise SystemExit(f"Timed out waiting for {host}:{port}: {last_error}")
PY
}

wait_for_http_health() {
  local url="$1"
  local timeout_s="$2"
  python3 - "$url" "$timeout_s" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
timeout_s = float(sys.argv[2])
deadline = time.time() + timeout_s
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True:
                sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1.0)

raise SystemExit(f"Timed out waiting for submission health: {last_error}")
PY
}

# --- 5. 启动仿真服务（评测机脚本，路径固定）---
echo "[start_infer] starting Environment service..." >&2
"${SAISDATA_ROOT}/env/run_hfm_socket_server.sh" "${MCR_ROOT}" >"${HFM_LOG}" 2>&1 &
HFM_PID=$!
wait_for_tcp 127.0.0.1 5558 120

# --- 6. 启动选手 submission 服务（F1 与 F2 分别使用两个策略服务）---
echo "[start_infer] starting submission service1 (policy1 for F1)..." >&2
python3 /app/submission/service1.py >"${SUBMISSION_LOG}" 2>&1 &
SUBMISSION_PID=$!
wait_for_http_health "${SERVICE_URL}" 30

echo "[start_infer] starting submission service2 (policy2 for F2a/F2b)..." >&2
SUBMISSION_LOG2="${SUBMISSION_LOG%.log}_2.log"
python3 /app/submission/service2.py >"${SUBMISSION_LOG2}" 2>&1 &
SUBMISSION_PID2=$!
wait_for_http_health "${SERVICE_URL2}" 30

# --- 7. 运行评测（评测机 run_test.py，输出路径固定为 /saisresult/infer_result.json）---
echo "[start_infer] running inference evaluation..." >&2
python3 "${SAISDATA_ROOT}/inference/run_test.py" \
  --no-build \
  --service-url "${SERVICE_URL}" \
  --service-url2 "${SERVICE_URL2}" \
  --output "${INFER_RESULT_PATH}" \
  "$@" 2>&1 | tee -a "${RUN_TEST_LOG}"
