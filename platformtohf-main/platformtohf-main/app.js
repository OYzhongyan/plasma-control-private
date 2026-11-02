const MAX_JSON_BYTES = 128 * 1024 * 1024;
const $ = id => document.getElementById(id);
let worker = null;
let jobId = 0;
let pending = null;
let resultValue = null;

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").dataset.error = String(error);
}

function setBusy(busy) {
  for (const id of ["runButton", "inferFile", "targetFile", "tagInput"]) $(id).disabled = busy;
  $("cancelButton").disabled = !busy;
}

function clearResult() {
  resultValue = null;
  $("downloadButton").disabled = true;
  $("totalScore").textContent = "—";
  $("effectiveSteps").textContent = "等待评分";
  $("metrics").replaceChildren();
  $("output").textContent = "选择轨迹和目标文件后开始评分。";
}

function displayResult(result) {
  resultValue = result;
  $("totalScore").textContent = result.score.toFixed(6);
  $("effectiveSteps").textContent = `有效步数 ${result.K_eff} / 500 · 满分 ${result.max_score} · γ = ${result.gamma}`;
  $("output").textContent = JSON.stringify(result, null, 2);
  const rows = Object.entries(result.task_metric_scores.F2b).map(([name, score]) => {
    const row = document.createElement("tr");
    for (const text of [name, Number(score).toFixed(6)]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    return row;
  });
  $("metrics").replaceChildren(...rows);
  $("downloadButton").disabled = false;
}

function stopWorker(message) {
  worker?.terminate();
  worker = null;
  if (pending) {
    const task = pending;
    pending = null;
    clearTimeout(task.timer);
    task.reject(new Error(message));
  }
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("./worker.js", document.baseURI), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      if (pending) setStatus(data.message);
      return;
    }
    if (!pending || data.id !== pending.id) return;
    const task = pending;
    pending = null;
    clearTimeout(task.timer);
    if (data.type === "error") task.reject(new Error(data.message));
    else task.resolve(data.result);
  };
  worker.onerror = event => {
    event.preventDefault();
    stopWorker(`Python Worker 加载失败：${event.message || "请检查网络及文件是否完整"}`);
  };
  return worker;
}

function validateText(text, label) {
  if (new TextEncoder().encode(text).length > MAX_JSON_BYTES) throw new Error(`${label} 超过 128 MiB`);
  const value = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} 顶层必须是对象`);
}

// Stable API used by submit.py. Arguments are data, never executable code.
window.scoreTrajectoryFiles = async (inferText, targetText, tag = "", timeoutMs = 300000) => {
  if (pending) throw new Error("已有评分正在运行，请先等待或取消");
  clearResult();
  setBusy(true);
  try {
    validateText(inferText, "轨迹 JSON");
    validateText(targetText, "目标 JSON");
    setStatus("正在启动浏览器 Python…");
    const result = await new Promise((resolve, reject) => {
      const id = ++jobId;
      const timer = setTimeout(() => stopWorker("评分超时，请检查 Pyodide CDN 网络后重试"), timeoutMs);
      pending = { id, resolve, reject, timer };
      try {
        getWorker().postMessage({ type: "compute", id, inferText, targetText });
      } catch (error) {
        stopWorker(String(error));
      }
    });
    if (tag) result.tag = String(tag);
    displayResult(result);
    setStatus("评分完成。计算在当前浏览器中执行，可下载结果 JSON。");
    return result;
  } catch (error) {
    $("output").textContent = "评分失败，未生成有效分数。";
    setStatus(String(error?.message || error), true);
    throw error;
  } finally {
    setBusy(false);
  }
};

for (const id of ["inferFile", "targetFile"]) {
  $(id).addEventListener("change", () => {
    clearResult();
    setStatus("文件已选择，点击开始评分。文件不会上传到评测服务器。");
  });
}
$("runButton").addEventListener("click", async () => {
  try {
    const infer = $("inferFile").files[0];
    const target = $("targetFile").files[0];
    if (!infer || !target) throw new Error("请选择轨迹 JSON 和目标 JSON 两个文件");
    if (infer.size > MAX_JSON_BYTES || target.size > MAX_JSON_BYTES) throw new Error("单个 JSON 文件不能超过 128 MiB");
    await window.scoreTrajectoryFiles(await infer.text(), await target.text(), $("tagInput").value.trim());
  } catch (error) {
    setStatus(String(error?.message || error), true);
  }
});
$("cancelButton").addEventListener("click", () => stopWorker("已取消评分，可重新开始"));
$("downloadButton").addEventListener("click", () => {
  if (!resultValue) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(resultValue, null, 2)], { type: "application/json;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "f2b_score.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
