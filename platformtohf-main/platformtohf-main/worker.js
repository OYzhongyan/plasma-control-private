// Match the module-worker API of pinned Pyodide; do not use importScripts().
const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
let runtimePromise;

async function initialize() {
  self.postMessage({ type: "status", message: "正在下载 Python 运行时，首次运行需要联网…" });
  const { loadPyodide } = await import(`${PYODIDE_BASE}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });
  self.postMessage({ type: "status", message: "正在加载 NumPy 和项目评分规则…" });
  await pyodide.loadPackage("numpy");
  for (const name of ["scorer_bundle.py", "compute.py"]) {
    const response = await fetch(new URL(name, import.meta.url), { cache: "no-cache" });
    if (!response.ok) throw new Error(`无法加载 ${name}: HTTP ${response.status}`);
    pyodide.FS.writeFile(`/home/pyodide/${name}`, await response.text(), { encoding: "utf8" });
  }
  pyodide.runPython("from compute import compute_json");
  return pyodide;
}

self.onmessage = async ({ data }) => {
  if (data.type !== "compute") return;
  try {
    if (!runtimePromise) runtimePromise = initialize().catch(error => {
      runtimePromise = null;
      throw error;
    });
    const pyodide = await runtimePromise;
    self.postMessage({ type: "status", message: "正在比较轨迹与目标，计算各指标和约束惩罚…" });
    pyodide.globals.set("infer_text", data.inferText);
    pyodide.globals.set("target_text", data.targetText);
    let text;
    try {
      text = await pyodide.runPythonAsync("compute_json(infer_text, target_text)");
    } finally {
      pyodide.globals.delete("infer_text");
      pyodide.globals.delete("target_text");
    }
    self.postMessage({ type: "result", id: data.id, result: JSON.parse(text) });
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, message: String(error?.message || error) });
  }
};
