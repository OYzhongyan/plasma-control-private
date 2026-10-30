#!/usr/bin/env python3
"""
主流程：加载配置与 reference，启动 submission 服务（一次），跑多个子任务，输出 infer_result.json。
子任务逻辑在 subtask_runner 中。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import obs_perturb  # noqa: E402
import submission_client  # noqa: E402
import subtask_runner  # noqa: E402

SAISDATA_ROOT = SCRIPT_DIR.parent
INFERENCE_DIR = SAISDATA_ROOT / "eval_files"
PROJECT_ROOT_DEFAULT = SAISDATA_ROOT
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "infer_result.json"

from score import config as score_config  # noqa: E402
from score import evaluate as score_evaluate  # noqa: E402


def _ensure_np(obj: Any) -> Any:
    """Convert nested lists from JSON into numpy-like arrays for score package."""
    import numpy as np

    if isinstance(obj, list):
        return np.asarray(obj, dtype=np.float64)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "lcfs_per_step" and isinstance(v, list):
                out[k] = [np.asarray(x, dtype=np.float64) for x in v]
            else:
                out[k] = _ensure_np(v)
        return out
    return obj


def _load_target_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(
            f"[run_test] target json not found at {p}, skip per-task score print.",
            flush=True,
        )
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_second_service_url(service_url: str) -> str:
    parsed = urlparse(service_url)
    if parsed.port is None:
        return service_url
    netloc = parsed.netloc.rsplit(":", 1)[0] + f":{parsed.port + 1}"
    return urlunparse(parsed._replace(netloc=netloc))


def _service_url_for_task(task_id: str, service_url_f1: str, service_url_f2: str) -> str:
    return service_url_f1 if task_id == "F1" else service_url_f2


def _format_metric_contributions(task_id: str, metric_scores: dict[str, float]) -> str:
    weights = score_config.get_task_metrics_and_weights(task_id)
    if not weights:
        return "metrics=none"
    return " ".join(
        f"{name}={float(metric_scores.get(name, 0.0)):.6f}/{float(weight):g}"
        for name, weight in weights.items()
    )


def _format_step_count(count: int) -> str:
    return "未触发" if count <= 0 else f"{count}步"


def _format_rho_summary(rho: Any, total_steps: int) -> str:
    import numpy as np

    if rho is None:
        return "未触发"
    arr = np.asarray(rho, dtype=float).reshape(-1)[:total_steps]
    zero_idx = np.flatnonzero(arr <= 0.5)
    if zero_idx.size == 0:
        return "未触发"
    first = int(zero_idx[0]) + 1
    count = int(total_steps - zero_idx[0])
    return f"从第{first}步起，共{count}步"


def _format_penalty_summary(task_id: str, res0: Any, total_steps: int) -> str:
    import numpy as np

    k_eff = int(getattr(res0, "K_eff", getattr(res0, "n_actual_steps", 0)))
    valid = slice(0, max(k_eff, 0))

    eta = getattr(res0, "eta", None)
    mu = getattr(res0, "mu", None)
    topo_mask = getattr(res0, "topo_mask", None)
    rho = getattr(res0, "rho", None)

    eta_count = 0
    if eta is not None and k_eff > 0:
        eta_count = int(np.sum(np.asarray(eta, dtype=float).reshape(-1)[valid] < 0.999))

    mu_count = 0
    if mu is not None and k_eff > 0:
        mu_count = int(np.sum(np.asarray(mu, dtype=float).reshape(-1)[valid] <= 0.5))

    if task_id == "F1":
        topo_text = "不适用"
    else:
        topo_count = 0
        if topo_mask is not None and k_eff > 0:
            topo_count = int(np.sum(np.asarray(topo_mask, dtype=float).reshape(-1)[valid] <= 0.5))
        topo_text = _format_step_count(topo_count)

    return (
        f"[penalty-summary] {task_id} | K_eff={k_eff}/{total_steps} | "
        f"位形类型不匹配惩罚: {_format_step_count(eta_count)}；"
        f"电流偏差>50kA熔断: {_format_step_count(mu_count)}；"
        f"线圈电流超限后清零: {_format_rho_summary(rho, total_steps)}；"
        f"XPT nX!=4专属指标清零: {topo_text}"
    )


def _print_task_score(task_id: str, infer_task: dict[str, Any], target_task: dict[str, Any]) -> None:
    trajectory = _ensure_np(infer_task["trajectory"])
    target_all = _ensure_np(target_task)
    timeout = bool(infer_task.get("timeout", False))

    result = score_evaluate(
        {task_id: {"trajectory": trajectory, "timeout": timeout}},
        target_all,
        task_ids=[task_id],
        timeouts={task_id: timeout},
    )
    res0 = result["per_task_results"][0]
    metric_scores = result.get("task_metric_scores", {}).get(task_id, {})
    task_score = float(result["task_scores"].get(task_id, 0.0))

    f_surv = float(infer_task.get("f_surv", 1.0))
    if timeout:
        reason = "timeout=True: /act 接口超时或返回异常，该子任务按规则记 0 分。"
    elif f_surv < 1.0:
        reason = (
            "timeout=False: 推理接口未超时（单步5s）；f_surv<1 说明等离子体提前终止，"
            "后续步无贡献。"
        )
    else:
        reason = "完整跑满该子任务步数。"

    metric_text = _format_metric_contributions(task_id, metric_scores)
    task_total = float(score_config.TASK_WEIGHTS.get(task_id, {}).get("total", 0.0))
    print(
        (
            f"[subtask-score] {task_id} score={task_score:.6f}/{task_total:g} | "
            f"{metric_text} | "
            f"K_eff={getattr(res0, 'K_eff', getattr(res0, 'n_actual_steps', 0))} "
            f"f_surv={f_surv:.6f} timeout={timeout} | {reason}"
        ),
        flush=True,
    )
    print(_format_penalty_summary(task_id, res0, score_config.get_total_steps(task_id)), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semi-final subtasks against submission service, write infer_result.json"
    )
    parser.add_argument(
        "--config",
        default=str(SAISDATA_ROOT / "configs/env_default.yaml"),
        help="Env config YAML",
    )
    parser.add_argument(
        "--task-config",
        default=str(INFERENCE_DIR / "task_references.yaml"),
        help="Task references YAML",
    )
    parser.add_argument("--service-url", default="http://127.0.0.1:10000")
    parser.add_argument(
        "--service-url2",
        default="http://127.0.0.1:10001",
        help="Second submission service for F2a/F2b. Defaults to service-url port + 1.",
    )
    parser.add_argument(
        "--act-timeout-s",
        type=float,
        default=5.0,
        help="Per-step /act HTTP timeout (seconds) from the 2nd env step onward; ",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="infer_result.json path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from task YAML (quick test)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not build/start submission container; assume service already running",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT_DEFAULT),
        help="Project root for building submission image (submission/Dockerfile)",
    )
    parser.add_argument(
        "--image-name",
        default="submission-eval",
        help="Docker image name for submission",
    )
    parser.add_argument(
        "--obs-perturb",
        default=str(INFERENCE_DIR / "obs_perturb.yaml"),
        help="YAML for observation delay/noise (trajectory stays true)",
    )
    parser.add_argument(
        "--target-json",
        default=str(SAISDATA_ROOT / "evaluation/results/target.json"),
        help="Target reference JSON for per-task score print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_config = subtask_runner.load_config(args.config)
    task_specs, max_steps = subtask_runner.load_task_specs(args.task_config)
    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    env_config["max_steps"] = max_steps
    obs_perturb_cfg = obs_perturb.load_obs_perturb_config(args.obs_perturb)
    target_path = Path(args.target_json)
    target_raw = _load_target_json(str(target_path))
    if target_raw is not None:
        print(f"[run_test] scoring target: {target_path}", flush=True)

    service_url_f1 = args.service_url
    service_url_f2 = args.service_url2 or _derive_second_service_url(args.service_url)
    print(f"[run_test] service map: F1 -> {service_url_f1}, F2a/F2b -> {service_url_f2}", flush=True)

    container_id = None
    if not args.no_build:
        print("[run_test] building and starting submission container...", flush=True)
        container_id = subtask_runner.start_submission_service(
            Path(args.project_root),
            service_url_f1,
            service_url_f2,
            args.image_name,
        )
        print("[run_test] submission service ready.", flush=True)
    else:
        submission_client.wait_for_health(service_url_f1)
        if service_url_f2 != service_url_f1:
            submission_client.wait_for_health(service_url_f2)
        print("[run_test] using existing submission services.", flush=True)

    results = {}
    try:
        for task_id in subtask_runner.TASK_IDS:
            if task_id not in task_specs:
                raise KeyError(f"Missing task config for {task_id} in {args.task_config}")
            task_service_url = _service_url_for_task(task_id, service_url_f1, service_url_f2)
            results[task_id] = subtask_runner.run_single_task(
                task_id=task_id,
                task_spec=task_specs[task_id],
                service_url=task_service_url,
                env_base_config=env_config,
                act_timeout_s=args.act_timeout_s,
                obs_perturb_cfg=obs_perturb_cfg,
            )
            if target_raw is not None:
                fallback = getattr(score_config, "TARGET_FALLBACK", {}).get(task_id)
                if task_id in target_raw or (fallback and fallback in target_raw):
                    _print_task_score(task_id, results[task_id], target_raw)
                else:
                    print(
                        f"[run_test] target missing task {task_id}, skip score print.",
                        flush=True,
                    )
        if target_raw is not None:
            total_result = score_evaluate(results, _ensure_np(target_raw), task_ids=subtask_runner.TASK_IDS)
            print(f"[total-score] {float(total_result['total_score']):.6f}", flush=True)
    finally:
        if container_id:
            submission_client.stop_submission_container(container_id)
            print("[run_test] submission container stopped.", flush=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SAISDATA_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[run_test] saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
