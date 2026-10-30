#!/usr/bin/env python3
# coding=utf-8
"""
复赛评测入口：读取 target（标准答案）与 infer_result（选手结果），按 README 规范输出 JSON。
用法: python evaluate.py <target_path> <infer_result_path>
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from score import config, evaluate

TASK_IDS = config.TASK_IDS  # ["F1", "F2a", "F2b"]


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def _to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def main() -> None:
    output = {
        "errorMsg": "user input is wrong, please check !",
        "score": 0,
        "scoreJson": {},
        "success": False,
    }

    log_dir = EVAL_DIR / "logs"
    logger = _setup_logging(log_dir)

    try:
        if len(sys.argv) < 3:
            output["errorMsg"] = "Arguments missing. Usage: python evaluate.py <target_path> <infer_result_path>"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        std_path = sys.argv[1]
        user_path = sys.argv[2]

        if not os.path.isfile(std_path):
            output["errorMsg"] = f"Target file not found: {std_path}"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return
        if not os.path.isfile(user_path):
            output["errorMsg"] = f"Infer result file not found: {user_path}"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        with open(std_path, "r", encoding="utf-8") as f:
            target_raw = json.load(f)
        with open(user_path, "r", encoding="utf-8") as f:
            infer_raw = json.load(f)

        # 校验 infer_result 必须包含全部子任务
        for tid in TASK_IDS:
            if tid not in infer_raw:
                output["errorMsg"] = f"Infer result missing task: {tid}"
                logger.warning(output["errorMsg"])
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return

        # target 允许 F2b 缺失（fallback 到 F2a）
        for tid in TASK_IDS:
            if tid not in target_raw:
                fallback = config.TARGET_FALLBACK.get(tid)
                if not (fallback and fallback in target_raw):
                    output["errorMsg"] = f"Target missing task: {tid} (and no fallback)"
                    logger.warning(output["errorMsg"])
                    print(json.dumps(output, ensure_ascii=False, indent=2))
                    return

        result = evaluate(infer_raw, target_raw, task_ids=TASK_IDS)

        task_scores = result["task_scores"]
        task_metric_scores = result["task_metric_scores"]
        total = float(result["total_score"])
        final_score = round(total, 6)

        part_scores = {
            f"score{i}": round(float(task_scores.get(tid, 0.0)), 6)
            for i, tid in enumerate(TASK_IDS, start=1)
        }

        detail = {
            "total_score": total,
            "task_scores": {k: float(v) for k, v in task_scores.items()},
            "task_metric_scores": {
                k: {m: float(s) for m, s in v.items()} for k, v in task_metric_scores.items()
            },
            "K_eff": {r.task_id: r.K_eff for r in result["per_task_results"]},
            "gamma": {r.task_id: r.gamma for r in result["per_task_results"]},
        }
        logger.info("Evaluation detail: %s", json.dumps(detail, ensure_ascii=False, indent=2, default=_to_jsonable))

        # 按指标分项铺平到 scoreJson（仅顶层数字字段）
        flat_metric = {}
        for tid in TASK_IDS:
            for m, s in task_metric_scores.get(tid, {}).items():
                flat_metric[f"{tid}_{m}"] = round(float(s), 6)

        score_json = {
            "score": final_score,
            **part_scores,
            **flat_metric,
        }
        success_output = {
            "score": final_score,
            **part_scores,
            "scoreJson": score_json,
            "errorMsg": "",
            "success": True,
        }
        print(json.dumps(success_output, ensure_ascii=False, indent=2))

    except Exception as e:
        output["errorMsg"] = str(e)
        logger.exception("Evaluation failed")
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
