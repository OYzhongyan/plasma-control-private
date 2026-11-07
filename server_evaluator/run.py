"""Section 4.5 worker: download ONNX bundles, simulate on CPU and report scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from server_evaluator.client import PlatformClient, PlatformError
from server_evaluator.model_grader import ModelGrader
from server_evaluator.store import Store, instance_lock
from server_evaluator.worker import Worker

LOG = logging.getLogger("server_evaluator")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default=os.environ.get("PLATFORM_PROBLEM_ID"), help="Platform string problem id, from the teacher API help")
    parser.add_argument("--platform-url", default=os.environ.get("PLATFORM_BASE_URL", "https://competition.opensii.ai"))
    parser.add_argument("--target", type=Path, default=ROOT / "evaluation/results/target.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "server_evaluator/runtime-reference-v1")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env_default.yaml")
    parser.add_argument("--reference-config", type=Path, default=ROOT / "eval_files/references.yaml")
    parser.add_argument("--obs-perturb-config", type=Path, default=ROOT / "eval_files/obs_perturb.yaml")
    parser.add_argument("--hfm-host", default="127.0.0.1")
    parser.add_argument("--hfm-port", type=int, default=5558)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    parser.add_argument("--evaluation-timeout", type=float, default=1800.0, help="Hard wall-clock limit for each evaluation process")
    parser.add_argument("--inference-timeout", type=float, default=1.0, help="Maximum observed wall time per model inference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onnx-threads", type=int, default=1)
    parser.add_argument("--memory-limit-mb", type=int, default=4096, help="Linux child address-space limit in MiB (minimum 256)")
    parser.add_argument("--sources", default="offline,manual", help="Comma-separated sources: offline,manual")
    parser.add_argument("--limit", type=int, default=100, help="Pending records per source, 1..500")
    parser.add_argument("--poll-interval", type=float, default=15)
    parser.add_argument("--http-timeout", type=float, default=30)
    parser.add_argument("--max-file-mb", type=float, default=128)
    parser.add_argument("--download-retries", type=int, default=3, help="Cycles before a persistent download failure is reported as failed")
    parser.add_argument("--once", action="store_true", help="One poll/grade/report cycle; nonzero if results remain outstanding")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Validate local configuration only; no login, HFM connection or score uploads")
    mode.add_argument("--model-file", type=Path, help="Evaluate one local model ZIP using HFM; no platform credentials or uploads")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = tuple(dict.fromkeys(s.strip() for s in args.sources.split(",") if s.strip()))
    if not args.model_file and not args.check_config and (not args.problem_id or not args.problem_id.strip()):
        parser.error("--problem-id or PLATFORM_PROBLEM_ID is required for platform polling")
    if not sources or set(sources) - {"offline", "manual"}: parser.error("--sources must contain offline and/or manual")
    if not 1 <= args.limit <= 500: parser.error("--limit must be 1..500")
    if args.download_retries < 1: parser.error("--download-retries must be positive")
    for name in ("poll_interval", "http_timeout", "max_file_mb", "socket_timeout", "evaluation_timeout", "inference_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0: parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    client = None
    stop = Event()
    def request_stop(signum, frame): stop.set()
    try:
        grader = ModelGrader(args.target, args.data_dir, env_config=args.env_config,
                             reference_config=args.reference_config, obs_perturb_config=args.obs_perturb_config,
                             hfm_host=args.hfm_host, hfm_port=args.hfm_port, socket_timeout=args.socket_timeout,
                             evaluation_timeout=args.evaluation_timeout, seed=args.seed,
                             onnx_threads=args.onnx_threads, memory_limit_mb=args.memory_limit_mb,
                             inference_timeout=args.inference_timeout, stop_event=stop)
        LOG.info("Model interface=%s; power supply=%s; seed=%s",
                 grader.config["model_interface"], grader.config["power_supply"], args.seed)
        if args.check_config:
            print(json.dumps({"ok": True, "problem_id": args.problem_id, "sources": sources, "target": str(args.target.resolve()),
                              "metric_names": grader.metric_names, "max_score": 30, "rules_fingerprint": grader.fingerprint,
                              "submission_format": "onnx-model-zip-v1", "evaluation": grader.config}, ensure_ascii=False))
            return 0
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if args.model_file:
            with instance_lock(args.data_dir):
                result = grader.grade_file(args.model_file)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        client = PlatformClient(args.platform_url, args.problem_id, username=os.environ.get("PLATFORM_USERNAME"),
                                password=os.environ.get("PLATFORM_PASSWORD"), token=os.environ.get("PLATFORM_TOKEN"), timeout=args.http_timeout)
        fingerprint = hashlib.sha256(json.dumps({"rules": grader.fingerprint, "platform": client.base_url,
                                                "problem": args.problem_id, "sources": sources}, sort_keys=True).encode()).hexdigest()
        with instance_lock(args.data_dir), Store(args.data_dir, fingerprint) as store:
            worker = Worker(client, store, grader, args.data_dir / "downloads", sources=sources, limit=args.limit,
                            max_bytes=int(args.max_file_mb * 1024 * 1024), download_retries=args.download_retries,
                            stop_event=stop)
            LOG.info("Offline grader started problem=%s sources=%s metrics=%s", args.problem_id, ",".join(sources), ",".join(grader.metric_names))
            while not stop.is_set():
                try:
                    outstanding = worker.cycle()
                    LOG.info("Cycle finished; locally outstanding=%s", outstanding)
                    if args.once: return 1 if outstanding else 0
                except PlatformError as exc:
                    LOG.warning("Platform operation deferred: %s", exc)
                    if args.once: return 1
                if not args.once: stop.wait(args.poll_interval)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        LOG.error("Worker stopped: %s", exc)
        return 1
    finally:
        if client: client.close()


if __name__ == "__main__":
    raise SystemExit(main())
