"""Local integration test between env and submission service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", default="http://127.0.0.1:18002")
    parser.add_argument("--service-url2", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "env_default.yaml"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--launch-service",
        choices=["none", "local", "docker"],
        default="docker",
        help="How to launch submission service before testing.",
    )
    return parser.parse_args()


def _derive_second_service_url(service_url: str) -> str:
    parsed = urlparse(service_url)
    if parsed.port is None:
        return service_url
    netloc = parsed.netloc.rsplit(":", 1)[0] + f":{parsed.port + 1}"
    return parsed._replace(netloc=netloc).geturl()


def start_services(
    mode: str,
    service_url: str,
    service_url2: str,
) -> tuple[list[subprocess.Popen[str]], str | None]:
    parsed = urlparse(service_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    parsed2 = urlparse(service_url2)
    port2 = parsed2.port or port + 1
    if mode == "none":
        return [], None
    if mode == "local":
        env = dict(os.environ)
        env["SUBMISSION_HOST"] = host
        env["SUBMISSION_PORT"] = str(port)
        env["SUBMISSION_PORT2"] = str(port2)
        processes = [
            subprocess.Popen(
                [sys.executable, str(ROOT / "submission" / "service1.py")],
                cwd=str(ROOT / "submission"),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            ),
            subprocess.Popen(
                [sys.executable, str(ROOT / "submission" / "service2.py")],
                cwd=str(ROOT / "submission"),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            ),
        ]
        return processes, None
    if mode == "docker":
        image_name = "fusion-control-comp"
        subprocess.run(
            ["docker", "build", "-f", "submission/Dockerfile", "-t", image_name, "."],
            cwd=str(ROOT),
            check=True,
        )
        container_id = (
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "-p",
                    f"{port}:8000",
                    "-p",
                    f"{port2}:8001",
                    "-v",
                    f"{ROOT}:/saisdata/33/standalone-env:ro",
                    image_name,
                ],
                cwd=str(ROOT),
                text=True,
            )
            .strip()
        )
        subprocess.run(
            ["docker", "exec", "-d", container_id, "python3", "/app/submission/service1.py"],
            check=True,
        )
        subprocess.run(
            ["docker", "exec", "-d", container_id, "python3", "/app/submission/service2.py"],
            check=True,
        )
        return [], container_id
    raise ValueError(f"Unsupported launch mode: {mode}")


def wait_for_health(service_url: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{service_url.rstrip('/')}/health", timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Submission service health check failed: {last_error}")


def post_json(service_url: str, path: str, payload: dict, timeout_s: float = 10.0) -> dict:
    request = urllib.request.Request(
        f"{service_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_env_steps(env, service_url: str, *, episode_id: str, steps: int) -> None:
    reset_payload = post_json(
        service_url,
        "/reset",
        {"episode_id": episode_id, "options": {"reference_mode": "hold"}},
    )
    print(f"{episode_id} reset:", reset_payload)
    if not reset_payload.get("ok", False):
        raise RuntimeError(reset_payload)

    obs, info = env.reset(options={"reference_mode": "hold"})
    print(f"{episode_id} env reset shot_id:", info["shot_id"])

    for step in range(steps):
        payload = post_json(
            service_url,
            "/act",
            {"observation": {k: v.tolist() for k, v in obs.items()}},
        )
        print(f"{episode_id} step {step} act:", payload)
        if not payload.get("ok", False):
            raise RuntimeError(payload)
        action = payload["action"]
        obs, reward, terminated, truncated, _ = env.step(action)
        print(
            f"{episode_id} step {step} env: reward={reward:.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            break


def main() -> None:
    from environment import HFMSimulator

    args = parse_args()
    processes: list[subprocess.Popen[str]] = []
    container_id = None
    service_url2 = args.service_url2 or _derive_second_service_url(args.service_url)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = HFMSimulator(config)

    try:
        processes, container_id = start_services(
            args.launch_service,
            args.service_url,
            service_url2,
        )
        print("service1 health:", wait_for_health(args.service_url.rstrip("/")))
        print("service2 health:", wait_for_health(service_url2.rstrip("/")))

        run_env_steps(env, args.service_url, episode_id="local-test-F1", steps=args.steps)
        run_env_steps(env, service_url2, episode_id="local-test-F2", steps=args.steps)
        print("two-service integration test passed")
    finally:
        env.close()
        if container_id is not None:
            subprocess.run(["docker", "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
