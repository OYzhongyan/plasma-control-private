"""Build image, run container, start both semifinal services, probe /health /reset /act."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
submission_dir = root / "submission"
model_dir = submission_dir / "model"
inference1_script = submission_dir / "inference1.py"
inference2_script = submission_dir / "inference2.py"
service1_script = submission_dir / "service1.py"
service2_script = submission_dir / "service2.py"
dockerfile = submission_dir / "Dockerfile"
requirements = submission_dir / "requirements.txt"
image_name = "fusion-control-comp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--port2", type=int, default=None)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if submission/model/ does not contain an ONNX file.",
    )
    return parser.parse_args()


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"health check failed: {last_error}")


def _post_json(base_url: str, path: str, payload: dict, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _probe_service(base_url: str, episode_id: str, example_obs: dict) -> None:
    health = _wait_for_health(base_url)
    if not health.get("ok", False):
        raise RuntimeError(f"{base_url}/health returned {health}")

    reset_payload = _post_json(base_url, "/reset", {"episode_id": episode_id})
    if not reset_payload.get("ok", False):
        raise RuntimeError(f"{base_url}/reset returned {reset_payload}")

    act_payload = _post_json(base_url, "/act", {"observation": example_obs})
    if not act_payload.get("ok", False):
        raise RuntimeError(f"{base_url}/act returned {act_payload}")
    if len(act_payload.get("action", [])) != 12:
        raise RuntimeError(f"{base_url}/act must return 12D action, got {act_payload}")


def main():
    args = parse_args()
    host_port = args.port
    host_port2 = args.port2 or host_port + 1
    if not submission_dir.is_dir():
        print("Missing submission/")
        sys.exit(1)
    for required in [inference1_script, inference2_script, service1_script, service2_script]:
        if not required.is_file():
            print(f"Missing {required.relative_to(root)}")
            sys.exit(1)
    if not dockerfile.is_file():
        print("Missing submission/Dockerfile")
        sys.exit(1)
    if not requirements.is_file():
        print("Missing submission/requirements.txt")
        sys.exit(1)
    if not model_dir.is_dir():
        print("Missing submission/model/")
        sys.exit(1)
    if args.require_model and not any(model_dir.glob("*.onnx")):
        print("Missing ONNX model in submission/model/ (expected *.onnx)")
        sys.exit(1)
    if not any(model_dir.glob("*.onnx")):
        print("Warning: no ONNX model found, fallback zero-action policy will be checked.")

    example_obs = {
        "Ip": [0.0],
        "R": [0.0],
        "Z": [0.0],
        "reference_Ip": [0.0],
        "reference_R": [0.0],
        "reference_Z": [0.0],
        "lcfs_points": [[0.0, 0.0]] * 32,
        "reference_lcfs_points": [[0.0, 0.0]] * 32,
        "reference_rX": [0.0] * 4,
        "reference_zX": [0.0] * 4,
        "reference_x_valid": [0.0] * 4,
        "reference_strike_r": [0.0] * 8,
        "reference_strike_z": [0.0] * 8,
        "reference_strike_valid": [0.0] * 8,
        "reference_nX": [0.0],
        "reference_n_strike": [0.0],
        "I_PF": [0.0] * 12,
    }
    base_url = f"http://127.0.0.1:{host_port}"
    base_url2 = f"http://127.0.0.1:{host_port2}"
    container_id = None

    try:
        subprocess.run(
            ["docker", "build", "-f", "submission/Dockerfile", "-t", image_name, "."],
            cwd=str(root),
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
                    f"{host_port}:8000",
                    "-p",
                    f"{host_port2}:8001",
                    "-v",
                    f"{root}:/saisdata/33/standalone-env:ro",
                    image_name,
                ],
                cwd=str(root),
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

        _probe_service(base_url, "check-submission-F1", example_obs)
        _probe_service(base_url2, "check-submission-F2", example_obs)

        print("Submission docker build and two-service API check passed.")
    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
