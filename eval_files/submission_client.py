"""与 submission 服务通信：健康检查、POST 请求、构建/启停 """

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional


def build_submission_image(project_root: Path, image_name: str) -> None:
    subprocess.run(
        ["docker", "build", "-f", "submission/Dockerfile", "-t", image_name, "."],
        cwd=str(project_root),
        check=True,
    )


def start_submission_container(
    image_name: str,
    host_port: int,
    host_port2: int | None = None,
) -> str:
    host_port2 = int(host_port2 if host_port2 is not None else host_port + 1)
    command = (
        "python3 /app/submission/service1.py & "
        "pid1=$!; "
        "python3 /app/submission/service2.py & "
        "pid2=$!; "
        "wait $pid1 $pid2"
    )
    return (
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
                "-e",
                "SUBMISSION_HOST=0.0.0.0",
                "-e",
                "SUBMISSION_PORT=8000",
                "-e",
                "SUBMISSION_PORT2=8001",
                "--entrypoint",
                "/bin/bash",
                image_name,
                "-lc",
                command,
            ],
            text=True,
        )
        .strip()
    )


def stop_submission_container(container_id: str | None) -> None:
    if container_id:
        subprocess.run(
            ["docker", "stop", container_id],
            check=False,
            stdout=subprocess.DEVNULL,
        )


def wait_for_health(service_url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{service_url.rstrip('/')}/health", timeout=2
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"submission service health check failed: {last_error}")


def post_json(
    service_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """
    timeout=None：不限制单次请求等待（用于 /reset、首步 /act 等初始化）。
    传入正数则为 urllib 读超时秒数。
    """
    request = urllib.request.Request(
        f"{service_url.rstrip('/')}/{path.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
