"""Generate docker compose and start HFM simulator containers."""

import argparse
import os
import subprocess
import sys

DEFAULT_CPUS_PER_CONTAINER = 4
DEFAULT_MEMORY_PER_CONTAINER = "4G"
DEFAULT_BASE_PORT = 5558
RESERVE_PERCENTAGE = 10

current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(current_dir, "docker-compose.generated.yml")


def start_docker_compose():
    parser = argparse.ArgumentParser(description="Docker Compose Generator for HFM servers")
    parser.add_argument("-c", "--cpus", type=int, default=DEFAULT_CPUS_PER_CONTAINER)
    parser.add_argument("-m", "--memory", type=str, default=DEFAULT_MEMORY_PER_CONTAINER)
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("-n", "--num", type=int, help="Number of containers")
    parser.add_argument("-y", "--yes", action="store_true", help="Start without prompting")
    args = parser.parse_args()

    if args.cpus <= 0 or not (1 <= args.port <= 65535):
        print("Invalid cpus or port.")
        sys.exit(1)

    if args.num is not None:
        if args.num <= 0:
            sys.exit(1)
        num_containers = args.num
    else:
        total_cores = os.cpu_count() or 1
        reserved = max(1, int(total_cores * RESERVE_PERCENTAGE / 100))
        usable = total_cores - reserved
        num_containers = usable // args.cpus

    if num_containers < 1:
        print("Not enough cores for one container.")
        sys.exit(1)

    content = [
        "# Auto-generated",
        "version: '3.8'",
        "",
        "services:",
    ]
    for i in range(1, num_containers + 1):
        cpu_start = (i - 1) * args.cpus
        cpu_end = cpu_start + args.cpus - 1
        port = args.port + i - 1
        content.append(
            f"""  hfm_server_{i}:
    container_name: hfm_server_{i}
    image: hfm-matlab-server:competition
    ports:
      - "{port}:5558"
    networks:
      - frontend_net
    deploy:
      resources:
        limits:
          cpus: '{args.cpus}.0'
          memory: {args.memory}
    cpuset: "{cpu_start}-{cpu_end}"
    restart: unless-stopped
    environment:
      MATLAB_SOCKET_TIMEOUT: 600
"""
        )
    content.append("networks:")
    content.append("  frontend_net:")
    content.append("    driver: bridge")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(content))

    print(f"Generated {OUTPUT_FILE} with {num_containers} services.")

    should_start = args.yes or (
        sys.stdin.isatty() and input("Start containers? (y/n) ").lower() == "y"
    )
    if should_start:
        project_name = "hfm_matlab_service"
        try:
            subprocess.run(
                ["docker", "compose", "-p", project_name, "-f", OUTPUT_FILE, "up", "-d", "--build"],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(
                ["docker-compose", "-p", project_name, "-f", OUTPUT_FILE, "up", "-d", "--build"],
                check=True,
            )


if __name__ == "__main__":
    start_docker_compose()
