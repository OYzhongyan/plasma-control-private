"""Stop simulator containers started by start_simulator.py."""

import os
import shutil
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(current_dir, "docker-compose.generated.yml")


def find_compose_command():
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if shutil.which("docker-compose"):
        try:
            subprocess.run(
                ["docker-compose", "version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ["docker-compose"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return None


def main():
    if not os.path.exists(yaml_file_path):
        print(f"Warning: {yaml_file_path} not found. Skip.")
        sys.exit(0)
    compose_cmd = find_compose_command()
    if not compose_cmd:
        print("Error: docker compose not found.", file=sys.stderr)
        sys.exit(1)
    project_name = "hfm_matlab_service"
    subprocess.run(compose_cmd + ["-p", project_name, "-f", yaml_file_path, "down"], check=True)
    print("Done.")


if __name__ == "__main__":
    main()
