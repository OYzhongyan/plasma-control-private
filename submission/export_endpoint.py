"""Generate the connection file uploaded by A to platform B; no model upload."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from submission.remote_protocol import make_endpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/endpoint.json")
    parser.add_argument("--token-env", default="SUBMISSION_ACCESS_TOKEN",
                        help="Read A's token from this environment variable; never print it")
    args = parser.parse_args(argv)
    value = make_endpoint(args.service_url, os.environ.get(args.token_env, ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=".endpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        temporary.chmod(0o600)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Connection file ready: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
