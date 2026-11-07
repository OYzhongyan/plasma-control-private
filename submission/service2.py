"""Participant A HTTP agent. Only C owns the official HFM and score."""
from __future__ import annotations
import argparse
import os
import secrets
import sys
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from submission.inference2 import Policy
from submission.agent_session import AgentSession, SessionConflict
from submission.remote_protocol import PROTOCOL, MAX_REQUEST_BYTES, checked_action, strict_json


def _coerce_observation(observation):
    return {key: np.asarray(value, dtype=np.float64) for key, value in observation.items()}


def create_app(model_dir=None, *, policy=None, access_token=None, allow_legacy=False,
               backend=None, model_path=None, vecnormalize_path=None, adapter_factory=None, device="cpu"):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    policy = policy if policy is not None else Policy(model_dir=model_dir, backend=backend, model_path=model_path,
                    vecnormalize_path=vecnormalize_path, adapter_factory=adapter_factory, device=device)
    session = AgentSession(policy)
    token = os.environ.get("SUBMISSION_ACCESS_TOKEN", "") if access_token is None else access_token
    app.extensions["agent_policy"] = policy
    app.extensions["agent_session"] = session

    @app.before_request
    def authenticate():
        expected = "Bearer " + token
        if token and not secrets.compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
            return jsonify(ok=False, error="Unauthorized"), 401

    @app.get("/health")
    def health():
        status = dict(policy.status())
        status.update(protocol=PROTOCOL, ok=bool(policy.ready))
        if not policy.ready:
            status["error"] = "Policy not ready; inspect A service startup logs"
        return jsonify(status), 200 if policy.ready else 503

    def operation(reset):
        if not policy.ready:
            return jsonify(ok=False, error="Policy not ready"), 503
        legacy_request = False
        try:
            payload = strict_json(request.get_data(cache=False))
            if allow_legacy and "protocol" not in payload:
                legacy_request = True
                if reset:
                    policy.reset(episode_id=payload.get("episode_id"), options=payload.get("options"))
                    return jsonify(ok=True)
                return jsonify(ok=True, action=checked_action(policy.act(_coerce_observation(payload["observation"]))))
            result = session.reset(payload) if reset else session.act(payload, _coerce_observation)
            return jsonify(result)
        except SessionConflict as exc:
            return jsonify(ok=False, error=str(exc)), 409
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            if session.failed or legacy_request:
                app.logger.exception("Participant inference failed")
                return jsonify(ok=False, error="Agent inference failed; reset with a new episode_id"), 500
            return jsonify(ok=False, error=str(exc)), 400
        except Exception:
            app.logger.exception("Participant inference failed")
            return jsonify(ok=False, error="Agent inference failed; inspect A service logs"), 500

    @app.post("/reset")
    def reset():
        return operation(True)

    @app.post("/act")
    def act():
        return operation(False)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Checkpoint file or training run directory")
    parser.add_argument("--vecnormalize", type=Path, help="Matching VecNormalize.pkl")
    parser.add_argument("--adapter-factory", help="Local module:function returning reset/encode/decode adapter")
    parser.add_argument("--backend", choices=("checkpoint", "onnx"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default=os.environ.get("SUBMISSION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SUBMISSION_PORT2", "8001")))
    args = parser.parse_args(argv)
    app = create_app(args.model_dir, backend=args.backend, model_path=args.model,
                     vecnormalize_path=args.vecnormalize, adapter_factory=args.adapter_factory, device=args.device)
    policy = app.extensions["agent_policy"]
    if not policy.ready:
        parser.error(policy.load_error or "Participant policy failed to load")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    app = create_app()
