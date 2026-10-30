"""HTTP submission service for policy1 (F1).

Endpoints:
- GET /health
- POST /reset
- POST /act
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference1 import Policy  # noqa: E402


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _coerce_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value, dtype=np.float32) for key, value in obs.items()}


def create_app() -> Flask:
    app = Flask(__name__)
    policy = Policy()

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True})

    @app.post("/reset")
    def reset() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            policy.reset()
            return jsonify(
                {
                    "ok": True,
                    "received": {
                        "episode_id": payload.get("episode_id"),
                        "options": _to_serializable(payload.get("options")),
                    },
                }
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/act")
    def act() -> Any:
        payload = request.get_json(force=True)
        if "observation" not in payload:
            return jsonify({"ok": False, "error": "Missing observation"}), 400

        try:
            observation = _coerce_observation(payload["observation"])
            action = np.asarray(policy.act(observation), dtype=np.float32).reshape(-1)
            if action.size != 12:
                return jsonify(
                    {"ok": False, "error": f"Policy must return 12D action, got size {action.size}"}
                ), 400
            return jsonify({"ok": True, "action": action.tolist()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("SUBMISSION_HOST", "0.0.0.0")
    port = int(os.environ.get("SUBMISSION_PORT", "8000"))
    app.run(host=host, port=port)
