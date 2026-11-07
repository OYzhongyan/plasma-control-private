"""Bounded direct HTTP transport to A; never imports contestant model code."""
from __future__ import annotations

import http.client
import ipaddress
import json
import math
import socket
import ssl
from urllib.parse import urlsplit
import uuid

import numpy as np

from submission.remote_protocol import (PROTOCOL, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
    canonical_origin, checked_action, make_endpoint, strict_json, wire_value)


class RemoteAgentError(ValueError):
    pass


class RemoteAgent:
    def __init__(self, endpoint, *, allowed_origins=(), timeout=15., retries=2, on_request=None):
        try:
            if set(endpoint) - {"version", "protocol", "service_url", "access_token"}:
                raise ValueError("Unknown endpoint fields")
            if type(endpoint.get("version")) is not int or endpoint["version"] != 1 or endpoint.get("protocol") != PROTOCOL:
                raise ValueError("Unsupported endpoint protocol")
            self.endpoint = make_endpoint(endpoint["service_url"], endpoint.get("access_token", ""))
            self.allowed_origins = {canonical_origin(value) for value in allowed_origins}
        except (ValueError, TypeError, KeyError) as exc:
            raise RemoteAgentError("Invalid remote endpoint configuration") from exc
        if not math.isfinite(timeout) or timeout <= 0 or type(retries) is not int or not 0 <= retries <= 5:
            raise ValueError("Remote timeout must be positive and retries in 0..5")
        self.timeout, self.retries = float(timeout), retries
        self.url = urlsplit(self.endpoint["service_url"])
        self.port = self.url.port or (443 if self.url.scheme == "https" else 80)
        self.on_request = on_request
        self.episode_id = None
        self.step_id = 0
        self.audit = {"retry_count": 0, "request_count": 0}
        self.model_id = None
        response = self._request("/health", None, -2)
        self._check_response(response)
        self.model_id = response.get("model_id")
        if self.model_id is not None and (not isinstance(self.model_id, str) or len(self.model_id) > 256):
            raise RemoteAgentError("Invalid model identity")

    def _address(self):
        try:
            answers = socket.getaddrinfo(self.url.hostname, self.port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise RemoteAgentError("Cannot resolve participant service") from exc
        if not answers:
            raise RemoteAgentError("Participant service has no addresses")
        for answer in answers:
            address = ipaddress.ip_address(answer[4][0])
            address = getattr(address, "ipv4_mapped", None) or address
            if address.is_unspecified or address.is_multicast or address.is_link_local:
                raise RemoteAgentError("Link-local, multicast and unspecified destinations are prohibited")
            if not address.is_global and self.endpoint["service_url"] not in self.allowed_origins:
                raise RemoteAgentError("Private service origin must be explicitly allowed by C's operator")
        return answers[0][4][0]

    def _request(self, path, payload, step):
        raw = None if payload is None else json.dumps(wire_value(payload), allow_nan=False,
                    separators=(",", ":"), sort_keys=True).encode()
        if raw is not None and len(raw) > MAX_REQUEST_BYTES:
            raise RemoteAgentError("Observation request is too large")
        if self.on_request:
            self.on_request(path, step)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.endpoint.get("access_token"):
            headers["Authorization"] = "Bearer " + self.endpoint["access_token"]
        for attempt in range(self.retries + 1):
            address = self._address()
            cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
            kwargs = {"context": ssl.create_default_context()} if self.url.scheme == "https" else {}
            connection = cls(self.url.hostname, self.port, timeout=self.timeout, **kwargs)
            # Keep Host / TLS SNI and certificate validation tied to the original
            # hostname, but connect ONLY to the numeric address checked above.
            connection._create_connection = lambda _address, timeout, source_address=None: socket.create_connection(
                (address, self.port), timeout, source_address)
            self.audit["request_count"] += 1
            try:
                connection.request("GET" if payload is None else "POST", path, body=raw, headers=headers)
                response = connection.getresponse()
                if response.status != 200:
                    raise RemoteAgentError(f"Participant service returned HTTP {response.status} at {path}")
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise RemoteAgentError("Invalid or oversized participant response")
                # read(size) may silently return a short body on premature EOF.
                # Treat it as lost transport, so A can replay the cached action.
                if response.length is not None and response.length > 0:
                    raise http.client.IncompleteRead(data, response.length)
                try:
                    return strict_json(data, maximum=MAX_RESPONSE_BYTES)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise RemoteAgentError("Invalid or oversized participant response") from exc
            except (OSError, http.client.HTTPException) as exc:
                if attempt == self.retries:
                    raise RemoteAgentError(f"Participant connection failed at {path} after {attempt + 1} attempts") from exc
                self.audit["retry_count"] += 1
            finally:
                connection.close()

    def _check_response(self, value, *, step=None):
        if value.get("ok") is not True or value.get("protocol") != PROTOCOL:
            raise RemoteAgentError("Participant response has invalid protocol or status")
        if self.episode_id is not None and value.get("episode_id") != self.episode_id:
            raise RemoteAgentError("Participant response has wrong episode")
        if step is not None and (type(value.get("step_id")) is not int or value["step_id"] != step):
            raise RemoteAgentError("Participant response has wrong step")
        if self.model_id is not None and value.get("model_id") != self.model_id:
            raise RemoteAgentError("Participant model identity changed during evaluation")

    def reset(self):
        self.episode_id, self.step_id = uuid.uuid4().hex, 0
        value = self._request("/reset", {"protocol": PROTOCOL, "episode_id": self.episode_id}, -1)
        self._check_response(value)

    def act(self, observation):
        if self.episode_id is None:
            raise RemoteAgentError("Reset is required before inference")
        response = self._request("/act", {"protocol": PROTOCOL, "episode_id": self.episode_id,
                "step_id": self.step_id, "observation": observation}, self.step_id)
        self._check_response(response, step=self.step_id)
        try:
            action = np.asarray(checked_action(response.get("action")), dtype=np.float32)
            if not np.all(np.isfinite(action)):
                raise ValueError("Action overflows float32")
        except (ValueError, TypeError, OverflowError) as exc:
            raise RemoteAgentError("Participant must return 12 finite physical voltages") from exc
        self.step_id += 1
        return action
