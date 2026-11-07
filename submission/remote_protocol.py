"""Data-only A/B/C protocol shared by participant services and the trusted grader."""
from __future__ import annotations
import ipaddress
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

PROTOCOL = "hfm-remote-policy-v1"
MAX_ENDPOINT_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024


def strict_json(raw, maximum=MAX_REQUEST_BYTES):
    if len(raw) > maximum:
        raise ValueError("JSON exceeds its size limit")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Non-finite JSON number")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    return value


def canonical_origin(url):
    if not isinstance(url, str) or any(ord(c) <= 32 for c in url) or "?" in url or "#" in url:
        raise ValueError("service_url must be an HTTP(S) root address")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"}):
            raise ValueError("service_url must be an HTTP(S) root address without credentials or a path")
        host = parsed.hostname
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535 or "%" in host:
            raise ValueError("Invalid service_url host or port")
        try:
            address = ipaddress.ip_address(host)
            host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        except ValueError:
            host = host.encode("idna").decode("ascii").lower()
            if (len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
                                       for part in host.split("."))):
                raise ValueError("Invalid service_url hostname")
        suffix = "" if port == (443 if parsed.scheme == "https" else 80) else f":{port}"
        return f"{parsed.scheme}://{host}{suffix}"
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid service_url; use an HTTP(S) root address") from exc


def make_endpoint(service_url, access_token=""):
    if (not isinstance(access_token, str) or len(access_token) > 4096
            or any(not 33 <= ord(c) <= 126 for c in access_token)):
        raise ValueError("access_token must contain printable ASCII without whitespace")
    value = dict(version=1, protocol=PROTOCOL, service_url=canonical_origin(service_url))
    if access_token:
        value["access_token"] = access_token
    return value


def read_endpoint(path):
    with Path(path).open("rb") as stream:
        value = strict_json(stream.read(MAX_ENDPOINT_BYTES + 1), MAX_ENDPOINT_BYTES)
    if set(value) - {"version", "protocol", "service_url", "access_token"}:
        raise ValueError("Endpoint file contains unsupported fields")
    if type(value.get("version")) is not int or value["version"] != 1 or value.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported endpoint version or protocol")
    return make_endpoint(value.get("service_url"), value.get("access_token", ""))


def wire_value(value):
    """JSON null denotes an unavailable/non-finite observation, not a fabricated zero."""
    if isinstance(value, dict):
        return {key: wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire_value(item) for item in value]
    if hasattr(value, "tolist"):
        return wire_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def checked_action(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (not isinstance(value, list) or len(value) != 12
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("Action must be exactly 12 finite physical voltages")
    return [float(v) for v in value]


def valid_envelope(value):
    if value.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported agent protocol")
    episode = value.get("episode_id")
    if not isinstance(episode, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", episode):
        raise ValueError("Invalid episode_id")
    return episode
