"""Versioned, authenticated data-only descriptors and shared-file packets."""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
import uuid

from submission.remote_protocol import strict_json, wire_value, MAX_ENDPOINT_BYTES, MAX_REQUEST_BYTES

SHARED_PROTOCOL = 'hfm-shared-policy-v1'
DEFAULT_SHARED_ROOT = '/inspire/hdd/project/26-nuclear-fusion/public/plasma-agents'
MAX_TTL = 3600.


def validate_descriptor(value):
    expected = {'version', 'protocol', 'transport', 'agent_id', 'access_token'}
    if (set(value) != expected or type(value.get('version')) is not int or value['version'] != 1
            or value.get('protocol') != SHARED_PROTOCOL or value.get('transport') != 'shared_directory'
            or not isinstance(value.get('agent_id'), str) or not re.fullmatch('[0-9a-f]{32}', value['agent_id'])
            or not isinstance(value.get('access_token'), str) or not re.fullmatch('[A-Za-z0-9_-]{32,128}', value['access_token'])):
        raise ValueError('Invalid shared-directory descriptor; regenerate endpoint.json')
    return value


def make_descriptor():
    return dict(version=1, protocol=SHARED_PROTOCOL, transport='shared_directory',
                agent_id=uuid.uuid4().hex, access_token=secrets.token_urlsafe(32))


def read_descriptor(path):
    with Path(path).open('rb') as stream:
        return validate_descriptor(strict_json(stream.read(MAX_ENDPOINT_BYTES+1), MAX_ENDPOINT_BYTES))


def write_descriptor(path, descriptor):
    validate_descriptor(descriptor)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.endpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(descriptor, indent=2).encode()+b'\n')
        temporary.chmod(0o600); os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def canonical(value):
    return json.dumps(wire_value(value), allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def digest(raw): return hashlib.sha256(raw).hexdigest()


def encode_packet(descriptor, request_id, direction, payload, *, expires_at, request_digest=None):
    value = dict(protocol=SHARED_PROTOCOL, agent_id=descriptor['agent_id'], request_id=request_id,
                 direction=direction, expires_at=expires_at, request_digest=request_digest, payload=wire_value(payload))
    value['signature'] = hmac.new(descriptor['access_token'].encode(), canonical(value), hashlib.sha256).hexdigest()
    return canonical(value)


def decode_packet(raw, descriptor, direction, *, request_id=None, request_digest=None, now=None, maximum=MAX_REQUEST_BYTES):
    value = strict_json(raw, maximum)
    expected = {'protocol', 'agent_id', 'request_id', 'direction', 'expires_at', 'request_digest', 'payload', 'signature'}
    if set(value) != expected: raise ValueError('Invalid shared message fields')
    signature = value.pop('signature')
    computed = hmac.new(descriptor['access_token'].encode(), canonical(value), hashlib.sha256).hexdigest()
    if not isinstance(signature, str) or not hmac.compare_digest(signature.encode(), computed.encode()):
        raise ValueError('Invalid shared message signature')
    now = time.time() if now is None else now
    expiry = value['expires_at']
    if (type(expiry) not in (int, float) or not math.isfinite(expiry)
            or not now < expiry <= now + MAX_TTL):
        raise ValueError('Shared message expired or has invalid expiry')
    if (value['protocol'] != SHARED_PROTOCOL or value['agent_id'] != descriptor['agent_id']
            or value['direction'] != direction or not isinstance(value['payload'], dict)
            or not isinstance(value['request_id'], str) or not re.fullmatch('[0-9a-f]{32}', value['request_id'])
            or request_id is not None and value['request_id'] != request_id
            or request_digest is not None and value['request_digest'] != request_digest):
        raise ValueError('Shared message correlation mismatch')
    return value
