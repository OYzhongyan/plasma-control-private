"""Participant A: load a local checkpoint and serve signed shared-file requests."""
import argparse
from contextlib import ExitStack
import logging
import math
import os
from pathlib import Path
import re
import signal
import sys
from threading import Event
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from submission.agent_session import AgentSession
from submission.remote_protocol import PROTOCOL, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, strict_json, read_endpoint
from submission.shared_channel import (DEFAULT_SHARED_ROOT, make_descriptor, read_descriptor,
    write_descriptor, encode_packet, decode_packet, digest)
from submission.shared_files import Mailbox

LOG = logging.getLogger(__name__)


class SharedWorker:
    def __init__(self, root, descriptor, policy):
        self.root, self.descriptor, self.policy = Path(root), descriptor, policy
        self.service_id = uuid.uuid4().hex
        self.session = AgentSession(policy)
        self.stack = ExitStack()
        self.last_gc = 0.
        try:
            self.mailbox = self.stack.enter_context(Mailbox(root, descriptor['agent_id'], create=True))
            self.stack.enter_context(self.mailbox.lock('agent.lock'))
            # Initialize the stable shared lock inode with project-group access.
            with self.mailbox.lock('client.lock'): pass
        except BaseException:
            self.close(); raise

    def dispatch(self, payload):
        try:
            if (not {'method', 'body'}.issubset(payload) or set(payload)-{'method', 'body', 'service_id'}
                    or not isinstance(payload['body'], dict)):
                raise ValueError('Invalid request')
            method, body = payload['method'], payload['body']
            if method != 'health' and payload.get('service_id') != self.service_id:
                raise ValueError('Request belongs to a previous service incarnation')
            if method == 'health':
                result = dict(self.policy.status(), protocol=PROTOCOL)
                result['ok'] = bool(self.policy.ready)
            elif not self.policy.ready:
                raise RuntimeError('Policy not ready')
            elif method == 'reset': result = self.session.reset(body)
            elif method == 'act':
                result = self.session.act(body, lambda obs: {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()})
            else: raise ValueError('Unknown request method')
            return {**result, 'service_id': self.service_id}
        except Exception:
            LOG.exception('Agent request failed')
            return dict(ok=False, protocol=PROTOCOL, service_id=self.service_id,
                        error='Agent request failed; inspect A service logs')

    def poll_once(self):
        handled = 0
        for name in self.mailbox.requests.names():
            if not re.fullmatch('[0-9a-f]{32}\\.json', name): continue
            try:
                raw = self.mailbox.requests.read(name, MAX_REQUEST_BYTES)
                packet = decode_packet(raw, self.descriptor, 'request', request_id=name[:-5])
            except FileNotFoundError: continue
            except (ValueError, RecursionError, UnicodeError):
                self.mailbox.requests.unlink(name)
                continue
            request_hash = digest(raw)
            try:
                cached = self.mailbox.responses.read(name, MAX_RESPONSE_BYTES)
                response = decode_packet(cached, self.descriptor, 'response', request_id=name[:-5],
                                         request_digest=request_hash, maximum=MAX_RESPONSE_BYTES)
                if response['payload'].get('service_id') == self.service_id: continue
            except (FileNotFoundError, ValueError, RecursionError, UnicodeError): pass
            result = self.dispatch(packet['payload'])
            raw_response = encode_packet(self.descriptor, name[:-5], 'response', result,
                                        expires_at=packet['expires_at'], request_digest=request_hash)
            if len(raw_response) > MAX_RESPONSE_BYTES:
                raise ValueError('Agent response exceeded size limit')
            self.mailbox.responses.write(name, raw_response)
            handled += 1
        if time.monotonic() - self.last_gc > 1.:
            self.last_gc = time.monotonic()
            self.collect_expired()
        return handled

    def collect_expired(self):
        for directory, limit in ((self.mailbox.requests, MAX_REQUEST_BYTES), (self.mailbox.responses, MAX_RESPONSE_BYTES)):
            for name in directory.names():
                if not re.fullmatch('[0-9a-f]{32}\\.json', name): continue
                try:
                    packet = strict_json(directory.read(name, limit), limit)
                    expiry = packet.get('expires_at')
                    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry < time.time():
                        directory.unlink(name)
                except FileNotFoundError: pass
                except (ValueError, RecursionError, UnicodeError): directory.unlink(name)

    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vecnormalize', type=Path)
    parser.add_argument('--adapter-factory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--shared-root', type=Path, default=Path(os.environ.get('SHARED_AGENT_ROOT', DEFAULT_SHARED_ROOT)))
    parser.add_argument('--output', type=Path, default=ROOT/'results/endpoint.json')
    parser.add_argument('--poll-interval', type=float, default=.05)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_interval) or not 0 < args.poll_interval <= 10:
        parser.error('--poll-interval must be in (0, 10]')
    if args.output.resolve().is_relative_to(args.shared_root.resolve()):
        parser.error('Secret endpoint.json must be saved outside the public mailbox root')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from submission.inference2 import Policy
    policy = Policy(backend='checkpoint', model_path=args.model, vecnormalize_path=args.vecnormalize,
                    adapter_factory=args.adapter_factory, device=args.device)
    if not policy.ready: parser.error(policy.load_error or 'Checkpoint is not ready')
    stop = Event()
    def request_stop(*args): stop.set()
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    try:
        if args.output.exists():
            try: descriptor = read_descriptor(args.output)
            except ValueError:
                read_endpoint(args.output)  # Only migrate a valid legacy HTTP descriptor.
                descriptor = make_descriptor()
        else: descriptor = make_descriptor()
        with SharedWorker(args.shared_root, descriptor, policy) as worker:
            write_descriptor(args.output, descriptor)
            print(f'Upload this file to B: {args.output.resolve()}', flush=True)
            print(f'Shared agent ready: {descriptor["agent_id"]}; no IP or listening port required', flush=True)
            while not stop.is_set():
                worker.poll_once(); stop.wait(args.poll_interval)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__': raise SystemExit(main())
