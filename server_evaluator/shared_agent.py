"""C's authenticated shared-directory policy client; no contestant imports."""
from contextlib import ExitStack, contextmanager
import math
from pathlib import Path
import time
import uuid

import numpy as np

from submission.remote_protocol import PROTOCOL, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, checked_action
from submission.shared_channel import SHARED_PROTOCOL, MAX_TTL, validate_descriptor, encode_packet, decode_packet, digest
from submission.shared_files import Directory, Mailbox
from .grader import GradingUnavailable


class SharedAgentError(ValueError): pass


class SharedAgent:
    protocol = SHARED_PROTOCOL

    def __init__(self, descriptor, *, shared_root, timeout=15., retries=2, poll_interval=.05, on_request=None):
        validate_descriptor(descriptor)
        if (not math.isfinite(timeout) or timeout <= 0 or type(retries) is not int or not 0 <= retries <= 5
                or timeout*(retries+1)+5 > MAX_TTL or not math.isfinite(poll_interval) or not 0 < poll_interval <= 10):
            raise ValueError('Invalid shared agent timeout/retry/poll configuration')
        self.descriptor, self.timeout, self.retries = descriptor, timeout, retries
        self.poll_interval, self.on_request = poll_interval, on_request
        self.model_id = self.service_id = self.episode_id = None
        self.step_id = 0
        self.current_step = -3
        self.audit = dict(transport='shared_directory', request_count=0, retry_count=0)
        self.stack = ExitStack()
        try:
            with self._storage():
                try:
                    root = Directory(Path(shared_root).resolve()); root.close()
                except OSError as exc: raise GradingUnavailable('C shared root is unavailable; check mount and permissions') from exc
                try: self.mailbox = self.stack.enter_context(Mailbox(shared_root, descriptor['agent_id']))
                except (FileNotFoundError, ValueError) as exc:
                    raise SharedAgentError('Agent mailbox is missing or unsafe; start A with the same shared root') from exc
                except OSError as exc: raise GradingUnavailable('Cannot open shared mailbox; check filesystem and project-group permissions') from exc
                try: self.stack.enter_context(self.mailbox.lock('client.lock'))
                except RuntimeError as exc: raise GradingUnavailable('Shared agent already has an active C client') from exc
                except ValueError as exc: raise SharedAgentError('Agent lock file is unsafe') from exc
            response = self._request('health', {}, -2)
            self._check_response(response)
            self.model_id, self.service_id = response.get('model_id'), response.get('service_id')
            if not isinstance(self.service_id, str) or len(self.service_id) > 128:
                raise SharedAgentError('Agent did not return a service incarnation ID')
        except BaseException:
            self.close(); raise

    def _request(self, method, body, step):
        self.current_step = step
        if self.on_request: self.on_request(method, step)
        rid = uuid.uuid4().hex
        payload = dict(method=method, body=body)
        if method != 'health': payload['service_id'] = self.service_id
        raw = encode_packet(self.descriptor, rid, 'request', payload,
                            expires_at=time.time()+self.timeout*(self.retries+1)+5)
        if len(raw) > MAX_REQUEST_BYTES: raise GradingUnavailable('C observation exceeds the shared request size limit')
        name, request_hash = rid+'.json', digest(raw)
        try:
            for attempt in range(self.retries+1):
                if attempt: self.audit['retry_count'] += 1
                self.audit['request_count'] += 1
                with self._storage(): self.mailbox.requests.write(name, raw)
                deadline = time.monotonic()+self.timeout
                while time.monotonic() < deadline:
                    try:
                        with self._storage(): reply = self.mailbox.responses.read(name, MAX_RESPONSE_BYTES)
                        packet = decode_packet(reply, self.descriptor, 'response', request_id=rid,
                                               request_digest=request_hash, maximum=MAX_RESPONSE_BYTES)
                        return packet['payload']
                    except (FileNotFoundError, ValueError, RecursionError, UnicodeError):
                        # Unauthenticated files cannot force acceptance or a policy reset.
                        time.sleep(min(self.poll_interval, max(0., deadline-time.monotonic())))
            raise SharedAgentError(f'A {method} timed out waiting for a valid signed response')
        except OSError as exc:
            raise GradingUnavailable('Shared storage I/O failed; submission remains pending') from exc
        finally:
            try:
                with self._storage():
                    self.mailbox.requests.unlink(name)
                    self.mailbox.responses.unlink(name)
            except OSError as exc:
                raise GradingUnavailable('Cannot clean shared message files; check storage') from exc

    def _check_response(self, value, *, step=None):
        if self.service_id is not None and value.get('service_id') != self.service_id:
            raise SharedAgentError('A service incarnation changed during evaluation')
        if value.get('ok') is not True or value.get('protocol') != PROTOCOL:
            raise SharedAgentError('A policy request failed; inspect A service logs')
        if self.model_id is not None and value.get('model_id') != self.model_id:
            raise SharedAgentError('A model identity changed during evaluation')
        if self.episode_id is not None and value.get('episode_id') != self.episode_id:
            raise SharedAgentError('A response has wrong episode ID')
        if step is not None and (type(value.get('step_id')) is not int or value['step_id'] != step):
            raise SharedAgentError('A response has wrong step ID')

    def reset(self):
        self.episode_id, self.step_id = uuid.uuid4().hex, 0
        value = self._request('reset', dict(protocol=PROTOCOL, episode_id=self.episode_id), -1)
        self._check_response(value)

    def act(self, observation):
        if self.episode_id is None: raise SharedAgentError('Reset required before act')
        value = self._request('act', dict(protocol=PROTOCOL, episode_id=self.episode_id,
            step_id=self.step_id, observation=observation), self.step_id)
        self._check_response(value, step=self.step_id)
        try:
            action = np.asarray(checked_action(value.get('action')), dtype=np.float32)
            if not np.all(np.isfinite(action)): raise ValueError('Overflowing voltage')
        except (ValueError, TypeError, OverflowError) as exc:
            raise SharedAgentError('A must return 12 finite physical voltages') from exc
        self.step_id += 1
        return action

    @contextmanager
    def _storage(self):
        if self.on_request: self.on_request('storage', self.current_step)
        try: yield
        finally:
            if self.on_request: self.on_request('wait', self.current_step)

    def close(self):
        with self._storage(): self.stack.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
