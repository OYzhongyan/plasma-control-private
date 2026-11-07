"""Optional lifecycle of an organizer-owned local Linux MATLAB runtime."""
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Event
import time

from .grader import GradingUnavailable


def _ensure_free_port(host, port):
    # Bind rather than connect: an HFM accept must not consume a bogus client.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as exc:
            raise GradingUnavailable('Managed HFM port is occupied; use an existing service without --hfm-runtime-dir') from exc


class ManagedHFM:
    def __init__(self, runtime_dir, data_dir, *, host='127.0.0.1', port=5558,
                 startup_timeout=180., stop_event=None):
        if host not in {'localhost', '127.0.0.1'}:
            raise ValueError('Managed HFM requires a local IPv4 loopback host')
        if not 1 <= port <= 65535 or not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('Invalid managed HFM port/startup timeout')
        self.runtime_dir = Path(runtime_dir).resolve()
        self.log_path = Path(data_dir).resolve() / 'hfm-runtime.log'
        self.port, self.startup_timeout = port, startup_timeout
        self.stop_event = stop_event if stop_event is not None else Event()
        self.process = self.log = None

    def start(self):
        if not sys.platform.startswith('linux'):
            raise GradingUnavailable('--hfm-runtime-dir requires Linux or WSL')
        script = self.runtime_dir / 'run_hfm_server.sh'
        if not script.is_file():
            raise GradingUnavailable('C HFM runtime is missing run_hfm_server.sh')
        _ensure_free_port('127.0.0.1', self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open('wb')
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith('PLATFORM_')}
        env.update(LISTEN_PORT=str(self.port), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            self.process = subprocess.Popen(['bash', str(script)], cwd=self.runtime_dir,
                env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            while not self.stop_event.is_set():
                if self.process.poll() is not None:
                    raise GradingUnavailable(f'HFM exited during startup; inspect {self.log_path}')
                # No TCP readiness probe: it would consume HFM's client accept.
                with self.log_path.open('rb') as stream:
                    stream.seek(max(0, self.log_path.stat().st_size - 65536))
                    ready = b'Waiting for a client to connect' in stream.read(65536)
                if ready:
                    return
                if time.monotonic() >= deadline:
                    raise GradingUnavailable(f'HFM startup timed out; inspect {self.log_path}')
                self.stop_event.wait(.2)
            raise GradingUnavailable('Worker stopped during HFM startup')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _stop_process(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def close(self):
        if self.process is not None:
            self._stop_process(self.process)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
