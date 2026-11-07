"""Restart only the C grader on configuration invalidation; external HFM stays owned by its operator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from threading import Event, Thread
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from server_evaluator.store import instance_lock

SUPERVISED_ENV = 'PLASMA_GRADER_SUPERVISED'


def restart_reason(line):
    prefix = r'(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} )?'
    line = line.rstrip('\r\n')
    change = 'Server code/configuration changed; restart with a new data directory'
    mismatch = 'Data directory fingerprint differs (target/rules/metric/platform/problem changed); use a new data directory'
    if re.fullmatch(prefix + r'WARNING submission=\S+ evaluation deferred: ' + re.escape(change), line):
        return 'Server code/configuration changed'
    if re.fullmatch(prefix + r'ERROR Worker stopped: ' + re.escape(mismatch), line):
        return 'Data directory fingerprint differs'
    return None


class Supervisor:
    def __init__(self, data_dir, worker_args, *, restart_delay=5., shutdown_timeout=45.,
                 worker_command=None, stop_event=None):
        if not math.isfinite(restart_delay) or restart_delay < 0:
            raise ValueError('restart_delay must be nonnegative and finite')
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError('shutdown_timeout must be positive and finite')
        self.root = Path(data_dir).resolve()
        self.invocation_dir = Path.cwd()
        self.runs = self.root/'runs'
        self.worker_args = list(worker_args)
        self.worker_command = list(worker_command or [sys.executable, '-u', str(ROOT/'server_evaluator/run.py')])
        self.restart_delay, self.shutdown_timeout = restart_delay, shutdown_timeout
        self.stop = stop_event if stop_event is not None else Event()

    def _new_run(self):
        return Path(tempfile.mkdtemp(prefix='run-', dir=self.runs))

    def _resume(self):
        state = self.root/'active.json'
        if not state.exists(): return self._new_run()
        try:
            value = json.loads(state.read_text(encoding='utf-8'))
            name = value['active_run']
            if value.get('version') != 1 or not isinstance(name, str) or not re.fullmatch(r'run-[A-Za-z0-9_-]+', name):
                raise ValueError('Invalid run name')
            path = self.runs/name
            if path.is_symlink() or not path.is_dir() or path.resolve().parent != self.runs.resolve():
                raise ValueError('Invalid run directory')
            return path
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('Invalid watchdog active state; preserve history and repair active.json') from exc

    def _state(self, current, status, pid=None, exit_code=None):
        value = dict(version=1, active_run=current.name, active_data_dir=str(current),
                     status=status, worker_pid=pid, exit_code=exit_code,
                     updated_at=datetime.now(timezone.utc).isoformat())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root, prefix='.active-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, indent=2); stream.write('\n')
            temporary.chmod(0o600)
            os.replace(temporary, self.root/'active.json')
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

    def _stop_worker(self, process):
        if os.name == 'nt':
            if process.poll() is None:
                # Windows local tests: stop only this child process tree.
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                if process.poll() is None: process.kill()
        else:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired: pass
            # The worker and its evaluation children share our dedicated group.
            # HFM is external and was never started in this group.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        process.wait(timeout=10)

    def _run_worker(self, current):
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        environment[SUPERVISED_ENV] = '1'
        options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
                   else {'start_new_session': True})
        command = [*self.worker_command, '--data-dir', str(current), *self.worker_args]
        process = subprocess.Popen(command, cwd=self.invocation_dir, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', **options)
        reason = []; reader_errors = []
        try:
            self._state(current, 'running', process.pid)
            self.log.info('C worker started pid=%s data_dir=%s', process.pid, current)
            def read_output():
                try:
                    with (current/'worker.log').open('a', encoding='utf-8') as output:
                        for line in process.stdout:
                            output.write(line); output.flush()
                            sys.stdout.write(line); sys.stdout.flush()
                            detected = restart_reason(line)
                            if detected and not reason: reason.append(detected)
                except Exception as exc:
                    reader_errors.append(exc); self.stop.set()
            reader = Thread(target=read_output, daemon=True)
            reader.start()
            while process.poll() is None and not self.stop.is_set() and not reason:
                self.stop.wait(.1)
        finally:
            self._stop_worker(process)
            if 'reader' in locals():
                reader.join(timeout=10)
                if reader.is_alive(): raise RuntimeError('Worker log reader did not stop')
            if process.stdout: process.stdout.close()
        if reader_errors: raise RuntimeError('Cannot preserve worker output; check watchdog storage') from reader_errors[0]
        return process.returncode, reason[0] if reason else None

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.runs.is_symlink(): raise ValueError('Watchdog runs directory must not be a symlink')
        self.runs.mkdir(exist_ok=True, mode=0o700)
        with instance_lock(self.root/'supervisor-lock'):
            self.log = logging.getLogger('grader.watchdog.'+uuid.uuid4().hex)
            self.log.setLevel(logging.INFO); self.log.propagate = False
            handlers = [logging.FileHandler(self.root/'supervisor.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)]
            for handler in handlers:
                handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                self.log.addHandler(handler)
            current = None
            try:
                current = self._resume()
                while not self.stop.is_set():
                    code, reason = self._run_worker(current)
                    if self.stop.is_set():
                        self._state(current, 'stopped', exit_code=code)
                        self.log.info('Watchdog stopped; external HFM remains unmanaged')
                        return 0
                    if reason:
                        self._state(current, 'restarting', exit_code=code)
                        self.log.warning('Restart requested: %s; preserve old data_dir=%s', reason, current)
                        if self.stop.wait(self.restart_delay):
                            self._state(current, 'stopped', exit_code=code)
                            return 0
                        current = self._new_run()
                        continue
                    self._state(current, 'stopped' if code == 0 else 'failed', exit_code=code)
                    self.log.info('C exited code=%s without configuration invalidation; no automatic restart', code)
                    return code if code >= 0 else 1
                return 0
            finally:
                for handler in handlers:
                    self.log.removeHandler(handler); handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'server_evaluator/runtime-watchdog',
                        help='Stable supervisor root; effective grader caches are runs/run-*')
    parser.add_argument('--restart-delay', type=float, default=5.)
    parser.add_argument('--shutdown-timeout', type=float, default=45.)
    args, forwarded = parser.parse_known_args(argv)
    from server_evaluator.run import build_parser
    worker = build_parser().parse_args(forwarded)
    if worker.hfm_runtime_dir is not None:
        parser.error('HFM is external: do not pass --hfm-runtime-dir to watchdog')
    if worker.once or worker.check_config or worker.endpoint_file or worker.model_file:
        parser.error('Watchdog requires continuous platform polling, not one-shot evaluation')
    if worker.data_dir is not None:
        parser.error('Pass --data-dir exactly; do not abbreviate the supervisor data directory option')
    stop = Event()
    def request_stop(*args): stop.set()
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    try:
        return Supervisor(args.data_dir, forwarded, restart_delay=args.restart_delay,
                          shutdown_timeout=args.shutdown_timeout, stop_event=stop).run()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Watchdog stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
