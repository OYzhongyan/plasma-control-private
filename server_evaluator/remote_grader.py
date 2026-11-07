"""Stage a connection descriptor, supervise C's simulation, report its scores."""
import hashlib
import math
from pathlib import Path
import sys
import tempfile

from submission.remote_protocol import canonical_origin, read_endpoint
from .grader import GradingUnavailable, InvalidSubmission
from .model_grader import ModelGrader


class RemoteGrader(ModelGrader):
    download_suffix = ".json"
    read_submission = staticmethod(read_endpoint)

    def __init__(self, target_path, data_dir, *, agent_timeout=15., agent_retries=2,
                 allowed_origins=(), **kwargs):
        if not math.isfinite(agent_timeout) or agent_timeout <= 0:
            raise ValueError("agent_timeout must be positive and finite")
        if type(agent_retries) is not int or not 0 <= agent_retries <= 5:
            raise ValueError("agent_retries must be in 0..5")
        remote = dict(timeout=float(agent_timeout), retries=agent_retries,
                      allowed_origins=sorted({canonical_origin(value) for value in allowed_origins}))
        # Parent also enforces an absolute RPC budget, including DNS / slow bodies.
        kwargs['inference_timeout'] = agent_timeout * (agent_retries + 1) + 5.
        super().__init__(target_path, data_dir, _remote_config=remote, **kwargs)

    def _command(self, endpoint, job_dir, config_path):
        return [sys.executable, '-m', 'server_evaluator.remote_rollout',
                '--endpoint-file', str(endpoint), '--config', str(config_path),
                '--trajectory', str(job_dir / 'trajectory.json'),
                '--result', str(job_dir / 'result.json')]

    def grade_file(self, path):
        self._check_configuration()
        path = Path(path).resolve()
        try:
            self.read_submission(path)
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            raise InvalidSubmission('Invalid remote endpoint JSON; regenerate endpoint.json on A') from exc
        except OSError as exc:
            raise GradingUnavailable('Cannot read downloaded endpoint JSON') from exc
        parent = self.data_dir / 'evaluations' / hashlib.sha256(str(path).encode()).hexdigest()
        if len(path.stem) == 64 and all(c in '0123456789abcdef' for c in path.stem):
            parent = self.data_dir / 'evaluations' / path.stem
        parent.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
        endpoint = job / 'endpoint.json'
        try:
            with endpoint.open('xb') as stream:
                endpoint.chmod(0o600)
                stream.write(path.read_bytes())
            return self._grade_staged(endpoint, job)
        finally:
            endpoint.unlink(missing_ok=True)
