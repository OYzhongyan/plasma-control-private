"""Grade a private descriptor through a trusted, administrator-chosen mailbox root."""
import math
from pathlib import Path
import sys

from submission.shared_channel import read_descriptor, DEFAULT_SHARED_ROOT, MAX_TTL
from .model_grader import ModelGrader
from .remote_grader import RemoteGrader


class SharedGrader(RemoteGrader):
    read_submission = staticmethod(read_descriptor)

    def __init__(self, target_path, data_dir, *, shared_root=DEFAULT_SHARED_ROOT, agent_timeout=15.,
                 agent_retries=2, agent_poll_interval=.05, **kwargs):
        if (not math.isfinite(agent_timeout) or agent_timeout <= 0 or type(agent_retries) is not int
                or not 0 <= agent_retries <= 5 or agent_timeout*(agent_retries+1)+5 > MAX_TTL
                or not math.isfinite(agent_poll_interval) or not 0 < agent_poll_interval <= 10):
            raise ValueError('Invalid shared agent timeout/retries/poll interval')
        remote = dict(transport='shared_directory', shared_root=str(Path(shared_root).resolve()),
                      timeout=agent_timeout, retries=agent_retries, poll_interval=agent_poll_interval)
        kwargs['inference_timeout'] = agent_timeout*(agent_retries+1)+5
        ModelGrader.__init__(self, target_path, data_dir, _remote_config=remote, **kwargs)

    def _command(self, endpoint, job_dir, config_path):
        return [sys.executable, '-m', 'server_evaluator.shared_rollout', '--endpoint-file', str(endpoint),
                '--config', str(config_path), '--trajectory', str(job_dir/'trajectory.json'),
                '--result', str(job_dir/'result.json')]
