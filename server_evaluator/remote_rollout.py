"""C owns HFM and scoring; A only receives observations and returns voltages."""
import argparse
from pathlib import Path

from submission.remote_protocol import read_endpoint
from .grader import GradingUnavailable, InvalidSubmission
from .model_rollout import (_run_model, _phase, _read_json, _apply_memory_limit,
                            _atomic_json, _memory_failure)
from .remote_agent import RemoteAgent, RemoteAgentError


def run_remote(endpoint_file, config, trajectory_path):
    try:
        try:
            endpoint = read_endpoint(endpoint_file)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise InvalidSubmission('Invalid remote endpoint JSON') from exc
        settings = config['remote_agent']
        agent = RemoteAgent(endpoint, **settings,
            on_request=lambda path, step: _phase(config, 'inference', step))
        return _run_model(None, config, trajectory_path, policy=agent)
    except RemoteAgentError as exc:
        raise InvalidSubmission(str(exc)) from exc
    except Exception as exc:
        if _memory_failure(exc):
            raise InvalidSubmission('Remote evaluation exceeded the child memory limit') from exc
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-file', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = _read_json(args.config)
        config['_progress_path'] = str(args.result.parent / 'phase.json')
        _apply_memory_limit(config)
        result = run_remote(args.endpoint_file, config, args.trajectory)
        envelope = dict(kind='completed', result=result)
    except InvalidSubmission as exc:
        envelope = dict(kind='invalid', error=str(exc))
    except MemoryError:
        envelope = dict(kind='invalid', error='Remote evaluation exceeded the child memory limit')
    except Exception as exc:
        envelope = dict(kind='unavailable', error=f'{type(exc).__name__}: {exc}')
    _atomic_json(args.result, envelope)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
