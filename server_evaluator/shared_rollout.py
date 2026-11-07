"""C-owned HFM and scoring with signed shared-file policy inference on A."""
from submission.shared_channel import read_descriptor
from .grader import InvalidSubmission
from .model_rollout import _run_model, _phase, _memory_failure
from .shared_agent import SharedAgent, SharedAgentError


def run_shared(endpoint_file, config, trajectory_path):
    try:
        try: descriptor = read_descriptor(endpoint_file)
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            raise InvalidSubmission('Invalid shared-directory endpoint JSON; restart A to regenerate it') from exc
        settings = {k: v for k, v in config['remote_agent'].items() if k != 'transport'}
        with SharedAgent(descriptor, **settings,
                         on_request=lambda method, step: _phase(config, 'shared_io' if method == 'storage' else 'inference', step)) as agent:
            result = _run_model(None, config, trajectory_path, policy=agent)
        _phase(config, 'scoring')
        return result
    except SharedAgentError as exc: raise InvalidSubmission(str(exc)) from exc
    except Exception as exc:
        if _memory_failure(exc): raise InvalidSubmission('Shared evaluation exceeded child memory limit') from exc
        raise


def main(argv=None):
    from .remote_rollout import main as run_child
    return run_child(argv, runner=run_shared)


if __name__ == '__main__': raise SystemExit(main())
