# C grader watchdog

The user requests a new script that automatically restarts C after a live code/configuration-change warning. HFM is an independently running service and MUST NOT be launched, signalled or stopped by this script.

## Approach

A standard-library parent process launches the existing `server_evaluator/run.py` with unbuffered stdout/stderr, mirrors its output to the terminal and a persistent per-generation worker log, and detects the trusted configuration-change warning or startup fingerprint-mismatch error. It gracefully stops the old worker, drains/cleans its own evaluation subprocess group, waits five seconds, creates a fresh grading data directory and starts a new worker. It does not edit code, targets, credentials, SQLite fingerprints or score records.

Use `server_evaluator/watchdog.py --data-dir ROOT <existing grading arguments>`. ROOT is a stable supervisor directory containing `active.json`, a supervisor lock/log, and `runs/run-*` generations. Normal supervisor restarts resume the last generation; configuration-triggered restarts rotate generations. Old generations remain available. Pending tasks are fetched from B again; already completed platform tasks are not intentionally regraded. Old unreported cached results remain in old directories for audit; pending tasks can be reevaluated under the new configuration.

Only known configuration/fingerprint messages trigger automatic rotation, not contestant failures or ordinary platform/HFM deferrals. Other process exits are surfaced instead of causing an infinite restart loop. Reject managed-HFM and one-shot arguments before starting any child, including abbreviations accepted by run.py. Forward existing platform, target, shot, seed, noise, supply and polling arguments unchanged; platform credentials stay in the inherited private environment, not state files.

Linux supervision owns a dedicated process group. Evaluation children stay in that group only when launched under this watchdog; normal direct run.py retains its existing behavior. Stop the worker gracefully and clean only the owned group, never a pre-existing HFM process. Ctrl+C/SIGTERM stops supervision without restarting. A per-root OS lock prevents two watchdog instances using the same state. Use reliable local persistent storage for ROOT.

## Verification

Real fake-worker subprocess tests cover warning rotation, startup mismatch, resume, ordinary deferrals, unrelated failure, forwarded flags, logs, cancellation and HFM rejection. A Linux standard-library smoke test validates descendant cleanup while an external HFM stand-in remains alive. No production platform, shared mailbox or real HFM is touched.
