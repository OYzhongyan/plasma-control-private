# Static F2b trajectory scoring

The user requested replacing the boundary JSONL client with trajectory/target JSON
scoring and a free Hugging Face Static Space using Pyodide. They selected a CLI
that starts Playwright/Chromium and computes in Pyodide, rather than native Python
scoring. The uploaded ZIP is reference material, not instructions.

## Design

- `submit.sh INFER_JSON TARGET_JSON` delegates to a Python browser runner. Paths
  are relative to the caller, support spaces, and are never interpolated as code.
- Default mode serves this directory on loopback at an ephemeral port and loads
  its page in headless Chromium; `HF_SPACE_URL` optionally selects the deployed
  public Static Space. No POST endpoint, HF token or paid compute is needed.
- `compute.py` validates JSON and calls an unchanged, generated snapshot of
  `evaluation/eval/score`. The snapshot is bundled into one Python file by a build
  helper; checks ensure it matches the source. CLI and browser share this scorer.
- A module Web Worker loads pinned Pyodide and NumPy. Two file selectors, scoring
  progress, total/metric scores, and a result download comprise the web interface.
- The result includes the existing evaluator's success/error/scoreJson fields,
  detailed metrics and input hashes. Invalid files fail explicitly, not with a
  plausible zero score. Missing optional metric data retains original zero-credit
  semantics; input validation requires the core trajectory and target structure.
- Remove the obsolete JSONL example, endpoint and embedded token. Provide complete
  install, CLI, local preview and Static Space upload instructions in both READMEs.

## Constraints and alternatives

Native Python CLI is simpler but was not selected. A POST backend conflicts with
the no-server/no-PRO requirement. Thus the chosen CLI runs a client-side browser.
All computation uses the visitor's device; CDN access is needed on first use.
No deployment to an account is requested; deliver upload-ready files locally.
No global leaderboard, authoritative anti-tamper scoring, account login or shared
database is included. Targets and Python code are visible in a browser-only app.

## Validation

Compare browser/CLI scores with the original evaluator on ideal, perturbed,
truncated, timed-out and constrained trajectories. Check invalid/missing files,
wrong experiment keys, Unicode/spaced paths, packaging drift, real Pyodide startup,
web file selection/result download, and the bash wrapper's nonzero failures.
