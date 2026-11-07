# Grading reference interface synchronization

> Execute inline with superpowers:executing-plans; keep the existing root checkout, as requested. Do not commit or push automatically.

**Goal:** Accept models exported by main (`4e73010`) with the identical observation encoding, metadata, actions, and power supply dynamics.

**Architecture:** Copy the shared reference wrapper/adapter and exporter from main, preserving server ONNX CPU session options and existing grading safeguards. Use main's physical supply delays (PF 2–5 ms; VS 0–1 ms), independently of observation perturbations.

**Tech stack:** Python, Gymnasium, NumPy, ONNX Runtime CPU, pytest.

**Spec:** User request: synchronize main observation encoding, metadata and action conversion to server/offline-grading and unify supply delays. Only current reference models are supported.

## Tasks

- [x] Add regression tests accepting strict 90D reference metadata, rejecting malformed/legacy layouts, comparing action conversion and seeded supply responses; run them before changing production code.
- [x] Synchronize shared reference components and update server metadata validation. Preserve server CPU limits, security checks, reference shots and scoring.
- [x] Update dependent training/export/evaluation imports without retaining historical observation support; verify the real exported model bundle loads and scores against a simulated socket boundary.
- [x] Document the input order, action units/mapping, physical delays, seed policy, new grading cache and handling already reported failures.
- [x] Run focused tests, review the changes and report validation limits.

## Review focus

- Required metadata types/order/normalization must be validated, rather than relabeling legacy weights.
- Absolute and delta actions must match main, including clipping, accumulation and reset.
- Identical seed and command sequences must produce identical delayed supply responses.
- The server must keep CPU session limits and artifact/security validation.
- An updated grader must not reuse cached scores from the old interface/dynamics.

## Validation

- Final focused suite: 118 passed, 1 skipped (bundled historical export), 35 Torch export warnings.
- Real PPO actor export ZIP, frozen non-identity normalization, safe unpack, CPU ONNX load and fake-HFM rollout/scoring verified.
- AST/source parity with main verified for reference/action wrappers, metadata/adapter, power supply and exporter; inference differs only in CPU session configuration.
- Independent review completed; stale external shot-catalogue startup regression fixed with a red/green CLI regression.
- No live HFM or production platform requests in this task. Changes remain uncommitted on server/offline-grading.
