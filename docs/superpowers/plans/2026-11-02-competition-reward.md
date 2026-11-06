# Competition reward implementation plan

> Execute inline with superpowers:executing-plans and test-driven-development.

**Goal:** Per-step raw reward contributions sum to the original F2b score.
**Spec:** ../specs/2026-11-02-competition-reward-design.md
**Stack:** Python, NumPy, existing official scorer and trajectory conversion.

## Constraints and review focus

No official scoring formula changes. No artificial termination/negative shaping.
Check step/reference indexing, strict > coil bounds, persistent masks, early reset,
end-of-episode invalidation, missing timings, horizons other than 500 and target
alignment. Preserve user changes and the separate server branch.

- [x] Write parity and lifecycle tests; observe failure against current reward.
- [x] Implement streaming reward using official functions and converter.
- [x] Integrate training/evaluation/reset/logs and simplify default config.
- [x] Document target selection, 30-point scale and PPO transformation limits.
- [x] Verify regression tests and saved-trajectory parity; review changes.

## Evidence

2026-11-02: the configured test venv ran competition reward, training references,
single-experiment, F2b cleanup, historical submission/export and XPT utility tests:
56 passed, 5 existing Torch ONNX exporter warnings, no skips. Reward tests include
a 16-step PPO update via the real training environment factory and a replay of
saved HFM observations matched against the saved trajectory's official score.
Perfect episodes sum to 30; coil equality does not trip the fuse; timeout/timing
invalidations reverse previous credit; reset and parallel environments isolate
state. Original scorer functions were not modified. Python 3.9 syntax and git
whitespace checks pass. This change did not rerun live MATLAB/HFM training.

Ruling: include eval_files and the evaluation score package in Python packaging,
because the new reward imports their shared converter/scorer. No copies of their
formulas are introduced. Preserve original class-name/constructor compatibility
while removing configurable shaping from the default Hydra reward node.
