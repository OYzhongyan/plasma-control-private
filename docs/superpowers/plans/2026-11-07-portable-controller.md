# Portable ONNX controller implementation plan

> Execute inline with superpowers:executing-plans. Use the root checkout; the user declined worktrees. Save branch-specific local commits; do not push.

**Goal:** Submit one data-only ZIP containing a complete controller, so grading never imports contestant training code.

**Spec:** ../specs/2026-11-07-portable-controller-design.md, amended by the user's approval of requested observation fields and optional explicit state.

**Architecture:** controller.json declares simulator field names/shapes and optional state. ONNX embeds feature extraction, normalization, actor and physical action conversion. Runtime only transports named fields/state; organizer retains simulator, power supply and scoring.

**Tech stack:** Torch/SB3 locally; NumPy, ONNX/ORT CPU and Python standard library at runtime.

## Constraints and review focus

- ZIP exactly controller.onnx + controller.json; preserve size/path/security limits.
- Inputs are requested raw simulator fields, not a required 4803D vector. State is optional, reset to zero and updated only after successful inference.
- Output is 12 finite physical voltage setpoints before organizer's power supply model.
- Server cannot import train; custom factories run only on contestant's export machine.
- Default controller must match original PPO + frozen VecNormalize + action wrappers in both modes over multiple steps and resets.
- Preserve unrelated untracked SII-585-THU directory, existing scoring, HFM configuration, delays, resource limits and platform semantics.

## Tasks

- [x] T1: Write failing contract/runtime/package tests. Implement independent field schema, manifest validation, data-only two-file bundle and generic ONNX runtime. Remove runtime imports of train.
- [x] T2: Write failing export parity/custom-controller tests. Implement local Torch controller factory and full export, embed frozen statistics and state, verify ORT parity. Move training metadata helpers into train.
- [x] T3: Update main submission docs, Docker runtime and dependent tests. Validate and save a scoped main commit; preserve user files.
- [x] T4: Transfer common changes to server/offline-grading without a worktree. Update model validation/supervision/fingerprints, new cache default and administrator docs. Test malformed packages and real export plus fake-HFM scoring with train imports blocked.
- [x] T5: Request one independent final code review, fix important findings through regression tests, verify both branches, save server commit and report limits and commit IDs.

## Execution ledger

- Design ruling: follow the user's latest approval: named observation inputs and optional state replace the initial fixed 4803D proposal.

- T1/T2 verification: 68 focused tests passed, including both action modes, non-identity normalization, reset, custom field/state factories, archive hardening and subprocess train-import prohibition.
- Ruling: state_dtype supports float32/float64; baseline delta uses float64 to match the existing NumPy accumulator.
- Main had a deleted sensor helper still imported by evaluation; restored its existing server behavior in a training-only sensor_perturbation module, keeping it outside submission.
- Wider legacy regression check: 31 passed, 7 failed because main already lacks train.features and train.legacy_train_ppo; leave obsolete-tool cleanup outside this task.

- T3 main implementation saved as fb6ba23; 89 focused tests passed.
- T4 server integration: 161 passed, 1 skipped, including full train-import-prohibited fake-HFM grading. HFM/official score/shot/delay source unchanged.
- Final independent review found one direct-script import regression; reproduced red with empty PYTHONPATH, restored root bootstrap, all 22 runtime tests green. Reviewer confirmed no remaining important findings.

- Final post-review server suite: 162 passed, 1 skipped, 37 warnings. Main-specific service startup fix backported; final main suite: 90 passed, 32 warnings. No live HFM or production platform requests made.
