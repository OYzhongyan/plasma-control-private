# Reference-only observation implementation plan

**Goal:** Actor and Critic consume the same current 90-dimensional target vector, with no observation or action history.

**Contract selected by the user:** `reference_Ip` (1), `reference_lX` (1), `reference_lcfs_points` (32×2), `reference_rX` (4), `reference_zX` (4), `reference_strike_r` (8), `reference_strike_z` (8), in that order. Other reference fields, validity masks and counts are outside this selected 90-dimensional contract.

**Architecture:** The existing `HistoricalOBservationWrapper` name becomes the current-reference wrapper. New training metadata uses `reference_only_v1`; inference and the server accept this contract only. Per the user's subsequent correction, remove the former historical wrapper/metadata/adapter instead of retaining old-checkpoint support. Action scaling and delta accumulation remain unchanged.

**Execution:** Apply the training-side change on `main` in the project root, per the user's correction. Do not create linked worktrees, modify server branch history, commit, or push. Preserve main's competition reward, configured training duration, and automatic ZIP packaging.

1. Add regression tests for reference order/dimension, reference-only dictionaries, history independence, changing targets, shape errors and metadata.
2. Implement the reference wrapper and remove the historical encoding/adapter.
3. Connect main's training, evaluation, ONNX export and inference; test real PPO actor/critic input, ONNX parity and automatic ZIP packaging.
4. Document the new contract and retraining requirement. Run focused regression checks and report unrelated existing failures separately.

Old 812-dimensional model weights and normalization statistics cannot become 90-dimensional weights by changing metadata. New training starts a new run; old artifacts are rejected.

## Verification and review

- The implementation was initially placed on `server/offline-grading`, then included in user commit `7e89f2e`. The user clarified that the training change belongs on `main`; it was ported with a three-way merge in the root checkout. Existing server commits and deployment files remain intact.
- Independent review found that sensor delay also delayed references, and that evaluation ignored required/disabled normalization. Added regression tests and fixed both; evaluation also restores the model's absolute/delta action mode from metadata.
- Export and inference validate the complete ordered reference contract, not only its dimension.
- Full `pytest test server_evaluator/tests`: 179 passed, 1 skipped, 1 existing failure (`test_power_supply.py::test_reset_resamples_delay`, default delays are fixed at zero). New PPO/ONNX tests use a small real CPU model; server integration uses an HFM test double. No new real HFM experiment was run.
- Main migration verification: reference input, export, ZIP packaging, normalization and submission inference tests: 53 passed. Full main `pytest test`: 107 passed, 2 skipped, 28 existing failures (27 Static scoring tests refer to absent `platformtohf-main` files; one power-supply delay test). Main's reward implementation and training duration were preserved. No server branch commits were rewritten and no new commit/push was made.
