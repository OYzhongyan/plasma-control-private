# Model submission grading implementation plan

**Goal:** Accept one exported model ZIP, run organizer-controlled HFM rollouts on a CPU worker, and report the existing seven metric contributions.

**Architecture:** Retain section 4.5 polling, SQLite caching and platform reporting. A bounded subprocess validates the model and uses trusted inference and simulator code with an organizer-owned target. Infrastructure failures remain pending for retry.

**Tech stack:** Python, ONNX Runtime CPU, NumPy, HFM socket runtime, existing scorer, pytest.

**Scope:** Current checkout is server/offline-grading; main is isolated in results/main-submission. Package format is delegated by the user: ONNX, JSON metadata and NPZ normalization in one ZIP. Never deserialize uploaded PPO/pickle or execute uploaded scripts. Do not submit to the real platform or push branches during tests.

- [x] Add shared bundle creation and bounded ZIP validation to both branches; test malformed archives and round trips.
- [x] Implement CPU inference and HFM rollout subprocess; test target consistency, close behavior and failures.
- [x] Integrate model grading, retries and cache fingerprints; test process timeouts and fake-platform reporting.
- [x] Replace main README final section with export, bundle and upload commands.
- [x] Update server dependencies, deployment docs and service template; run targeted verification.

Review focus: path traversal, oversized archives, ONNX external data, feature/stat dimensions, unavailable simulators, stale caches, cancellation, branch separation.

Verification: server/inference/bundle tests 93 passed, 4 optional SB3 tests skipped; main bundle tests 23 passed. Both worktrees pass git diff --check. No real platform submissions or score writes were performed. Real HFM audit is stored separately under results/model-grader-smoke.

Real offline verification completed: Windows CPU ONNX Runtime plus the full WSL HFM runtime, 205/500 valid steps before physical early termination, status completed, total score 0.03596237279729657/30. The temporary HFM process group was stopped and port 15560 verified closed. Artifact hashes and seven metric contributions are in results/model-grader-smoke/verification_summary.json. Linux address-space enforcement was unit-tested; the Windows smoke run did not apply RLIMIT_AS.
