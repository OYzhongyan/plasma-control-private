# Restore history baseline and run 10m: implementation plan

> Execute inline using superpowers:executing-plans. This task and push/training launch are already explicitly authorized by the user; no extra worktree is created.

**Goal:** Restore SII-585-THU's F2 history inputs and dense reward, publish main, and start a recorded full baseline on Inspire.
**Spec:** experiments/20261109_f2b_history_baseline_10m_seed0/DESIGN.md
**Stack:** Python, Gymnasium, SB3 PPO, Hydra, HFM MATLAB socket pool, Inspire Jupyter SDK.

## Tasks
- [x] T0: Read both READMEs; capture source commit/file snapshots and live resource/runtime inventory. Source is clean except a deleted screenshot. Container quota is 54 CPU/300 GiB; existing 32 HFM sockets have no clients.
- [x] T1: RED tests for history features/state, adapters/metadata and reward; restore F2 components and synchronized consumers. Verify original-source reward parity and train/inference sequence parity.
- [x] T2: Restore reward CSV writes, mandatory CSV/TensorBoard, correct rollout checkpoint cadence and resume-step flag. Test meaningful local tiny PPO logging.
- [x] T3: Experiment runner records config/code/environment/status, runs training then seen/held-out validation, saves summaries/plots. Test bookkeeping and failure handling without HFM. Keep owned files disjoint if delegated.
- [x] T4: Update README, record upstream provenance, run relevant tests/review, commit and push main to OYzhongyan/plasma-control-private.
- [ ] T5: Create a new clone in the requested remote parent; install missing packages in competition_env without proxies/SSH; preserve existing HFM. Short real preflight then detached 10m run. Record PID, commit, start/update/checkpoint evidence and reproduction commands.

## Review focus
History reset and first-action padding must match upstream; physical 12D history must not double-accumulate delta actions; metadata and checkpoint stats must match; dense reward is not official score; training errors must not be reported as success; logs survive W&B failures; independent services must not be killed; archive/token/private target files must not be committed.

## Ledger
- User correction: restore upstream reward too, superseding the initial proposal to preserve competition reward as training objective.
- Source snapshot is saved privately under ignored results/baseline_restore; commit and hashes will be recorded in public provenance without including private keys or runtime artifacts.
- Ruling: preserve the 90D interface only for loading existing artifacts, while all new baseline defaults use history. This avoids breaking existing remote submission fixtures and does not weaken new artifact validation.

- T1/T2/T3 release verification: 117 passed, 4 skipped (Windows/platform conditions), 35 warnings in 70.03s. Real tiny PPO logger and 812D checkpoint/ONNX parity included. Reward oracle matches 8 sequential frames exactly.
- Independent read-only review: no Critical/Important blockers; wrapper AST matches upstream. Seed routing narrow review confirmed config without seed preserves prior behavior and no seed enters MATLAB init.
- Ruling: evaluation includes upstream README sensor noise (Ip 2300 A, R .008 m) in addition to delay; explicitly label this stress test. Single fixed target only, no multi-target reference selection in this baseline.
- Reproducibility repair: per-env power-supply RNG receives env seed +30000; unchanged delay distribution, continuous stream across resets. Test observed RED then GREEN.

- T4 published main b17fad608fd48fcc8822d7899f1408fd43aad6c2; HTTPS clone created at /inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-baseline-20261109. Missing scipy 1.17.1 installed without upgrading existing torch/SB3.
- T5 real preflight: 1024 requested/actual steps on two HFM environments; both PPO rollout collection and actor/critic updates completed; final model and matching normalization saved. Runtime HFM timeout count 0.
- Found/fixed in real preflight: saved configs.pkl contains save_dir, rejected by Hydra struct during evaluation merge. Reproduced RED with saved runtime field, fixed by merging into an unstructured deep copy; 49 affected tests GREEN. Preserve initial failure log. Use the existing short model for three 16-step/one-episode scoring probes; these are pipeline verification, not benchmark scores. Full baseline remains 10m with final 500-step/3-episode evaluations.
