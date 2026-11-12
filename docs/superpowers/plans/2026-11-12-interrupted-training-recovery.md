# Interrupted baseline recovery implementation plan

> Execute inline using superpowers:executing-plans and test-driven-development. User has authorized checking, restarting both existing experiments and publishing current code. Keep main in the root workspace; no worktrees.

**Goal:** Recover the two interrupted 10m experiments and preserve inspectable local curves.

**Architecture:** Restore the newest valid checkpoint and its matching VecNormalize, retaining cumulative steps and the original global learning-rate horizon. Preserve old training files and place the continuation in a distinct segment of the same experiment. Restart an isolated 32-port HFM pool in each recreated Notebook; never use shared cache/log paths across instances.

**Stack:** Existing PPO/Hydra/SB3 2.8.0/PyTorch 2.10.0, Inspire Jupyter only, TensorBoard and W&B offline.

**Spec:** User request on 2026-11-12 and original experiment DESIGN.md files. Cumulative target remains 10,000,000, rounded to full 8192-transition rollouts; model, optimizer, normalization, actor/critic interface, dense reward, shots, perturbations, seeds and step-500 bootstrap unchanged. Simulator/RNG trajectories cannot be exactly restored because they were not checkpointed. Latest CNN saved state is 901120 although logs reached925696.

## Tasks
- [x] Audit shared checkpoints, progress, TensorBoard/W&B files, new Notebook status and matching environment packages.
- [x] Add optional absolute resume target with RED→GREEN tests. Existing additional-step semantics remain the default. New mode computes remaining learn budget while retaining global schedule horizon.
- [x] Publish tested minimal recovery support, update both remote clones, verify unchanged physical/feature/reward sources.
- [x] Restart each Notebook's HFM services with separate cache/logs. Validate actual checkpoint, optimizer and normalization loading.
- [x] Launch detached recorded continuation segments, auto-evaluate same three splits at completion, check 32 STEP streams and new local event files.
- [x] Read TensorBoard scalar samples, summarize old progress and latest recoverable states, archive reproducibility records without smoke artifacts, and push records.

## Review focus
Do not add another10m steps to a saved model. Do not reset warmup/cosine decay. Never overwrite old CSV/event/checkpoint files. Do not conflate training scalar competition_score with formal held-out30-point scores. Detect already-running recovery before duplicate launch. Do not reveal credentials or sync W&B externally without a request.

## Ledger

- RED: missing budget helper, then GREEN:29 relevant regression tests. Fresh read-only review found no Critical/Important defects; minor early-validation gap was reproduced with a failing entrypoint test and fixed. Final budget suite:8 passed, including actual PPO save/load/continuation to a global target. Tests use temporary directories deleted after completion.
- New absolute-target CLI uses `+resume.target_total_timesteps`, keeping the base config unchanged so archived baseline comparisons remain exact. No actor, critic, feature encoding, reward, shot, power-supply or simulator implementation changes are made.
- Recovery source94c517e published and both clones fast-forwarded. Older generated experiment/root README records were archived before update; byte-identical untracked documentation was adopted from Git. Physical, wrapper, dense reward, references and formal scorer hashes match original manifests. Independent HFM caches now live in each container's /tmp; stdout is under each continuation training directory.
- Both actual checkpoints, optimizer states (13 baseline /23 CNN entries) and VecNormalize statistics loaded successfully. New HFM pools each have32 progressing STEP streams. Baseline resumed 2026-11-12 14:39:29 Beijing; CNN resumed14:39:05. First new PPO dump reached2220032 and909312 respectively; new TensorBoard/W&B files contain data. Old TensorBoard files contain36 scalar tags each,270 baseline /113 CNN reward samples. W&B remains offline.
- EventAccumulator readback confirms new event-file scalar steps2220032 and909312; both new offline W&B files grew from headers to32768 bytes. Original and recovery CSV/events stay separate. Initial PPO episode means reuse saved buffers and the first reward-component window is short; do not interpret those first values as fresh full-episode evaluation. Temporary validation directories are absent.
