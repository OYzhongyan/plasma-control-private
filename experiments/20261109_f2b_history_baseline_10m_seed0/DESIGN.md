# F2b history baseline: design and reproducibility contract

## Purpose
Restore the current F2 implementation from `/inspire/qb-ilm/project/ai-for-fusion/czxs25220173/SII-585-THU` (commit `ffda09ab4621ede477ff71247547f8cbc43d133a`) into the participant `main` branch. The user explicitly requests restoration of both the history observation wrapper and that repository's reward, a new GitHub clone on Inspire, and a full 10,000,000-timestep baseline with reproducible local records.

## Decisions
- Keep the single-experiment `train/` layout and the current 12-voltage shared-service submission protocol. Do not restore F1/F2a selectors or replace the trusted competition scorer.
- Restore the upstream F2 feature equations, four frames of 188 features and five 12D physical-action slots in delta mode: 812 policy inputs. Use class name `ObservationWrapper`. Preserve the existing 90D reference interface only for already exported artifacts; new training defaults to the history baseline.
- Restore the upstream F2 dense reward mathematically, its published weights and smooth score (`score_lower_bound=false`, `use_competition_score=false`). Specialize unreachable F1 branches away; keep formal competition reward available as a separate comparison utility. Record reward/scorer provenance and hashes.
- PPO: seed 0, 32 environments, 256 rollout steps, batch 1024, six epochs, learning rate 1e-4 with existing warmup/cosine schedule, 256x256 Tanh actor/critic, CPU. Requested 10,000,000 steps; actual may round upward to complete a rollout.
- Generalization: train on `15892_400` and `15892_500`, reset signeo/bp/q0 perturbations +/-5% sampled anew each episode, 500-step horizon, one observation-step delay as upstream F2b README. Fixed target geometry is the tracked public reference. Hold out `14795_400` for validation and `13705_500` for a separate cross-initial-condition test. Report seen-shot and held-out scores separately, without claiming single-seed evidence proves broad generalization.
- Local CSV and TensorBoard are mandatory regardless of W&B. W&B defaults to offline for a long run so login/network cannot block training. No secrets are copied into records.
- Reproducibility repair: pass environment seed + 30000 into the existing PowerSupplyModel RNG; distribution/equations and per-step resampling remain unchanged. Each environment's stream advances continuously across resets. MATLAB/BLAS cross-platform bitwise equivalence is not promised.
- Evaluation follows the upstream README's `eval.obs_perturb.enabled=true`: one-step dictionary delay plus Gaussian Ip std=2300 A and R std=0.008 m. Training uses the upstream delay-only default. This deliberate sensor-noise stress test is reported explicitly and is not a same-distribution evaluation. Inference kernel latency is not yet measured by this runner; no claim is made about satisfying the optional 0.5 ms rule.
- Each experiment lives under `experiments/<id>/`: purpose/changes in Markdown; config, provenance, environment and status in JSON/YAML; scalar curves in CSV/TensorBoard; validation artifacts and summaries in evaluation subdirectories; figures in `picture/`. Heavy generated artifacts are kept on disk and ignored by Git.
- C's services and unrelated processes remain untouched. Use Jupyter transport only, no SSH or network proxy. The currently idle HFM training pool on 2223-2254 may be reused after checking no active client. Do not stop it. Isolated short preflight precedes the detached full run.

## Approach considered
Replacing the entire old repository would restore obsolete task selection and remove current submission reliability. Replacing only the wrapper would break metadata, validation and serving. Restore the F2 mathematical components and synchronize their consumers instead.

## Completion evidence
Tests cover sequence/action/reset parity, metadata and frozen normalization, restored reward parity, real nonempty local scalar logs, and experiment bookkeeping/failure status. Before pushing, run relevant existing submission/reward tests. On Inspire verify new clone commit, package/runtime versions and input/output shapes, run a short real HFM update and scoring check, then launch and observe the full run advance. Final validation/plots execute automatically after training; final score remains unknown until it completes.
