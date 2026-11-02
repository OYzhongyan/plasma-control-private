# Per-step competition reward on the contestant baseline

The user approved retaining per-step rewards and replacing the current shaping
formula with the official competition scoring rules. Work stays on `main`;
server grading, official score formulas and unrelated dirty files are preserved.

## Design

Replace the default reward with a streaming adapter over the existing official
metric/penalty/scoring functions and the same raw-observation-to-trajectory
converter used by rollout evaluation. Each normal step contributes 1/500 of its
weighted score, with a perfect 500-step episode summing to 30. Do not recompute
the entire trajectory every step.

Use the official current schedule (not observation.reference_Ip), fixed divertor
topology, nX==4 gating, strict coil over-limit checks with a persistent fuse, and
original seven weights. Geometry is frozen from the active reference for the
episode, or a supplied explicit target. Equality with a server score requires
the same target; document selecting the official target via the existing
target.default_reference_path config.

Global timeout or explicitly supplied inference-time measurements can invalidate
previously awarded credit; return a correction when that condition is observed.
Do not fabricate inference timings when none exist. No extra negative failure
penalty or reward-induced termination. A real simulator failure/step horizon
still terminates/truncates normally. Reset state on every successful env reset,
including an episode interrupted before termination. Score after step 500 remains
zero except for any newly observed global invalidation.

Keep the SemifinalDenseReward name and legacy constructor keywords compatible
for old configs, while new training/eval use CompetitionReward explicitly. Old
reward_weights/score_lower_bound no longer control the competition reward. Update
default config, logs and README. Leave old dedicated legacy entrypoints alone.

Raw un-discounted reward sums match the official scorer on the same serialized
trajectory and target, within floating precision. PPO's gamma=0.995 and reward
normalization are separate transformations and remain as configured.

## Verification

Test streaming-vs-official parity for perfect/offset/missing-topology/current/coil
boundary/early termination/timeout/timing/horizon/reset cases, and a saved HFM
trajectory when available. Verify env reset and factory configuration without
requiring MATLAB. Run relevant baseline import/reference/history/export tests.
