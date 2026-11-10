# CNN critic implementation plan

> Execute in this session, with independent policy and observation modules delegated under superpowers:dispatching-parallel-agents. User has approved design, publishing and experiment launch. Do not request approval again. No worktrees; root main is the requested workspace. Final fresh-context review after integration.

**Goal:** Implement the approved four-frame flux CNN critic and launch a matched 10m experiment on plasma-control2.
**Architecture:** Exact baseline actor; isolated CNN critic with flux history, original auxiliary features and masks. New normalization/metadata mode selected only by an opt-in critic config.
**Stack:** SB32.8/PyTorch/Gymnasium/Hydra/HFM/Inspire Jupyter.
**Spec:** DESIGN.md in this directory.

## Tasks
- [x] T0: Inspect Notebook2 resource quota, packages, HFM runtime and idle ports. Preserve Notebook1 baseline. Publish pending authorized cleanup/docs separately.
- [x] T1: RED→GREEN tests for exact actor initialization and RNG, actor-only inference, value shape/gradient isolation and SB3 save/load; implement new flux critic policy.
- [x] T2: RED→GREEN tests for original actor-feature parity, C-order flux history/masks/reset, global flux normalization and terminal/save-load handling; implement new wrapper and VecNormalize subclass.
- [x] T3: RED→GREEN integration tests; opt-in train/eval factories, metadata, normalization snapshots, full-checkpoint inference and actor-only ONNX export. Preserve old contracts.
- [x] T4: Extend recorded runner with opt-in critic variant. Compare actual baseline config recursively, permit only critic fields. Verify same actor seed/RNG; physical reward/environment and formal scorer files remain unchanged.
- [x] T5: Relevant regression suite and fresh-context review; fix important findings. Commit/push tested code, pin clone on Notebook2 at that commit.
- [x] T6: Verify and reuse the distinct idle HFM pool on Notebook2 only; real temporary preflight (train/save/eval/bootstrap). Delete all smoke outputs. Launch matched detached 10m; verify actual simulation/learning progress, local logs and experiment ledger.

## Review focus
Actor and critic parameter separation; identical baseline actor initialization and sampling RNG; flux padding masks; scalar normalization across valid pixels; no duplicate stats updates; correct final-state Dict bootstrap before reset; serialization across training/eval/A-side inference; actor export excludes CNN and uses actor-only stats; exact non-critic config equality; no HFM/process actions on Notebook1; no retained smoke artifacts or credentials.

## Ledger
- User explicitly approved earlier design and requested timely GitHub updates. Existing pending changes are authorized cleanup/reference/cancelled analysis records, to be published separately from this feature.
- Ruling: preserve raw actor feature float64 values exactly as inherited, since early float32 casting changed RMS arithmetic. The Box/model contract remains float32 after VecNormalize. Flux and masks are float32. Malformed actor-required Fx retains baseline validation failures; nonfinite full-sized critic frames are masked.
- Local relevant regression suite: 148 passed, 4 skipped, 24 warnings in 83.20s. RED observed for each new subsystem. Independent whole-feature read-only review found no Critical/Important blockers. Local SB3 is 2.9.2; remote runtime preflight will use matching baseline SB3 2.8.0.
- Notebook2 has a separate process namespace (baseline train PID absent), CPU quota110 and memory500GiB. Its 32 HFM listeners are idle; no existing services have been stopped. Fresh independent HTTPS clone created at /inspire/qb-ilm/project/ai-for-fusion/czxs25220173/plasma-control-cnn-critic. Core dependencies are being aligned to baseline versions before launch.
- Core runtime aligned: PyTorch 2.10.0+cu128, SB3 2.8.0, W&B 0.30.0 and SciPy 1.17.1. Direct package mirrors were used for slow downloads; no proxy or SSH was used. Remote new-feature tests: 37 passed. The first temporary launcher selected base Python; corrected the launcher interpreter, deleted that temporary directory, and reran. No production code change was required.
- Real HFM temporary PPO training completed, saved/reloaded checkpoint, checked actual parameter counts, and exercised all three evaluation shots. All temporary models, scores, plots, trajectories and logs were deleted; no smoke result files are retained. Unit tests separately verify real SB3 timeout bootstrap at the final observation.
- Full training launched 2026-11-10T14:03:31Z (22:03:31 Beijing), runner3690091 / train3690338, source8a5b9db on Notebook2. Full requested config comparison passes except critic; 32 HFM connections and independently increasing CPU counters with STEP commands verified. Initial formal run is still collecting its first8192-transition rollout; final10m results are pending. Snapshot is historical evidence, not a live monitor.
