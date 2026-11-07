# Remote checkpoint agent implementation plan

> Execute inline with superpowers:executing-plans. Root checkout only, per user preference. The A/B/C design and participant file changes were approved in conversation. No further design confirmation is needed.

**Goal:** A serves its own checkpoint and wrappers; B stores endpoint JSON; C owns HFM, hidden shot and scoring.

**Spec:** ../specs/2026-11-07-remote-agent-design.md

**Stack:** Flask + SB3/Torch on A; standard HTTP/TLS client + existing simulator/worker on C.

## Tasks

- [x] T1: Add failing protocol/session tests. Implement strict endpoint JSON, optional bearer auth, episode/step idempotence, bounded JSON wire format.
- [x] T2: Add real PPO checkpoint/wrapper parity tests. Implement frozen normalization, buffered actual wrapper reuse and custom local adapter factory; checkpoint-default service/CLI and A-only launcher.
- [ ] T3: Update main participant README/requirements, generate endpoint.json, run focused tests and save a scoped local commit.
- [ ] T4: Transfer common files to server branch. Add destination-pinned bounded HTTP client and tests for retries, redirects, address policy, auth, output validation and secret redaction.
- [ ] T5: Add remote grader/rollout integration, optional C HFM startup, CLI/default cache, platform loopback tests and operator docs. Preserve official scoring and infrastructure retry semantics.
- [ ] T6: Independent final review, fix important findings through regression tests, verify both branches/shared files and save local commits; no pushes or live platform requests.

## Global constraints

- Endpoint file never controls C shot, target, HFM address or score. No model/code downloads from A.
- C remote execution never imports train/Torch/SB3. A reuses its real wrappers without a second feature implementation.
- Same episode/step retry never advances A twice. Bad action/state poisons the session until new reset.
- No redirects or environment proxies; DNS address validation and connection pinning prevent accidental/internal target access. Private origins require explicit C permission.
- Preserve unrelated files, legacy ONNX tools, source scoring and physical supply settings.

## Review focus

Checkpoint normalization pairing; adapter history/reset parity; retry after lost response; concurrent/mismatched IDs; HFM versus participant fault classification; transport bounds, SSRF and credential separation; branch-specific documentation.

## Execution ledger

- Defaults are implementation decisions within the approved design: checkpoint backend, endpoint protocol v1, public destinations by default, explicit allow-origin for private evaluation networks.

- T1 protocol/session: retries, conflicting IDs, authentication, concurrency and reset failure invalidation covered.
- T2 actual PPO/wrapper parity covers absolute/delta, frozen non-identity statistics, changed 3D observations, matching latest checkpoint stats, disabled normalization and invalid stats shapes.
