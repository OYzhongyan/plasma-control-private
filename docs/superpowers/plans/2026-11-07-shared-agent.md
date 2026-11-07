# Shared-directory Agent Implementation Plan

> Execute inline with superpowers:executing-plans and TDD. The user explicitly approved this route and asked to continue implementation; do not re-request design or execution approval. Root checkout only, no push.

**Goal:** Replace official HTTP observation/action exchange with signed shared files and one-command participant startup.

**Architecture:** Reuse checkpoint Policy and AgentSession on A, and trusted rollout/worker on C. Add a bounded signed filesystem channel with atomic I/O, locking, expiry and per-request correlation.

**Tech Stack:** Python standard library, NumPy, existing PPO/HFM/scorer.

**Spec:** ../specs/2026-11-07-shared-agent-design.md

- [x] T1: Write failing local channel/session tests. Implement confined filesystem operations, descriptor schema, signatures, locks and atomic bounded messages.
- [x] T2: Add A worker and CLI, automatic descriptor generation and launcher. Verify real exchanges, duplicate actions, expiry, malformed messages, service restarts and single-client locking. Update main README/submission README and commit.
- [ ] T3: Sync common files into server branch. Add shared C client/grader/rollout and default CLI, preserve trusted scoring/delay and infrastructure retry classification. Test fake platform + A files + fake HFM and no training imports.
- [ ] T4: Independent whole-change review; reproduce and fix important findings, verify both branches, commit locally, no push.

## Review focus

Public-directory symlink/TOCTOU confinement; secret disclosure and signed response correlation; expired request replay; duplicate inference; cleanup after timeout; shared FS permissions and process locks; C storage versus participant fault classification; preserving HFM/scoring and branch-specific docs.

## Ledger

- Pre-flight: common descriptor/mailbox schema is consumed by A and C; keep shared modules byte-identical. Reuse existing AgentSession internal protocol while using a new signed outer shared-file protocol.
- Ruling: keep legacy HTTP/ONNX explicit development modes; official default is shared-directory transport. Cost: retained compatibility code needs distinct documentation.
- T1/T2: module absence RED then GREEN; real checkpoint CLI tested. Main suite 71 passed, 3 Windows skips. Linux native-filesystem smoke reproduced restrictive-umask failure, then passed permissions, private descriptor, directory-swap confinement and locking after fix. WSL lacks pytest; standard-library smoke used instead.
- Migration: valid old HTTP endpoint is regenerated automatically on successful shared service startup; malformed unknown files fail rather than being overwritten. Regression RED then GREEN.
