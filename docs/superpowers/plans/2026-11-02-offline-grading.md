# Offline grading implementation plan

> Execute inline using superpowers:executing-plans. User selected section 4.5.

**Goal:** A deployable CPU worker on `server/offline-grading`, isolated from the
contestant baseline branch `main`.

**Architecture:** Poll/download -> original scorer -> SQLite -> batch return,
with restart recovery, single-worker lock, and no inbound service.

**Tech stack:** Python 3.9+, requests, NumPy, SQLite, systemd (optional Linux).

**Spec:** ../specs/2026-11-02-offline-grading-design.md

## Constraints / review focus

Platform token must never accompany file downloads or appear in persistent task
data. Enforce this problem/source/pending scope. Confirm partial batch acknowledgments
and uncertain network responses. Preserve cached results across restarts. Reject
invalid JSON/target and bound download sizes. Keep all credentials out of commits.

## Tasks

- [x] Write failing scorer, durable-state and fake-platform tests.
- [x] Implement scorer and persistent task/settings storage with single-instance lock.
- [x] Implement platform auth/poll/download/report client and polling CLI.
- [x] Add setup, deployment, metric configuration and systemd instructions.
- [x] Verify parity, malformed inputs, auth refresh, token-free downloads, report
  retry/partial errors, restart deduplication and branch-scoped clean diff.
- [x] Commit only server-related files on the new branch; do not push or merge.

## Verification evidence

2026-11-02: `pytest server_evaluator/tests test/test_static_trajectory_scoring.py
-q` passed 41 tests (14 server tests and 27 existing scoring/CLI checks). The
real run.py --once path was exercised twice against a local fake HTTP platform,
including password login, file download and batch reporting, with no duplicate
download/report. Verified lost-response reconciliation, partial batch errors,
download retry/size limits, foreign/auto exclusion, auth refresh, no credentials
on download requests or in persisted state, malformed/deep JSON, single-instance
locking, source/target fingerprinting and offline config checks. Python 3.9 syntax
and git whitespace checks passed. No live platform accounts were accessed.
