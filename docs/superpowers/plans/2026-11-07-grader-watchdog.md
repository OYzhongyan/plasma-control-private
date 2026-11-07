# C Grader Watchdog Implementation Plan

> Use superpowers:executing-plans inline and TDD. The user explicitly specified automatic C-only restart with persistent external HFM; implement directly in the root checkout, no worktrees, no push.

**Goal:** Automatically recover C after code/configuration invalidation while leaving external HFM untouched.

**Architecture:** Standard-library process supervisor with log detection, fresh data generations, resumable active state and owned-process cleanup. Preserve scoring and platform worker behavior.

**Spec:** ../specs/2026-11-07-grader-watchdog-design.md

- [x] T1: Write failing fake-worker and argument-validation tests; implement watchdog, state, lock, logging and targeted restart.
- [x] T2: Preserve evaluation children in the supervisor-owned group, verify cancellation/descendant cleanup and external HFM survival on Linux; update operator README and systemd example.
- [x] T3: Independent final review, reproduce/fix material findings, run appropriate regressions and save local server-branch commit; no push.

## Ledger and review focus

- Shared interface: watchdog supplies active --data-dir and a process-group environment marker; ModelGrader only changes child session creation under that marker. It never changes the physical simulator algorithm.
- Ruling: unexpected exits without a recognized configuration warning are surfaced, not restarted forever. Cost: unrelated fatal setup errors require operator repair.
- Ruling: retain old generation caches without importing old scores into a new rules fingerprint. Cost: still-pending platform tasks may be reevaluated, and old unreported cache records need operator audit.
- Review: message false positives, generation path confinement, stale or corrupt state, stop/restart races, leaked descendants, external-HFM isolation, argument abbreviations and credential leakage.
- T1: missing-module RED; real fake-worker tests 8 passed, including immediate startup mismatch exit, rotation/resume, unrelated deferral/failure, cancellation and managed-HFM argument abbreviations.
- T2: Linux smoke passed actual process-group option, restart, descendant cleanup and survival of an external HFM stand-in. Real run.py + fake platform test also passed config mutation and subsequent pending-task processing. Existing/full server suite before review fixes: 118 passed, 1 skipped.
- Final review: two Important findings (relative argument cwd and systemd retry policy) plus embedded-log matching labeled Minor. All reproduced RED.
- Ruling: regrade embedded-log recognition as Important because it can stop an active grading worker and repeatedly rotate caches on unrelated report errors. Cost if left unfixed: unintended interrupted evaluations and restart loops. Fix uses complete trusted log events, not arbitrary embedded text.
- Final fixes: relative-path cwd, systemd unexpected-exit retry and embedded-log matching all RED then GREEN; watchdog12passed, final full server suite121passed1skipped5warnings in58.07s. Final Linux smoke again verified owned descendant cleanup and external HFM stand-in survival.
- Ruling: production platform/HFM integration and shared deployment behavior remain operator smoke checks; Linux is the deployment target, Windows process-tree cleanup is local-test support. Cost: these isolated tests cannot guarantee all production service or Windows-orphan scenarios.
- Finish: scoped local server/offline-grading commit, no main changes, no worktree and no push. Existing external HFM was never accessed or signalled by tools.
