# Static trajectory scoring implementation plan

> Execute inline using superpowers:executing-plans; no delegated agents required.

**Goal:** Score F2b trajectory and target JSON in free Static Space/Pyodide,
including a headless-browser command-line interface selected by the user.

**Architecture:** Shared compute adapter + generated unchanged scoring snapshot;
module Worker; static HTML/JS; Playwright runner with optional hosted page URL.

**Tech stack:** Python, NumPy, Pyodide 314.0.7, Playwright Chromium, Bash, HTML/JS.

**Spec:** ../specs/2026-11-02-static-trajectory-scoring-design.md

## Global constraints

- Edit the existing platformtohf directory and corresponding README; preserve
  training/inference code and the official scoring formulas.
- No server compute, HTTP POST scoring, credentials in frontend, or paid hosting.
- Support full JSON documents and two input paths; return errors and exit nonzero.

## Review focus

Wrong task keys; malformed/ragged arrays; lost optional score penalties; path/code
injection; stale outputs after errors; Worker initialization/retry and CDN failure.

## Tasks

- [x] Write failing parity/validation/CLI and bundle freshness tests.
- [x] Generate scorer snapshot from original source; implement shared JSON adapter.
- [x] Build static two-file UI, module Worker, progress, cancellation and download.
- [x] Replace Bash submission script with Playwright runner and requirements.
- [x] Document packaging/deployment/CLI; remove old JSONL/token/endpoint material.
- [x] Run scoring parity tests, real Pyodide browser/CLI and UI tests, Bash syntax,
  Python/JS checks and diff checks. Review only requested changes and report limits.

## Evidence

2026-11-02: 27 tests passed including real Chromium/Pyodide, eight scoring cases,
the saved HFM rollout when available, UI upload/download/cancel/retry, both local
and supplied-page CLI modes, Unicode/spaced paths, bad input rejection and bundle
freshness. Bash syntax and missing-file exit checks passed in WSL; JS/Python syntax
and git whitespace checks passed. Desktop/mobile screenshots reviewed locally.
HF deployment is left to the owner; no remote changes were made. No authoritative
official target is present locally, so fixtures are explicitly synthetic targets.
