# Shared-directory policy transport

The user approved replacing HTTP with a shared directory under `/inspire/hdd/project/26-nuclear-fusion/public`. A retains its checkpoint, frozen normalization and actual wrappers. B receives one private `endpoint.json`. C retains HFM, hidden shots and official scoring. No numerical algorithm changes are part of this task.

## Chosen approach

Use signed request/response JSON files, replacing HTTP for official submissions. HTTP registration alone still needs network routing; shared files remove the need to choose IPs or ports. Existing HTTP/ONNX development tools may remain explicit compatibility options.

A single command starts `submission/shared_service.py --model results/YOUR_RUN/checkpoints`. It creates a random service ID, a local secret, a per-service shared mailbox, and local `results/endpoint.json`. The uploaded descriptor contains only version, protocol, service ID and secret; C supplies the trusted shared root. No submitted filesystem path is accepted.

The shared root defaults to `/inspire/hdd/project/26-nuclear-fusion/public/plasma-agents`, configurable on A and C with `--shared-root`. Administrators grant both accounts access through the project group/ACL. Directory permissions do not replace signatures. The descriptor and secret remain outside the public mailbox; observations are not confidential on a public filesystem.

Requests and replies carry random request IDs, direction, expiry, payload and HMAC-SHA256 signatures. Replies bind the request digest. Atomic temporary-file replacement prevents partial reads. Existing episode/step idempotence prevents duplicate policy advancement. Each process has a service incarnation ID to detect restarts. Expired/unsigned messages never reach the policy. Completed message files are removed; orphan messages expire and are collected.

Linux directory-descriptor-relative opens with no-follow flags confine access to the configured mailbox even during symlink replacement. Regular files and bounded JSON are required. OS locks serialize A instances and C clients without stale lock directories. Windows fallback is for local tests; deployment is Linux with shared-filesystem locking support.

C distinguishes participant timeouts/invalid replies from shared-storage I/O failures (pending). Trusted root absence is infrastructure failure; an absent contestant mailbox is submission failure. Requests never contain official file paths, platform credentials or scores. C scores only its own trajectory.

## Delivery

Update main and server/offline-grading through the root checkout, preserving unrelated work. Save local commits, no push/worktrees. Tests use local shared folders, small PPO fixtures and fake HFM/platform, not the production public directory or live platform. Actual shared filesystem permissions, locking, latency and HFM require deployment smoke testing.
