# Server-side platform grading (manual section 4.5)

The user explicitly selected section 4.5 and requested a new Git branch plus a
new directory containing the deployable worker. `main` remains the contestant
baseline. Work takes place on `server/offline-grading`; existing unrelated dirty
README/test changes remain untouched and outside the server commit.

## Responsibilities

The competition platform owns student authentication, teams, submissions, history
and rankings. This worker only polls this problem's pending offline/manual tasks,
downloads a JSON trajectory, compares it against a fixed local official target
using the existing F2b score package, and posts completed/failed results.

Use a plain Python CLI plus NumPy, requests and SQLite. No HTTP listener, HF,
browser, GPU, HFM or student code execution is required. Platform credentials are
provided by environment variables, never embedded or written to task files.
Downloads use a separate unauthenticated request path, bounded streaming and HTTP
timeouts. Only full JSON documents are accepted, not arbitrary archives/code.

## Durable operation

- A single-instance OS file lock prevents two workers sharing the same data dir.
- SQLite keys submissions by the platform's string id; saved scores survive
  restarts, network timeouts and uncertain batch responses.
- Save scores before reporting. Cache the target/rules/metric fingerprint and
  refuse to mix runs against different settings in one data directory.
- Reconcile cached unreported tasks with platform state before reporting. Never
  rescore or overwrite already completed platform records. Treat unknown state
  conservatively and retry; absence from a list does not prove completion.
- Parse batch `updated`, `failed`, `results`, `errors`; only mark confirmed items
  acknowledged. Handle partial failures independently. Refresh authentication on
  401 if username/password are available.

## Delivery

`server_evaluator/` contains CLI, platform client, original-score adapter, durable
store and instance lock, dependency list, systemd example, tests and server README.
The root README on this branch identifies the server role and links to deployment
instructions. Configuration includes problem id, fixed target path, metric name,
data directory, sources, limits and polling interval. `--once` supports smoke tests.

Default metric is `得分`, with the existing 0–30 scale; platform names must match.
No platform account is accessed during development; all network integration uses
a local fake API. The supplied manual is reference specification, not permission
to use its embedded credentials. Live platform validation requires the owner to
configure environment credentials and the actual problem id on their server.
