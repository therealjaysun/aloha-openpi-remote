# Phase 05 — Observability and reliability

- **Objective:** Preserve actionable, low-overhead evidence for every run and fail safely across expected network/process problems.
- **Scope:** Local JSONL/summary, remote low-frequency GPU metrics, bounded reconnect/retry policy built on phase 03 finite I/O, partial-result reporting, performance/profile comparison.
- **Non-goals:** Hosted telemetry, databases, dashboards, per-step SSH/`nvidia-smi`, silent exception swallowing, infinite retry.
- **Dependencies:** Phase 04 local runtime interfaces for pure implementation; live phase 04 runtime and remote lifecycle only for hardware evidence; ignored output directories.
- **Planned files:** `tools/remote_aloha/telemetry.py`, `tests/test_telemetry.py`, `scripts/collect_gpu_metrics.sh`, buffered-policy reconnect/run/lifecycle integration and focused tests, docs/plans.
- **Planned commits:** `feat(telemetry): record local runtime events`; `feat(telemetry): sample remote GPU metrics`; `fix(runtime): bound retries and preserve failure evidence`; `docs(perf): summarize policy profile performance`.
- **Branch:** `codex/05-observability`.
- **PR base:** `codex/04-end-to-end-control`.
- **PR title:** `feat(telemetry): record control and GPU metrics`.
- **Acceptance criteria:** Timestamped per-run JSONL and Markdown/CSV summary; profile/commit/versions included; inference/sim/frequency/wait/retry/reward/GPU metrics captured; telemetry overhead measured and low; Ctrl+C/failures preserve valid lines/video; all spawned processes cleaned; π₀ and π₀.₅ summarized separately.
- **Test commands:** `make test`; `make metrics`; per-profile `make run`; interrupt/failure scenarios; parse every JSONL line; inspect summaries; verify no output tracked.
- **Risks:** Telemetry blocks loop, corrupt line on crash, clock mismatch Mac/PC, GPU sampler orphan, sensitive paths in logs, retry causes stale actions.
- **Rollback:** Disable subscribers/sampler while retaining core control loop; stop validated metrics PID; revert phase; keep partial ignored artifacts.
- **Current status:** Plan complete; implementation not started.
- **Actual results:** No telemetry or performance evidence exists.
- **Deviations:** None.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Machine handoff

Keep both machines on while collecting correlated telemetry. At phase end, stop and validate the owned GPU sampler/server/tunnel processes; do not announce that the PC may power off until any phase 06 hardware evidence is complete.
