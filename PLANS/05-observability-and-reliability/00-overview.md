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
- **Current status:** Complete; open for review.
- **Actual results:** Exact candidate `de63e19c37d8fd76fbda5b5a07a8f5a0b19b42d6` passed 273 tests with one platform skip, lint, Bash syntax, fail-closed secret scanning, and three hardware episodes per profile. Both profiles had complete GPU coverage and zero request retries/failures. π₀ task success was 3/3; experimental π₀.₅ task success was 0/3 without an infrastructure failure. Active rates averaged 48.21 Hz and 47.56 Hz respectively, so sustained 50 Hz is not claimed. Exact server, tunnel, sampler, and listener cleanup passed on Mac, Windows, and WSL.
- **Deviations:** Automatic retry is deliberately limited to client construction/connect/metadata before reset or inference. Once an inference may have been sent, replay is unsafe; the episode aborts and preserves partial evidence. The first hardware run exposed an orphaned WSL sampler when the Mac SSH client was terminated; the final implementation added an explicit remote ownership record/stop and the exact-candidate rerun passed cleanup.
- **PR:** [PR 6](https://github.com/therealjaysun/pi-robotics/pull/6).
- **Final commit SHA:** Hardware implementation `de63e19c37d8fd76fbda5b5a07a8f5a0b19b42d6`; final evidence is at branch HEAD.

## Machine handoff

Hardware collection and cleanup are complete. Phase 06 has no planned PC-only acceptance gate; the PC may be powered off unless an unexpected regression requires a hardware rerun.
