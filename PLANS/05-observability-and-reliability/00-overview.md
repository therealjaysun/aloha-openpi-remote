# Phase 05 — Observability and reliability

- **Objective:** Preserve actionable, low-overhead evidence for every run and fail safely across expected network/process problems.
- **Scope:** Local JSONL/summary, post-step 14-joint actual/command trajectories and atomic episode plots, remote low-frequency GPU metrics, bounded reconnect/retry policy built on phase 03 finite I/O, partial-result reporting, performance/profile comparison.
- **Non-goals:** A second logger, hosted telemetry, databases, dashboards, per-step persistence/network/SSH/`nvidia-smi`, silent exception swallowing, infinite retry.
- **Dependencies:** Phase 04 local runtime interfaces for pure implementation; live phase 04 runtime and remote lifecycle only for hardware evidence; ignored output directories.
- **Planned files:** `tools/remote_aloha/telemetry.py`, a small trajectory plot helper and focused test, `tools/remote_aloha/run.py`, `tests/test_telemetry.py`, `scripts/collect_gpu_metrics.sh`, buffered-policy reconnect/run/lifecycle integration, docs/plans.
- **Planned commits:** Existing Phase 5 commits plus `feat(telemetry): plot joint trajectories` and refreshed exact-candidate evidence.
- **Branch:** `codex/05-observability`.
- **PR base:** `codex/04-end-to-end-control`.
- **PR title:** `feat(telemetry): record control and GPU metrics`.
- **Acceptance criteria:** Existing criteria plus one private atomic plot per episode from the existing step JSONL; one finite actual and commanded 14-vector per applied step with exact step/monotonic elapsed time; authoritative fixed-range normalization; partial plotting; safe plot IDs and coverage fields in publishable summaries; all 14 actual series visually verified for both profiles.
- **Test commands:** `make test`; `make lint`; `make secret-scan`; focused synthetic/interrupt/plot/overhead tests; `make metrics`; the existing three-episode π₀ and π₀.₅ `make run` sequence; inspect six plots and confirm outputs remain untracked.
- **Risks:** Larger step rows exceed the 1 ms telemetry budget, plots obscure 14 series, bad limits mislead, interrupted rows diverge from applied steps, plot metadata leaks a path; existing network/process risks remain.
- **Rollback:** Disable subscribers/sampler while retaining core control loop; stop validated metrics PID; revert phase; keep partial ignored artifacts.
- **Current status:** Joint-trajectory amendment in progress on the existing Phase 5 candidate; prior evidence remains a baseline, not acceptance for the amendment.
- **Actual results:** Baseline candidate `de63e19c37d8fd76fbda5b5a07a8f5a0b19b42d6` passed the original Phase 5 gates. New local and exact-candidate hardware results remain pending and will replace this completion claim only after both profile runs, six plot checks, telemetry-overhead measurement, and cleanup pass.
- **Deviations:** Automatic retry is deliberately limited to client construction/connect/metadata before reset or inference. Once an inference may have been sent, replay is unsafe; the episode aborts and preserves partial evidence. The first hardware run exposed an orphaned WSL sampler when the Mac SSH client was terminated; the final implementation added an explicit remote ownership record/stop and the exact-candidate rerun passed cleanup.
- **PR:** [PR 6](https://github.com/therealjaysun/pi-robotics/pull/6).
- **Final commit SHA:** Pending amended exact-candidate hardware validation; prior hardware baseline `de63e19c37d8fd76fbda5b5a07a8f5a0b19b42d6`.

## Machine handoff

Keep the PC off during plan/code/local-test work. Request power-on only after the amended Phase 5 SHA is clean, pushed, secret-scanned, and ready for the existing two-profile hardware sequence.
