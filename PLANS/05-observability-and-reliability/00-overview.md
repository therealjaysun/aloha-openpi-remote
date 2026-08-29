# Phase 05 — Observability and reliability

- **Objective:** Preserve actionable, low-overhead evidence for every run and fail safely across expected network/process problems.
- **Scope:** Local JSONL/summary, post-step 14-joint trajectories and atomic plots, ordered overhead/left-wrist/right-wrist episode MP4s, remote low-frequency GPU metrics, bounded reconnect/retry, partial-result reporting, and performance/profile comparison.
- **Non-goals:** A second logger, hosted telemetry, databases, dashboards, per-step persistence/network/SSH/`nvidia-smi`, silent exception swallowing, infinite retry.
- **Dependencies:** Phase 04 local runtime interfaces for pure implementation; live phase 04 runtime and remote lifecycle only for hardware evidence; ignored output directories.
- **Planned files:** `tools/remote_aloha/telemetry.py`, a small trajectory plot helper and focused test, `tools/remote_aloha/run.py`, `tests/test_telemetry.py`, `scripts/collect_gpu_metrics.sh`, buffered-policy reconnect/run/lifecycle integration, docs/plans.
- **Planned commits:** Existing Phase 5 commits plus `feat(telemetry): plot joint trajectories` and refreshed exact-candidate evidence.
- **Branch:** `codex/05-observability`.
- **PR base:** `codex/04-end-to-end-control`.
- **PR title:** `feat(telemetry): record control and GPU metrics`.
- **Acceptance criteria:** Existing criteria plus one private atomic plot and one atomic horizontal three-camera MP4 per active episode; exact applied-step coverage; authoritative trajectory normalization; safe publishable IDs/metadata; all 14 actual series and all three ordered video panels visually verified.
- **Test commands:** `make test`; `make lint`; `make secret-scan`; focused synthetic/interrupt/plot/overhead tests; `make metrics`; the existing three-episode π₀ and π₀.₅ `make run` sequence; inspect six plots and confirm outputs remain untracked.
- **Risks:** Larger rows or composite frames exceed telemetry/RAM budgets, plots obscure series, interrupted artifacts diverge from applied steps, metadata leaks paths, or staged prompt boundaries reuse stale buffered actions.
- **Rollback:** Disable subscribers/sampler while retaining core control loop; stop validated metrics PID; revert phase; keep partial ignored artifacts.
- **Current status:** Historical Phase 5 is complete. The S0827 extension reuses these artifacts and is pending exact-candidate staged-run validation; Phase 5 is not restarted.
- **Actual results:** Exact candidate `2065dd9d5a5e7f21ea40a940944d48ac08c6da20` passed 297 tests with one platform skip, lint/format/Bash/secret gates, and hosted PR checks. RTX 3090 runs recorded exact 761/761 π₀ and 900/900 π₀.₅ step/sample coverage, 14 actual plus 14 commanded series, and six passing plots. π₀ infrastructure/task success was 3/3 and 2/3; experimental π₀.₅ was 3/3 and 0/3. Maximum episode telemetry-write p95 was 0.197 ms, below the 1 ms budget. All raw rows/plots are ignored, and final stop/residue checks passed.
- **Deviations:** Automatic retry is deliberately limited to client construction/connect/metadata before reset or inference. Once an inference may have been sent, replay is unsafe; the episode aborts and preserves partial evidence. The first hardware run exposed an orphaned WSL sampler when the Mac SSH client was terminated; the final implementation added an explicit remote ownership record/stop and the exact-candidate rerun passed cleanup.
- **PR:** [PR 6](https://github.com/therealjaysun/pi-robotics/pull/6).
- **Final commit SHA:** Hardware-validated implementation `2065dd9d5a5e7f21ea40a940944d48ac08c6da20`; completion evidence is at branch HEAD.

## Machine handoff

The historical Phase 5 profile campaign remains complete. The S0827 extension adds only one π₀ Scenario 1 diagnostic after local/hosted gates; it does not rerun the Phase 5 campaign.
