# 05.04 — Performance summary

- **Objective:** Produce an honest, compact infrastructure and task report for each policy profile.
- **Inputs/prerequisites:** Complete per-profile runs plus GPU/local telemetry.
- **Implementation tasks:** Aggregate count/mean/p50/p95/max for cold/warm inference, active sim step/rate, wall-clock episode rate, buffer waits, dropped-leading actions, GPU memory/utilization; evaluate warmed p95 against the prefetch budget before making a 50 Hz claim; report request count/retries/failures/reward/success; identify verified profile, checkpoint, source SHAs, explicit seeds, package versions; calculate telemetry coverage/overhead; keep π₀ and π₀.₅ rows separate; link videos by local run ID only.
- **Files expected to change:** Telemetry aggregator/test fixtures, `README.md` status/results, plan actual-results fields; generated summaries ignored unless deliberately sanitized as tiny fixtures.
- **Validation:** Known fixture calculations; empty/partial/single-sample runs; cross-check event counts and episode steps; ensure no machine identifiers.
- **Acceptance:** Reviewer can determine whether infrastructure met the contract and compare both profiles without conflating transfer success with system health.
- **Planned commit:** `docs(perf): summarize remote ALOHA runs`.
- **Actual findings:** Candidate `de63e19` completed three episodes per profile. π₀: infrastructure/task 3/3, 731 steps, 38 requests, warmed inference mean/p95/max 327.28/392.83/448.95 ms, active-rate mean/p95 48.21/48.24 Hz, and GPU memory/utilization max 15,403 MiB/46%. Experimental π₀.₅: infrastructure 3/3, task 0/3, 900 steps, 40 requests, warmed inference mean/p95/max 418.27/515.54/672.15 ms, active-rate mean/p95 47.56/48.13 Hz, and GPU memory/utilization max 15,859 MiB/39%. Neither profile supports an uninterrupted 50 Hz claim.
- **Remaining blockers:** None.
- **Completion status:** Complete. Ignored summaries are `outputs/phase05/20260828T170424.866182Z/pi0_aloha_sim/performance-summary.json` and `outputs/phase05/20260828T170602.263300Z/pi05_aloha_base/performance-summary.json`; `make metrics` reproduced both.
