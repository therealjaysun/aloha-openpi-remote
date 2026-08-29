# 05.04 — Performance summary

- **Objective:** Produce an honest, compact infrastructure and task report for each policy profile.
- **Inputs/prerequisites:** Complete per-profile runs plus GPU/local telemetry.
- **Implementation tasks:** Aggregate count/mean/p50/p95/max for cold/warm inference, active sim step/rate, wall-clock episode rate, buffer waits, dropped-leading actions, GPU memory/utilization; evaluate warmed p95 against the prefetch budget before making a 50 Hz claim; report request count/retries/failures/reward/success; identify verified profile, checkpoint, source SHAs, explicit seeds, package versions; calculate telemetry coverage/overhead; keep π₀ and π₀.₅ rows separate; link videos by local run ID only.
- **Files expected to change:** Telemetry aggregator/test fixtures, `README.md` status/results, plan actual-results fields; generated summaries ignored unless deliberately sanitized as tiny fixtures.
- **Validation:** Known fixture calculations; empty/partial/single-sample runs; cross-check event counts and episode steps; ensure no machine identifiers.
- **Acceptance:** Reviewer can determine whether infrastructure met the contract and compare both profiles without conflating transfer success with system health.
- **Planned commit:** `docs(perf): summarize remote ALOHA runs`.
- **Actual findings:** No measurements exist; no latency or success estimate is claimed.
- **Remaining blockers:** Live episodes and GPU samples.
- **Completion status:** Planned.
