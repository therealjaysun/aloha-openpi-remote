# 05.04 — Performance summary

- **Objective:** Produce an honest, compact infrastructure and task report for each policy profile.
- **Inputs/prerequisites:** Complete per-profile runs plus GPU/local telemetry.
- **Implementation tasks:** Retain existing metrics/trajectory plots. Save each active episode as one atomic `224×672` MP4 with synchronized overhead, left-wrist, and right-wrist panels in that order; record the ordered views/layout in private manifests and safe camera metadata. Failed wrist capture may use black unavailable panels only in the honest failed/partial final frame.
- **Files expected to change:** Telemetry aggregator/test fixtures, `README.md` status/results, plan actual-results fields; generated summaries ignored unless deliberately sanitized as tiny fixtures.
- **Validation:** Existing trajectory checks plus synthetic panel order/shape/dtype, exact video/applied-step coverage, atomic finalization, and visual inspection of all three panels in the staged diagnostic MP4.
- **Acceptance:** Existing comparison remains intact; every episode with valid trajectory rows has a readable atomic plot, full coverage for passing runs, and only safe compact trajectory metadata in publishable summaries.
- **Planned commit:** `docs(perf): summarize remote ALOHA runs`.
- **Actual findings:** Candidate `2065dd9d` completed three episodes per profile. π₀: infrastructure/task 3/3 and 2/3, 761 steps, 42 requests, warmed inference mean/p95/max 310.75/359.21/475.38 ms, active-rate mean/p95/max 46.91/47.02/47.03 Hz, and GPU memory/utilization max 15,401 MiB/39%. Experimental π₀.₅: infrastructure/task 3/3 and 0/3, 900 steps, 42 requests, warmed inference mean/p95/max 405.43/502.35/577.85 ms, active-rate mean/p95/max 48.13/48.38/48.41 Hz, and GPU memory/utilization max 15,857 MiB/100%. Both summaries report 14 joints, exact 1.0 step coverage, safe plot IDs, and 3/3 passing plots. Visual inspection confirmed all 14 actual and 14 dashed commanded series in each of six plots. Neither profile supports an uninterrupted 50 Hz claim.
- **Remaining blockers:** None.
- **Completion status:** Complete; ignored summary hashes are `ed1edb598ef2e224fdb9905e995430ba9dec5cb4d01810c1516ffa59a5c91bcd` (π₀) and `79115916ef7067957e1693243453608ce316f86ccdd62b416cc76f5b616bc911` (π₀.₅).
