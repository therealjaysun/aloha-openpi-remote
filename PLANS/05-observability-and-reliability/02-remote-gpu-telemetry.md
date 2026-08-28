# 05.02 — Remote GPU telemetry

- **Objective:** Sample RTX 3090 utilization/memory independently at low frequency and correlate it with runs.
- **Inputs/prerequisites:** WSL server PID; `nvidia-smi`; remote ignored runtime directory; clock metadata.
- **Implementation tasks:** At sampler start/end record Mac UTC, WSL UTC, monotonic elapsed values, midpoint clock-offset estimate, and round-trip uncertainty; start one foreground sampler over a dedicated Mac-owned authenticated SSH child per run/profile using supported `nvidia-smi` fields at the configured interval; normalize samples to UTC/relative elapsed in GPU JSONL and attach run ID/profile/server PID/source SHA; verify the existing server ownership record before each sample and protect the sampler with a per-run lock; stop the owned SSH child in `finally`; copy raw metrics once into the ignored local run directory and produce an allowlisted summary, never SSH per step.
- **Files expected to change:** `scripts/collect_gpu_metrics.sh`, server/run lifecycle scripts, telemetry aggregation/tests, docs.
- **Validation:** Start/duplicate/stop/stale PID; sample cadence near configured 1 s; process associated with intended GPU/server; no SSH in Mac step loop.
- **Acceptance:** GPU samples span inference and correlate by relative interval despite clock skew; sampler cleans up; profile/server/run/SHA correlation is unambiguous; raw identifiers remain ignored and are not committed.
- **Planned commit:** `feat(telemetry): sample remote GPU metrics`.
- **Actual findings:** Real WSL `nvidia-smi` device-level sampling works and captured earlier failed π₀ attempts. WSL does not provide reliable per-process GPU attribution here, so Phase 05 reports device memory/utilization plus verified policy-server host RSS from `/proc/<pid>/status`.
- **Remaining blockers:** Correlated exact-candidate runs for both profiles.
- **Completion status:** Sampler implementation and pure validation complete; hardware evidence pending.
