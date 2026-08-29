# 05.02 — Remote GPU telemetry

- **Objective:** Sample RTX 3090 utilization/memory independently at low frequency and correlate it with runs.
- **Inputs/prerequisites:** WSL server PID; `nvidia-smi`; remote ignored runtime directory; clock metadata.
- **Implementation tasks:** At sampler start/end record Mac UTC, WSL UTC, monotonic elapsed values, and measured clock offset; start one validated remote sampler per run/profile using supported `nvidia-smi` fields at the configured interval; normalize samples to UTC/relative elapsed in GPU JSONL and attach run ID/profile/server PID/source SHA; use the shared atomic ownership record and protect against duplicate/stale/reused PID; stop in `finally`; copy raw metrics once into the ignored local run directory and produce an allowlisted summary, never SSH per step.
- **Files expected to change:** `scripts/collect_gpu_metrics.sh`, server/run lifecycle scripts, telemetry aggregation/tests, docs.
- **Validation:** Start/duplicate/stop/stale PID; sample cadence near configured 1 s; process associated with intended GPU/server; no SSH in Mac step loop.
- **Acceptance:** GPU samples span inference and correlate by relative interval despite clock skew; sampler cleans up; profile/server/run/SHA correlation is unambiguous; raw identifiers remain ignored and are not committed.
- **Planned commit:** `feat(telemetry): sample remote GPU metrics`.
- **Actual findings:** Real WSL `nvidia-smi` sampling and policy-process GPU-memory attribution work and captured the failed π₀ attempts. The run-correlated long-lived sampler remains unimplemented.
- **Remaining blockers:** Successful inference and the Phase 05 run lifecycle.
- **Completion status:** Hardware interface proven; feature implementation planned.
