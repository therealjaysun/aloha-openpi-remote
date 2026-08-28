# 05.01 — Local telemetry

- **Objective:** Record structured Mac-side control/inference events without materially disturbing 50 Hz stepping.
- **Inputs/prerequisites:** Working runtime and buffer; timestamped ignored run directory.
- **Implementation tasks:** Implement line-buffered JSONL writer with UTC ISO time; instrument the phase 04 environment result accessor, policy request latency/server timing/chunk length, request/result steps, dropped-leading actions, active step interval/rate, wall-clock episode rate, buffer waits, errors/retries; associate `server_timing.prev_total_ms` with request N-1 (or exclude it from current-request aggregates); include episode/step/profile/verified source SHAs/package versions; copy relevant raw server logs into the ignored run directory; write metadata first and terminal/partial status in `finally`; aggregate concise Markdown or CSV after run; sanitize only publishable summaries, not by mutating raw local evidence.
- **Files expected to change:** `tools/remote_aloha/telemetry.py`, `tests/test_telemetry.py`, `tools/remote_aloha/run.py`, `.gitignore`.
- **Validation:** Round-trip every event as JSON; NumPy scalar/NaN rejection; partial final line handling; aggregation math fixtures; benchmark subscriber p95 overhead under 1 ms on Mac.
- **Acceptance:** Required run/output fields in `REQUIREMENTS.md` are available; valid lines survive interruption; no fsync/network per step; raw output/logs stay ignored; publishable summary is allowlisted and labels model profile.
- **Planned commit:** `feat(telemetry): record local runtime events`.
- **Actual findings:** Stock server returns `server_timing.infer_ms` plus previous-request total timing. The implementation associates previous total time with request N-1, records monotonic/UTC events in private JSONL, preserves partial valid lines, measures write overhead, and emits allowlisted JSON/Markdown summaries. At the audit pin stock Runtime uses `time.time`; phase 04 owns the monotonic runtime correction.
- **Remaining blockers:** None.
- **Completion status:** Complete. Both exact-candidate profile runs produced valid private JSONL and allowlisted summaries. Telemetry writes averaged 0.168 ms for π₀ and 0.175 ms for π₀.₅, below the 1 ms acceptance limit.

Minimal instrumentation: time the existing calls and expose metrics to one subscriber; do not introduce a logging framework or database.
