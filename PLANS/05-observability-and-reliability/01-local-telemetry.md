# 05.01 — Local telemetry

- **Objective:** Record structured Mac-side control/inference events without materially disturbing 50 Hz stepping.
- **Inputs/prerequisites:** Working runtime and buffer; timestamped ignored run directory.
- **Implementation tasks:** Extend the existing line-buffered `step` event after each successful `environment.step`: record one-based applied step, monotonic elapsed seconds, finite post-step `agent_pos`, and the finite command just applied, each exactly 14 values. Derive names/order from pinned gym-aloha 0.1.1. Normalize arm radians with the fixed MuJoCo XML joint ranges and grippers with gym-aloha's documented `[0,1]` close/open coordinate; never use observed extrema. Preserve valid partial rows and the existing no-fsync/no-network step path.
- **Files expected to change:** `tools/remote_aloha/telemetry.py`, `tools/remote_aloha/run.py`, a minimal trajectory helper, focused tests, and the already-used lightweight dependency lock only if needed for installed Matplotlib.
- **Validation:** Synthetic 14-joint round trip; shape/NaN/infinity rejection; exact step/sample coverage; monotonic elapsed-time checks; Ctrl+C/failure partial rows; 300-row writer benchmark with both vectors remains below 1 ms p95 on Mac.
- **Acceptance:** Trajectory rows equal successfully applied steps for passing episodes; valid partial rows survive interruption; no per-step fsync/network/SSH; raw vectors and plot paths stay only in ignored private outputs; publishable output contains only bounded counts/status and a safe local ID.
- **Planned commit:** `feat(telemetry): record local runtime events`.
- **Actual findings:** Stock server returns `server_timing.infer_ms` plus previous-request total timing. The implementation associates previous total time with request N-1, records monotonic/UTC events in private JSONL, preserves partial valid lines, measures write overhead, and emits allowlisted JSON/Markdown summaries. At the audit pin stock Runtime uses `time.time`; phase 04 owns the monotonic runtime correction.
- **Remaining blockers:** Local implementation/validation, then exact-candidate π₀ and π₀.₅ reruns and plot inspection.
- **Completion status:** Amendment in progress. Prior writes averaged 0.168 ms for π₀ and 0.175 ms for π₀.₅; the larger trajectory rows must independently remain below 1 ms p95.

Minimal instrumentation: time the existing calls and expose metrics to one subscriber; do not introduce a logging framework or database.
