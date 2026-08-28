# 05.01 — Local telemetry

- **Objective:** Record structured Mac-side control/inference events without materially disturbing 50 Hz stepping.
- **Inputs/prerequisites:** Working runtime and buffer; timestamped ignored run directory.
- **Implementation tasks:** Extend the existing line-buffered `step` event after each successful `environment.step`: record one-based applied step, monotonic elapsed seconds, finite post-step `agent_pos`, and the finite command just applied, each exactly 14 values. Derive names/order from pinned gym-aloha 0.1.1. Normalize arm radians with the fixed MuJoCo XML joint ranges and grippers with gym-aloha's documented `[0,1]` close/open coordinate; never use observed extrema. Preserve valid partial rows and the existing no-fsync/no-network step path.
- **Files expected to change:** `tools/remote_aloha/telemetry.py`, `tools/remote_aloha/run.py`, a minimal trajectory helper, focused tests, and the already-used lightweight dependency lock only if needed for installed Matplotlib.
- **Validation:** Synthetic 14-joint round trip; shape/NaN/infinity rejection; exact step/sample coverage; monotonic elapsed-time checks; Ctrl+C/failure partial rows; 300-row writer benchmark with both vectors remains below 1 ms p95 on Mac.
- **Acceptance:** Trajectory rows equal successfully applied steps for passing episodes; valid partial rows survive interruption; no per-step fsync/network/SSH; raw vectors and plot paths stay only in ignored private outputs; publishable output contains only bounded counts/status and a safe local ID.
- **Planned commit:** `feat(telemetry): record local runtime events`.
- **Actual findings:** The existing `step` event now records the exact zero-based simulation step, one-based applied step, monotonic elapsed time, and finite 14-value actual/commanded vectors immediately after each successful step. Synthetic 300-row p95 was 0.110 ms; hardware episode p95 stayed at or below 0.162 ms for π₀ and 0.197 ms for π₀. All 1,661 hardware step rows matched applied-step order and survived in ignored line-buffered JSONL.
- **Remaining blockers:** None.
- **Completion status:** Complete at hardware candidate `2065dd9d`; focused shape/finite/coverage/interruption tests and both exact-candidate profile runs passed.

Minimal instrumentation: time the existing calls and expose metrics to one subscriber; do not introduce a logging framework or database.
