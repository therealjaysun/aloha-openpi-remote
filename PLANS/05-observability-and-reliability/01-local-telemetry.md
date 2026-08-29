# 05.01 — Local telemetry

- **Objective:** Record structured Mac-side control/inference events without materially disturbing 50 Hz stepping.
- **Inputs/prerequisites:** Working runtime and buffer; timestamped ignored run directory.
- **Implementation tasks:** Keep the existing line-buffered `step` event and exact 14-joint vectors. For the opt-in staged diagnostic, add only safe stage IDs/boundaries/events; never record prompt text. At each boundary drain the old in-flight request, discard every old queued action, then fetch the new stage before applying its first step. Preserve partial rows and the no-fsync/no-network step path.
- **Files expected to change:** `tools/remote_aloha/telemetry.py`, `tools/remote_aloha/run.py`, a minimal trajectory helper, focused tests, and the already-used lightweight dependency lock only if needed for installed Matplotlib.
- **Validation:** Existing trajectory tests plus exact stage coverage, invalid schedule/ID rejection, old-prefetch flushing, safe-summary omission of prompts, and unchanged sub-1-ms telemetry-write budget.
- **Acceptance:** Trajectory rows equal successfully applied steps for passing episodes; valid partial rows survive interruption; no per-step fsync/network/SSH; raw vectors and plot paths stay only in ignored private outputs; publishable output contains only bounded counts/status and a safe local ID.
- **Planned commit:** `feat(telemetry): record local runtime events`.
- **Actual findings:** The existing `step` event now records the exact zero-based simulation step, one-based applied step, monotonic elapsed time, and finite 14-value actual/commanded vectors immediately after each successful step. Synthetic 300-row p95 was 0.110 ms; hardware episode p95 stayed at or below 0.162 ms for π₀ and 0.197 ms for π₀. All 1,661 hardware step rows matched applied-step order and survived in ignored line-buffered JSONL.
- **Remaining blockers:** None.
- **Completion status:** Complete at hardware candidate `2065dd9d`; focused shape/finite/coverage/interruption tests and both exact-candidate profile runs passed.

Minimal instrumentation: time the existing calls and expose metrics to one subscriber; do not introduce a logging framework or database.
