# 05.03 — Failure handling

- **Objective:** Bound waits, avoid stale actions, clean owned processes, and preserve evidence for recoverable failures.
- **Inputs/prerequisites:** Runtime/tunnel/server/metrics lifecycle and telemetry.
- **Implementation tasks:** Reuse phase 03's finite client open/metadata/inference timeouts and `close()`; add validated bounded retry counts/backoff around connection-class errors only, recreating and identity-checking the client; never step with invalid, prior-generation, elapsed-leading, or repeated stale action; preserve partial JSONL/video/result summary; handle Ctrl+C; use shared PID identity validation and best-effort global cleanup; report exact next command for port/render/GPU/checkpoint/routing failures. If a checkpoint looks partial/corrupt, record its exact private path/error locally and request approval before targeted cache deletion.
- **Files expected to change:** Buffered policy/run/lifecycle scripts; config/tests/docs. No second client timeout implementation.
- **Validation:** Server absent/start slow; inference timeout; disconnect mid-chunk; invalid response; buffer underrun; Ctrl+C; stale/unrelated PID; partial video/write error; retry exhaustion.
- **Acceptance:** No infinite wait, broad catch, hidden failure, stale action replay, or unrelated kill; partial results remain readable; cleanup is idempotent.
- **Planned commit:** `fix(runtime): bound remote failures and preserve partial runs`.
- **Actual findings:** Stock client retries connection-refused forever and calls `recv()` without a project-level timeout; phase 03 owns the minimal OpenPI client patch. WebSockets 14.1 already supports the required bounded API.
- **Remaining blockers:** Live failure timing and retry policy require hardware validation; finite I/O design is not blocked.
- **Completion status:** Planned.
