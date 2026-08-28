# 05.03 — Failure handling

- **Objective:** Bound waits, avoid stale actions, clean owned processes, and preserve evidence for recoverable failures.
- **Inputs/prerequisites:** Runtime/tunnel/server/metrics lifecycle and telemetry.
- **Implementation tasks:** Reuse phase 03's finite client open/metadata/inference timeouts and `close()`; apply validated bounded retries/backoff only to client construction/connect/metadata before environment reset or any inference; recreate and identity-check each attempted client. After an inference may have been sent, a timeout/disconnect has an ambiguous server outcome: close transport, discard buffered actions, abort the episode, and never replay automatically. Never step with invalid, prior-generation, elapsed-leading, or repeated stale action; preserve partial JSONL/video/result summary; handle Ctrl+C; use exact owned-process validation and cleanup; report exact next command for port/render/GPU/checkpoint/routing failures. If a checkpoint looks partial/corrupt, record its exact private path/error locally and request approval before targeted cache deletion.
- **Files expected to change:** Buffered policy/run/lifecycle scripts; config/tests/docs. No second client timeout implementation.
- **Validation:** Server absent/start slow with bounded constructor retry; identity mismatch without retry; inference timeout/disconnect after submit with exactly one request and no later environment step; invalid response; buffer underrun; Ctrl+C; stale/unrelated PID; partial video/write error; retry exhaustion.
- **Acceptance:** No infinite wait, broad catch, hidden failure, stale action replay, or unrelated kill; partial results remain readable; cleanup is idempotent.
- **Planned commit:** `fix(runtime): bound remote failures and preserve partial runs`.
- **Actual findings:** Stock client retries connection-refused forever and calls `recv()` without a project-level timeout; phase 03 owns the minimal OpenPI client patch. WebSockets 14.1 already supports the required bounded API.
- **Remaining blockers:** Live startup/retry timing and owned sampler cleanup require exact-candidate hardware validation.
- **Completion status:** Implementation and injected-failure validation complete; hardware evidence pending.
