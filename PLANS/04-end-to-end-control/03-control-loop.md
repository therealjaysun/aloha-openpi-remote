# 04.03 — Control loop

- **Objective:** Connect validated observations → tunneled inference → buffered actions → MuJoCo using stock OpenPI runtime pieces.
- **Inputs/prerequisites:** Ready server/tunnel; buffer/contract tests; selected profile.
- **Implementation tasks:** Construct a fresh raw Gym ALOHA environment per explicit seed; use a bounded WebSocket client at `127.0.0.1:<validated-port>` and verify tunnel plus handshake profile/config/checkpoint/SHA before action execution. Use the buffered policy in a direct monotonic loop that checks the cap before each action, never catches up after a wait, validates the action immediately before Gym, records post-step terminal frames, and stops on source termination/truncation. Make `VideoSaver` idempotently write available frames through a same-directory temporary file and atomic rename. `finally` closes policy/socket/worker and environment and preserves a mode-600 manifest even on interruption or encoding failure.
- **Files expected to change:** `tools/remote_aloha/run.py`, `scripts/run_aloha.sh`, config, `Makefile`, `examples/aloha_sim/saver.py`, focused tests.
- **Validation:** Short controlled run with exact requested actions; exact seed passed to the sole reset; interruption/connection loss preserves exact partial step count; finalizer idempotence and atomic output; no catch-up pacing; no orphan client worker.
- **Acceptance:** Correct data flow with no schema/network termination; exact full episode runs; selected server identity matches; complete or explicitly partial video/result manifest saved; process exits in bounded time and releases local resources.
- **Planned commit:** `feat(runtime): connect ALOHA simulation to remote policy`.
- **Actual findings:** Stock ALOHA main exposes host/port/task/seed/horizon/display but only one episode and no profile/telemetry. Stock environment derives even its first Gym seed from an RNG. Stock Runtime supports episode count/max steps, but shortened runs currently execute `max_episode_steps + 1` actions, timing uses the wall clock, subscribers finalize only on normal completion, and a final reset always runs.
- **Remaining blockers:** None for implementation; Phase 02 bounded conversion and Phase 03 real actions are complete.
- **Completion status:** Implemented and locally tested; real episodes pending.
