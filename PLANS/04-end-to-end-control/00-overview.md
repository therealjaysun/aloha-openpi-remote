# Phase 04 — End-to-end control

- **Objective:** Execute remote policy action chunks in Mac ALOHA simulation at a 50 Hz target for complete seeded episodes with either policy profile.
- **Scope:** Observation/action contracts, step-aware action buffer, one background inference request, direct exact-step simulation loop, episode result/video finalization, per-profile episodes.
- **Non-goals:** Training, temporal ensembling, multi-server failover, success-rate optimization, changing ALOHA physics or policy transforms.
- **Dependencies:** Phase 03 local client/tunnel interfaces for pure implementation. Hardware acceptance additionally requires passing Mac simulation and both profile smoke tests through the tunnel.
- **Planned files:** `tools/remote_aloha/{observation_contract,action_buffer,buffered_policy,run}.py`, focused tests, `examples/aloha_sim/saver.py`, `scripts/run_aloha.sh`, `Makefile`, config/docs/plans.
- **Planned commits:** `feat(runtime): validate ALOHA policy contracts`; `feat(runtime): execute prefetched action chunks`; `feat(runtime): connect ALOHA simulation to remote policy`; `test(runtime): run complete remote-policy episodes`.
- **Branch:** `codex/04-end-to-end-control`.
- **PR base:** `codex/03-secure-connectivity`.
- **PR title:** `feat(runtime): connect ALOHA simulation to OpenPI`.
- **Acceptance criteria:** Strict schema/dtype/finite checks; `1 <= prefetch < execution horizon <= 50`; per-episode client lifetime plus request-step tracking prevents prior-episode or elapsed-leading action replay; warmed p95 inference plus 100 ms margin fits the prefetch budget and telemetry has zero underruns before uninterrupted 50 Hz is claimed; active and wall-clock rates/waits are distinct; exact step counts; graceful Ctrl+C and idempotent partial finalization; three explicit one-episode seeds per profile; videos/rewards/completion saved; infrastructure and task success reported separately.
- **Test commands:** `make test`; `make smoke-sim`; for each profile/backend run `make server`, `make tunnel`, `make smoke-policy`, `ALOHA_SEED=0 ALOHA_EPISODES=3 OPENPI_POLICY_BACKEND=pytorch make run`, inspect summaries/videos, then `make stop`; finish with `git status --short`.
- **Risks:** Synchronous inference stalls 50 Hz; prefetch chunk becomes stale; buffer underrun; action discontinuity; network loss; invalid/NaN action damages simulated state; output loss on interruption.
- **Rollback:** Select stock synchronous `ActionChunkBroker` path for diagnosis or revert phase; stop simulation/tunnel/server; retain partial ignored outputs.
- **Current status:** Implementation and Mac validation complete; two-profile hardware acceptance pending.
- **Actual results:** 205 tests plus lint/format/shell gates pass; the real Mac simulator completed seeds 0–2 for 900 total steps at 13.76 ms aggregate p95 with a decoded 300-frame, 50 fps video. No policy-controlled episode has run on this candidate yet.
- **Deviations:** Both π₀ and experimental π₀.₅ profiles are first-class runtime selections; evaluation labels remain distinct. A direct fresh-per-episode loop replaces planned patches to the generic upstream Runtime, eliminating unrelated shared-runtime behavior and cross-episode state from this demo path.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Minimal runtime design

Reuse the WebSocket protocol, ALOHA transforms, Gym task, and `VideoSaver`. A thin buffered policy blocks for the first chunk, executes at most 30 of 50 actions, and prefetches with 25 remaining. Each future is tagged with its observation step; when it returns, elapsed leading actions are dropped and the fresh slice atomically replaces—not appends behind—the old buffer. If nothing fresh remains, inference waits without stepping. One `ThreadPoolExecutor(max_workers=1)` is sufficient.

Use a direct monotonic one-episode loop with one fresh Gym environment, WebSocket client, buffer, and output directory per explicit seed. This avoids patching or depending on the stock generic Runtime's max+1/final-reset/finalizer behavior and makes prior-episode action reuse impossible by construction. The simulator seeds are explicit; policy sampling is not claimed deterministic because the remote API exposes no RNG seed.

## Machine handoff

This is a both-machines phase: keep the PC on/awake for inference while the Mac runs MuJoCo. Restart server and tunnel after any PC reboot; do not require a second workspace or manual source copy.
