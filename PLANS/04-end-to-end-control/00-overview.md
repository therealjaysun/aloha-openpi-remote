# Phase 04 — End-to-end control

- **Objective:** Execute remote policy action chunks in Mac ALOHA simulation at a 50 Hz target for complete seeded episodes with either policy profile.
- **Scope:** Observation/action contracts, generation/step-aware action buffer, one background inference request, minimal shared Runtime correctness/cleanup fixes, episode result/video finalization, per-profile episodes.
- **Non-goals:** Training, temporal ensembling, multi-server failover, success-rate optimization, changing ALOHA physics or policy transforms.
- **Dependencies:** Phase 03 local client/tunnel interfaces for pure implementation. Hardware acceptance additionally requires passing Mac simulation and both profile smoke tests through the tunnel.
- **Planned files:** `tools/remote_aloha/{observation_contract,action_buffer,buffered_policy,run}.py`, `tests/test_{observation_contract,action_buffer}.py`, minimal focused changes/tests in `packages/openpi-client/.../runtime.py` and `examples/aloha_sim/{env,saver}.py`, `scripts/run_aloha.sh`, `Makefile`, config/docs/plans.
- **Planned commits:** `feat(runtime): validate ALOHA policy contracts`; `feat(runtime): execute prefetched action chunks`; `fix(runtime): correct episode accounting and finalization`; `feat(runtime): connect ALOHA simulation to remote policy`; `test(runtime): run complete remote-policy episodes`.
- **Branch:** `codex/04-end-to-end-control`.
- **PR base:** `codex/03-secure-connectivity`.
- **PR title:** `feat(runtime): connect ALOHA simulation to OpenPI`.
- **Acceptance criteria:** Strict schema/dtype/finite checks; `1 <= prefetch < execution horizon <= 50`; request generation/step tracking prevents prior-episode or elapsed-leading action replay; warmed p95 inference fits the prefetch budget before uninterrupted 50 Hz is claimed; active and wall-clock rates/waits are distinct; exact step counts; graceful Ctrl+C and idempotent partial finalization; three explicit one-episode seeds per profile; videos/rewards/completion saved; infrastructure and task success reported separately.
- **Test commands:** `make test`; `make smoke-sim`; `make smoke-policy` per profile; `OPENPI_POLICY_PROFILE=<profile> make run` per profile; inspect summaries/videos; `git status --short`.
- **Risks:** Synchronous inference stalls 50 Hz; prefetch chunk becomes stale; buffer underrun; action discontinuity; network loss; invalid/NaN action damages simulated state; output loss on interruption.
- **Rollback:** Select stock synchronous `ActionChunkBroker` path for diagnosis or revert phase; stop simulation/tunnel/server; retain partial ignored outputs.
- **Current status:** Plan complete; implementation not started.
- **Actual results:** No policy-controlled episode ran.
- **Deviations:** Both π₀ and experimental π₀.₅ profiles are first-class runtime selections; evaluation labels remain distinct.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Minimal runtime design

Reuse `AlohaSimEnvironment`, `PolicyAgent`, `Runtime`, WebSocket protocol, ALOHA transforms, and `VideoSaver`. Add a thin buffered policy: block for the first chunk, initially execute 10 of 50 actions, and initially prefetch with five remaining. Tag the request with episode generation and observation step; when it returns, discard results from an old generation, drop elapsed leading actions, and atomically replace—not append behind—the old buffer. If nothing fresh remains, re-infer from the latest observation. One `ThreadPoolExecutor(max_workers=1)` is sufficient; no async framework or custom scheduler.

Patch the shared Runtime once to use monotonic scheduling, stop after exactly `max_episode_steps`, and finalize subscribers in `finally`; suppress its real-robot-oriented final reset for this simulation path. Patch the sim environment so its first reset uses the constructor seed verbatim (later resets may use its RNG), then run a fresh one-episode environment for each seed so reported 0/1/2 values are the actual Gym seeds.

## Machine handoff

This is a both-machines phase: keep the PC on/awake for inference while the Mac runs MuJoCo. Restart server and tunnel after any PC reboot; do not require a second workspace or manual source copy.
