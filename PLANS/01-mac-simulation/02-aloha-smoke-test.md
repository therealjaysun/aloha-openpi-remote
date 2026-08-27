# 01.02 — ALOHA smoke test

- **Objective:** Prove `gym_aloha/AlohaTransferCube-v0` resets, observes, acts, and terminates correctly on Mac.
- **Inputs/prerequisites:** Working native environment; stock gym registration.
- **Implementation tasks:** Instantiate `obs_type="pixels_agent_pos"`, `render_mode="rgb_array"`; reset with configured seed; assert keys/shapes/dtypes; assert action space `(14,)`; inspect and record the pinned task's info/reward success semantics; step at least 200 times using the current absolute `agent_pos` as a stable no-op when accepted, with deterministic finite fallback actions only if evidence requires it; do not blindly clip future policy commands to the nominal Gym box; stop/reset on termination; always close; report reward and only a source-defined success metric without asserting no-op policy success.
- **Files expected to change:** `tools/remote_aloha/sim_smoke_test.py`, `scripts/smoke_sim.sh`, one small test for pure validation logic.
- **Validation:** `make smoke-sim`; repeat seed 0; confirm exit code, step count, no orphan process.
- **Acceptance:** Raw top image `(480,640,3)` uint8, `agent_pos (14,)`, action `(14,)`, finite state/action, ≥200 steps, clean exit.
- **Planned commit:** `feat(sim): add native ALOHA simulation smoke test`.
- **Actual findings:** gym-aloha source registers TransferCube with a 300-step TimeLimit; metadata and physics use 50 Hz. The declared action bounds are `[-1,1]`, but the environment's own absolute start pose includes an elbow value above 1, so this box is not a safe universal clipping rule for policy outputs.
- **Remaining blockers:** Runtime package compatibility and physical render context untested.
- **Completion status:** Planned.
