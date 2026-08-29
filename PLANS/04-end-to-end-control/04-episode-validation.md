# 04.04 — Episode validation

- **Objective:** Demonstrate complete reproducible policy-controlled episodes for π₀ and π₀.₅ profiles.
- **Inputs/prerequisites:** Stable control loop; output directory ignored; profile-specific server restart procedure.
- **Implementation tasks:** For each profile launch a fresh one-episode environment for explicit seeds 0, 1, 2 through the 300-step TimeLimit unless the pinned task's source-defined success condition terminates sooner; capture video, read-only reward maximum/termination/raw info and source-defined success when available, request count, chunk/horizon/prefetch, dropped-leading actions, waits/errors, active step rate, wall-clock episode rate, package/project/upstream commits; verify artifacts; compare infrastructure pass and transfer success separately; do not pool profile results or invent `is_success` when the environment does not provide it.
- **Files expected to change:** Run summary generation, `README.md`, `PLANS/STATUS.md`, overview actual-results fields; generated evidence remains ignored.
- **Validation:** Six run manifests/videos (three per profile) or exact hardware blockers; replay artifact metadata; no tracked outputs.
- **Acceptance:** Each available profile completes three episodes with valid actions and no fatal schema/network errors; actual success rate and limitations are explicit.
- **Planned commit:** `test(runtime): validate complete remote-policy episodes`.
- **Actual findings:** Not run. Expected π₀ profile is task-specific; π₀.₅ base profile may have low/zero success and that does not invalidate connectivity infrastructure.
- **Remaining blockers:** Phase 01 is complete; Phase 02 cannot return actions on current memory, so Phases 03–04 cannot yet run.
- **Completion status:** Blocked by E-PC-JAX/E-PC-CONVERT, not by PC availability.
