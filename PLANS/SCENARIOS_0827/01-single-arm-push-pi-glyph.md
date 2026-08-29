# S0827.01 — Single-arm Push-PI glyph

- **Objective:** Evaluate one active ALOHA arm pushing one rigid Greek π body into a visible matching target.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); native scenario smoke; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Select the shared rigid-π descriptor; activate the left six joint targets; project the right six joints to the post-settle home pose and both grippers to the frozen pusher value before stepping; use an explicit left-arm-only prompt; sample one supported non-goal π pose from the calibrated comparison region; use the same seed-derived sampled/settled pose, target, and scene hash as S0827.02; expose exact target-footprint coverage and timing through existing telemetry.
- **Files expected to change:** Only the shared pure scenario module, shared environment/registry, existing runner/config/telemetry/display integrations, and focused tests listed in the overview; no single-arm simulator fork or 7-D policy transform.
- **Validation:** Keep the historical paired-seed checks. For the coverage amendment, run π₀ seed 0 once at the exact candidate; require one finite `[0,100]` coverage sample per applied step, exact final/best row linkage, earliest-best monotonic time, valid video/trajectory, and clean stop.
- **Acceptance:** Report `(best coverage descending, time-to-best ascending)`, initial/final coverage, total elapsed time, and existing success/safety fields. The measured value may be 0%; one episode measures this candidate but does not prove optimality.
- **Planned commit:** Included in `feat(sim): add shared Push-PI environment` and `feat(runtime): integrate Push-PI scenarios and display`.
- **Actual findings:** The current ALOHA policy ABI cannot accept a literal 7-D state/action. Holding the inactive half is the minimal compatible experiment.
- **Actual validation:** Six exact-SHA hardware episodes completed 1,800 applied steps and matching trajectory rows with six valid videos and six inspected 14-joint plots. All reached the time limit with no contact, lift, fall, off-table event, or task success.
- **Remaining blockers:** None. PR 7 ancestry remains a merge-order dependency, not a technical blocker.
- **Completion status:** Coverage amendment pending E-MAC-S0827-COVERAGE and E-PC-S0827-COVERAGE; prior six episodes remain historical.

`ponytail:` the visible right robot remains in the scene; remove it only if measured occlusion or contact changes the experiment.
