# S0827.01 — Single-arm Push-PI glyph

- **Objective:** Evaluate one active ALOHA arm pushing one rigid Greek π body into a visible matching target.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); native scenario smoke; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Select the shared rigid-π descriptor; activate the left six joint targets; project the right six joints to the post-settle home pose and both grippers to the frozen pusher value before stepping; use an explicit left-arm-only prompt; sample one supported non-goal π pose from the calibrated comparison region; use the same seed-derived sampled/settled pose, target, and scene hash as S0827.02; expose exact target-footprint coverage and timing through existing telemetry.
- **Files expected to change:** Only the shared pure scenario module, shared environment/registry, existing runner/config/telemetry/display integrations, and focused tests listed in the overview; no single-arm simulator fork or 7-D policy transform.
- **Validation:** Keep the historical paired-seed checks. For the coverage amendment, run π₀ seed 0 once at the exact candidate; require one finite `[0,100]` coverage sample per applied step, exact final/best row linkage, earliest-best monotonic time, valid video/trajectory, and clean stop.
- **Acceptance:** Report `(best coverage descending, time-to-best ascending)`, initial/final coverage, total elapsed time, and existing success/safety fields. The measured value may be 0%; one episode measures this candidate but does not prove optimality.
- **Planned commit:** Included in `feat(sim): add shared Push-PI environment` and `feat(runtime): integrate Push-PI scenarios and display`.
- **Actual findings:** The current ALOHA policy ABI cannot accept a literal 7-D state/action. Holding the inactive half is the minimal compatible experiment.
- **Actual validation:** The historical six episodes remain unchanged. The exact coverage candidate's π₀ seed-0 run completed 300/300 rows with `0.0%` coverage and no meaningful displacement. A separate experimental π₀.₅ base diagnostic completed 6,000/6,000 rows over 120 simulated seconds, also at `0.0%`; the arm swept the π farther from its target around step 960, then remained near joint limits. Both videos/plots passed inspection and both cleanup checks left the policy port free. The extended result is not pooled with acceptance evidence.
- **Remaining blockers:** None. PR 7 ancestry remains a merge-order dependency, not a technical blocker.
- **Completion status:** Complete; E-MAC/E-PC-S0827-COVERAGE plus separate E-MAC/E-PC-S0827-LONG diagnostic. Prior six episodes remain historical.

`ponytail:` the visible right robot remains in the scene; remove it only if measured occlusion or contact changes the experiment.
