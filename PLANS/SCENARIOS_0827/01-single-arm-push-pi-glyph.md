# S0827.01 — Single-arm Push-PI glyph

- **Objective:** Evaluate one active ALOHA arm pushing one rigid Greek π body into a visible matching target.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); native scenario smoke; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Select the shared rigid-π descriptor; activate the left six joint targets; project the right six joints to the post-settle home pose and both grippers to the frozen pusher value before stepping; sample one supported non-goal π pose from the calibrated comparison region; use the same seed-derived sampled/settled pose, prompt, target, and scene hash as S0827.02; expose the fixed bounded scenario state in existing step telemetry.
- **Files expected to change:** Only the shared pure scenario module, shared environment/registry, existing runner/config/telemetry/display integrations, and focused tests listed in the overview; no single-arm simulator fork or 7-D policy transform.
- **Validation:** Reset/step/render seeds 0, 1, and 2; verify finite `(14,)` observation/applied action while adversarial discarded right-arm commands yield projected trajectory rows and keep actual right joints within the calibrated tolerance; verify exact pairing with S0827.02; check supported random poses, fixed-gripper projection, exact-goal success, perturbed/off-table/lift/fall failure, explicit prompt, bounded info, live view, and complete/partial MP4.
- **Acceptance:** Three episodes per required profile end in `success`, `off_table`, `fallen`, or an exact 300-step `time_limit`; partial runs do not count and must be rerun. Every start is fully supported and inside the Mac-calibrated operational region; the right arm remains parked; videos/manifests are valid; actual `push_success` and `lifted_ever` are reported without requiring nonzero success.
- **Planned commit:** Included in `feat(sim): add shared Push-PI environment` and `feat(runtime): integrate Push-PI scenarios and display`.
- **Actual findings:** The current ALOHA policy ABI cannot accept a literal 7-D state/action. Holding the inactive half is the minimal compatible experiment.
- **Actual validation:** Six exact-SHA hardware episodes completed 1,800 applied steps and matching trajectory rows with six valid videos and six inspected 14-joint plots. All reached the time limit with no contact, lift, fall, off-table event, or task success.
- **Remaining blockers:** None. PR 7 ancestry remains a merge-order dependency, not a technical blocker.
- **Completion status:** Complete; E-MAC-S0827 and E-PC-S0827.

`ponytail:` the visible right robot remains in the scene; remove it only if measured occlusion or contact changes the experiment.
