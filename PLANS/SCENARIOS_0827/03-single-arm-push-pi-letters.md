# S0827.03 — Single-arm Push-`PI` letters

- **Objective:** Evaluate one active ALOHA arm pushing two independently initialized rigid uppercase `P` and `I` bodies onto their respective targets.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); primitive letter geometry and target smoke; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Create exactly two named free bodies from the frozen compound-box descriptor: an uppercase `P` and a single-piece uppercase `I` with no dot; activate the left six joint targets, project the right six to post-settle home, and fix both grippers; deterministically rejection-sample two supported, upright, non-overlapping, non-goal poses in the calibrated region; require each named body to match its own fixed target for held success; share scene hash, sampled/settled poses, prompt, and goals with S0827.04.
- **Files expected to change:** Only the shared Push-PI environment/registry and focused geometry/reset/outcome tests; no per-letter environment classes, mesh assets, or new geometry dependency.
- **Validation:** For seeds 0, 1, and 2 verify deterministic named sampled/settled poses, complete footprint support, clearance, stable hold settling, scene-hash pairing with S0827.04, and policy-resolution color-mask visibility; verify one correct letter is insufficient, swapped letters fail, exact targets succeed, pose/lift/fall/off-table rules work, discarded right-arm commands never reach applied telemetry, the fixed prompt reaches the first request, and complete/partial live/video outputs contain both letters and targets.
- **Acceptance:** Three episodes per required profile end in `success`, `off_table`, `fallen`, or an exact 300-step `time_limit`; partial runs do not count and must be rerun. Two valid random bodies start fully on the table, the right arm stays parked, results contain per-letter and joint success metrics, and each video/manifest is readable even when neither letter reaches its goal.
- **Planned commit:** Included in `feat(sim): add shared Push-PI environment` and `feat(runtime): integrate Push-PI scenarios and display`.
- **Actual findings:** Gym-Aloha has no multi-letter task or general object-pose sampler; its global one/two-object `BOX_POSE` path should not be extended for this suite.
- **Remaining blockers:** Exact-candidate local/hosted gates and both hardware matrices. Zero-shot sequencing remains an experiment result.
- **Completion status:** Implementation and Mac calibration pass with uppercase dotless `P`/`I`; hardware validation pending.

`ponytail:` primitive glyphs are sufficient for the first experiment; add authored meshes only if top-camera recognition is demonstrably inadequate.
