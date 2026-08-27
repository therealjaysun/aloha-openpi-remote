# 04.01 — Observation contract

- **Objective:** Reject malformed simulator/policy data before it crosses the network or reaches MuJoCo.
- **Inputs/prerequisites:** Audited ALOHA transform and gym spaces.
- **Implementation tasks:** Validate mapping keys; preserve and record stock Gym numeric finite `state (14,) float64` unless live contract evidence requires a documented conversion; require `cam_high` and allowed image names only, uint8 CHW `(3,224,224)`; validate optional prompt; validate response `actions` observed dtype, numeric finite `(N,14)` with `N >= execution_horizon`; do not clip absolute policy joint commands to Gym's nominal box; verify profile/config/checkpoint/SHA metadata; preserve compliant arrays without redundant copies; produce actionable errors.
- **Files expected to change:** `tools/remote_aloha/observation_contract.py`, its one focused test file, call sites in smoke/runtime.
- **Validation:** Valid stock observation/action plus missing key, wrong layout/dtype/shape, unknown camera, NaN/Inf, short chunk, identity mismatch, and nominal-box-exceeding but finite absolute action cases.
- **Acceptance:** Same validator is used by policy smoke and runtime; errors name field/expected/actual; no server call on invalid input and no sim step on invalid action.
- **Planned commit:** `feat(runtime): validate ALOHA policy contracts`.
- **Actual findings:** Stock sim produces only `cam_high`; wrist cameras are optional/masked by `AlohaInputs`. Both profiles share the same ALOHA input/output transform and 50×14 wire action shape.
- **Remaining blockers:** None for implementation; real response metadata must be observed.
- **Completion status:** Planned.
