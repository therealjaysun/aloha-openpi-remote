# S0827.02 — Two-arm Push-π glyph

- **Objective:** Evaluate the same rigid Greek π task with both ALOHA arms available; record whether the policy actually uses one or both.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); passing S0827.01 environment/reset tests; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Select the same rigid-π descriptor and RNG stream as S0827.01; enable both six-joint arm command groups while projecting both grippers to the identical pusher value; keep scene hash, sampled/settled pose, target, camera, prompt, success rules, episode length, and seeds fixed so the arm mask is the only pair delta; record the overview's fixed per-arm travel/contact/participation/interference fields without making them success requirements.
- **Files expected to change:** Only the shared Push-π environment/registry and matrix tests; do not fork geometry, reset, reward, display, video, or runner code from S0827.01.
- **Validation:** Assert paired seeds produce identical scene hashes and sampled/settled/target π poses; verify both six-joint groups reach the simulator only after full-response validation and gripper projection; test order-independent named contacts and participation; rerun exact-goal/perturbation/lift/fall/off-table checks; exercise the Mac display smoke and complete/partial MP4 path without redundant display-on GPU runs.
- **Acceptance:** Three episodes per required profile complete with both arm command groups available and valid artifacts; paired manifests prove identical scene initial conditions to S0827.01 and report actual per-arm participation; report per-profile paired success/time/raw-error deltas without treating three seeds as a statistical claim.
- **Planned commit:** Included in `feat(sim): add shared Push-pi environment` and `test(scenarios): validate Push-pi matrix workflow`.
- **Actual findings:** The installed MuJoCo/Gym-Aloha model is already bimanual and its policy/state contract already matches both model profiles.
- **Remaining blockers:** Preferred branch base awaits the seven-PR merge; no technical design blocker remains. Zero-shot coordination competence is an experiment result.
- **Completion status:** Ready for implementation after the branch gate; not started.
