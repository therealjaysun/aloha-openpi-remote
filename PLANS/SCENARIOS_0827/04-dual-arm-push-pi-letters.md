# S0827.04 — Two-arm Push-`PI` letters

- **Objective:** Evaluate the same independently initialized uppercase `P` and `I` task with both ALOHA arms available; record whether the policy actually uses one or both.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); passing S0827.03 geometry/reset tests; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Reuse S0827.03 descriptor, RNG, support/clearance checks, targets, camera, prompt, reward, and outcome rules unchanged; enable both six-joint command groups while projecting both grippers to the frozen pusher value; keep scene hash and sampled/settled poses byte-equivalent within each pair; record per-letter errors plus the overview's fixed travel/contact/participation/interference fields without making participation a success requirement.
- **Files expected to change:** Only the shared scenario mapping and matrix tests; no fork of the letter scene, sampler, display, saver, or control loop.
- **Validation:** Assert scene-hash and sampled/settled reset identity with S0827.03; verify both six-joint groups reach the simulator after projection and commanded telemetry matches exactly; rerun named-target, swapped-letter, one-letter-only, support/lift/fall/off-table, order-independent contact, and termination tests; exercise the Mac display and decoded complete/partial MP4 path without duplicate GPU display runs.
- **Acceptance:** Three episodes per required profile complete with both arm command groups available and valid artifacts; paired manifests permit direct single/two-arm comparison for each model/seed and report actual per-arm participation; joint and per-letter success remain separate and may be zero.
- **Planned commit:** Included in `feat(sim): add shared Push-pi environment` and `test(scenarios): validate Push-pi matrix workflow`.
- **Actual findings:** This is the highest-complexity zero-shot condition because the policy must coordinate two arms and sequence two semantic targets; that affects expected task quality, not the inference contract.
- **Remaining blockers:** Preferred branch base awaits the seven-PR merge; no technical design blocker remains. Zero-shot sequencing is an experiment result.
- **Completion status:** Ready for implementation after the branch gate; not started.
