# S0827.01 — Single-arm Push-PI glyph

- **Objective:** Prompt one ALOHA arm to push one rigid Greek π body into a visible matching target while leaving the complete bimanual action under model control.
- **Inputs/prerequisites:** Shared contracts in [`00-overview.md`](00-overview.md); native scenario smoke; Phase 04 buffered runner; selected policy profile and explicit scenario prompt.
- **Implementation tasks:** Keep the one-go left-arm prompt, but pass all 14 model commands through unchanged. “Single arm” is requested behavior only; record any right-arm participation or interference rather than suppressing it.
- **Files expected to change:** Only the shared action projection/matrix validation, focused identity/runner tests, descriptor provenance, and documentation; no loader, telemetry, environment, or simulator change.
- **Validation:** Keep historical locked results labeled. Run one exact-candidate unlocked fixed-prompt π₀ block-transfer seed-0 diagnostic; require complete 14-D command pass-through, no prompt transitions, exact natural-terminal coverage/joint/video rows under the 6,000-step ceiling, a `224×672` ordered MP4, valid plot/GPU metrics, visual inspection across all panels, and clean stop.
- **Acceptance:** Report `(best coverage descending, time-to-best ascending)`, initial/final coverage, total elapsed time, and existing success/safety fields. The measured value may be 0%; one episode measures this candidate but does not prove optimality.
- **Planned commit:** Included in `feat(sim): add shared Push-PI environment` and `feat(runtime): integrate Push-PI scenarios and display`.
- **Actual findings:** The policy ABI is 14-D, so “left arm only” is expressed in the prompt, not by discarding valid right-arm or gripper commands. The unlocked model moved the right arm more than the prompted left arm but still made no object contact.
- **Actual validation:** Exact candidate `1c0604e` completed 6,000/6,000 coverage/joint/video samples over 120 simulated/180.05 wall seconds. Every row was finite 14-D at exact steps 0–5,999 with monotonic time; every row commanded the formerly parked right arm and both formerly fixed grippers away from their old values. Coverage stayed `0.0%`; there was no contact, lift, fall, off-table event, or visible π displacement. Right/left travel was `31.2387/9.3381 rad`; all three views and 14 actual plus 14 commanded series passed inspection. GPU coverage, local gates, two stops, and final free-port doctor passed.
- **Remaining blockers:** None for the scoped unlocked Scenario 1 diagnostic. Historical matrices do not certify unlocked hardware behavior across all scenarios.
- **Completion status:** Complete; earlier locked-action evidence remains historical.

`ponytail:` the visible right robot remains in the scene; remove it only if measured occlusion or contact changes the experiment.
