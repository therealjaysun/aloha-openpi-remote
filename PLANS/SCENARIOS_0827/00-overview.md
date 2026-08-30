# Scenario suite 0827 — Push-PI

> This is a custom 3-D ALOHA **Push-PI** experiment inspired by PushT. It is not the standard PushT benchmark and its scores are not comparable to PushT results. The glyph scenario uses Greek π; the letter scenario uses uppercase dotless `P` and `I`.

- **Objective:** Evaluate one shared ALOHA tabletop-pushing environment across four seeded conditions and measure exact matching-target footprint coverage plus elapsed time.
- **Scope:** Stock Transfer Cube plus four fixed Push-PI scenario IDs; mandatory overhead and both wrists; one shared environment; identity pass-through of all 14 model commands; fixed scenario prompts plus the historical Scenario 1 staged diagnostic; existing buffered control, JSONL, trajectories, composite videos, GPU sampling, tunnel, and optional Mac viewer.
- **Non-goals:** Standard PushT compatibility or score; pixel/raster/pose-error progress heuristics; a new coverage-based success threshold; training/fine-tuning; a literal 7-D/single-arm robot; planar end-effector constraints or a general IK system; custom meshes; free-text or mid-episode prompt GUI; browser dashboard; parallel policy servers; nonzero zero-shot success as an infrastructure requirement.
- **Dependencies:** Completed phases 00–06; pinned OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; installed `gym-aloha==0.1.1`; converted PyTorch π₀ and π₀.₅ artifacts.
- **Planned files:** `tools/remote_aloha/scenarios.py` for pure descriptors/sampling/outcomes/projection; `examples/aloha_sim/push_pi_env.py` for the simulator adapter and four registrations; focused pure and simulator tests; small changes to config, run, smoke, telemetry, metrics, saver/display, Makefile, `.env.example`, README, and plans. Do not edit site-packages or add a geometry/GUI framework.
- **Planned commits:** `feat(sim): add shared Push-PI environment`; `feat(runtime): integrate Push-PI scenarios and display`; `test(scenarios): validate Push-PI matrix workflow`.
- **Branch/PR:** Preferred: after PRs 1–7 merge, create `codex/push-pi-scenarios` from updated `main` and open one PR to `main`. If explicitly scheduled before that merge, branch from `codex/06-hardening-docs`, base the PR there, and retarget only after the stack merges. Never implement from the old standalone `main`.
- **Acceptance:** Stock remains the default task and π₀.₅ is the default profile. Every finite 14-D model action reaches MuJoCo unchanged in stock and all custom scenarios; “single arm” is prompted and measured, not software-enforced. Standalone Scenario 1 and 2 diagnostics keep seed 0, three cameras, one fixed prompt, and the 6,000-step ceiling; they remain separate from 300-step acceptance evidence.
- **Test commands:** Run focused prompt/descriptor/three-view tests plus local lint/secret/public gates; GitHub Actions is disabled. Sync the exact candidate once, then run doctor/setup/server/smoke and one seed-0 headless 6,000-step fixed-prompt episode for each explicitly selected profile/scenario. Inspect metrics/artifacts, stop between profiles, stop twice at the end, and run final doctor. The two profile matrices remain historical.
- **Risks:** The π₀ simulator fine-tune used only the overhead camera, so wrist views may help, do nothing, or hurt; three real views increase inference latency and VRAM; novel-task zero-shot failure; compound bodies snag or tip; interrupted video encoding may fail even though JSONL/manifest data survives.
- **Rollback:** Leave or set `ALOHA_SCENARIO=transfer_cube`, which routes to the unchanged stock environment and identity action projection; disable the optional display; retain ignored partial artifacts; use `make stop` for owned server/tunnel cleanup.
- **Current status:** Complete on PR 8. Exact unlocked candidate `42a9e10` passed local gates, PC sync, both-profile Scenario 2 runs, artifact inspection, and cleanup; the earlier matrices remain historical locked-action evidence.
- **Actual results:** Candidate `1c0604e` proved no-lock Scenario 1 action pass-through with one exact 6,000-step π₀ run. Candidate `42a9e10` then sent Scenario 2's three ordered instructions once and completed exact 6,000-step runs for default π₀.₅ base and π₀ simulation-checkpoint. Both recorded 6,000 finite 14-D step/trajectory rows and three-view frames, zero prompt transitions, 0% coverage, no named contact/lift/fall/off-table, verified 14+14 series plots, and full GPU/cleanup evidence. π₀.₅ moved aggressively (`76.0704/232.1311 rad` left/right travel) and worsened target error by `0.00274457 m`; π₀ moved `9.7060/32.7359 rad` and changed error by only `-0.00000318 m`.
- **Deviations:** “Single arm” is prompt intent only. Neither arm nor gripper is project-locked. The new standalone diagnostics are descriptive, stochastic measurements; they do not replace the historical locked-action four-scenario matrices or prove causality.
- **PR:** [PR 8](https://github.com/therealjaysun/pi-robotics/pull/8).
- **Validated implementation SHA:** Unlocked Scenario 2 pair `42a9e10088650750ac0a940b13fbc324912d497a`; unlocked Scenario 1 `1c0604e97dbbd9333fd5fd9ed156582ab3334f1d`; locked fixed-prompt baseline `595bc4c067948b1f74c21de313bd832076197871`; locked matrix `7c2ec5927ad200e5aaf30bed0db4ef61cb9e2ba4`.

## Benchmark boundary

Standard PushT is a 2-D Pymunk task: a circular pusher receives a 2-D Cartesian target at 10 Hz and succeeds above 95% T-shape goal coverage. This suite borrows only the randomized-object/fixed-goal planar-pushing idea. Push-PI instead uses MuJoCo at 50 Hz, ALOHA's 14 absolute joint/gripper targets, fixed robot reset, 3-D failure rules, and project-defined held pose tolerances.

Primary references:

- [Official Diffusion Policy PushT environment](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py)
- [Official gym-pusht environment](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)
- Pinned local gym-aloha 0.1.1 source and assets installed by `examples/aloha_sim/requirements.txt`

Do not call a Push-PI result a PushT benchmark result, reuse PushT's 95% score, or imply that these OpenPI checkpoints were trained for either PushT or Push-PI.

## Fixed scenario and configuration contract

| `ALOHA_SCENARIO` | Gymnasium ID | Bodies | Prompt intent | Prompt |
| --- | --- | --- | --- | --- |
| `transfer_cube` (default) | `gym_aloha/AlohaTransferCube-v0` | Stock cube | Stock behavior | Existing profile behavior |
| `push_pi_single` | `pi_robotics/PushPiSingleArm-v0` | One rigid π | Left arm requested; all 14 commands pass | `Using only the left arm, first tilt the wrist down to see the pi-shaped block and its matching outline, then lower the gripper close to the table beside the block and make short incremental pushes that move the block into the outline; recheck alignment after each push and do not lift the block.` |
| `push_pi_dual` | `pi_robotics/PushPiBimanual-v0` | One rigid π | Both arms requested; all 14 commands pass | `Using both arms, first tilt both wrists down to see the pi-shaped block and its matching outline; then lower both grippers close to the table on opposite sides of the block without lifting it; finally make short coordinated incremental pushes that move the block into the outline, rechecking alignment after each push.` |
| `push_letters_single` | `pi_robotics/PushLettersSingleArm-v0` | Separate rigid uppercase `P` and `I` | Left arm requested; all 14 commands pass | `Using only the left arm, push the P and I blocks onto their matching targets.` |
| `push_letters_dual` | `pi_robotics/PushLettersBimanual-v0` | Separate rigid uppercase `P` and `I` | Both arms requested; all 14 commands pass | `Using both arms, push the P and I blocks onto their matching targets.` |

- `ALOHA_SCENARIO` accepts exactly the enum above and `ALOHA_DISPLAY` accepts `0|1`; unknown, blank-explicit, boolean-like, import-path, XML-path, or prompt values fail closed. Derive the task only from the selected scenario; the redundant `ALOHA_TASK` override is removed.
- Resolve scenario → Gym ID, descriptor, prompted arm mode, and fixed prompt once in `MacSimConfig`. A scenario prompt overrides the profile default and is immutable for an episode.
- For every scenario, convert the pinned MuJoCo `top`, `left_wrist`, and `right_wrist` cameras to `cam_high`, `cam_left_wrist`, and `cam_right_wrist`. Missing, extra, wrong-shaped, wrong-dtype, or non-three-view requests fail closed. Record the safe ordered camera set in run metadata.
- The optional Matplotlib window is display-only. It has no prompt text box, makes no network connection, and never changes policy input.
- Add `make scenario-matrix` for one fail-closed four-scenario/three-seed batch under the selected profile and `make scenario-metrics` for its rollup. A single `ALOHA_SCENARIO=<id> make run` remains useful for development.
- Keep 300 steps as the fixed acceptance limit. `ALOHA_EPISODE_STEPS` may extend only a standalone diagnostic to at most 6,000 steps; the matrix rejects that override and extended evidence is never pooled with acceptance results.

## Minimal implementation and data flow

1. Keep pure scenario descriptors, layout sampling, pose tests, outcome rules, and action projection in one simulator-independent module so public CPU CI does not need dm-control.
2. Add one repository-owned Gymnasium environment/task implementation and register the four fixed IDs with `max_episode_steps=300`. Import its registry before `gymnasium.make()` in both the active runner and simulation smoke path. Preserve `pixels.top` as HWC `uint8 (480,640,3)`, capture both pinned wrist cameras at the same size on reset/every step, preserve `agent_pos` as finite `float64 (14,)`, `info["is_success"]` as boolean on reset/step, 50 Hz metadata, and finite `(14,)` actions.
3. Build physics from the pinned transfer-cube XML: remove the cube body and its keyframe, inject project-owned primitive bodies/targets, and compile the XML string with a sorted in-memory mapping of the installed include/mesh assets. Never modify or copy the installed robot assets.
4. After the buffered policy validates a complete `(50,14)` response and pops one finite action, copy all 14 values unchanged, validate the applied `(14,)` command, then call `environment.step()`. The existing step JSONL and commanded trajectory record exactly that complete model command for every scenario.
5. Return a fixed, flat, bounded scenario `info` schema. Copy approved numeric/boolean pose, support, lift, fall, and contact values into the existing per-step JSONL event; do not create another logger. Keep detailed poses in ignored manifests/events and publish only allowlisted aggregates.
6. Compose `cam_high`, `cam_left_wrist`, and `cam_right_wrist` left-to-right in every active episode MP4/display frame. Complete/partial encoding stays atomic and one frame per applied step.
7. On reset and every applied step, compute exact planar overlap between each rigid glyph footprint and its named target from the frozen component rectangles. Record the per-body and target-area-weighted coverage fractions in existing scenario info; do not change reward, success, or termination.
8. Route scenario outputs to ignored `outputs/scenarios_0827/<batch-id>/<profile>/<scenario>/seed-<seed>/`. Produce per-scenario summaries first, then a matrix rollup that fails unless exact candidate/profile, all four IDs, seeds 0–2, scene pairing, artifact coverage, and safe IDs agree. Do not let the first scenario's metadata stand in for the matrix.

## Mac-only calibration gate

No PC or policy inference is used until one committed descriptor freezes all geometry/physics/evaluation values. No value may vary by model profile or arm mode.

- Freeze each primitive's body-relative offset, box half-size, origin, thickness, density, friction, color, resting height, footprint radius, target pose, display decimation, and every pose/lift/fall/hold threshold.
- The pinned tabletop is a 12-triangle rectangular mesh with SHA-256 `76a1571d1aa36520f2bd81c268991b99816c2a7819464d718e0fd9976fe30dce`; its transformed top support is `x=[-0.6096,0.6096]`, `y=[0.219,0.981]`, `z=0`. Regression-check the asset hash and these source-derived bounds; do not infer authority from `geom_aabb` or observed object positions.
- Operationally calibrate the conservative spawn polygon with deterministic 300-step waypoint sweeps that bracket its inset X/Y limits for both arms and all movable types. This proves only the tested region, not continuous analytical reachability. Bound physics-invalid reset retries and report scenario/seed on exhaustion.
- During reset, initialize both grippers to the same finite neutral value, settle deterministically, and capture the post-settle 14-D home state for provenance/calibration. After reset, do not park an arm or replace gripper commands; all model values pass through. The scripted calibration may still deliberately command home/neutral values and measure drift.
- Render every canonical/random reset at policy resolution and require deterministic color-mask visibility for every movable body and target plus human inspection of the four canonical reset PNGs.
- The canonical fixture exists only for scripted smoke calibration. Scored episodes always use seeded randomized layouts.
- Compute `scene_hash = SHA256(object kind + generated XML bytes + sorted relative asset-name/content hashes)`. The generated XML already carries physical geometry/target values; exclude seed, prompt, and prompted arm mode so paired single/dual scenarios share a hash.

## Reset and outcome contract

- `sample_layout(object_kind, seed)` uses only local `numpy.random.default_rng(seed)` before arm-mode logic. Set named free joints in MuJoCo order `[x,y,z,qw,qx,qy,qz]` with yaw quaternion `[cos(yaw/2),0,0,sin(yaw/2)]`; never use trailing `qpos` slices or global `BOX_POSE`.
- Randomize object position/yaw inside the calibrated region inset by full compound-body footprints. Reject body/body, body/goal, body/robot, already-successful, unsupported, and unstable layouts with a bounded attempt count. Record sampled and settled named poses; paired arm modes require byte-identical values.
- Project-defined first-run success thresholds remain XY `<=0.03 m`, wrapped yaw `<=15°`, absolute roll/pitch `<=10°`, and COM height within `0.005 m` of rest, all true for five consecutive applied steps. These are Push-PI thresholds, not PushT authority.
- `lifted_ever` becomes permanently true if COM rises more than `0.01 m`; it prevents success. Terminate unsuccessfully if a transformed footprint leaves the source-derived tabletop boundary or roll/pitch exceeds `30°`. Otherwise truncate at 300 steps. Reward is `1` only on held success, else `0`.
- For letters, uppercase `P` must match the `P` target and uppercase `I` the `I` target; one correct or swapped bodies fail. Treat the dotless uppercase `I` as 180° yaw-symmetric; `P` and Greek π retain ordinary 360° wrapped yaw.
- Coverage is `100 × area(actual planar footprint ∩ named target footprint) / area(named target footprint)`. Union overlapping descriptor rectangles exactly and weight the overall value by target area. Never normalize from observed motion or pixels. Report initial/final/best-applied coverage, the earliest applied step and monotonic elapsed time attaining the best value, and total elapsed time. Rank runs by higher best coverage, then lower time-to-best; one run is a measurement, not proof of optimality.

## Participation, policy, and evidence contract

- Per-arm `contact_ever` means any contact between a named movable-object geom and either named gripper-finger geom for that arm, independent of MuJoCo contact ordering. `both_arms_participated = left_contact_ever and right_contact_ever`. “Interference” means both arms contact the same named body on the same applied step. Record each arm's six-joint travel separately; motion alone is not participation.
- No-lift scoring evaluates the requested pushing behavior, but neither arm nor gripper is mechanically constrained by project code. Never call the controller planar, push-only, or single-arm-enforced.
- Validate both existing profiles separately. `pi0_aloha_sim` is Transfer-Cube-fine-tuned; `pi05_aloha_base` is an experimental ALOHA base profile. This is not a clean model-generation comparison and zero task successes are allowed.
- Keep the GPU server warm within one profile, but create a fresh client and `BufferedPolicy` for every episode as the runner already does. One pre-matrix `make smoke-policy` validates/warmups infrastructure. Within every scored episode, execute the first valid chunk normally; label only its latency cold and exclude only that latency from warmed p95.
- Extend publishable telemetry with fixed scenario/task allowlists and compact safe fields only: scenario, scene hash, episode/sample/artifact counts, terminal counts, push-success count, bounded coverage/error/contact values, monotonic durations, and safe local IDs. Require one coverage sample per applied step and preserve valid partial maxima. Reject arbitrary prompts/paths and omit raw poses, machine identifiers, absolute paths, and detailed JSONL.
- Keep the existing `<1 ms` per-step telemetry-write p95 budget with scenario fields.

## Validation and machine logistics

1. **Mac/pure:** Unknown config rejection; descriptor/hash; deterministic paired sampling; geometry/support/settling; exact success/perturbation/hold/lift/fall/off-table rules; order-independent contacts; action identity; applied-command/trajectory equality; bounded info/sanitizer behavior; partial JSONL/video/manifest; telemetry overhead; stock Transfer Cube regression. Simulator integration tests may skip only when the pinned Mac stack is absent and must pass locally.
2. **Mac/simulator:** Gym checker/API smoke, canonical fixed-waypoint push for each object layout, a hold-action 300-step run for every ID, four display-on visual smokes, decoded complete/partial videos, and clean ignored outputs. Freeze and commit the descriptor before GPU work.
3. **Candidate gate:** Finish Mac tests/lint/public/secret audits; commit, secret-scan, and push the exact candidate for PC synchronization. GitHub Actions is disabled by user choice for this local-only project. Verify Mac SHA = WSL SHA; reuse converted weights only when profile/runtime identity matches. Do not run general CI on the PC.
4. **Historical π₀ hardware baseline:** The completed top-only evidence used one 12-episode matrix and is not rerun for this amendment.
5. **Historical π₀.₅ hardware baseline:** The completed top-only 12-episode matrix remains historical and is not rerun for this amendment.
6. **Coverage amendment:** Pure tests cover 0/partial/rotated/100% overlap, compound unions, finite/range rejection, exact sample coverage, earliest-best timing, partial preservation, sanitizer bounds, and `<1 ms` telemetry p95. Physics, reset, action, controller, or server changes restart both 12-episode matrices. Prompt-only behavior experiments update the descriptor hash but rerun only their explicitly selected Scenario 1 candidate and remain separate from matrix evidence.
7. **Completion:** Inspect the Scenario 1 video, trajectory plot, telemetry row linkage, coverage summary, and cleanup. Preserve partial artifacts but keep the amendment pending until a complete or honest natural terminal passes exact-candidate local validation.
8. **Three-view amendment:** Contract-test stock and all four custom scenarios through the shared observation path; verify three named, nonblank, distinct views and sub-1 MiB requests; inspect representative stock/custom reset triptychs; then run one headless π₀ `push_pi_single` seed-0 episode. Compare it with the historical top-only run only as descriptive context because the implementation SHA and stochastic samples differ; record latency, VRAM, cadence, underruns, coverage/contact, artifacts, and cleanup.
9. **Unlocked-action experiment:** Prove identity pass-through for stock and all four custom scenario descriptors locally. Per user scope, do not rerun the historical matrices; rerun only π₀ block-transfer Scenario 1 seed 0 with the same one-go prompt, cameras, and 6,000-step ceiling. Inspect all panels, plot, telemetry linkage, both-arm/gripper commands, participation, coverage/timing, GPU metrics, and cleanup before marking SP23 complete.

Three seeds provide paired demonstrations, not statistical or causal evidence. Report actual zero-shot task results even when every episode is unsuccessful.

## Scenario plans

1. [`01-single-arm-push-pi-glyph.md`](01-single-arm-push-pi-glyph.md)
2. [`02-dual-arm-push-pi-glyph.md`](02-dual-arm-push-pi-glyph.md)
3. [`03-single-arm-push-pi-letters.md`](03-single-arm-push-pi-letters.md)
4. [`04-dual-arm-push-pi-letters.md`](04-dual-arm-push-pi-letters.md)
