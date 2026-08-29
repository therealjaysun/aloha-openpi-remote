# Scenario suite 0827 — Push-PI

> This is a custom 3-D ALOHA **Push-PI** experiment inspired by PushT. It is not the standard PushT benchmark and its scores are not comparable to PushT results. The glyph scenario uses Greek π; the letter scenario uses uppercase dotless `P` and `I`.

- **Objective:** Evaluate one shared ALOHA tabletop-pushing environment across four seeded conditions and measure exact matching-target footprint coverage plus elapsed time.
- **Scope:** Four fixed scenario IDs; one project-owned Gymnasium environment; primitive compound bodies and visible targets; deterministic paired resets; fixed-gripper action projection; fixed scenario prompts; existing buffered control, JSONL telemetry, trajectories, videos, GPU sampling, SSH tunnel, and optional display-only Mac viewer.
- **Non-goals:** Standard PushT compatibility or score; pixel/raster/pose-error progress heuristics; a new coverage-based success threshold; training/fine-tuning; a literal 7-D/single-arm robot; planar end-effector constraints or a general IK system; custom meshes; free-text or mid-episode prompt GUI; browser dashboard; parallel policy servers; nonzero zero-shot success as an infrastructure requirement.
- **Dependencies:** Completed phases 00–06; pinned OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; installed `gym-aloha==0.1.1`; converted PyTorch π₀ and π₀.₅ artifacts.
- **Planned files:** `tools/remote_aloha/scenarios.py` for pure descriptors/sampling/outcomes/projection; `examples/aloha_sim/push_pi_env.py` for the simulator adapter and four registrations; focused pure and simulator tests; small changes to config, run, smoke, telemetry, metrics, saver/display, Makefile, `.env.example`, README, and plans. Do not edit site-packages or add a geometry/GUI framework.
- **Planned commits:** `feat(sim): add shared Push-PI environment`; `feat(runtime): integrate Push-PI scenarios and display`; `test(scenarios): validate Push-PI matrix workflow`.
- **Branch/PR:** Preferred: after PRs 1–7 merge, create `codex/push-pi-scenarios` from updated `main` and open one PR to `main`. If explicitly scheduled before that merge, branch from `codex/06-hardening-docs`, base the PR there, and retarget only after the stack merges. Never implement from the old standalone `main`.
- **Acceptance:** Stock Transfer Cube remains the default and regresses cleanly; all four scenarios preserve finite `(14,)` state/applied-action and `(50,14)` model-chunk contracts; paired arm modes have identical scene hashes and named sampled/settled poses for each seed; fixed-gripper projection is reflected exactly in commanded trajectory rows; Mac calibration/smokes pass before GPU work; π₀ and π₀.₅ each complete one 12-episode headless matrix, 24 episodes total, without contract/process failures; infrastructure and task success remain separate.
- **Test commands:** `make scenario-calibrate`; `ALOHA_SCENARIO=transfer_cube ALOHA_DISPLAY=0 ALOHA_SEED=0 ALOHA_EPISODES=3 make smoke-sim`; four custom hold runs with `ALOHA_DISPLAY=0` and seeds 0–2; four custom one-episode visual runs with `ALOHA_DISPLAY=1`; `make test lint secret-scan public-audit`; hosted checks; after the PC gate, `make scenario-matrix scenario-metrics` once per profile.
- **Risks:** Novel-task zero-shot failure; compound bodies snag or tip; insufficient 224×224 recognition; empirically calibrated region is not a mathematical reachability proof; optional display slows control; interrupted video encoding may fail even though JSONL/manifest data survives.
- **Rollback:** Leave or set `ALOHA_SCENARIO=transfer_cube`, which routes to the unchanged stock environment and identity action projection; disable the optional display; retain ignored partial artifacts; use `make stop` for owned server/tunnel cleanup.
- **Current status:** Complete. E-MAC-S0827-COVERAGE and E-PC-S0827-COVERAGE validate the exact evaluator candidate and bounded π₀ Scenario 1 measurement; prior 24-episode evidence remains historical.
- **Actual results:** All four Gym IDs compile/reset on the pinned Mac stack. The frozen descriptor is `7d41cc0ddfb88f46eb0c33139f8e82c92ccc823853e07a80a5413faf135ad2d9`. Four canonical 300-step pushes achieved `+0.117..+0.215 m` Y motion, exact named contact, at most `0.00638 m` height deviation, and at most `0.00882 rad` parked-arm error. Every randomized/canonical body and target exceeded 20 policy pixels; four reset images and four display-on smokes passed. The two historical hardware matrices completed 24/24 infrastructure-valid episodes, 7,200/7,200 steps and trajectory samples, and 24/24 videos and 14-joint plots, with zero task success. The exact coverage amendment's π₀ Scenario 1 run completed 300/300 coverage/joint/video samples and measured initial/final/best coverage `0.0%`, step 1/`0.604606 s` as the earliest zero tie, `6.968412 s` total elapsed time, no contact, and only `0.000000669 m` target-distance change. This is an honest zero-shot result, not an infrastructure failure.
- **Deviations:** “Single arm” means one active six-joint arm in the stock bimanual scene; both gripper slots remain fixed. A literal one-arm morphology is deferred unless measured occlusion/contact makes the controlled ablation invalid.
- **PR:** [PR 8](https://github.com/therealjaysun/pi-robotics/pull/8).
- **Validated implementation SHA:** `4516422a95e3d3572997cace51b6a9b718cb8794` (coverage evaluator); historical matrix SHA `7c2ec5927ad200e5aaf30bed0db4ef61cb9e2ba4`.

## Benchmark boundary

Standard PushT is a 2-D Pymunk task: a circular pusher receives a 2-D Cartesian target at 10 Hz and succeeds above 95% T-shape goal coverage. This suite borrows only the randomized-object/fixed-goal planar-pushing idea. Push-PI instead uses MuJoCo at 50 Hz, ALOHA's 14 absolute joint/gripper targets, fixed robot reset, 3-D failure rules, and project-defined held pose tolerances.

Primary references:

- [Official Diffusion Policy PushT environment](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py)
- [Official gym-pusht environment](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)
- Pinned local gym-aloha 0.1.1 source and assets installed by `examples/aloha_sim/requirements.txt`

Do not call a Push-PI result a PushT benchmark result, reuse PushT's 95% score, or imply that these OpenPI checkpoints were trained for either PushT or Push-PI.

## Fixed scenario and configuration contract

| `ALOHA_SCENARIO` | Gymnasium ID | Bodies | Active joints | Prompt |
| --- | --- | --- | --- | --- |
| `transfer_cube` (default) | `gym_aloha/AlohaTransferCube-v0` | Stock cube | Stock behavior | Existing profile behavior |
| `push_pi_single` | `pi_robotics/PushPiSingleArm-v0` | One rigid π | Left six | `Using only the left arm, push the pi-shaped block onto its matching target.` |
| `push_pi_dual` | `pi_robotics/PushPiBimanual-v0` | One rigid π | Left and right six | `Using both arms, push the pi-shaped block onto its matching target.` |
| `push_letters_single` | `pi_robotics/PushLettersSingleArm-v0` | Separate rigid uppercase `P` and `I` | Left six | `Using only the left arm, push the P and I blocks onto their matching targets.` |
| `push_letters_dual` | `pi_robotics/PushLettersBimanual-v0` | Separate rigid uppercase `P` and `I` | Left and right six | `Using both arms, push the P and I blocks onto their matching targets.` |

- Add `ALOHA_SCENARIO` as exactly the enum above and `ALOHA_DISPLAY=0|1`; unknown, blank-explicit, boolean-like, import-path, XML-path, or prompt values fail closed. Keep `ALOHA_TASK` pinned and backward compatible.
- Resolve scenario → Gym ID, descriptor, arm mask, and arm-explicit prompt once in `MacSimConfig`. A scenario prompt overrides the profile default and is immutable for an episode.
- The optional Matplotlib window is display-only. It has no prompt text box, makes no network connection, and never changes policy input.
- Add `make scenario-matrix` for one fail-closed four-scenario/three-seed batch under the selected profile and `make scenario-metrics` for its rollup. A single `ALOHA_SCENARIO=<id> make run` remains useful for development.
- Keep 300 steps as the fixed acceptance limit. `ALOHA_EPISODE_STEPS` may extend only a standalone diagnostic to at most 6,000 steps; the matrix rejects that override and extended evidence is never pooled with acceptance results.

## Minimal implementation and data flow

1. Keep pure scenario descriptors, layout sampling, pose tests, outcome rules, and action projection in one simulator-independent module so public CPU CI does not need dm-control.
2. Add one repository-owned Gymnasium environment/task implementation and register the four fixed IDs with `max_episode_steps=300`. Import its registry before `gymnasium.make()` in both the active runner and simulation smoke path. Preserve `pixels.top` as HWC `uint8 (480,640,3)`, `agent_pos` as finite `float64 (14,)`, `info["is_success"]` as boolean on reset/step, 50 Hz metadata, and finite `(14,)` actions.
3. Build physics from the pinned transfer-cube XML: remove the cube body and its keyframe, inject project-owned primitive bodies/targets, and compile the XML string with a sorted in-memory mapping of the installed include/mesh assets. Never modify or copy the installed robot assets.
4. After the buffered policy validates a complete `(50,14)` response and pops one finite action, apply the pure scenario projector, validate the projected `(14,)` command, then call `environment.step()`. The existing step JSONL and commanded trajectory must record the projected command actually applied, never the discarded raw arm/gripper values. Stock Transfer Cube uses identity projection.
5. Return a fixed, flat, bounded scenario `info` schema. Copy approved numeric/boolean pose, support, lift, fall, and contact values into the existing per-step JSONL event; do not create another logger. Keep detailed poses in ignored manifests/events and publish only allowlisted aggregates.
6. Save exactly one post-step 224×224 frame after every successfully applied step. Feed that same frame to video and the optional main-thread display; HUD overlays exist only on a display copy. Complete or partial encoding is atomic. Record `no_frames` or `encode_failed` honestly while preserving JSONL/manifest evidence.
7. On reset and every applied step, compute exact planar overlap between each rigid glyph footprint and its named target from the frozen component rectangles. Record the per-body and target-area-weighted coverage fractions in existing scenario info; do not change reward, success, or termination.
8. Route scenario outputs to ignored `outputs/scenarios_0827/<batch-id>/<profile>/<scenario>/seed-<seed>/`. Produce per-scenario summaries first, then a matrix rollup that fails unless exact candidate/profile, all four IDs, seeds 0–2, scene pairing, artifact coverage, and safe IDs agree. Do not let the first scenario's metadata stand in for the matrix.

## Mac-only calibration gate

No PC or policy inference is used until one committed descriptor freezes all geometry/physics/evaluation values. No value may vary by model profile or arm mode.

- Freeze each primitive's body-relative offset, box half-size, origin, thickness, density, friction, color, resting height, footprint radius, target pose, display decimation, and every pose/lift/fall/hold threshold.
- The pinned tabletop is a 12-triangle rectangular mesh with SHA-256 `76a1571d1aa36520f2bd81c268991b99816c2a7819464d718e0fd9976fe30dce`; its transformed top support is `x=[-0.6096,0.6096]`, `y=[0.219,0.981]`, `z=0`. Regression-check the asset hash and these source-derived bounds; do not infer authority from `geom_aabb` or observed object positions.
- Operationally calibrate the conservative spawn polygon with deterministic 300-step waypoint sweeps that bracket its inset X/Y limits for both arms and all movable types. This proves only the tested region, not continuous analytical reachability. Bound physics-invalid reset retries and report scenario/seed on exhaustion.
- During reset, set both gripper actuator targets to one frozen finite pusher value in gym-aloha's normalized `[0,1]` range, record its corresponding physical finger target, settle deterministically, then capture the post-settle 14-D home state. Park only the inactive arm's six joints from that state; both gripper commands stay independently fixed. Freeze a parked-arm maximum-error tolerance from the scripted hold test.
- Render every canonical/random reset at policy resolution and require deterministic color-mask visibility for every movable body and target plus human inspection of the four canonical reset PNGs.
- The canonical fixture exists only for scripted smoke calibration. Scored episodes always use seeded randomized layouts.
- Compute `scene_hash = SHA256(object kind + generated XML bytes + sorted relative asset-name/content hashes)`. The generated XML already carries physical geometry/target values; exclude seed, prompt, and arm mask so paired single/dual scenarios share a hash.

## Reset and outcome contract

- `sample_layout(object_kind, seed)` uses only local `numpy.random.default_rng(seed)` before arm-mode logic. Set named free joints in MuJoCo order `[x,y,z,qw,qx,qy,qz]` with yaw quaternion `[cos(yaw/2),0,0,sin(yaw/2)]`; never use trailing `qpos` slices or global `BOX_POSE`.
- Randomize object position/yaw inside the calibrated region inset by full compound-body footprints. Reject body/body, body/goal, body/robot, already-successful, unsupported, and unstable layouts with a bounded attempt count. Record sampled and settled named poses; paired arm modes require byte-identical values.
- Project-defined first-run success thresholds remain XY `<=0.03 m`, wrapped yaw `<=15°`, absolute roll/pitch `<=10°`, and COM height within `0.005 m` of rest, all true for five consecutive applied steps. These are Push-PI thresholds, not PushT authority.
- `lifted_ever` becomes permanently true if COM rises more than `0.01 m`; it prevents success. Terminate unsuccessfully if a transformed footprint leaves the source-derived tabletop boundary or roll/pitch exceeds `30°`. Otherwise truncate at 300 steps. Reward is `1` only on held success, else `0`.
- For letters, uppercase `P` must match the `P` target and uppercase `I` the `I` target; one correct or swapped bodies fail. Treat the dotless uppercase `I` as 180° yaw-symmetric; `P` and Greek π retain ordinary 360° wrapped yaw.
- Coverage is `100 × area(actual planar footprint ∩ named target footprint) / area(named target footprint)`. Union overlapping descriptor rectangles exactly and weight the overall value by target area. Never normalize from observed motion or pixels. Report initial/final/best-applied coverage, the earliest applied step and monotonic elapsed time attaining the best value, and total elapsed time. Rank runs by higher best coverage, then lower time-to-best; one run is a measurement, not proof of optimality.

## Participation, policy, and evidence contract

- Per-arm `contact_ever` means any contact between a named movable-object geom and either named gripper-finger geom for that arm, independent of MuJoCo contact ordering. `both_arms_participated = left_contact_ever and right_contact_ever`. “Interference” means both arms contact the same named body on the same applied step. Record each arm's six-joint travel separately; motion alone is not participation.
- Fixed grippers plus no-lift scoring make this a **fixed-gripper, push-scored** experiment; they do not physically prevent an arm from lifting. Never call the controller planar or push-only.
- Validate both existing profiles separately. `pi0_aloha_sim` is Transfer-Cube-fine-tuned; `pi05_aloha_base` is an experimental ALOHA base profile. This is not a clean model-generation comparison and zero task successes are allowed.
- Keep the GPU server warm within one profile, but create a fresh client and `BufferedPolicy` for every episode as the runner already does. One pre-matrix `make smoke-policy` validates/warmups infrastructure. Within every scored episode, execute the first valid chunk normally; label only its latency cold and exclude only that latency from warmed p95.
- Extend publishable telemetry with fixed scenario/task allowlists and compact safe fields only: scenario, scene hash, episode/sample/artifact counts, terminal counts, push-success count, bounded coverage/error/contact values, monotonic durations, and safe local IDs. Require one coverage sample per applied step and preserve valid partial maxima. Reject arbitrary prompts/paths and omit raw poses, machine identifiers, absolute paths, and detailed JSONL.
- Keep the existing `<1 ms` per-step telemetry-write p95 budget with scenario fields.

## Validation and machine logistics

1. **Mac/pure:** Unknown config rejection; descriptor/hash; deterministic paired sampling; geometry/support/settling; exact success/perturbation/hold/lift/fall/off-table rules; order-independent contacts; projection identity; projected-command/trajectory equality; bounded info/sanitizer behavior; partial JSONL/video/manifest; telemetry overhead; stock Transfer Cube regression. Simulator integration tests may skip only when the pinned Mac stack is absent and must pass locally.
2. **Mac/simulator:** Gym checker/API smoke, canonical fixed-waypoint push for each object layout, a hold-action 300-step run for every ID, four display-on visual smokes, decoded complete/partial videos, and clean ignored outputs. Freeze and commit the descriptor before GPU work.
3. **Candidate gate:** Finish Mac gates and local CI/public/secret audits; commit; secret-scan and push that exact candidate; open the standalone draft PR; require green hosted checks; only then tell the user to turn on the PC. Verify Mac SHA = WSL SHA; reuse converted weights only when profile/runtime identity matches. Do not run general CI on the PC.
4. **π₀ hardware:** `make server`, one `make smoke-policy`, one headless `make scenario-matrix` (4 scenarios × seeds 0–2 = 12 scored episodes), `make scenario-metrics`, inspect artifacts, then `make stop`.
5. **π₀.₅ hardware:** Repeat the same 12 scored episodes under `pi05_aloha_base`, then stop and verify no owned listener/tunnel/sampler remains. The four Mac display smokes are not repeated on GPU.
6. **Coverage amendment:** Pure tests cover 0/partial/rotated/100% overlap, compound unions, finite/range rejection, exact sample coverage, earliest-best timing, partial preservation, sanitizer bounds, and `<1 ms` telemetry p95. Because this amendment changes only evaluation/telemetry/tests/docs, rerun only Scenario 1 (`push_pi_single`, π₀, seed 0, one headless episode) at the exact Mac/WSL SHA; any descriptor, physics, reset, prompt, action, controller, or server change restarts both 12-episode matrices.
7. **Completion:** Inspect the Scenario 1 video, trajectory plot, telemetry row linkage, coverage summary, and cleanup. Preserve partial artifacts but keep the amendment pending until a complete or honest 300-step terminal run passes exact-candidate validation and local/hosted gates.

Three seeds provide paired demonstrations, not statistical or causal evidence. Report actual zero-shot task results even when every episode is unsuccessful.

## Scenario plans

1. [`01-single-arm-push-pi-glyph.md`](01-single-arm-push-pi-glyph.md)
2. [`02-dual-arm-push-pi-glyph.md`](02-dual-arm-push-pi-glyph.md)
3. [`03-single-arm-push-pi-letters.md`](03-single-arm-push-pi-letters.md)
4. [`04-dual-arm-push-pi-letters.md`](04-dual-arm-push-pi-letters.md)
