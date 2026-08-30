# Push-PI on LIBERO with optimized π₀.5 — 0830

## Status

- **Branch:** `codex/libero-push-pi-0830` from `main` at `856636e`.
- **Plan:** Executed and verified on 2026-08-29 PDT.
- **Scenarios:** One LIBERO Panda arm runs `push_pi` and `push_p_i`.
- **Duration:** 30 policy seconds by default (600 actions at LIBERO's 20 Hz); smoke mode is 6 policy seconds (120 actions). The requested extended runs use 300 policy seconds (6,000 actions). The existing 10 stabilization actions do not count toward any duration.
- **Model:** Official `pi05_libero` weights. These custom shapes and prompts are out of distribution, so completion requires valid full-duration runs and artifacts, not task success.

## Reuse boundary

1. Reuse the existing PI/P/I `BodyDescriptor` parts, colors, physics values, dotted-outline generation, prompts, and planar coverage calculation from `tools/remote_aloha/scenarios.py`. Adapt only placement coordinates to the LIBERO table.
2. Reuse `examples/libero/main.py` for LIBERO's Panda, two-camera observation preprocessing, 8-D state, 7-D OSC action execution, five-step replanning, and video output.
3. Reuse the existing remote checkpoint conversion, WebSocket-over-SSH lifecycle, GPU validation, timing, and cleanup code by adding one `pi05_libero` profile.
4. Reuse S1 and S5B unchanged. They already apply to every π₀.5 PyTorch configuration through inference mode, the exact denoise loop, precomputed denoise inputs, `noise.clone()`, and the compiled denoise step.
5. Do not modify the LIBERO submodule, fork the model, add an inference framework, or duplicate the ALOHA 14-D controller.

## Implementation

1. Add a project-side LIBERO Push-PI environment module. Register PI/P/I `CompositeObject` blocks and non-contact dotted target fixtures from the shared descriptors, plus two tabletop tasks whose success check reuses exact planar footprint coverage against each target fixture's actual pose.
2. Extend the existing LIBERO runner with `push_pi` / `push_p_i`, `--smoke`, and a 30-second default. Continue through the configured limit even after sticky success. Reuse the existing telemetry, trajectory, video, GPU sampling, manifest, and summary formats, writing the full artifact set under `outputs/libero_0829/<timestamp>/pi05_libero/<scenario>/`.
3. Add `pi05_libero` to the existing project profile, conversion, server, process, and tunnel allowlists. Derive server action metadata from the profile (`10×7`) instead of the current ALOHA-only `50×14` constants.
4. Make the bounded policy smoke construct the native LIBERO request (`observation/image`, `observation/wrist_image`, 8-D state, prompt) while retaining the ALOHA smoke path.
5. Document the exact simulator and remote commands in `examples/libero/README.md`.

## Gates and execution order

1. Pure checks: focused profile/metadata/smoke tests, LIBERO scenario self-check, Ruff, and shell syntax.
2. Simulator smoke without policy inference: instantiate and step both tasks for 120 actions, verify two distinct camera views, finite 7-D actions, 120 frames, and readable videos.
3. Remote gate: commit and push the exact candidate, sync the WSL checkout, prefetch/convert `pi05_libero`, start the PyTorch server, and pass the optimized policy smoke. Confirm one compiled denoise graph, no steady-state recompiles, RTX 3090 placement, finite `(10, 7)` output, and timing evidence.
4. Connected smoke: run `push_pi` and `push_p_i` for exactly 120 policy actions each.
5. Extended runs: after the connected smoke passes, run each scenario for exactly 6,000 policy actions / 300 simulated seconds, validate telemetry, joint traces, full video decode, GPU coverage, metadata, and server timing, then stop the owned tunnel and server.

## Execution record

- Local gates passed: 450 tests (1 skipped), Ruff and shell lint, secret scan, scenario self-check, simulator checks, exact target-placement checks, and an end-to-end artifact test.
- Remote setup and full-float32 `pi05_libero` conversion passed on the RTX 3090 at source `3db19aa1e1eb6cc32077131e22b26b3cd197002b`; model SHA-256 is `bc6831059bc6062bca25226f07d51a0af06d4a1ef003982c1eb2e4f67f04f206`.
- Optimized policy smoke passed with finite `(10, 7)` actions. After the one-time 23.7 s S5B compile, server inference was 175 ms, including 56 ms prefix-KV and 64 ms denoise timing; subsequent simulator requests had no compile-sized latency spikes.
- The corrected connected smoke completed 120 actions / 6.0 seconds with the full artifact set and GPU coverage at source `1511d6058f53ee607a5232eb165ace1ebf0c0cf1`.
- `push_pi` completed 6,000 actions / 300.0 seconds and 1,200 policy requests. Warm round-trip inference was 239.6 ms mean / 316.2 ms p95; server policy inference was 167.5 ms mean / 193.2 ms p95. Best coverage was 39.83%, final coverage was 16.38%, and the task was unsuccessful. Output: `outputs/libero_0829/20260830T061115.580609Z/pi05_libero/push_pi`.
- `push_p_i` completed 6,000 actions / 300.0 seconds and 1,200 policy requests. Warm round-trip inference was 234.1 ms mean / 310.5 ms p95; server policy inference was 168.2 ms mean / 193.5 ms p95. Best/final coverage was 0.0%, and the task was unsuccessful. Output: `outputs/libero_0829/20260830T061815.907879Z/pi05_libero/push_p_i`.
- Each run has 8,402 telemetry events, a terminal event, 6,000 actual seven-joint samples, a readable 6,000-frame dual-camera MP4 at 20 fps, policy/server timing summaries, metadata, the first policy observation, and complete GPU sampling coverage (313 / 318 samples respectively).
- The custom glyph tasks are out of distribution for the official checkpoint. The runs are valid integration and speed evaluations, but their unsuccessful outcomes are not standard LIBERO benchmark scores.
- All artifacts are preserved under ignored `outputs/libero_0829/`; the owned policy server and tunnel were stopped after both extended runs.

## Acceptance

- The corrected telemetry smoke has exactly 120 actions/frames and 6.0 simulated policy seconds.
- Both extended runs have exactly 6,000 actions/frames and 300.0 simulated policy seconds.
- Every policy response is finite `(10, 7)` and every applied action is finite `(7,)`.
- The server uses `pi05_libero` PyTorch weights on the RTX 3090 with the retained S1/S5B path and no steady-state recompile or eager fallback.
- Each run preserves joint traces, a readable dual-camera MP4, metadata/manifest, raw and summarized step/policy telemetry, GPU metrics, timing summaries, server tail, and the first policy observation; observed success or failure is reported without implying PushT or standard-LIBERO benchmark performance.
- Owned remote processes and the loopback tunnel are stopped at completion.

## Rollback

- Existing LIBERO benchmark mode remains the default when no custom scenario is selected.
- Existing ALOHA profiles and runners remain unchanged in behavior.
- Stop remote resources with `make stop`; custom outputs remain ignored under `outputs/libero_0829`.
