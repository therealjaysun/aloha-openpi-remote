# Runtime continuity and inference speed — 0829

## Handoff

- **Status:** Ready for implementation; P0 repository hygiene is the first required step.
- **Base candidate:** `42a9e10088650750ac0a940b13fbc324912d497a`.
- **Default profile:** `pi05_aloha_base`; `pi0_aloha_sim` remains an explicit option and must pass the same infrastructure checks.
- **Order:** P0 clean baseline → implement B0 instrumentation plus disabled R1/R2 switches → B0 measurements → isolated R1/R2 trials → S1–S6 measured speed work → final hardware validation.
- **Ownership:** One integrating agent owns shared files and final validation. Read-only agents may inspect independent candidates.
- **PC boundary:** Finish Mac B0/R1/R2 code and tests, with baseline settings `crossfade=0` and `30/25`, then push the exact candidate. Ask the user to start the PC before B0 and stop it after final evidence is copied back.

## Objective

1. Remove visible command jolts when a fresh action chunk replaces the old chunk.
2. Test whether a larger execution buffer reduces ordinary underruns.
3. Reduce warmed model inference time on the RTX 3090 without changing checkpoints, cameras, image resolution, prompts, or the default ten denoising steps.

R1 and R2 improve motion continuity or latency tolerance; they do **not** make an individual model inference faster. Only S1–S6 count as inference-speed work.

## Current evidence

The latest comparable 120-second Scenario 2 runs used three cameras, one fixed prompt, horizon/prefetch `30/25`, and no joint locks:

| Profile | Warm tunneled p95 | Server infer p95 | Underruns | Replacement command-jump p95 |
| --- | ---: | ---: | ---: | ---: |
| `pi05_aloha_base` | 649.82 ms | 398.25 ms | 5 | 23.208% of joint range |
| `pi0_aloha_sim` | 621.02 ms | 362.86 ms | 2 | 7.655% of joint range |

The repeated jolts align with wholesale chunk replacement, not dropped video frames. Fixed 50 fps playback makes them look more abrupt because the runs advanced at about 33–34 wall-clock steps/s, but every applied step still has one recorded frame.

## Fixed constraints

- Keep BF16 weights, both checkpoints, all three `224×224` camera inputs, 14 unlocked controls, the 50-action model output, one in-flight request, WebSocket over SSH, and existing telemetry/output systems.
- A fixed scenario sends one prompt once. Crossfade only time-aligned chunks produced by the same checkpoint and prompt stage.
- Never crossfade across a prompt-stage transition; clear the old buffer first.
- Every applied command must contain exactly 14 finite values and remain attributable to model output.
- Add no inference framework, dashboard, per-step network work, or new dependency.
- Keep raw benchmarks, telemetry, plots, and videos ignored. Publishable summaries must remain free of machine IDs and absolute paths.
- GitHub Actions remains disabled. Use local gates only.

## Acceptance

### Runtime continuity

- Trial `ALOHA_CHUNK_CROSSFADE_STEPS=5` and horizon/prefetch `45/40` independently before combining them. Promote them to defaults only after their hardware gates pass; `0` keeps raw replacement for A/B diagnosis.
- Crossfade reduces replacement-step normalized command-jump p95 to at most 5% for π0.5 and does not regress π0.
- The selected buffer configuration records zero underruns and no stale, repeated, or out-of-order action.
- Buffer qualification uses observed depth at request submission, not `prefetch_steps × 20 ms`: elapsed-prefix removal often leaves fewer than 40 usable actions.
- Initial requests, empty/late slices, failures, Ctrl+C, and prompt transitions preserve existing fail-closed behavior.
- Telemetry records the exact crossfaded command applied at every successful simulation step.

### Inference speed

- Use 5 warmups plus 50 synchronized measured requests per profile on a captured three-camera observation and fixed noise.
- Keep a speed candidate only if warmed server-inference p95 improves by at least 3% on one profile and regresses by no more than 2% on the other. Repeat results within that noise band.
- Combined target: at least 10% lower warmed server-inference p95 for both profiles versus B0.
- Preserve fixed-input/fixed-noise actions exactly when operation order is unchanged; otherwise declare a tight numeric tolerance before the A/B run.
- No OOM, process exit, steady-state recompile, silent kernel/backend fallback, or checkpoint/task-contract change.

The runtime work may complete even if no speed candidate survives measurement; report that result instead of retaining ineffective complexity.

## Work packages

### P0 — Commit a clean documentation baseline

1. Verify the intentional move into `PLANS/ORIGINAL_IMPLEMENTATION_0828/` retained every original plan file and repair repository links that still target the old locations.
2. Review and separate the pending scenario/README evidence edits from runtime source changes.
3. Exclude the untracked `PLANS/INF_OPT_0829/.Rhistory`; it is not project documentation.
4. Commit and push the documentation-only baseline, then require a clean working tree before runtime implementation or PC synchronization.

**Exit:** The plan move is reviewable, Markdown links resolve, `.Rhistory` cannot enter a candidate, and `git status --short` is empty.

### B0 — Correct baseline and timing

1. Reuse existing request telemetry; add only enough CUDA-event timing to separate input transfer, three-camera SigLIP, Gemma prefix/KV, ten-step denoising, and device-to-host output.
2. Include the final CUDA-to-CPU synchronization that the current `Policy.infer` timer misses.
3. Record buffer depth at request submission and at result receipt, usable fresh actions, elapsed-prefix count, and replacement-step command delta.
4. Capture one valid observation per profile and run the acceptance benchmark locally in WSL and through the Mac tunnel.
5. Record cold load/warmup separately from warmed requests.

**Exit:** Stage totals reconcile with synchronized request time, fixed-noise replay is stable, and evidence names the exact source SHA and active profile.

### R1 — Five-step same-prompt chunk crossfade

**Decision:** Approved for isolated implementation and validation on 2026-08-29.

Extend existing atomic buffer replacement; do not add a second controller.

1. Validate the returned `(50, 14)` chunk and drop its elapsed prefix as today.
2. Pair remaining old and fresh actions for `N = min(5, old remainder, fresh slice)` future steps.
3. For paired action `i`, apply `old[i] * (1 - alpha) + fresh[i] * alpha`, where `alpha = (i + 1) / N`. The last crossfaded action is fully fresh.
4. Keep the fresh slice's original length and FIFO order. Never append extra steps, shift indices, or exceed the execution horizon.
5. Do not crossfade the initial chunk, an empty side, a fully elapsed/failed response, an underrun with no old remainder, close, or a prompt-stage transition. Drain a staged transition with crossfade explicitly disabled before clearing the old buffer.
6. Accept only `ALOHA_CHUNK_CROSSFADE_STEPS=0` or `5`; use 0 for the baseline and promote 5 after validation.
7. Keep the scenario projector's `identity-14d` descriptor: crossfading happens earlier in the action buffer. Record configured/applied crossfade length separately; no joint is parked or locked.
8. Reuse telemetry for replacement deltas. Raw per-step commanded rows already capture the applied result.

**Focused checks:** Exact five-step weights on synthetic 14D chunks; elapsed alignment; unchanged buffer length; partial/no overlap; finite/shape rejection; no cross-stage blend; applied-command telemetry; failure and bounded close behavior.

### R2 — Trial larger buffer/prefetch

**Decision:** Approved for isolated implementation and validation on 2026-08-29.

1. Test execution horizon/prefetch `45/40` via environment overrides while retaining `1 <= prefetch < horizon <= 50`.
2. Keep one worker and one in-flight request. Do not append a late chunk, repeat an action, or step while empty.
3. Preserve elapsed-prefix alignment before R1 crossfading.
4. Measure actual request-submission depth, request count, usable returned suffix, GPU duty, underruns, and observation age.
5. Promote `45/40` only if it beats `30/25` on the runtime gate without harmful request churn or staleness.

`40 × 20 ms` is only a nominal threshold, not guaranteed runway: a response arriving after about 16–17 successful simulation steps can leave only 33–34 usable actions from the 50-action wire chunk and trigger the next request immediately.

**Focused checks:** Both configurations accepted; invalid boundaries rejected; one request in flight; correct elapsed slice; zero stale/repeated actions; qualification from observed depth.

### S1 — Remove exact sampler overhead

**Decision:** Approved for implementation on 2026-08-29.

1. Replace the CUDA-tensor-controlled `while` with `for _ in range(num_steps)` while preserving the current tensor timestep update sequence. The project default remains `num_steps=10`; do not hard-code it again.
2. Hoist suffix masks, prefix offsets, position IDs, and attention configuration that are invariant during those iterations.
3. Serve under `torch.inference_mode()`; `no_grad()` already exists, so measure the incremental value.

**Keep only if:** Exactly `num_steps` denoise calls (ten by default), fixed-noise parity, and the speed gate passes.

### S2 — Batch the three vision encodes

Stack the ordered overhead/left-wrist/right-wrist tensors, call the existing SigLIP encoder once, then restore original camera/token order.

**Keep only if:** Pixels, masks, order, output shape, and declared parity hold; peak VRAM remains safe; vision-stage and warmed p95 improve.

### S3 — Trim masked language padding

Crop only trailing language positions whose masks are false for the whole batch. Never truncate a valid token. If compilation later needs stable shapes, derive the smallest buckets from observed valid lengths.

**Keep only if:** Retained tokens/masks match baseline, no prompt truncation occurs, and warmed p95 improves without bucket churn. Measure π0 and π0.5 separately.

### S4 — Test native fused attention

Test PyTorch SDPA independently for SigLIP, Gemma prefix attention, and the action expert. Combine only passing components.

**Keep only if:** Profiler evidence proves a fused CUDA kernel ran, numeric parity holds, and warmed p95 improves. A silent eager/math fallback fails.

### S5 — Compile only the stable denoise step

After S1, test `torch.compile(..., mode="default")` on the fixed-shape denoise step. Warm it before readiness and retain explicit eager selection. Do not retry the previously failing whole-sampler `max-autotune` path.

**Keep only if:** Cold compile time/memory are reported separately; there is no OOM, graph churn, or silent eager fallback; warmed p95 passes.

### S6 — Remove proven host-copy overhead

Only if B0 shows material CPU/input-transfer cost, remove redundant `np.array` copies and consolidate host-to-device transfers. Use pinned/nonblocking copies only when profiling proves real overlap and safe ownership.

**Keep only if:** Input tensors are byte-identical, lifetimes remain safe, and synchronized warmed p95 improves.

## Deferred because they do not fit this pass

- WebSocket/SSH compression: may reduce transport time, not model inference time.
- Loading/restoration and memory-only changes: do not speed warmed inference.
- Video replay-rate changes: presentation only.
- Quantization: explicitly owned by another task.
- Fewer/lower-resolution cameras: violates the three-view requirement.
- Fewer denoising steps, INT4/INT8, pruning, distillation, or shorter model horizon: behavior-changing and unnecessary until quality-preserving work plateaus. π0.5 also lacks a positive task-success baseline for that comparison.
- Concurrent requests or cross-request image/KV reuse: creates stale/out-of-order observations; per-request prefix KV caching already exists.

## Validation and evidence sequence

1. **Clean baseline:** Complete P0 and verify the pushed documentation-only SHA.
2. **Mac before PC:** Implement B0 instrumentation and R1/R2 as selectable settings, keeping `crossfade=0` and `30/25` for exact baseline behavior. Run focused CPU tests, `make test`, `make lint`, `make secret-scan`, `make public-audit`, and `git diff --check`; commit/push the exact candidate.
3. **PC baseline:** User starts the PC. Sync the exact SHA, run doctor/setup/smoke, then B0 for both profiles.
4. **Isolate runtime changes:** Compare `30/25 + crossfade 0`, `30/25 + 5`, and `45/40 + 0` with short identical episodes. Test combined `45/40 + 5` only after both independent candidates pass.
5. **Speed A/B:** Apply S1–S6 one at a time. Benchmark both profiles after each; remove losers before continuing. Do not run a 120-second episode per experiment.
6. **Combined smoke:** Run the existing four-call policy smoke plus one 300-step three-camera episode per profile. Verify finite 14D actions, crossfade/buffer metrics, videos, trajectories, and cleanup.
7. **Final hardware run:** Run the same 120-second Scenario 2 seed/profile pair used by the baseline, once for π0.5 and once for π0. Inspect both videos and all 14 trajectory series; compare task coverage/time honestly.
8. **Closeout:** Re-run local gates on the exact pushed SHA, update status/evidence only after the final candidate passes, run `make stop`, confirm the policy port is free, and tell the user the PC can be switched off.

## Minimal file ownership

- Runtime: `tools/remote_aloha/action_buffer.py`, `buffered_policy.py`, `config.py`, `run.py`, existing tests/config/docs.
- Model speed: `src/openpi/models_pytorch/pi0_pytorch.py`, `src/openpi/policies/policy.py`, `scripts/serve_policy.py`, one focused parity/benchmark test.
- Evidence: existing telemetry and summaries only.

Each retained work package gets one reviewable commit with its focused check and before/after result. Revert failed experiments rather than carrying dormant branches or abstractions.
