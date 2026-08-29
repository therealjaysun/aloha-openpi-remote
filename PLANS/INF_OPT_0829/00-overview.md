# Runtime continuity and inference speed — 0829

## Handoff

- **Status:** P0/B0 and short R1/R2 trials are complete. S1 is retained; S2 and S3 were reverted after failing exact π₀.₅ action parity. S4 is next.
- **Base candidate:** main after merged plan PR #9 (`3ebf8b017384e1e38ba97261d65a723966596b45`).
- **Optimization scope:** `pi05_aloha_base` only. By user decision on 2026-08-29, do not run, benchmark, optimize, or qualify π₀. Keep existing π₀ support unchanged for compatibility.
- **Order:** P0 clean baseline → implement B0 instrumentation plus disabled R1/R2 switches → B0 measurements → isolated R1/R2 trials → S1–S6 measured speed work → final hardware validation.
- **Ownership:** One integrating agent owns shared files and final validation. Read-only agents may inspect independent candidates.
- **PC boundary:** The S3 server is stopped. Inspect S4 locally, sync only a focused fused-attention candidate, then A/B with the same π₀.₅ capture and noise seed.

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

The repeated jolts align with wholesale chunk replacement, not dropped video frames. Fixed 50 fps playback makes them look more abrupt because the runs advanced at about 33–34 wall-clock steps/s, but every applied step still has one recorded frame.

Exact candidate `e00768e` adds a timing-only π₀.₅ smoke result of 659.85 ms warmed tunnel p95 and 448.19 ms server-inference p95 over 50 measured requests. Its matched 300-step seed-0 runtime trials were:

| Horizon/prefetch | Crossfade | Underruns | Receipt depth p50 | Command-jump p95 |
| --- | ---: | ---: | ---: | ---: |
| `30/25` | 0 | 1 | 5 | 31.299% |
| `30/25` | 5 | 1 | 6 | 6.052% |
| `45/40` | 0 | 0 | 12 | 22.347% |
| `45/40` | 5 | 0 | 13 | 6.554% |

All four preserved 300/300 step/trajectory/video coverage. `45/40 + 5` addresses both observed failure modes, but its jump p95 still misses the 5% gate; no runtime default is promoted yet.

Exact candidate `8faea85` completed B0 with one captured three-camera Scenario 2 observation, fixed noise seed 7, 5 warmups, and 50 measured π₀.₅ requests. Bitwise action replay passed (`observation ff33e919…`, `actions 38f302fd…`). Warmed server-inference p95 was 459.74 ms; denoise p95 was 302.91 ms. All 50 stage totals reconciled.

Retained S1 candidate `36bbf1a` preserved the same action digest in three 5+50 replays. Server-inference p95 was 373.51, 365.30, then 360.72 ms; the last two runs differ by 1.25% and improve on B0 by at least 20.5%. Their denoise p95 values were 228.01 and 229.55 ms.

## Fixed constraints

- Keep BF16 weights, the π₀.₅ checkpoint, all three `224×224` camera inputs, 14 unlocked controls, the 50-action model output, one in-flight request, WebSocket over SSH, and existing telemetry/output systems.
- A fixed scenario sends one prompt once. Crossfade only time-aligned chunks produced by the same checkpoint and prompt stage.
- Never crossfade across a prompt-stage transition; clear the old buffer first.
- Every applied command must contain exactly 14 finite values and remain attributable to model output.
- Add no inference framework, dashboard, per-step network work, or new dependency.
- Keep raw benchmarks, telemetry, plots, and videos ignored. Publishable summaries must remain free of machine IDs and absolute paths.
- GitHub Actions remains disabled. Use local gates only.

## Acceptance

### Runtime continuity

- Trial `ALOHA_CHUNK_CROSSFADE_STEPS=5` and horizon/prefetch `45/40` independently before combining them. Promote them to defaults only after their hardware gates pass; `0` keeps raw replacement for A/B diagnosis.
- Crossfade reduces replacement-step normalized command-jump p95 to at most 5% for π₀.₅.
- The selected buffer configuration records zero underruns and no stale, repeated, or out-of-order action.
- Buffer qualification uses each completed warm request's observed submission depth, not `prefetch_steps × 20 ms`: elapsed-prefix removal often leaves fewer than 40 usable actions.
- Initial requests, empty/late slices, failures, Ctrl+C, and prompt transitions preserve existing fail-closed behavior.
- Telemetry records the exact crossfaded command applied at every successful simulation step.

### Inference speed

- Use 5 warmups plus 50 synchronized measured π₀.₅ requests on a captured three-camera observation and fixed noise.
- Keep a speed candidate only if warmed server-inference p95 improves by at least 3% versus B0 and a repeat remains within 2% of that result.
- Combined target: at least 10% lower warmed π₀.₅ server-inference p95 versus B0.
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

**Current boundary:** Complete on `8faea85`. The private captured observation and fixed local noise seed produced bitwise-identical actions across all warmup and measured calls; synchronized stage totals reconciled.

**Completed work; do not rerun B0:**

1. Reuse existing request telemetry; add only enough CUDA-event timing to separate input transfer, three-camera SigLIP, Gemma prefix/KV, ten-step denoising, and device-to-host output.
2. Include the final CUDA-to-CPU synchronization that the current `Policy.infer` timer misses.
3. Record buffer depth at request submission and at result receipt, usable fresh actions, elapsed-prefix count, and replacement-step command delta.
4. Capture one valid π₀.₅ observation and run the acceptance benchmark locally in WSL and through the Mac tunnel.
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

**Result:** Retained on `36bbf1a`; exact action parity passed and stable server-inference p95 improved by more than the 3% gate.

1. Replace the CUDA-tensor-controlled `while` with `for _ in range(num_steps)` while preserving the current tensor timestep update sequence. The project default remains `num_steps=10`; do not hard-code it again.
2. Hoist suffix masks, prefix offsets, position IDs, and attention configuration that are invariant during those iterations.
3. Serve under `torch.inference_mode()`; `no_grad()` already exists, so measure the incremental value.

**Keep only if:** Exactly `num_steps` denoise calls (ten by default), fixed-noise parity, and the speed gate passes.

### S2 — Batch the three vision encodes

Stack the ordered overhead/left-wrist/right-wrist tensors, call the existing SigLIP encoder once, then restore original camera/token order.

**Keep only if:** Pixels, masks, order, output shape, and declared parity hold; peak VRAM remains safe; vision-stage and warmed p95 improve.

**Result:** Rejected and reverted. Candidate `547bd0c` reduced vision p95 to 26.18 ms and server p95 to 350.26 ms, but changed the fixed-input action digest. Against S1, action error was max 0.003765, p95 0.003030, and at most 0.112% of a joint range: below 0.01-scale change but above the declared numerical-equivalence band.

### S3 — Trim masked language padding

Crop only trailing language positions whose masks are false for the whole batch. Never truncate a valid token. If compilation later needs stable shapes, derive the smallest buckets from observed valid lengths.

**Keep only if:** Retained tokens/masks match baseline, no prompt truncation occurs, and warmed π₀.₅ p95 improves without bucket churn.

**Result:** Rejected and reverted. Candidate `61be2fd` reduced prefix/KV p95 to 70.17 ms and server p95 to 348.37 ms, but changed the fixed-input action digest; numeric magnitude was not measured.

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

1. **Clean baseline — complete:** P0 and documentation-only SHA merged.
2. **Mac candidate — complete:** B0 instrumentation and selectable R1/R2 passed local gates and were pushed.
3. **PC timing baseline — complete:** Exact `8faea85` synced; doctor/setup, captured three-camera observation, and tunneled fixed-noise 5+50 replay passed for π₀.₅.
4. **Isolated runtime trials — complete:** Short matched baseline, R1, R2, and combined π₀.₅ runs are recorded above; no default is promoted yet.
5. **Speed A/B — active:** S1 retained on `36bbf1a`; S2 and S3 rejected and reverted. Apply S4–S6 one at a time; benchmark π₀.₅ after each and remove losers. Do not run a 120-second episode per experiment.
6. **Combined smoke:** Run the existing four-call π₀.₅ policy smoke plus one 300-step three-camera episode. Verify finite 14D actions, crossfade/buffer metrics, video, trajectory, and cleanup.
7. **Final hardware run:** Run the same 120-second Scenario 2 seed/profile pair used by the π₀.₅ baseline. Inspect its video and all 14 trajectory series; compare task coverage/time honestly.
8. **Closeout:** Re-run local gates on the exact pushed SHA, update status/evidence only after the final candidate passes, run `make stop`, confirm the policy port is free, and tell the user the PC can be switched off.

## Minimal file ownership

- Runtime: `tools/remote_aloha/action_buffer.py`, `buffered_policy.py`, `config.py`, `run.py`, existing tests/config/docs.
- Model speed: `src/openpi/models_pytorch/pi0_pytorch.py`, `src/openpi/policies/policy.py`, `scripts/serve_policy.py`, one focused parity/benchmark test.
- Evidence: existing telemetry and summaries only.

Each retained work package gets one reviewable commit with its focused check and before/after result. Revert failed experiments rather than carrying dormant branches or abstractions.
