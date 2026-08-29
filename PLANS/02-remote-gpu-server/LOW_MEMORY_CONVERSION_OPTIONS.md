# Phase 02 recovery options — low-memory checkpoint conversion

This decision plan exists because the stock JAX policy loads in BF16 but first inference exceeds RTX 3090 VRAM, while OpenPI's stock JAX→PyTorch converter restores the complete checkpoint as FP32 and exceeds the PC's roughly 11.7 GiB WSL RAM. It does not change Phase 02 acceptance: both profiles must still return finite `(50,14)` actions on the RTX GPU.

## Constraints

- WSL sees about 11.7 GiB RAM, 8 GiB swap, 24 GiB RTX VRAM, and ample disk.
- Preserve the direct converter's stock `full-float32` default; project orchestration defaults to automatic selection.
- No custom checkpoint format, allocator flags, public listener, destructive cache cleanup, or PC-side CI.
- A failed experiment may retain ignored logs but must publish no checkpoint directory.
- The explicit PyTorch demo backend disables optional `torch.compile` autotuning; the pinned JAX/default behavior remains unchanged.
- π₀ runs first. Do not convert π₀.₅ until π₀ converts, fresh-loads, and returns finite RTX actions through the explicit PyTorch backend.
- Orbax partial restore operates on stored leaves. Stacked leaves contain 18–27 model layers, so the source floor is one stored leaf; target slices are copied one layer at a time.

## Options

| Option | Current PC | Change size | Evidence/risk | Decision |
| --- | --- | --- | --- | --- |
| A. Retry whole-tree FP32 conversion | No | None | Already kernel-OOM-killed | Rejected |
| B. Retry whole-tree BF16 restore | No | One dtype change | Already kernel-OOM-killed; existing NumPy→Torch bridge cannot carry BF16 | Rejected |
| C. Partial BF16 restore → DLPack → GPU-resident PyTorch model → 1 GB SafeTensors shards | Passed for both profiles | Bounded project-local mode plus sharded loader | Largest stored BF16 leaf is about 2.25 GiB; stock mappings, strict load, and finite-action inference passed | Selected and passed |
| D. Stock converter on ≥32 GiB available RAM | No; requires another/upgraded host | No converter change | Evidence-based practical capacity target, not a validated guarantee; extra hardware required | Supported fallback |
| E. Download an official matching PyTorch checkpoint | Not currently available | None | No drop-in artifacts were found for the pinned profiles | Reconsider only if upstream publishes one |

## Selected experiment: `partial-bfloat16`

1. Require a clean, pushed, secret-scanned Phase 02 SHA and the existing verified SSH→Windows→WSL route.
2. Stop the policy server and acquire the conversion lock.
3. Run a bounded one-leaf proof: restore the real image-embedding leaf directly in BF16, transfer with DLPack, apply its real mapping, and compare its BF16 bits with FP32→BF16.
4. If the proof passes, instantiate the target model on `cuda:0`, restore one complete Orbax leaf at a time on the JAX CPU backend, and synchronously copy each mapped target slice once.
5. Fail on any unknown, missing, duplicate, wrong-shape, or wrong-dtype source/target. Preserve only the known tied base LM-head alias; disable the unused expert LM head before allocation accounting and serialization.
6. Save with the pinned Hugging Face helper as standard 1 GB sharded SafeTensors. Copy the source checkpoint's own assets.
7. Load the shards into a fresh model and require no unexpected keys and only the proven tied alias as a reported missing key.
8. Atomically rename the owned temporary directory only after validation. Record peak RSS, peak GPU memory, source SHA, profile, and a combined shard hash.
9. If any gate fails, remove only the marker-owned temporary output and keep the ≥32 GiB conversion-host fallback.

## Acceptance and stop rules

- Converter success means only that an artifact was safely produced and strict-load validated; it does not satisfy inference acceptance.
- π₀ conversion failure stops the experiment. Do not download or convert π₀.₅.
- π₀ conversion success proceeds to an uncompiled `OPENPI_POLICY_BACKEND=pytorch` server smoke, avoiding the measured first-call compiler process exit. Only a finite π₀ RTX action permits π₀.₅ to repeat the same conversion and smoke path.
- PyTorch smoke must prove the loaded model is on `cuda:0`, observe the 3090 during inference, sample WSL host RSS, and verify the owned server immediately and again through a second SSH session. Do not require the WSL `nvidia-smi` compute-process table when it omits the Torch Linux PID; retain that check for JAX.
- Any OOM, timeout, mapping mismatch, missing norm stats, loader mismatch, or unowned path is a hard failure with no published artifact.
- No PC-side CI is run. Mac pure tests/lint, secret scan, and hosted checks protect the candidate; the PC runs only the bounded hardware experiment.

## Outcome

Option C passed on the current PC for both profiles. π₀ and π₀.₅ each passed the one-leaf proof, full BF16 conversion, fresh sharded load, explicit PyTorch launch, four finite RTX actions, a second-session check while WSL remained active, and safe stop. Phase 03 separately owns persistence after the final Windows WSL client exits. The ≥32 GiB stock-converter fallback was not needed. Exact memory, latency, artifact-hash, and raw-evidence hashes are recorded in E-PC-BF16.

`make convert-pc` subsequently made the selection automatic: Linux `MemAvailable < 16 GiB` selects Option C; otherwise it selects the full-FP32 route. `OPENPI_CONVERSION_RESTORE_MODE` permits an intentional allowlisted override. Auto mode fails closed if available RAM cannot be measured and records both the measurement and selected mode.

## AI continuation capsule

- Historical recovery note: phases 00–06 are complete; current execution state lives only in [`../STATUS.md`](../STATUS.md).
- Recovery command after an artifact is removed or the pin changes: `OPENPI_POLICY_PROFILE=pi0_aloha_sim make convert-pc` after `make ci`, `make secret-scan`, push, `make doctor-pc`, and `make setup-pc`.
- Success state: completed for both profiles; retain the converted PC-local artifacts. No phase continuation is pending here.
- Every outcome records `E-PC-BF16`: machine, UTC, exact command/exit, project and upstream SHAs, profile, sanitized proof/conversion/load/inference result, RSS/GPU peaks, ignored artifact paths and hashes, produced checkpoint hash when applicable, and exact recovery. On failure, request an Ubuntu 22.04 conversion process with ≥32 GiB available RAM and ≥60 GiB disk.
- Never infer Phase 02 completion from conversion, checkpoint loading, or server health alone; require finite `(50,14)` actions for both profiles.
