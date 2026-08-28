# Phase 02 — Remote GPU server

- **Objective:** Reproducibly run selectable π₀ ALOHA-Sim and π₀.₅ ALOHA-base inference profiles inside WSL2 on the RTX 3090.
- **Scope:** Remote discovery, WSL/CUDA checks, pinned OpenPI install, profile selection, checkpoint cache, loopback server lifecycle, GPU evidence.
- **Non-goals:** Driver/firewall/WSL network/SSH-server changes, training/fine-tuning, public model port, claiming π₀.₅ is sim-fine-tuned.
- **Dependencies:** Phase 01 local branch for script/config staging. Remote validation additionally requires the secret-scanned phase 02 candidate SHA, completed P1 SSH/power handoff, user-authorized PC access, WSL2 Ubuntu, and the existing Windows NVIDIA driver. GitHub auth is optional because the documented Git-bundle transfer is available.
- **Implemented files:** bounded Mac→SSH→WSL orchestration in `tools/remote_aloha/remote.py`; fixed profile/response contracts and pure tests; WSL doctor/setup/start/check/smoke/stop scripts; minimal `scripts/serve_policy.py` host/metadata/GPU patch; config, Make, CI, ignore, and evidence gates.
- **Implementation commit:** `7f024035822c341acfc705c44842431a6fd57695` (`feat(remote): stage selectable WSL policy server`).
- **Branch:** `codex/02-remote-gpu-server`.
- **PR base:** `codex/01-mac-simulation`.
- **PR title:** `feat(remote): run OpenPI policy server in WSL`.
- **Acceptance criteria:** Detected distro (never guessed); Mac/WSL source SHAs match and contain the audited upstream base; RTX 3090 visible; a WSL-local request proves both profiles start on `127.0.0.1:8000`, report safe profile/config/checkpoint/SHA metadata, return finite `(50,14)` actions on GPU, survive setup SSH exit, and stop safely; π₀.₅ limitation is explicit.
- **Test commands:** `make doctor-pc`; `make setup-pc`; `OPENPI_POLICY_PROFILE=pi0_aloha_sim make convert-pc`; `OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make server`; `OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make smoke-policy`; `make stop`; repeat conversion/inference with `pi05_aloha_base` only after π₀ returns finite actions; retain `OPENPI_POLICY_BACKEND=jax` for the original path; remote `nvidia-smi`; `/healthz`; unit tests.
- **Risks:** No remote access; stale/partial checkpoint; WSL `nvidia-smi` limitations; 3090 CUDA/driver mismatch; GPU/system-memory pressure; unsafe Windows→WSL quoting; server binds wider than intended. The selected recovery experiment and fallback are compared in [`LOW_MEMORY_CONVERSION_OPTIONS.md`](LOW_MEMORY_CONVERSION_OPTIONS.md).
- **Rollback:** Stop only validated PID; remove project venv/cache only with explicit user request; revert scripts/host patch; never alter drivers, WSL, or firewall.
- **Current status:** Complete at 02.04. Both converted profiles passed bounded RTX 3090 inference, second-session survival, and identity-verified cleanup; later connectivity/control/observability phases are also complete.
- **Actual results:** The pinned JAX π₀ path still OOMs on first inference, and the stock converter still exceeds current WSL RAM. The partial-BF16 path succeeded for `pi0_aloha_sim` and `pi05_aloha_base`: one-leaf proofs, complete mapping, standard sharded SafeTensors, fresh-model loads, explicit uncompiled PyTorch selection, four finite `(50,14)` actions per profile, CUDA/3090 model placement, WSL host/GPU sampling, cross-session survival, and safe stops passed. Project orchestration now selects that path automatically when Linux `MemAvailable` is below 16 GiB while keeping explicit overrides and the direct converter's full-FP32 default. Final hardware candidate `38b5228418c729d39d1c4fe551ef5ddcbef9e49e`; detailed metrics and hashes are E-PC-BF16.
- **Deviations:** Added the user-requested π₀.₅ option as experimental `pi05_aloha` + `pi05_base` because upstream has no `pi05_aloha_sim`. Masked-camera compaction did not make JAX fit. The selected recovery uses direct BF16 leaf restore, disables the unused expert LM head, constructs/loads the target on CUDA, and disables optional PyTorch compilation because the measured autotune first call exited under the constrained host.
- **PR:** [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3); hardware acceptance passed and the PR is ready for human review.
- **Final implementation commit SHA:** Hardware candidate `38b5228418c729d39d1c4fe551ef5ddcbef9e49e`; final branch tip `6fef5e2700d07dd3c9eeef373bcf72a011e67a0a`.

## Policy profiles

| Profile | Config | Checkpoint | Status/meaning |
| --- | --- | --- | --- |
| `pi0_aloha_sim` (default) | `pi0_aloha_sim` | `gs://openpi-assets/checkpoints/pi0_aloha_sim` | Official task-specific ALOHA Transfer Cube sim policy |
| `pi05_aloha_base` | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` | Official π₀.₅ base model through ALOHA transforms; experimental in this sim, not task-fine-tuned |

Both return the same ALOHA wire action contract `(50,14)`. Record the profile in every run/video/telemetry summary and compare infrastructure plus task success separately.

## Machine handoff

The PC/SSH handoff and Phase 02 hardware acceptance are complete; both converted checkpoints remain in the PC's ignored OpenPI cache and the owned server is stopped. The PC can be powered off until Phase 03 connectivity work resumes. See `PLANS/STATUS.md` for the live gate.

## Authoritative references

- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf) — WSL uses the Windows NVIDIA driver; do not install a Linux display driver.
- [Microsoft WSL filesystems](https://learn.microsoft.com/en-us/windows/wsl/filesystems) and [interop](https://learn.microsoft.com/en-us/windows/wsl/interop) — Linux-side project placement and Windows→WSL command routing.
- [uv project sync](https://docs.astral.sh/uv/concepts/projects/sync/) — exact locked-environment synchronization.
- [OpenPI `serve_policy.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py) — pinned upstream server entry point kept compatible by the local host/metadata patch.
