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
- **Test commands:** `make doctor-pc`; `make setup-pc`; `OPENPI_POLICY_PROFILE=pi0_aloha_sim make server`; `make stop`; repeat with `pi05_aloha_base`; remote `nvidia-smi`; `/healthz`; unit tests.
- **Risks:** No remote access; stale/partial checkpoint; WSL `nvidia-smi` limitations; 3090 CUDA/driver mismatch; 24 GB memory pressure; unsafe Windows→WSL quoting; server binds wider than intended.
- **Rollback:** Stop only validated PID; remove project venv/cache only with explicit user request; revert scripts/host patch; never alter drivers, WSL, or firewall.
- **Current status:** Mac-side Phase 02 implementation, local validation, and hosted Linux CPU/secret-scan checks are complete in draft PR 3. The next gate is the explicit PC/SSH handoff; remote acceptance remains untested.
- **Actual results:** Both selectable profiles, generated Bash/PowerShell/cmd routes, candidate/secret-scan gating, managed-checkout setup, loopback lifecycle, PIDfd stop safety, GPU evidence, and WSL-local smoke contracts are implemented and locally tested. The remote environment remains unknown: no SSH, WSL, CUDA, checkpoint, or inference command has run.
- **Deviations:** Added the user-requested π₀.₅ option as an experimental `pi05_aloha` + `pi05_base` profile because upstream has no `pi05_aloha_sim` checkpoint.
- **PR:** [Draft PR 3](https://github.com/therealjaysun/aloha-openpi-remote/pull/3); keep draft until real RTX acceptance passes.
- **Final implementation commit SHA:** `7f024035822c341acfc705c44842431a6fd57695`; later evidence-only commits do not change the implementation.

## Policy profiles

| Profile | Config | Checkpoint | Status/meaning |
| --- | --- | --- | --- |
| `pi0_aloha_sim` (default) | `pi0_aloha_sim` | `gs://openpi-assets/checkpoints/pi0_aloha_sim` | Official task-specific ALOHA Transfer Cube sim policy |
| `pi05_aloha_base` | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` | Official π₀.₅ base model through ALOHA transforms; experimental in this sim, not task-fine-tuned |

Both return the same ALOHA wire action contract `(50,14)`. Record the profile in every run/video/telemetry summary and compare infrastructure plus task success separately.

## Machine handoff

Stage and locally test this phase while the PC is off, then record/push or bundle its exact candidate SHA. After the user replies `PC ready`, complete SSH trust and bounded diagnostics; emit `PC REMOTE WORK STARTED` only when those pass. Request `PC CONSOLE ACTION REQUIRED` only for a proven local/admin blocker.

## Authoritative references

- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf) — WSL uses the Windows NVIDIA driver; do not install a Linux display driver.
- [Microsoft WSL filesystems](https://learn.microsoft.com/en-us/windows/wsl/filesystems) and [interop](https://learn.microsoft.com/en-us/windows/wsl/interop) — Linux-side project placement and Windows→WSL command routing.
- [uv project sync](https://docs.astral.sh/uv/concepts/projects/sync/) — exact locked-environment synchronization.
- [OpenPI `serve_policy.py`](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/serve_policy.py) — upstream server entry point kept compatible by the local host/metadata patch.
