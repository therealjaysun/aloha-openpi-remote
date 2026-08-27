# Phase 02 — Remote GPU server

- **Objective:** Reproducibly run selectable π₀ ALOHA-Sim and π₀.₅ ALOHA-base inference profiles inside WSL2 on the RTX 3090.
- **Scope:** Remote discovery, WSL/CUDA checks, pinned OpenPI install, profile selection, checkpoint cache, loopback server lifecycle, GPU evidence.
- **Non-goals:** Driver/firewall/WSL network/SSH-server changes, training/fine-tuning, public model port, claiming π₀.₅ is sim-fine-tuned.
- **Dependencies:** Phase 01 local branch for script/config staging. Remote validation additionally requires the secret-scanned phase 02 candidate SHA, completed P1 SSH/power handoff, user-authorized PC access, WSL2 Ubuntu, and the existing Windows NVIDIA driver. GitHub auth is optional because the documented Git-bundle transfer is available.
- **Planned files:** `scripts/{doctor_pc,setup_pc,start_policy_server,stop_policy_server}.sh`, `tools/remote_aloha/config.py`, tests for config/command construction, `Makefile`, `.env.example`, minimal `scripts/serve_policy.py` host patch/test.
- **Planned commits:** `feat(remote): add WSL environment diagnostics`; `feat(remote): add selectable OpenPI policy server lifecycle`.
- **Branch:** `codex/02-remote-gpu-server`.
- **PR base:** `codex/01-mac-simulation`.
- **PR title:** `feat(remote): run OpenPI policy server in WSL`.
- **Acceptance criteria:** Detected distro (never guessed); Mac/WSL source SHAs match and contain the audited upstream base; RTX 3090 visible; a WSL-local request proves both profiles start on `127.0.0.1:8000`, report safe profile/config/checkpoint/SHA metadata, return finite `(50,14)` actions on GPU, survive setup SSH exit, and stop safely; π₀.₅ limitation is explicit.
- **Test commands:** `make doctor-pc`; `make setup-pc`; `OPENPI_POLICY_PROFILE=pi0_aloha_sim make server`; `make stop`; repeat with `pi05_aloha_base`; remote `nvidia-smi`; `/healthz`; unit tests.
- **Risks:** No remote access; stale/partial checkpoint; WSL `nvidia-smi` limitations; 3090 CUDA/driver mismatch; 24 GB memory pressure; unsafe Windows→WSL quoting; server binds wider than intended.
- **Rollback:** Stop only validated PID; remove project venv/cache only with explicit user request; revert scripts/host patch; never alter drivers, WSL, or firewall.
- **Current status:** Plan complete; implementation not started. Local staging is unblocked; remote acceptance is blocked on PC/SSH access.
- **Actual results:** Remote environment unknown; no SSH, WSL, CUDA, checkpoint, or inference command ran.
- **Deviations:** Added the user-requested π₀.₅ option as an experimental `pi05_aloha` + `pi05_base` profile because upstream has no `pi05_aloha_sim` checkpoint.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Policy profiles

| Profile | Config | Checkpoint | Status/meaning |
| --- | --- | --- | --- |
| `pi0_aloha_sim` (default) | `pi0_aloha_sim` | `gs://openpi-assets/checkpoints/pi0_aloha_sim` | Official task-specific ALOHA Transfer Cube sim policy |
| `pi05_aloha_base` | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` | Official π₀.₅ base model through ALOHA transforms; experimental in this sim, not task-fine-tuned |

Both return the same ALOHA wire action contract `(50,14)`. Record the profile in every run/video/telemetry summary and compare infrastructure plus task success separately.

## Machine handoff

Stage and locally test this phase while the PC is off, then record/push or bundle its exact candidate SHA. After the user replies `PC ready`, complete SSH trust and bounded diagnostics; emit `PC REMOTE WORK STARTED` only when those pass. Request `PC CONSOLE ACTION REQUIRED` only for a proven local/admin blocker.
