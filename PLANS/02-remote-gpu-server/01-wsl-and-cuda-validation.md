# 02.01 — WSL and CUDA validation

- **Objective:** Discover the actual remote shell/WSL/GPU state without changing system configuration.
- **Inputs/prerequisites:** P1 power-on; `robot-gpu` alias and independently fingerprint-verified host key per `EXECUTION_LOGISTICS.md`.
- **Implementation tasks:** Use bounded batch SSH; detect direct WSL Bash, Windows PowerShell, or Windows `cmd.exe`; if Windows, list installed distros and select only an explicit configured value or the single detected Ubuntu distro; run sanitized WSL version/disk/Python/uv/git checks; run `nvidia-smi`; verify GPU name includes RTX 3090; report exact recovery commands without installing drivers.
- **Files expected to change:** `scripts/doctor_pc.sh`, config/command-builder tests, `Makefile`, docs.
- **Validation:** `make doctor-pc`; captured exit codes; no committed identifiers; no private paths in output fixtures.
- **Acceptance:** Shell path, distro, Ubuntu version, driver/CUDA visibility, GPU model/memory, disk, port 8000 state, and next-action diagnostics are known.
- **Planned commit:** `feat(remote): add WSL and RTX diagnostics`.
- **Actual findings:** None; `robot-gpu` is not configured locally.
- **Remaining blockers:** SSH alias/PC access.
- **Completion status:** Blocked.

Never install a Linux NVIDIA driver in WSL; NVIDIA documents that WSL uses the Windows host driver.
