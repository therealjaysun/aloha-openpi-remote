# 02.01 — WSL and CUDA validation

- **Objective:** Discover the actual remote shell/WSL/GPU state without changing system configuration.
- **Inputs/prerequisites:** P1 power-on; `robot-gpu` alias and independently fingerprint-verified host key per `EXECUTION_LOGISTICS.md`.
- **Implementation tasks:** Use bounded batch SSH; detect direct WSL Bash, Windows PowerShell, or Windows `cmd.exe`; if Windows, list installed distros and select only an explicit configured value or the single detected Ubuntu distro; run sanitized WSL version/disk/Python/uv/git checks; run `nvidia-smi`; verify GPU name includes RTX 3090; report exact recovery commands without installing drivers.
- **Files expected to change:** `scripts/doctor_pc.sh`, config/command-builder tests, `Makefile`, docs.
- **Validation:** `make doctor-pc`; captured exit codes; no committed identifiers; no private paths in output fixtures.
- **Acceptance:** Shell path, distro, Ubuntu version, driver/CUDA visibility, GPU model/memory, disk, port 8000 state, and next-action diagnostics are known.
- **Planned commit:** `feat(remote): add WSL and RTX diagnostics`.
- **Actual findings:** The Mac-side doctor detects direct WSL Bash, PowerShell, or cmd without shell interpolation; selects only an explicit distro or the sole detected Ubuntu distro; accepts upstream-supported Ubuntu 22.04 or the user's explicit experimental Ubuntu 24.04 target; and requires x86_64 WSL2, RTX 3090, `nvidia-smi`, `uv`, and the fixed tool set. Console preflight established Ubuntu 24.04 WSL2, x86_64, RTX 3090 with 24 GiB VRAM, working WSL GPU visibility, and ample disk; `uv` is missing. Generated route tests pass. `robot-gpu` has not been queried.
- **Remaining blockers:** Private SSH trust, `uv` installation in `Ubuntu-24.04`, and remote doctor execution.
- **Completion status:** Mac implementation complete; hardware acceptance blocked.

Never install a Linux NVIDIA driver in WSL; the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf) documents that WSL uses the Windows host driver.
