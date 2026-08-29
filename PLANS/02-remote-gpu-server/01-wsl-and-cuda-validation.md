# 02.01 — WSL and CUDA validation

- **Objective:** Discover the actual remote shell/WSL/GPU state without changing system configuration.
- **Inputs/prerequisites:** P1 power-on; `robot-gpu` alias and independently fingerprint-verified host key per `EXECUTION_LOGISTICS.md`.
- **Implementation tasks:** Use bounded batch SSH; detect direct WSL Bash, Windows PowerShell, or Windows `cmd.exe`; if Windows, list installed distros and select only an explicit configured value or the single detected Ubuntu distro; run sanitized WSL version/disk/Python/uv/git checks; run `nvidia-smi`; verify GPU name includes RTX 3090; report exact recovery commands without installing drivers.
- **Files expected to change:** `scripts/doctor_pc.sh`, config/command-builder tests, `Makefile`, docs.
- **Validation:** `make doctor-pc`; captured exit codes; no committed identifiers; no private paths in output fixtures.
- **Acceptance:** Shell path, distro, Ubuntu version, driver/CUDA visibility, GPU model/memory, disk, port 8000 state, and next-action diagnostics are known.
- **Planned commit:** `feat(remote): add WSL and RTX diagnostics`.
- **Actual findings:** Strict key-based SSH and host-key verification pass. The doctor detects the Windows cmd route, normalizes WSL's UTF-16 distro list and CRLF script stream, explicitly selects `Ubuntu-24.04`, and reports x86_64 WSL2, Python 3.12.3, `uv` 0.12.6, RTX 3090 with 24 GiB VRAM, driver 591.86, about 929 GiB free, and port 8000 free. Ubuntu 24.04 remains an explicit experimental OpenPI target. The doctor now also checks the compiler and Linux input headers needed to build the locked `evdev` source package.
- **Remaining blockers:** None for WSL/CUDA discovery. Ubuntu 24.04 remains outside upstream's supported Ubuntu 22.04 target, so its passing locked checks are recorded as experimental evidence rather than an upstream support claim.
- **Completion status:** Complete; hardware validation passed.

Never install a Linux NVIDIA driver in WSL; the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf) documents that WSL uses the Windows host driver.
