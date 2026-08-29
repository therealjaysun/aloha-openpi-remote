# 02.01 — WSL and CUDA validation

- **Objective:** Discover the actual remote shell/WSL/GPU state without changing system configuration.
- **Inputs/prerequisites:** P1 power-on; `robot-gpu` alias and independently fingerprint-verified host key per `EXECUTION_LOGISTICS.md`.
- **Implementation tasks:** Use bounded batch SSH; detect/select WSL without guessing; report and consistency-check sanitized Linux `MemTotal`/`MemAvailable`; run version/disk/Python/uv/git and RTX 3090/CUDA checks; report exact recovery without mutating Windows configuration or installing drivers.
- **Files expected to change:** `scripts/doctor_pc.sh`, config/command-builder tests, `Makefile`, docs.
- **Validation:** `make doctor-pc`; captured exit codes; no committed identifiers; no private paths in output fixtures.
- **Acceptance:** Shell path, distro, effective Linux total/available RAM, Ubuntu version, driver/CUDA visibility, GPU model/memory, disk, port state, and next-action diagnostics are known.
- **Planned commit:** `feat(remote): add WSL and RTX diagnostics`.
- **Actual findings:** Strict key-based SSH and host-key verification pass. After the 2026-08-29 PC upgrade, the user-owned Windows WSL2 config uses a 32 GB memory ceiling plus 8 GB swap; the updated doctor reported 32,866,932 KiB total, 31,981,904 KiB available, and automatic `full-float32`. It detects the Windows route, explicitly selects `Ubuntu-24.04`, and reports x86_64 WSL2, toolchain, disk, port, RTX 3090, driver, and effective RAM without exposing Windows paths.
- **Remaining blockers:** None for WSL/CUDA discovery. Ubuntu 24.04 remains outside upstream's supported Ubuntu 22.04 target, so its passing locked checks are recorded as experimental evidence rather than an upstream support claim.
- **Completion status:** Complete; hardware validation passed.

Never install a Linux NVIDIA driver in WSL; the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/pdf/CUDA_on_WSL_User_Guide.pdf) documents that WSL uses the Windows host driver.
