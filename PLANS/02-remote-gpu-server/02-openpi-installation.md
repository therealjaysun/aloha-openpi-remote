# 02.02 — OpenPI installation

- **Objective:** Install the audited OpenPI source and GPU runtime in the detected WSL distro.
- **Inputs/prerequisites:** Passing WSL/GPU diagnostics; secret-scanned local candidate commit; absolute WSL project/cache paths resolved inside WSL; network/disk access.
- **Implementation tasks:** Preserve the existing bounded setup flow and mode-600 evidence while streaming safe source/submodule/environment/GPU milestones to the Mac terminal; resolve `OPENPI_REMOTE_DIR` to an absolute WSL POSIX path; fetch/check out only the exact clean candidate; validate upstream/submodules/SHA, disk/cache, locked `uv sync`, and JAX GPU without copying Mac secrets or resetting remote changes.
- **Files expected to change:** `scripts/setup_pc.sh`, config/remote-command tests, `Makefile`, docs.
- **Validation:** Remote `git status`, remotes, SHA, submodules; `uv run python` imports OpenPI/JAX; `jax.devices()` contains GPU; disk/cache checks.
- **Acceptance:** Exact secret-scanned project SHA runs in an isolated WSL env and contains the audited upstream base; no CPU-only fallback; sufficient disk and no unexplained partial download; no driver/system security changes; rerun is safe.
- **Planned commit:** `feat(remote): install pinned OpenPI in WSL`.
- **Actual findings:** Setup accepted the exact secret-scanned candidate, created its marked managed checkout, verified both submodules, installed pinned CPython 3.11.16, resolved all 279 locked packages, and preserved the checkpoint cache. The initial `evdev==1.9.2` header failure was repaired with Ubuntu's userspace Linux headers; the rerun completed and JAX identified the RTX 3090 rather than falling back to CPU. The lock includes JAX CUDA 12 and torch 2.7.1.
- **Remaining blockers:** None. The locked JAX installation and both bounded PyTorch conversions passed; see E-PC-BF16.
- **Completion status:** Complete; exact-candidate setup and JAX GPU validation passed.

The environment step follows the official [uv locked sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/); project and cache paths stay in the WSL filesystem per [Microsoft's WSL filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).
