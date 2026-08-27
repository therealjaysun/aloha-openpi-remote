# 01.01 — Native environment

- **Objective:** Create the smallest Mac-only environment that matches the official ALOHA Sim client.
- **Inputs/prerequisites:** `uv`; arm64 Mac; upstream example lock; phase config values.
- **Implementation tasks:** Reuse the ignored Python 3.10 `examples/aloha_sim/.venv` initialized in phase 00 (create it only if absent); sync official requirements unchanged first; editable-install only `packages/openpi-client`; reinstall `requirements/project-test.txt` after the sim sync so pytest/Ruff remain; verify interpreter/platform/imports; keep `MUJOCO_GL` unset initially; add Homebrew GLFW library path only if the captured error requires it; never run root `uv sync` on Mac.
- **Files expected to change:** `scripts/setup_mac.sh`, `scripts/doctor_mac.sh`, `Makefile`, `.gitignore`, possibly a Mac-specific requirements constraint only after proven failure.
- **Validation:** `uv run --python 3.10`/venv Python reports arm64; import `mujoco`, `dm_control`, `gym_aloha`, `openpi_client`; print sanitized versions; create a minimal MuJoCo model.
- **Acceptance:** Re-runnable setup; native arm64 packages plus runnable pure project tests; no CUDA/JAX/model install; exact failure and recovery shown on error.
- **Planned commit:** `feat(sim): add native Mac environment setup`.
- **Actual findings:** Upstream example explicitly uses Python 3.10. MuJoCo 2.3.x has CPython 3.10 macOS arm64 wheels. dm-control documents Homebrew Python plus GLFW library-path handling. Full OpenPI is unsupported outside Ubuntu.
- **Remaining blockers:** Actual resolution/import/render test not run.
- **Completion status:** Planned.

Escalation ladder: official lock → compatible current package version in an isolated experimental venv → source build if practical → Rosetta only after native failure is recorded. Create `05-native-blocker.md` if native remains blocked.
