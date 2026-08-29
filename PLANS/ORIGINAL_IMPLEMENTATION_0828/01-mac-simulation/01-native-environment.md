# 01.01 — Native environment

- **Objective:** Create the smallest Mac-only environment that matches the official ALOHA Sim client.
- **Inputs/prerequisites:** `uv`; arm64 Mac; upstream example lock; phase config values.
- **Implementation tasks:** Reuse the ignored Python 3.10 `examples/aloha_sim/.venv` initialized in phase 00 (create it only if absent); sync official requirements unchanged first; editable-install only `packages/openpi-client`; reinstall `requirements/project-test.txt` after the sim sync so pytest/Ruff remain; verify interpreter/platform/imports; keep `MUJOCO_GL` unset initially; add Homebrew GLFW library path only if the captured error requires it; never run root `uv sync` on Mac.
- **Files expected to change:** `scripts/setup_mac.sh`, `scripts/doctor_mac.sh`, `Makefile`, `.gitignore`, possibly a Mac-specific requirements constraint only after proven failure.
- **Validation:** `uv run --python 3.10`/venv Python reports arm64; import `mujoco`, `dm_control`, `gym_aloha`, `openpi_client`; print sanitized versions; create a minimal MuJoCo model.
- **Acceptance:** Re-runnable setup; native arm64 packages plus runnable pure project tests; no CUDA/JAX/model install; exact failure and recovery shown on error.
- **Planned commit:** `feat(sim): add native Mac environment setup`.
- **Actual findings:** `make setup-mac` is rerunnable with managed native arm64 Python 3.10.20, the official simulator lock, editable `openpi-client`, and the hashed test lock. `make doctor-mac` passed MuJoCo model creation and a 64×64 native render. `uv pip check` passed and no JAX/model runtime was installed. Pinned `imageio-ffmpeg==0.5.1` had no usable arm64 executable and was narrowly replaced by 0.6.0. Pinned MuJoCo 2.3.7 hard-coded the legacy `/System/Library/OpenGL.framework/OpenGL` path, which macOS 26 no longer resolves; setup corrects it to the current framework location without changing engine version.
- **Remaining blockers:** None for Phase 01. Rendering must run in the logged-in Mac desktop session, not a headless SSH session.
- **Completion status:** Complete at validated implementation SHA `44e1d5f229c787d7d1af24bf323a968bce33dfcf`.

Escalation ladder: official lock → compatible current package version in an isolated experimental venv → source build if practical → Rosetta only after native failure is recorded. Create `05-native-blocker.md` if native remains blocked.
