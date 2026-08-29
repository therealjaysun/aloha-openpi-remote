# Phase 01 — Native Mac simulation

- **Objective:** Prove stock dual-arm ALOHA simulation, 50 Hz stepping, offscreen rendering, and video on the M3 Pro without installing model inference.
- **Scope:** Isolated Python 3.10 environment; MuJoCo/gym-aloha/OpenPI client; TransferCube smoke test; video; native diagnostics.
- **Non-goals:** OpenPI model/JAX/CUDA on Mac, Rosetta by default, remote server/tunnel, interactive viewer unless offscreen is stable first.
- **Dependencies:** Published Phase 00 baseline; Apple Silicon macOS; `uv`; native imageio-ffmpeg path or documented narrow fallback.
- **Planned files:** `scripts/{doctor_mac,setup_mac,smoke_sim}.sh`, `tools/remote_aloha/{__init__,config,sim_smoke_test}.py`, tests for config/smoke helpers, `Makefile`, `.gitignore`; conditional `05-native-blocker.md`.
- **Planned commits:** `feat(sim): add native ALOHA simulation smoke test`; `feat(render): record offscreen ALOHA episodes`.
- **Branch:** `codex/01-mac-simulation`.
- **PR base:** `codex/00-bootstrap`.
- **PR title:** `feat(sim): run native ALOHA simulation on macOS`.
- **Acceptance criteria:** Native arm64 Python 3.10 imports dependencies; TransferCube reset/shape checks pass; at least 200 safe steps and one complete 300-step episode run; 50 fps frames render; MP4 opens and has frames; p95 step+render+policy-image conversion is at most 20 ms; cleanup succeeds; no model runtime installed. A slower result is an explicit performance blocker, not a 50 Hz pass, though later pure implementation may continue.
- **Test commands:** `make doctor-mac`; `make setup-mac`; `make smoke-sim`; `make test`; `make lint`; `file examples/aloha_sim/.venv/bin/python`; `ffprobe <generated-video>` when available.
- **Risks:** Old official lock vs newer macOS; dm-control GLFW library discovery; Python 3.14 default accidentally used; render nondeterminism; disk growth from video.
- **Rollback:** Remove only the ignored phase venv/output directories; revert phase commits; no system driver/security changes.
- **Current status:** Complete, published, and open for human review with hosted CI passing.
- **Actual results:** At validated implementation SHA `44e1d5f229c787d7d1af24bf323a968bce33dfcf`, native arm64 Python 3.10.20 imported the pinned simulator, rendered successfully, and completed two repeat runs of seeds 0–2 at 300 steps each. Aggregate step+render+224-conversion p95 was 11.880 ms and 11.750 ms, below the 20 ms gate. Each run decoded a 300-frame 224×224 MP4 at 50 fps. `make ci` passed 18 tests, Ruff, formatting, and shell syntax in both the simulator environment and a fresh nine-package lightweight environment.
- **Deviations:** The planned Mac-only `imageio-ffmpeg==0.6.0` override was required. macOS 26 no longer resolves the legacy OpenGL path hard-coded by pinned MuJoCo 2.3.7, so setup applies a narrow installed-file path correction while retaining the pinned physics engine; doctor and full simulation validate it on every fresh setup.
- **PR:** [#2](https://github.com/therealjaysun/pi-robotics/pull/2), open for human review, base `codex/00-bootstrap`, head `codex/01-mac-simulation`, not draft. Hosted `pure-checks` and `secret-scan` passed.
- **Final commit SHA:** Validated implementation `44e1d5f229c787d7d1af24bf323a968bce33dfcf`; final branch tip `db02575b8eb94db7f7a175dd38a0b607bef4fc26`.

## Machine handoff

The RTX PC stays off throughout this phase. When it passes, continue through phase 02 local script/config/unit-test staging and create the secret-scanned remote-test candidate commit. Emit `PC ACTION REQUIRED — POWER ON` only when that SHA is ready for WSL validation.
