# Phase 01 — Native Mac simulation

- **Objective:** Prove stock dual-arm ALOHA simulation, 50 Hz stepping, offscreen rendering, and video on the M3 Pro without installing model inference.
- **Scope:** Isolated Python 3.10 environment; MuJoCo/gym-aloha/OpenPI client; TransferCube smoke test; video; native diagnostics.
- **Non-goals:** OpenPI model/JAX/CUDA on Mac, Rosetta by default, remote server/tunnel, interactive viewer unless offscreen is stable first.
- **Dependencies:** Phase 00 local baseline; GitHub publication may remain externally blocked; Apple Silicon macOS; `uv`; native imageio-ffmpeg path or documented narrow fallback.
- **Planned files:** `scripts/{doctor_mac,setup_mac,smoke_sim}.sh`, `tools/remote_aloha/{__init__,config,sim_smoke_test}.py`, tests for config/smoke helpers, `Makefile`, `.gitignore`; conditional `05-native-blocker.md`.
- **Planned commits:** `feat(sim): add native ALOHA simulation smoke test`; `feat(render): record offscreen ALOHA episodes`.
- **Branch:** `codex/01-mac-simulation`.
- **PR base:** `codex/00-bootstrap`.
- **PR title:** `feat(sim): run native ALOHA simulation on macOS`.
- **Acceptance criteria:** Native arm64 Python 3.10 imports dependencies; TransferCube reset/shape checks pass; at least 200 safe steps and one complete 300-step episode run; 50 fps frames render; MP4 opens and has frames; p95 step+render+policy-image conversion is at most 20 ms; cleanup succeeds; no model runtime installed. A slower result is an explicit performance blocker, not a 50 Hz pass, though later pure implementation may continue.
- **Test commands:** `make doctor-mac`; `make setup-mac`; `make smoke-sim`; `make test`; `make lint`; `file examples/aloha_sim/.venv/bin/python`; `ffprobe <generated-video>` when available.
- **Risks:** Old official lock vs newer macOS; dm-control GLFW library discovery; Python 3.14 default accidentally used; render nondeterminism; disk growth from video.
- **Rollback:** Remove only the ignored phase venv/output directories; revert phase commits; no system driver/security changes.
- **Current status:** Plan complete; implementation not started.
- **Actual results:** Mac arm64/macOS 26.6.1/18 GB detected; default Python 3.14.5 lacks simulator packages; `uv` present; 58 GiB free at `2026-08-27T03:09:59Z` (volatile). No install or simulation ran.
- **Deviations:** None.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Machine handoff

The RTX PC stays off throughout this phase. When it passes, continue through phase 02 local script/config/unit-test staging and create the secret-scanned remote-test candidate commit. Emit `PC ACTION REQUIRED — POWER ON` only when that SHA is ready for WSL validation.
