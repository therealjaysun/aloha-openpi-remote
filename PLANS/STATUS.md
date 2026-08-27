# Project status

Planning baseline: 2026-08-26. Plans are complete; phases 00–01 are implemented, published, and validated. Phase 02 setup and lifecycle work is complete on the PC, but real inference is blocked by the current GPU/system-memory combination.

Final plan review: 2026-08-27. Independent code/test, pinned-source memory, plan-traceability, and repository/PR audits were reconciled after real PC testing.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Complete; open for review | `codex/00-bootstrap` | 1 | [PR 1](https://github.com/therealjaysun/pi-robotics/pull/1) | `main` | `codex/00-bootstrap` | Local fail-closed scan + hosted `secret-scan` passed; private upstream jobs skipped explicitly | None | `62083a5` |
| 01 Mac simulation | Complete; open for review | `codex/01-mac-simulation` | 2 | [PR 2](https://github.com/therealjaysun/pi-robotics/pull/2) | `codex/00-bootstrap` | `codex/01-mac-simulation` | 18 tests + Ruff/format/shell + doctor + two 900-step runs; hosted `pure-checks` + `secret-scan` passed | None | `44e1d5f` validated implementation; final evidence at branch HEAD |
| 02 Remote GPU | Blocked at real inference; draft | `codex/02-remote-gpu-server` | 3 | [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3) | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | 108 passed, 1 Linux-only skip on Mac; hosted checks green; WSL doctor/setup/lifecycle/cleanup passed; no inference action returned | This setup's π₀ JAX requests CUDA-OOM; official PyTorch conversion target is ≥32 GiB available RAM | `3c3f849` hardware candidate; final evidence at branch HEAD |
| 03 Connectivity | Planned | `codex/03-secure-connectivity` | — | — | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | Not run | Requires a policy that returns valid actions; SSH/cmd→WSL route is known | — |
| 04 End-to-end | Planned | `codex/04-end-to-end-control` | — | — | `codex/03-secure-connectivity` | `codex/04-end-to-end-control` | Not run | Depends on phases 01–03 | — |
| 05 Observability | Planned | `codex/05-observability` | — | — | `codex/04-end-to-end-control` | `codex/05-observability` | Not run | Depends on live inference | — |
| 06 Hardening/docs | Planned | `codex/06-hardening-docs` | — | — | `codex/05-observability` | `codex/06-hardening-docs` | Not run | Depends on all evidence | — |

## Audited state

- Volatile snapshot at `2026-08-27T03:09:59Z`: Apple Silicon arm64, macOS 26.6.1, 18 GB RAM, 58 GiB workspace disk free. Rerun `make doctor-mac` before capacity decisions.
- Tools: Python 3.14.5 default, `uv 0.12.1`, `gh 2.89.0`, Docker 28.0.4.
- Native arm64 Python 3.10.20 now exists only in ignored `examples/aloha_sim/.venv`; the default Python remains untouched and no model/JAX/CUDA stack is installed on Mac.
- PC capacity observed: RTX 3090 with 24,576 MiB VRAM, about 16 GB physical RAM, about 11.7 GiB visible to WSL, and 8 GiB swap. Source inspection supports ≥32 GiB available RAM as a practical converter target, not a measured upstream minimum.
- Source baseline remains upstream `215abfb217dbac7d5f1273282331b9b1866c0479`. Baseline commit is `13426ca`; validated Phase 01 implementation is `44e1d5f229c787d7d1af24bf323a968bce33dfcf`.
- Remotes: official OpenPI is fetch-only `upstream` with push disabled; public project `origin` is `https://github.com/therealjaysun/pi-robotics`.
- Submodules: ALOHA `d1dc83afd89ded4379851257fe5d85632d31d5ec`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- GitHub: authenticated access works. The public repository exists; Actions allows selected immutable-SHA actions, requires SHA pinning, uses read-only default workflow permissions, and cannot approve pull-request reviews. See [`00-bootstrap/04-github-blocker.md`](00-bootstrap/04-github-blocker.md).
- Remote: strict `robot-gpu` key/host trust passes through Windows cmd to explicitly selected Ubuntu 24.04 WSL2. The exact candidate `3c3f849b1033c581d6e649980446362cc99e35f9` passed locked setup with JAX on the RTX 3090; machine identifiers remain untracked.
- Public repository URL: https://github.com/therealjaysun/pi-robotics.

## Final plan review evidence

- Original planning review: `2026-08-27T03:17:02Z`, Mac, upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`, uncommitted planning working tree, profile N/A, raw artifact N/A.
- Independent specification, pinned-source technical, and operations/security re-reviews each reported no unresolved plan finding after amendments.
- Structural validator exited 0: 7 phase overviews and 29 subphase plans contain every required field; no root `IMPLEMENTATION_PLAN.md` exists.
- Relative-link and branch/base-stack validators exited 0; all 42 plan Markdown files resolve their local links.
- Traceability validator exited 0 at the original planning review. The current configuration table enumerates 29 keys; E-MAC02 preserves its historical 28-key validation claim from before `OPENPI_JAX_MEM_FRACTION` was added.
- Obsolete-rule, trailing-whitespace, and tracked README diff checks exited 0. No implementation, GitHub, simulator, SSH, WSL, GPU, or inference result is implied by these planning checks.
- Post-hardware review revalidated 42 plan files, 29 subphases, 29 configuration keys, 151 unique requirement IDs, all 30 definition-of-done IDs, and every local Markdown link. Hardware claims and recovery are grounded in E-PC-SETUP/JAX/CONVERT/STOP.

## Execution cursor

- Active subphase: 02.04 is blocked after the selected documented non-experimental JAX allocator modes failed and the official PyTorch conversion attempt exceeded available RAM.
- Machine gate: `PC SAFE TO POWER OFF` — final identity-verified stop exited 0; no tunnel or long-lived sampler was started.
- Exact user action: upgrade the RTX PC or provide an SSH-accessible CPU Ubuntu 22.04 host with ≥32 GiB RAM available to the conversion process and ≥60 GiB free disk (64 GB total host RAM preferred for Windows/WSL), then reply `conversion host ready`. Codex will convert pinned configs `pi0_aloha_sim` and `pi05_aloha`, copy/hash-check the weights and original assets, add explicit PyTorch checkpoint/backend selection, strict-load the artifacts, and smoke both profiles. Conversion is the next supported experiment, not a guaranteed inference fix.
- External recovery: upgrade the RTX PC or provide the SSH-accessible conversion host described above, then reply `conversion host ready`; GitHub authentication/publication need no recovery.
- Last verified Mac/WSL hardware candidate SHA: `3c3f849b1033c581d6e649980446362cc99e35f9`; upstream SHA: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- PR state: PRs 1–2 are open, non-draft, and green for human review; PR 3 is draft pending hardware acceptance; PRs 4–7 remain pending. No auto-merge is enabled.

Update this cursor immediately before pausing for GitHub login, `conversion host ready`, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Current gate: Phase 02 memory blocker. SSH trust, Windows→WSL routing, RTX detection, locked setup, loopback lifecycle, and verified cleanup are complete.
- The RTX PC may remain off when conversion uses a separate host. If the RTX PC itself is upgraded, power it on and reply `conversion host ready`.
- After conversion succeeds, Codex will ask for `PC ready`, copy and verify the artifacts, rerun bounded diagnostics, and validate both profiles through the PyTorch server path before Phase 03.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
