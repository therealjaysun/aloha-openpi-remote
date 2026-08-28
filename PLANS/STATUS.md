# Project status

Planning baseline: 2026-08-26. Plans are complete; phases 00–02 are implemented, published, and validated. Phase 03 secure connectivity is locally implemented; its first hardware run found and reproduced Windows idle-WSL teardown, and the integrated SSH-owned lifetime fix awaits final two-profile acceptance.

Final plan review: 2026-08-27. Independent code/test, pinned-source memory, plan-traceability, and repository/PR audits were reconciled after real PC testing.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Complete; open for review | `codex/00-bootstrap` | 1 | [PR 1](https://github.com/therealjaysun/pi-robotics/pull/1) | `main` | `codex/00-bootstrap` | Local fail-closed scan + hosted `secret-scan` passed; private upstream jobs skipped explicitly | None | `62083a5` |
| 01 Mac simulation | Complete; open for review | `codex/01-mac-simulation` | 2 | [PR 2](https://github.com/therealjaysun/pi-robotics/pull/2) | `codex/00-bootstrap` | `codex/01-mac-simulation` | 18 tests + Ruff/format/shell + doctor + two 900-step runs; hosted `pure-checks` + `secret-scan` passed | None | `44e1d5f` validated implementation; final evidence at branch HEAD |
| 02 Remote GPU | Complete; ready for review | `codex/02-remote-gpu-server` | 3 | [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3) | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | 118 Mac tests pass with one Linux-only skip; Ruff/format/shell/secret scan pass; both partial-BF16 conversions, finite-action smokes, second-session survival, and safe stops passed | None; JAX OOM remains a documented rejected path | Hardware candidate `38b5228`; final evidence at branch HEAD |
| 03 Connectivity | Local holder fix; final hardware pending | `codex/03-secure-connectivity` | 4 | [Draft PR 4](https://github.com/therealjaysun/pi-robotics/pull/4) | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | 170 Mac tests pass with one Linux-only skip; Ruff/format/shell pass; holder/tunnel/rollback/client/security tests pass | Final integrated π₀/π₀.₅ idle-survival, tunnel, listener, and cleanup acceptance | Candidate at branch HEAD pending secret scan/push |
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
- Remote: strict `robot-gpu` key/host trust passes through Windows cmd to explicitly selected Ubuntu 24.04 WSL2. Hardware candidate `38b5228418c729d39d1c4fe551ef5ddcbef9e49e` passed locked setup plus both converted-profile smokes on the RTX 3090; machine identifiers remain untracked.
- Public repository URL: https://github.com/therealjaysun/pi-robotics.

## Final plan review evidence

- Original planning review: `2026-08-27T03:17:02Z`, Mac, upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`, uncommitted planning working tree, profile N/A, raw artifact N/A.
- Independent specification, pinned-source technical, and operations/security re-reviews each reported no unresolved plan finding after amendments.
- Structural validator exited 0: 7 phase overviews and 29 subphase plans contain every required field; no root `IMPLEMENTATION_PLAN.md` exists.
- Relative-link and branch/base-stack validators exited 0; all 42 plan Markdown files resolve their local links.
- Traceability validator exited 0 at the original planning review, when the configuration table enumerated 29 keys; E-MAC02 preserves its earlier historical 28-key claim from before `OPENPI_JAX_MEM_FRACTION` was added.
- Obsolete-rule, trailing-whitespace, and tracked README diff checks exited 0. No implementation, GitHub, simulator, SSH, WSL, GPU, or inference result is implied by these planning checks.
- The partial-BF16/backend amendment revalidated 43 plan files, 29 subphases, 31 configuration keys, 153 unique requirement IDs, all 30 definition-of-done IDs, and every local Markdown link. E-PC-BF16 supplies the completed hardware proof; the final Phase 02 amendment makes the proven bounded path automatic below 16 GiB available RAM.

## Execution cursor

- Active subphase: 03.03 integrated SSH-owned WSL lifetime acceptance, then final 03.04 two-profile tunnel smokes.
- Machine gate: publish the exact secret-scanned Phase 03 candidate, then keep the RTX PC on and awake for bounded validation.
- Exact user action: none if the previously connected PC remains on; otherwise turn it on and reply `PC ready` when requested.
- Recovery: rerun the E-PC-BF16 commands only if either converted artifact is removed or the pinned model/runtime changes.
- Last verified Mac/WSL hardware candidate SHA: `38b5228418c729d39d1c4fe551ef5ddcbef9e49e`; upstream SHA: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- PR state: PRs 1–3 are open and green; PR 4 is a draft; PRs 5–7 remain pending. No auto-merge is enabled.

Update this cursor immediately before pausing for GitHub login, `conversion host ready`, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Phase 02 hardware coordination is complete. Both converted artifacts remain in the PC-local OpenPI cache.
- Phase 03 now needs the PC for final holder-integrated Windows-loopback, Mac tunnel, two-profile inference, listener non-exposure, and verified cleanup. No PC-side CI runs.
- Codex will use the existing private alias and request `PC ready` only if the machine is no longer reachable.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
