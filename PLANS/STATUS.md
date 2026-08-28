# Project status

Planning baseline: 2026-08-26. Phase 05 is reopened only for the requested joint-trajectory amendment; its existing architecture and PR remain authoritative. Phase 06 waits for the updated parent candidate.

Final plan review: 2026-08-28. Independent code/test, security, plan-traceability, and repository/PR audits were reconciled after real PC testing.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Complete; open for review | `codex/00-bootstrap` | 1 | [PR 1](https://github.com/therealjaysun/pi-robotics/pull/1) | `main` | `codex/00-bootstrap` | Local fail-closed scan + hosted `secret-scan` passed; private upstream jobs skipped explicitly | None | `62083a5` |
| 01 Mac simulation | Complete; open for review | `codex/01-mac-simulation` | 2 | [PR 2](https://github.com/therealjaysun/pi-robotics/pull/2) | `codex/00-bootstrap` | `codex/01-mac-simulation` | 18 tests + Ruff/format/shell + doctor + two 900-step runs; hosted `pure-checks` + `secret-scan` passed | None | `44e1d5f` validated implementation; final evidence at branch HEAD |
| 02 Remote GPU | Complete; ready for review | `codex/02-remote-gpu-server` | 3 | [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3) | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | 118 Mac tests pass with one Linux-only skip; Ruff/format/shell/secret scan pass; both partial-BF16 conversions, finite-action smokes, second-session survival, and safe stops passed | None; JAX OOM remains a documented rejected path | Hardware candidate `38b5228`; final evidence at branch HEAD |
| 03 Connectivity | Complete; ready for review | `codex/03-secure-connectivity` | 4 | [PR 4](https://github.com/therealjaysun/pi-robotics/pull/4) | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | 171 Mac tests pass with one Linux-only skip; Ruff/format/shell/secret scan and hosted checks pass; both exact-candidate tunneled profile smokes and cleanup pass | None | Hardware implementation `0c64187`; final evidence at branch HEAD |
| 04 End-to-end | Complete; open for review | `codex/04-end-to-end-control` | 5 | [PR 5](https://github.com/therealjaysun/pi-robotics/pull/5) | `codex/03-secure-connectivity` | `codex/04-end-to-end-control` | 205 local and isolated-lock tests plus lint/format/shell/secret scan pass; hosted `pure-checks`/`secret-scan` pass; Mac 900-step smoke and six exact-candidate policy episodes/videos pass with clean stops | None; sustained 50 Hz was not claimed (45.44–47.09 Hz measured) | Hardware implementation `0fca61f`; final evidence at branch HEAD |
| 05 Observability | Joint-trajectory amendment in progress in existing PR | `codex/05-observability` | 6 | [PR 6](https://github.com/therealjaysun/pi-robotics/pull/6) | `codex/04-end-to-end-control` | `codex/05-observability` | Prior 273-test/hardware baseline passed; amended local and hardware gates pending | Exact-candidate π₀/π₀.₅ trajectory plots and overhead must pass | Prior hardware `de63e19`; amended SHA pending |
| 06 Hardening/docs | Paused for updated Phase 05 parent | `codex/06-hardening-docs` | 7 | [PR 7](https://github.com/therealjaysun/pi-robotics/pull/7) | `codex/05-observability` | `codex/06-hardening-docs` | Prior Phase 06 was green; descendant revalidation pending | Updated Phase 05 ancestry/evidence | Prior branch tip `e38910b` |

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
- Final Phase 03 validation rechecked 43 tracked plan files, 7 phase overviews, 29 structured subphases, 31 configuration keys, 153 unique requirement IDs, all 30 definition-of-done IDs, and every tracked local Markdown link. Local code quality/secret checks, hosted PR checks, both exact-candidate profile smokes, loopback exposure checks, and zero-residue cleanup passed.

## Execution cursor

- Active subphase: 05.01 local trajectory telemetry, then 05.04 plot/performance summary; this extends the existing Phase 5 candidate.
- Machine gate: local plan/code/tests first; then the existing π₀ and π₀.₅ Phase 5 run sequence at one clean pushed SHA.
- Exact user action: none until Codex reports `PC ACTION REQUIRED — POWER ON` if the SSH host is unavailable.
- Recovery: rerun the E-PC-BF16 commands only if either converted artifact is removed or the pinned model/runtime changes.
- Last verified Mac/WSL hardware candidate SHA: `de63e19c37d8fd76fbda5b5a07a8f5a0b19b42d6`; upstream SHA: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- PR state: PRs 1–7 exist; PR 6 will receive the amended candidate, then PR 7 will be repaired onto that parent and both revalidated. No auto-merge is enabled.

Update this cursor immediately before pausing for GitHub login, `conversion host ready`, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Phase 02 hardware coordination is complete. Both converted artifacts remain in the PC-local OpenPI cache.
- Phase 03 hardware coordination is complete. Both converted artifacts remain in the PC-local cache, and no PC-side CI was run.
- Phase 04 hardware coordination is complete. Six policy-controlled episodes and both stop lifecycles passed; no PC-side CI was run.
- Phase 05 hardware rerun is pending only after the amended candidate passes local gates; it reuses the existing two profile runs and adds no redundant GPU campaign.
- Codex will use the existing private alias and request power-on only if the machine is no longer reachable.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
