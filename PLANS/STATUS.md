# Project status

Planning baseline: 2026-08-26. Plans and phases 00–06 are implemented, published, validated, and open for human review.

Final plan review: 2026-08-28. Independent code/test, security, plan-traceability, and repository/PR audits were reconciled after real PC testing.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Complete; open for review | `codex/00-bootstrap` | 1 | [PR 1](https://github.com/therealjaysun/pi-robotics/pull/1) | `main` | `codex/00-bootstrap` | Local fail-closed scan + hosted `secret-scan` passed; private upstream jobs skipped explicitly | None | `62083a5` |
| 01 Mac simulation | Complete; open for review | `codex/01-mac-simulation` | 2 | [PR 2](https://github.com/therealjaysun/pi-robotics/pull/2) | `codex/00-bootstrap` | `codex/01-mac-simulation` | 18 tests + Ruff/format/shell + doctor + two 900-step runs; hosted `pure-checks` + `secret-scan` passed | None | `db02575` |
| 02 Remote GPU | Complete; ready for review | `codex/02-remote-gpu-server` | 3 | [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3) | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | 118 Mac tests pass with one Linux-only skip; Ruff/format/shell/secret scan pass; both partial-BF16 conversions, finite-action smokes, second-session survival, and safe stops passed | None; JAX OOM remains a documented rejected path | `6fef5e2` |
| 03 Connectivity | Complete; ready for review | `codex/03-secure-connectivity` | 4 | [PR 4](https://github.com/therealjaysun/pi-robotics/pull/4) | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | 171 Mac tests pass with one Linux-only skip; Ruff/format/shell/secret scan and hosted checks pass; both exact-candidate tunneled profile smokes and cleanup pass | None | `e0a7021` |
| 04 End-to-end | Complete; open for review | `codex/04-end-to-end-control` | 5 | [PR 5](https://github.com/therealjaysun/pi-robotics/pull/5) | `codex/03-secure-connectivity` | `codex/04-end-to-end-control` | 205 local and isolated-lock tests plus lint/format/shell/secret scan pass; hosted `pure-checks`/`secret-scan` pass; Mac 900-step smoke and six exact-candidate policy episodes/videos pass with clean stops | None; sustained 50 Hz was not claimed (45.44–47.09 Hz measured) | `d7e9714` |
| 05 Observability | Complete; open for review | `codex/05-observability` | 6 | [PR 6](https://github.com/therealjaysun/pi-robotics/pull/6) | `codex/04-end-to-end-control` | `codex/05-observability` | 297 passed + 1 platform skip; lint/format/Bash/secret and hosted checks pass; exact π₀/π₀.₅ hardware runs produced 1,661/1,661 samples and six verified 14-joint plots | None | Hardware implementation `2065dd9`; completion evidence at branch HEAD |
| 06 Hardening/docs | Complete; open for review | `codex/06-hardening-docs` | 7 | [PR 7](https://github.com/therealjaysun/pi-robotics/pull/7) | `codex/05-observability` | `codex/06-hardening-docs` | 318 passed + 1 platform skip; 21 feasible upstream tests, lint/format/Bash, doctors, public/secret audits, plan/link/ancestry validators, hosted checks, and seven-PR status pass | None; final hardware proof remains Phase 5 `2065dd9`, with Phase 6 `90b0fed` smokes retained as historical evidence | Implementation `a8a3ca1`; final evidence at branch HEAD |
| S0827 Push-PI | Complete; open for review | `codex/push-pi-scenarios` | 8 | [PR 8](https://github.com/therealjaysun/pi-robotics/pull/8) | `codex/06-hardening-docs` | `codex/push-pi-scenarios` | 413 passed + 1 platform skip; lint/format/Bash/secret/public and hosted checks pass; exact-SHA three-view π₀ 300-step acceptance and 6,000-step diagnostic artifacts pass | None; prior matrices remain historical top-only evidence | Three-view implementation `6662a23`; completion evidence at branch HEAD |

## Active extension cursor

- S0827 composite-video/staged-prompt extension is in progress on `codex/push-pi-scenarios`; implementation tests pass, while full gates and one exact-SHA π₀ diagnostic remain. Prior matrix/acceptance evidence is unchanged.
- Mac state: full CI, exact clean-checkout calibration, stock regression, four three-seed hold runs, four visual smokes, and image inspection pass for the amended prompt descriptor.
- Identity: custom 3-D ALOHA Push-PI experiment inspired by PushT, not the standard PushT benchmark. The glyph task uses Greek π; the letter task uses uppercase dotless `P` and `I`.
- Machine gate: **OFF pending candidate gates**; no PC action is needed until the exact implementation is committed, pushed, and hosted checks pass.
- PR dependency: PR 8 is based on `codex/06-hardening-docs`; retarget to `main` only after PR 7 merges and verify the incremental diff.
- Plan: [`SCENARIOS_0827/00-overview.md`](SCENARIOS_0827/00-overview.md).

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
- Relative-link and branch/base-stack validators exited 0; all tracked project plan Markdown files resolve their local links.
- Traceability validator exited 0 at the original planning review, when the configuration table enumerated 29 keys; E-MAC02 preserves its earlier historical 28-key claim from before `OPENPI_JAX_MEM_FRACTION` was added.
- Obsolete-rule, trailing-whitespace, and tracked README diff checks exited 0. No implementation, GitHub, simulator, SSH, WSL, GPU, or inference result is implied by these planning checks.
- The partial-BF16/backend amendment revalidated 43 plan files, 29 subphases, 31 configuration keys, 153 unique requirement IDs, all 30 definition-of-done IDs, and every local Markdown link. E-PC-BF16 supplies the completed hardware proof; the final Phase 02 amendment makes the proven bounded path automatic below 16 GiB available RAM.
- Final Phase 03 validation rechecked 43 tracked plan files, 7 phase overviews, 29 structured subphases, 31 configuration keys, 153 unique requirement IDs, all 30 definition-of-done IDs, and every tracked local Markdown link. Local code quality/secret checks, hosted PR checks, both exact-candidate profile smokes, loopback exposure checks, and zero-residue cleanup passed.
- Final trajectory/Phase 6 validation rechecked 48 plan files, 7 overviews, 29 structured subphases, 28 live configuration keys, 157 requirement IDs, 30 DOD rows, every project-owned Markdown link, and the complete branch stack. All passed.

## Execution cursor

- Active subphase: S0827 composite episode video plus staged Scenario 1 diagnostic; phases 00–06 remain complete.
- Machine gate: OFF after successful cleanup and a final free-port doctor check.
- Exact user action: none; the PC may be powered off.
- Release gate: pending local/exact-commit/hosted checks, staged hardware run, artifact inspection, and cleanup.
- Recovery: camera-only amendments require global contract tests plus the bounded Scenario 1 hardware run; do not claim the historical matrices validate three-view behavior.
- Three-view implementation `6662a235278f01c6ee08dec976f58359c47181ec`; latest top-only diagnostic `6e6180c85b5ae279966f790ee7e96720b3285a6e`; coverage `4516422a95e3d3572997cace51b6a9b718cb8794`; matrix `7c2ec5927ad200e5aaf30bed0db4ef61cb9e2ba4`; upstream `215abfb217dbac7d5f1273282331b9b1866c0479`.
- PR state: PRs 1–7 remain open and stacked; PR 8 is the standalone S0827 extension based temporarily on PR 7.

Update this cursor immediately before pausing for GitHub login, `conversion host ready`, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Phase 02 hardware coordination is complete. Both converted artifacts remain in the PC-local OpenPI cache.
- Phase 03 hardware coordination is complete. Both converted artifacts remain in the PC-local cache, and no PC-side CI was run.
- Phase 04 hardware coordination is complete. Six policy-controlled episodes and both stop lifecycles passed; no PC-side CI was run.
- Phase 05 hardware coordination is complete: both existing profile runs include exact trajectory capture and plot verification, with no redundant GPU campaign.
- Phase 06's prior exact-SHA smokes passed at historical candidate `90b0fed`; final descendant `a8a3ca1` passed local/hosted revalidation. No redundant GPU run was added.
- S0827 hardware coordination is complete at `7c2ec592`: π₀ and experimental π₀.₅ each passed one four-scenario/three-seed matrix; all 24 videos and plots were inspected; no PC CI or display smoke was run; final cleanup passed.
- S0827 coverage coordination is complete at `4516422`: one π₀ seed-0 Scenario 1 run produced 300 exact coverage/joint/video samples, measured 0.0% best coverage, passed artifact/GPU/sanitizer inspection, and left the policy port free.
- S0827 three-view coordination is complete at `6662a23`: stock plus all four custom scenarios use overhead and both wrists; π₀ seed-0 Scenario 1 completed the 300-step acceptance run plus a separate 6,000-step/120-simulated-second diagnostic, both at 0.0% best coverage and no contact. Camera/artifact/GPU inspection and final free-port cleanup passed; no PC CI or redundant matrix was run.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
