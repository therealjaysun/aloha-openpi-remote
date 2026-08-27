# Project status

Planning baseline: 2026-08-26. Plans are complete; phases 00–01 are implemented, published, and validated; phase 02 Mac-side implementation is locally validated and awaiting its publication gate.

Final plan review: 2026-08-26. Independent specification, technical-source, and operations/security audits were reconciled; structural and link validation evidence is recorded by the completing turn.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Complete; open for review | `codex/00-bootstrap` | 1 | [PR 1](https://github.com/therealjaysun/pi-robotics/pull/1) | `main` | `codex/00-bootstrap` | Local fail-closed scan + hosted `secret-scan` passed; private upstream jobs skipped explicitly | None | `62083a5` |
| 01 Mac simulation | Complete; open for review | `codex/01-mac-simulation` | 2 | [PR 2](https://github.com/therealjaysun/pi-robotics/pull/2) | `codex/00-bootstrap` | `codex/01-mac-simulation` | 18 tests + Ruff/format/shell + doctor + two 900-step runs; hosted `pure-checks` + `secret-scan` passed | None | `44e1d5f` validated implementation; final evidence at branch HEAD |
| 02 Remote GPU | Mac/hosted staging complete; draft pending hardware | `codex/02-remote-gpu-server` | 3 | [PR 3](https://github.com/therealjaysun/pi-robotics/pull/3) | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | 100 passed, 1 Linux-only PIDfd test skipped on Mac; isolated CPU CI, Ruff/format/generated Bash/shell syntax, native sim, and scan green; hosted Linux pure-checks + secret-scan green | Explicit PC/SSH handoff | `7f02403` implementation; evidence at branch HEAD |
| 03 Connectivity | Planned | `codex/03-secure-connectivity` | — | — | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | Not run | Remote shell/routing mode unknown | — |
| 04 End-to-end | Planned | `codex/04-end-to-end-control` | — | — | `codex/03-secure-connectivity` | `codex/04-end-to-end-control` | Not run | Depends on phases 01–03 | — |
| 05 Observability | Planned | `codex/05-observability` | — | — | `codex/04-end-to-end-control` | `codex/05-observability` | Not run | Depends on live inference | — |
| 06 Hardening/docs | Planned | `codex/06-hardening-docs` | — | — | `codex/05-observability` | `codex/06-hardening-docs` | Not run | Depends on all evidence | — |

## Audited state

- Volatile snapshot at `2026-08-27T03:09:59Z`: Apple Silicon arm64, macOS 26.6.1, 18 GB RAM, 58 GiB workspace disk free. Rerun `make doctor-mac` before capacity decisions.
- Tools: Python 3.14.5 default, `uv 0.12.1`, `gh 2.89.0`, Docker 28.0.4.
- Native arm64 Python 3.10.20 now exists only in ignored `examples/aloha_sim/.venv`; the default Python remains untouched and no model/JAX/CUDA stack is installed on Mac.
- Source baseline remains upstream `215abfb217dbac7d5f1273282331b9b1866c0479`. Baseline commit is `13426ca`; validated Phase 01 implementation is `44e1d5f229c787d7d1af24bf323a968bce33dfcf`.
- Remotes: official OpenPI is fetch-only `upstream` with push disabled; public project `origin` is `https://github.com/therealjaysun/pi-robotics`.
- Submodules: ALOHA `d1dc83afd89ded4379851257fe5d85632d31d5ec`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- GitHub: authenticated access works. The public repository exists; Actions allows selected immutable-SHA actions, requires SHA pinning, uses read-only default workflow permissions, and cannot approve pull-request reviews. See [`00-bootstrap/04-github-blocker.md`](00-bootstrap/04-github-blocker.md).
- SSH: `robot-gpu` is not configured. No remote facts have been guessed.
- Public repository URL: https://github.com/therealjaysun/pi-robotics.

## Final plan review evidence

- `2026-08-27T03:17:02Z`, Mac, upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`, uncommitted planning working tree, profile N/A, raw artifact N/A.
- Independent specification, pinned-source technical, and operations/security re-reviews each reported no unresolved plan finding after amendments.
- Structural validator exited 0: 7 phase overviews and 29 subphase plans contain every required field; no root `IMPLEMENTATION_PLAN.md` exists.
- Relative-link and branch/base-stack validators exited 0; all 42 plan Markdown files resolve their local links.
- Traceability validator exited 0: 151 requirement rows have 151 unique IDs, all terminal requirement IDs exist, and all 28 configuration keys are enumerated.
- Obsolete-rule, trailing-whitespace, and tracked README diff checks exited 0. No implementation, GitHub, simulator, SSH, WSL, GPU, or inference result is implied by these planning checks.

## Execution cursor

- Active subphase: Phase 02 Mac/hosted staging complete in draft PR 3; stopped at the P1 power/SSH handoff before 02.01 hardware acceptance.
- Machine gate: M0 — Mac only.
- Exact next implementation action after the final evidence-only head is green: emit `PC ACTION REQUIRED — POWER ON`; after `PC ready`, run the SSH trust gate and `make doctor-pc`.
- External recovery: none; GitHub authentication and publication are verified.
- Last verified local implementation SHA: `44e1d5f229c787d7d1af24bf323a968bce33dfcf`; last verified WSL SHA: unknown.
- PR state: PRs 1–2 are open, non-draft, and green for human review; PR 3 is draft pending hardware acceptance; PRs 4–7 remain pending. No auto-merge is enabled.

Update this cursor immediately before pausing for GitHub login, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Current gate: M0 — Mac only. The RTX PC may remain off through phases 00–01 and phase 02 local staging.
- Next PC notification: `PC ACTION REQUIRED — POWER ON`, emitted only after the final Phase 02 branch SHA is pushed, secret-scanned, its draft PR exists, and hosted CI is green.
- After you reply `PC ready`, Codex completes the one-time SSH trust gate, then runs bounded remote diagnostics. `PC REMOTE WORK STARTED` is emitted only after those checks pass.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
