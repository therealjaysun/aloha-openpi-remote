# Project status

Planning baseline: 2026-08-26. Plans are complete; implementation has not started.

Final plan review: 2026-08-26. Independent specification, technical-source, and operations/security audits were reconciled; structural and link validation evidence is recorded by the completing turn.

| Phase | Status | Branch | PR number | PR URL | Base branch | Head branch | Tests | Blockers | Last commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 Bootstrap | Planned | `codex/00-bootstrap` | — | — | `main` | `codex/00-bootstrap` | Not run | GitHub token invalid; public origin not configured; repository existence unknown | — |
| 01 Mac simulation | Planned | `codex/01-mac-simulation` | — | — | `codex/00-bootstrap` | `codex/01-mac-simulation` | Not run | Native packages not installed/tested | — |
| 02 Remote GPU | Planned | `codex/02-remote-gpu-server` | — | — | `codex/01-mac-simulation` | `codex/02-remote-gpu-server` | Not run | SSH alias and PC access unavailable | — |
| 03 Connectivity | Planned | `codex/03-secure-connectivity` | — | — | `codex/02-remote-gpu-server` | `codex/03-secure-connectivity` | Not run | Remote shell/routing mode unknown | — |
| 04 End-to-end | Planned | `codex/04-end-to-end-control` | — | — | `codex/03-secure-connectivity` | `codex/04-end-to-end-control` | Not run | Depends on phases 01–03 | — |
| 05 Observability | Planned | `codex/05-observability` | — | — | `codex/04-end-to-end-control` | `codex/05-observability` | Not run | Depends on live inference | — |
| 06 Hardening/docs | Planned | `codex/06-hardening-docs` | — | — | `codex/05-observability` | `codex/06-hardening-docs` | Not run | Depends on all evidence | — |

## Audited state

- Volatile snapshot at `2026-08-27T03:09:59Z`: Apple Silicon arm64, macOS 26.6.1, 18 GB RAM, 58 GiB workspace disk free. Rerun `make doctor-mac` before capacity decisions.
- Tools: Python 3.14.5 default, `uv 0.12.1`, `gh 2.89.0`, Docker 28.0.4.
- Required Python 3.10 environments are not created; simulator imports are absent from the default Python by design.
- Source baseline: the official clone was clean at `215abfb217dbac7d5f1273282331b9b1866c0479`. The current planning working tree contains this `PLANS/` hierarchy and the README audit banner, plus unrelated untracked `.DS_Store` files that must not be staged.
- Remotes: current `origin` is official OpenPI; phase 00 must verify and rename it to `upstream` before creating the user's `origin`.
- Submodules: ALOHA `d1dc83afd89ded4379851257fe5d85632d31d5ec`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- GitHub: authentication check failed because the stored token is invalid. Repository existence was not queried. See [`00-bootstrap/04-github-blocker.md`](00-bootstrap/04-github-blocker.md).
- SSH: `robot-gpu` is not configured. No remote facts have been guessed.
- Public repository URL: pending.

## Final plan review evidence

- `2026-08-27T03:17:02Z`, Mac, upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`, uncommitted planning working tree, profile N/A, raw artifact N/A.
- Independent specification, pinned-source technical, and operations/security re-reviews each reported no unresolved plan finding after amendments.
- Structural validator exited 0: 7 phase overviews and 29 subphase plans contain every required field; no root `IMPLEMENTATION_PLAN.md` exists.
- Relative-link and branch/base-stack validators exited 0; all 42 plan Markdown files resolve their local links.
- Traceability validator exited 0: 151 requirement rows have 151 unique IDs, all terminal requirement IDs exist, and all 27 configuration keys are enumerated.
- Obsolete-rule, trailing-whitespace, and tracked README diff checks exited 0. No implementation, GitHub, simulator, SSH, WSL, GPU, or inference result is implied by these planning checks.

## Execution cursor

- Active subphase: planning final review complete; implementation has not started.
- Machine gate: M0 — Mac only.
- Exact next implementation command: `git switch -c codex/00-bootstrap` after confirming the planning changes are ready to become the phase baseline.
- External recovery (non-blocking for local work): `gh auth login -h github.com`.
- Last verified local SHA: `215abfb217dbac7d5f1273282331b9b1866c0479`; last verified WSL SHA: unknown.
- PR state: all seven pending; when created they remain draft until full phase acceptance or a user-approved named hardware exception.

Update this cursor immediately before pausing for GitHub login, `PC ready`, PC console work, or power-off.

## Hardware coordination

- Current gate: M0 — Mac only. The RTX PC may remain off through phases 00–01 and phase 02 local staging.
- Next PC notification: `PC ACTION REQUIRED — POWER ON`, emitted only after a secret-scanned phase 02 remote-test candidate SHA is ready.
- After you reply `PC ready`, Codex completes the one-time SSH trust gate, then runs bounded remote diagnostics. `PC REMOTE WORK STARTED` is emitted only after those checks pass.
- Full procedure: [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md).
