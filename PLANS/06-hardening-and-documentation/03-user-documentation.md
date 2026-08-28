# 06.03 — User documentation

- **Objective:** Make the Mac→SSH→Windows/WSL→RTX workflow reproducible without unexplained or private manual steps.
- **Inputs/prerequisites:** Final commands, measured outputs, blockers, public URL, profile behavior.
- **Implementation tasks:** Complete `REQUIREMENTS.md` DOC01–DOC22; document `OPENPI_POLICY_PROFILE` with π₀ default and π₀.₅ experimental caveat; add a compact “Differences from upstream” inventory covering the server host/metadata changes, client timeout/close, environment/runtime/finalizer changes, buffering, telemetry, and lifecycle scripts; write deeper architecture/data contracts and exact troubleshooting decision tree; keep placeholders and commands copyable.
- **Files expected to change:** `README.md`, `docs/ARCHITECTURE.md`, `docs/TROUBLESHOOTING.md`, `AGENTS.md`, `.env.example`, Makefile help.
- **Validation:** Fresh-shell command walkthrough on each available machine; link check; compare every documented target/flag/default to source and `make help`; search for private identifiers/paths.
- **Acceptance:** `REQUIREMENTS.md` DOC01–DOC22 pass or carry exact external blockers; preferred quick start works or stops with one exact next command; both profiles are selectable; substantial upstream differences are listed; no stale/invented/manual step.
- **Planned commit:** `docs(setup): document complete Mac and RTX workflow`.
- **Actual findings:** README, architecture, security, contribution, and troubleshooting documents now cover DOC01–DOC22: project identity/upstream link and pin, Mac/PC/SSH prerequisites, explicit Ubuntu selection, both profiles, partial-BF16 automatic selection, every stable Make target, output boundaries, measured Phase 05 results, honest cadence limits, recovery, attribution, and seven-PR review order. Every documented target exists and all project-owned Markdown links resolve; inherited unmodified upstream docs are outside this link claim.
- **Remaining blockers:** None.
- **Completion status:** Complete.

Planned profile examples:

```bash
OPENPI_POLICY_PROFILE=pi0_aloha_sim make server
OPENPI_POLICY_PROFILE=pi0_aloha_sim make run

OPENPI_POLICY_PROFILE=pi05_aloha_base make server
OPENPI_POLICY_PROFILE=pi05_aloha_base make run
```
