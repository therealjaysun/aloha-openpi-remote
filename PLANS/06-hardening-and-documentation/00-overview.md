# Phase 06 — Hardening and documentation

- **Objective:** Close test/security/documentation gaps and produce review-ready evidence for the complete two-profile workflow.
- **Scope:** Required unit/integration checks, public-repo audit, user/architecture/troubleshooting docs, CI verification, release checklist, final PR-stack metadata.
- **Non-goals:** New robot tasks, training, performance optimization without measurements, publishing weights/logs/videos, merging PRs.
- **Dependencies:** Locally implemented phase 00–05 interfaces for final integration. Tests/docs/security and blocker recording continue even when GitHub or hardware evidence is externally unavailable.
- **Planned files:** Required tests; `SECURITY.md`, `CONTRIBUTING.md`, `.github/**`, `docs/{ARCHITECTURE,TROUBLESHOOTING}.md`, `README.md`, `AGENTS.md`, Make/scripts/plans, final status updates in `PLANS/REQUIREMENTS.md`.
- **Planned commits:** `test(runtime): cover contracts and lifecycle edges`; `chore(security): audit public repository hygiene`; `docs(setup): document Mac and RTX workflow`; `docs(release): record final validation evidence`.
- **Branch:** `codex/06-hardening-docs`.
- **PR base:** `codex/05-observability`.
- **PR title:** `docs: harden and document the complete workflow`.
- **Acceptance criteria:** Every `REQUIREMENTS.md` row is Pass or Blocked with durable evidence and one exact recovery; README covers both profiles and all DOC rows; secret/public audit passes; no generated/private artifacts tracked; plans/dashboard/PRs contain actual SHAs/results/URLs; all seven PRs remain open; final branch contains complete project.
- **Test commands:** `make doctor`; `make test`; `make lint`; `make secret-scan`; `bash -n scripts/*.sh`; upstream non-manual tests feasible on CPU; CI checks; link/command review; `git diff --check`; staged/tracked-file audit.
- **Risks:** Docs drift from scripts, false hardware claims, secret-scan gaps, private paths in evidence, upstream full test suite too heavy, stacked diff contamination.
- **Rollback:** Revert phase docs/tests/audit changes; retain earlier functional stack; never erase evidence or merge/close PRs automatically.
- **Current status:** Plan complete; implementation not started.
- **Actual results:** Planning hierarchy and upstream audit exist; implementation/test/CI/security/release results do not.
- **Deviations:** Documentation and release matrix must cover both `pi0_aloha_sim` and experimental `pi05_aloha_base`.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Machine handoff

Most tests/docs/security work is Mac-only. Before the final GPU evidence, emit `PC ACTION REQUIRED — POWER ON` if needed; after final `make stop` and process validation, emit `PC SAFE TO POWER OFF`.
