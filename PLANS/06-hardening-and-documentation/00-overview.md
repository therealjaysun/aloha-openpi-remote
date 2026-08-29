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
- **Test commands:** `make doctor`; `make test`; `make lint`; `make secret-scan`; `make public-audit`; `make pr-status`; `bash -n scripts/*.sh`; upstream non-manual tests feasible on CPU; CI checks; project-owned link/command review; `git diff --check`; staged/tracked-file audit.
- **Risks:** Docs drift from scripts, false hardware claims, secret-scan gaps, private paths in evidence, upstream full test suite too heavy, stacked diff contamination.
- **Rollback:** Revert phase docs/tests/audit changes; retain earlier functional stack; never erase evidence or merge/close PRs automatically.
- **Current status:** Complete; open for review.
- **Actual results:** Final implementation candidate `a8a3ca14833b4e2f7f440e00cc0ea98b21a61a58` passed 318 project tests with one platform skip, Ruff/format/Bash, 21 feasible upstream OpenPI-client tests, desktop Mac and repository doctors, fail-closed secret/public-history audits, and project-owned Markdown links. All four hosted PR 7 checks passed. The final validator found 48 plan files, 7 phase overviews, 29 structured subphases, 28 live configuration keys, 157 requirement IDs, and 30 DOD rows; branch ancestry passed. `make pr-status` verified PRs 1–7 open, ready, correctly stacked, green, and without auto-merge. Phase 5 candidate `2065dd9d` supplies the newer exact-hardware trajectory proof.
- **Deviations:** The Phase 6 commits were rebased onto completed Phase 5 and reconciled rather than rebuilding the architecture. The heavyweight upstream model/training/GPU suite remains infeasible in the lightweight Mac/public-CI lane; 21 feasible client tests passed. The historical Phase 6 candidate `90b0fed` already passed both lightweight profile smokes, while the newer parent `2065dd9d` passed both full trajectory runs; no redundant GPU run was added because the final integration changed only local validation, cleanup ordering, documentation, and strict schema checks. Full-FP32 monolithic loading remains source-tested but not hardware-run because the available artifacts use validated sharded partial BF16.
- **PR:** [PR 7](https://github.com/therealjaysun/pi-robotics/pull/7).
- **Final commit SHA:** Validated implementation `a8a3ca14833b4e2f7f440e00cc0ea98b21a61a58`; final evidence is at branch HEAD.

## Machine handoff

Phase 05 supplies the final exact-hardware GPU, trajectory, and performance evidence. No further PC work is required; the PC is safe to power off.
