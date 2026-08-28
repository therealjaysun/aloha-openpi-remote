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
- **Actual results:** Candidate `6885787438537a949dad0482e520733d3c1d396f` passed 286 project tests with one Linux-only skip, Ruff/format/Bash, 21 feasible upstream OpenPI-client tests, the desktop Mac doctor, fail-closed secret/public-history audits, and local Markdown link validation. Hosted PR 7 checks passed. `make pr-status` verified PRs 1–7 open, non-draft, correctly stacked, green, and without auto-merge. README, architecture, security, contribution, and troubleshooting guidance cover both profiles and the complete Mac/WSL workflow.
- **Deviations:** Documentation and release evidence cover both `pi0_aloha_sim` and experimental `pi05_aloha_base`. The planned four small commits were collapsed into one coherent implementation commit plus this evidence commit because the code/tests/docs formed one release gate. The heavyweight upstream model/training/GPU suite was not installed on the Mac or public runner; 21 feasible upstream client tests passed and the exact infeasible lane is recorded.
- **PR:** [PR 7](https://github.com/therealjaysun/pi-robotics/pull/7).
- **Final commit SHA:** Validated implementation `6885787438537a949dad0482e520733d3c1d396f`; final evidence is at branch HEAD.

## Machine handoff

Phase 05 already supplied the final GPU evidence and verified zero Mac/Windows/WSL residue. Phase 06 required no PC rerun; the PC is safe to power off.
