# 06.04 — Release readiness

- **Objective:** Holistically verify definition-of-done evidence, stacked diffs, and the final handoff without merging.
- **Inputs/prerequisites:** All feasible phase work complete; all seven branches/PRs pushed; final security/test/docs results.
- **Implementation tasks:** Walk `REQUIREMENTS.md` DOD01–DOD30 and every other row; classify each Pass/Blocked with the durable evidence schema and next command; verify final branch ancestry/content; check each PR base/head/incremental diff/body/readiness; update every overview actual result/deviation/PR/SHA; update dashboard and merge guide; report commands/tests/hardware/security/blockers and one next command.
- **Files expected to change:** All phase overviews/subphase actual fields as needed, `PLANS/STATUS.md`, `PR_STACK.md`, `REVIEW_AND_MERGE.md`, `README.md`.
- **Validation:** `git log --graph --decorate --all`; `gh pr list/view/diff/checks` for 1–7; final `make doctor`, `make test`, `make lint`, `make secret-scan`, `make public-audit`, and `make pr-status`; project-owned file/link audit; verify PRs open and auto-merge off.
- **Acceptance:** No unexplained pending item or unsupported success claim; blockers contain sanitized exact logs and recovery commands; human can review/merge in order; final branch is complete.
- **Planned commit:** `docs(release): record final validation evidence`.
- **Actual findings:** Phases 00–06 are implemented and evidence-backed. Branch `codex/06-hardening-docs` contains the complete project. `make pr-status` verified all seven PRs open, ready, correctly based/headed, green, and without auto-merge. The public audit, project-owned docs/link review, local/hosted tests, feasible upstream lane, Phase 05 full-run hardware evidence, Phase 06 exact-SHA two-profile smokes, and cleanup evidence are reconciled in `REQUIREMENTS.md`.
- **Remaining blockers:** None. Human review and merge are intentionally outside implementation scope; exact merge/retarget recovery remains in `REVIEW_AND_MERGE.md`.
- **Completion status:** Complete.
