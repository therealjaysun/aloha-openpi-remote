# 06.04 — Release readiness

- **Objective:** Holistically verify definition-of-done evidence, stacked diffs, and the final handoff without merging.
- **Inputs/prerequisites:** All feasible phase work complete; all seven branches/PRs pushed; final security/test/docs results.
- **Implementation tasks:** Walk `REQUIREMENTS.md` DOD01–DOD30 and every other row; classify each Pass/Blocked with the durable evidence schema and next command; verify final branch ancestry/content; check each PR base/head/incremental diff/body/readiness; update every overview actual result/deviation/PR/SHA; update dashboard and merge guide; report commands/tests/hardware/security/blockers and one next command.
- **Files expected to change:** All phase overviews/subphase actual fields as needed, `PLANS/STATUS.md`, `PR_STACK.md`, `REVIEW_AND_MERGE.md`, `README.md`.
- **Validation:** `git log --graph --decorate --all`; `gh pr list/view/diff/checks` for 1–7; final `make doctor test lint secret-scan`; repository/file/link audit; verify PRs open and auto-merge off.
- **Acceptance:** No unexplained pending item or unsupported success claim; blockers contain sanitized exact logs and recovery commands; human can review/merge in order; final branch is complete.
- **Planned commit:** `docs(release): record final validation evidence`.
- **Actual findings:** Phases 00–04 are complete; E-PC-JAX/E-PC-CONVERT were resolved for the demo by the bounded partial-BF16 PyTorch path, and both profiles passed complete tunneled simulation episodes with honest cadence/task-result separation. The repository is public, PRs 1–5 remain open, and the current branch is not release-ready.
- **Remaining blockers:** Implement/validate Phases 05–06 and complete the seven-PR review stack.
- **Completion status:** Planned.
