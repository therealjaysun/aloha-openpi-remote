# Phase 00 — Bootstrap

- **Objective:** Turn the audited upstream clone into a safe, reproducible public derivative with complete plans and lightweight governance.
- **Scope:** Upstream audit, remote layout, public repository, ignore rules, project-facing docs, Make targets, CI, security/contribution files, PR template.
- **Non-goals:** Installing simulation/model dependencies, contacting the PC, running inference, or changing model/runtime behavior.
- **Dependencies:** Git and audited source at `215abfb217dbac7d5f1273282331b9b1866c0479`; authenticated `gh` is required only for origin/push/PR tasks and does not block local implementation.
- **Planned files:** `AGENTS.md`, `README.md`, `Makefile`, `.env.example`, `.gitignore`, `SECURITY.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/workflows/ci.yml`, `scripts/secret_scan.sh`, `PLANS/**`.
- **Planned commits:** `chore(plans): define phased implementation and PR stack`; `chore(security): add fail-closed pre-push scan`; `chore(repo): establish public repository baseline`; `ci: add lightweight public checks`.
- **Branch:** `codex/00-bootstrap`.
- **PR base:** `main`.
- **PR title:** `chore: establish remote ALOHA project baseline`.
- **Acceptance criteria:** Official history/submodules preserved; `upstream` is official; public project `origin` is verified; required root/governance files exist; no private or generated data is tracked; plan links resolve; CI uses no GPU; PR 1/7 is open.
- **Test commands:** `git status --short`; `git remote -v`; `git submodule status`; `make lint`; `make test`; `make secret-scan`; `git diff --check`; `gh pr view --json url,state,isDraft,baseRefName,headRefName`.
- **Risks:** Invalid GitHub auth, repository name collision, accidental staging of `.DS_Store`/secrets, upstream CODEOWNERS notifying unrelated maintainers, replacing upstream CI with weaker checks.
- **Rollback:** Revert the unmerged phase branch or remove the local project remote after verification; restore remote names; never rewrite `main` or upstream history. Deleting a public repository requires separate explicit user approval.
- **Current status:** Complete; public repository, hardened CI, and PR 1 are verified.
- **Actual results:** Official history/submodules are preserved; `upstream` is fetch-only with push disabled; public `origin` is `https://github.com/therealjaysun/pi-robotics`; `main` is pinned to upstream SHA `215abfb217dbac7d5f1273282331b9b1866c0479`. Gitleaks 8.30.1 passed the project range, staged changes, and all non-ignored candidates. Actions permits only selected immutable-SHA actions, requires SHA pinning, and grants read-only default workflow permissions. Hosted `secret-scan` passed; derivative-inapplicable upstream private-runner jobs skipped explicitly.
- **Deviations:** GitHub authentication delayed publication but did not block local Phase 01. The validated Phase 01 commit was preserved by merging this parent branch instead of rewriting it.
- **PR:** [#1](https://github.com/therealjaysun/pi-robotics/pull/1), open for human review, base `main`, head `codex/00-bootstrap`, not draft.
- **Final commit SHA:** Publication safeguard `076becccc075f44ad79f33044e0d7f205861cf20`; repository evidence `27d7cabe394ea56db80c0b99871bc43b8abe0ccb`; this final evidence update is branch HEAD.

## Machine handoff

Mac only. Keep the RTX PC off; GitHub/repository/planning work does not depend on it.
