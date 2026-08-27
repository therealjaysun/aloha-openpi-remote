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
- **Current status:** Plan complete; implementation not started; GitHub blocker active.
- **Actual results:** Official repository cloned with submodules and audited; planning hierarchy created. No public repo, branch, commit, push, or PR created.
- **Deviations:** User requested planning only, so execution stops before phase implementation.
- **PR:** Pending; number and URL not assigned.
- **Final commit SHA:** Pending.

## Machine handoff

Mac only. Keep the RTX PC off; GitHub/repository/planning work does not depend on it.
