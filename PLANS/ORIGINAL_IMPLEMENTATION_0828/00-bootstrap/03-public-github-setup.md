# 00.03 — Public GitHub setup

- **Objective:** Create or safely reuse the authenticated user's public project repository and set `origin`.
- **Inputs/prerequisites:** Valid `gh auth status`; verified authenticated login; official remote already renamed `upstream`; passing fail-closed scan and staged/candidate inspection.
- **Implementation tasks:** Inspect `gh repo create --help` and current GitHub Actions-permission API docs; resolve owner with `gh api user --jq .login`; inspect `OWNER/pi-robotics`; reuse only if clearly this project and fast-forward compatible, otherwise choose a collision-safe suffix and record it. Create the empty public repo without overwriting and immediately disable Actions before the first workflow-bearing push; if that setting cannot be verified, do not push. Set `origin`; rerun hygiene checks; explicitly `git push -u origin main` while Actions remain disabled; verify tracking/default branch `main`; never force an existing repository. Subphase 00.05 hardens workflows on the phase branch, sets default workflow permissions read-only, and enables Actions only before opening PR 1.
- **Files expected to change:** `README.md`, `PLANS/STATUS.md`; Git remote config (not tracked).
- **Validation:** `gh repo view OWNER/REPO --json nameWithOwner,url,visibility,defaultBranchRef`; `git remote -v`; `git ls-remote origin`; `git config --get branch.main.remote`; secret scan; `git diff --cached --name-status`.
- **Acceptance:** Repo is public, project-related, owned by authenticated account; `origin` is it; `upstream` remains official; URL recorded; baseline pushed with no sensitive/generated files and no inherited workflow execution; Actions remain disabled pending subphase 00.05.
- **Planned commit:** `chore(repo): record public repository`.
- **Actual findings:** Authentication is valid for the verified account. The public repository was created independently and later renamed to `https://github.com/therealjaysun/pi-robotics`; it is not a GitHub fork (`parent=null`). Official OpenPI remains fetch-only `upstream`, with its push URL disabled. Actions was disabled and verified before `main` was first pushed at `215abfb217dbac7d5f1273282331b9b1866c0479`.
- **Remaining blockers:** None.
- **Completion status:** Complete. Hardened branches were pushed while Actions was disabled; subphase 00.05 then enabled selected immutable-SHA actions with read-only workflow permissions and verified PR 1.

Recovery execution outline (replace `OWNER` only from `gh api`, never by guess):

```bash
gh auth status
OWNER="$(gh api user --jq .login)"
gh repo view "$OWNER/pi-robotics"
gh repo create "$OWNER/pi-robotics" --public --source=. --remote=origin
# Disable Actions through the verified GitHub API before pushing; stop if unverifiable.
make secret-scan
git push -u origin main
```
