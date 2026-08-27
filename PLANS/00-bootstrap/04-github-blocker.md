# 00.04 — Active GitHub blocker

- **Objective:** Preserve exact, sanitized recovery information for public repository creation.
- **Inputs/prerequisites:** GitHub CLI 2.89.0.
- **Implementation tasks:** Re-authenticate interactively; verify account; rerun collision inspection and setup from `03-public-github-setup.md`; meanwhile continue all local branches/commits and record every SHA ready to push; delete this file if no blocker remains or mark resolved with date/evidence.
- **Files expected to change:** This file, `PLANS/STATUS.md`, `README.md`.
- **Validation:** `gh auth status`; `gh api user --jq .login`; `gh repo view`.
- **Acceptance:** Valid auth and a verified public project repository; no token output captured.
- **Planned commit:** `chore(repo): resolve GitHub setup blocker`.
- **Actual findings:** `gh auth status` failed: stored token for the active account is invalid. Expected repository name: `aloha-openpi-remote`. No repository creation command was attempted.
- **Remaining blockers:** User must complete the browser/device authentication flow.
- **Completion status:** Active blocker.

Exact recovery:

```bash
gh auth login -h github.com
gh auth status
gh api user --jq .login
```

Then resume `03-public-github-setup.md`. GitHub publication is not a gate for local implementation. Local phase branches/commits ready to push: none (planning-only turn; update this line as work proceeds).
