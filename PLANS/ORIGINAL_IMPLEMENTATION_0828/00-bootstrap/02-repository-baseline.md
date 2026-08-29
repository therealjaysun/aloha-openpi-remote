# 00.02 — Repository baseline

- **Objective:** Establish a derivative baseline without losing history or unrelated work.
- **Inputs/prerequisites:** Completed upstream audit; clean index; current official `origin` verified by URL and commit.
- **Implementation tasks:** Preserve untracked `.DS_Store`; add ignore entry; rename verified official `origin` to `upstream`; create phase branch; add compact project docs/config; before any public push create fail-closed `scripts/secret_scan.sh` plus `make secret-scan`, bootstrap gitleaks or stop with its install command, and test staged/project-range/non-ignored-candidate coverage; inspect upstream CODEOWNERS and remove derivative-inapplicable owner assignments rather than notifying upstream maintainers; preserve licenses/submodules/history.
- **Files expected to change:** `.gitignore`, `AGENTS.md`, `README.md`, `Makefile`, `.env.example`, `scripts/secret_scan.sh`, `PLANS/**`, `.github/CODEOWNERS` deletion.
- **Validation:** `git status --short`; `git remote get-url upstream`; `git log --oneline -1`; `git submodule status`; `make secret-scan`; `git diff --check`; staged-file inspection.
- **Acceptance:** `upstream` is exactly official OpenPI; no user files overwritten; baseline identifies project as independent; generated/private paths ignored; the fail-closed scan target exists and passes before subphase 00.03 can push.
- **Planned commits:** `chore(plans): define phased implementation and PR stack`; `chore(security): add fail-closed pre-push scan`; `chore(repo): establish derivative baseline`.
- **Actual findings:** The derivative baseline is complete. `origin` is the standalone public `therealjaysun/pi-robotics` project; official OpenPI is fetch-only `upstream` with push disabled. Licenses, submodules, and Git history are retained; upstream CODEOWNERS was removed; generated `.DS_Store` files remain ignored.
- **Remaining blockers:** None.
- **Completion status:** Complete.
