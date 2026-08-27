# 06.02 — Security and public-repo audit

- **Objective:** Prove the public Git history/staging/config contains no secrets, machine identifiers, weights, or generated data and exposes no model port.
- **Inputs/prerequisites:** Final branch and origin; secret scanner installed/selected; all outputs stopped.
- **Implementation tasks:** Preserve `LICENSE`, `LICENSE_GEMMA.txt`, submodule licensing, and modified-source notices; fail closed if gitleaks is unavailable; before every push scan staged files, all project-added commits, and non-ignored candidate files with gitleaks plus explicit tracked-file patterns; inspect `.env`/keys/private addresses/OS or SSH identities/absolute paths/logs/videos/checkpoints/telemetry; verify loopback listeners and SSH settings; audit Actions permissions/dependencies; confirm derivative disclaimer and substantial-changes inventory.
- **Files expected to change:** `.gitignore`, `scripts/secret_scan.sh`, `SECURITY.md`, README attribution/security, CI, sanitizer tests; audit results in plans.
- **Validation:** `make secret-scan`; `git ls-files`; `git diff --cached --name-status`; `git log -p main..HEAD` scan; listener checks; clean generated-file status.
- **Acceptance:** `REQUIREMENTS.md` PH01–PH20 pass; scanner has no suppressions hiding real findings; no sensitive/generated files exist in any phase commit; server/tunnel are loopback-only; no host-key/firewall weakening; attribution is intact.
- **Planned commit:** `chore(security): audit public repository hygiene`.
- **Actual findings:** Upstream has no root `NOTICE`; both root license files remain. Upstream CODEOWNERS was removed, the standalone public origin exists, and fail-closed Gitleaks plus explicit candidate checks pass through Phase 02.
- **Remaining blockers:** Repeat the audit against the final Phase 06 history and live listener state.
- **Completion status:** Planned.
