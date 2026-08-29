# 03.01 — SSH discovery

- **Objective:** Validate the `robot-gpu` alias and discover the remote command environment without exposing config values.
- **Inputs/prerequisites:** User-configured SSH host alias and independently fingerprint-verified host key per `EXECUTION_LOGISTICS.md`.
- **Implementation tasks:** Use `ssh -G` only to report configured/not configured; connect with `BatchMode=yes` and bounded timeout; identify direct Bash/WSL, Windows PowerShell, or Windows `cmd.exe` using fixed probes; never print SSH config, keys, address, username, or hostname; return the SSH trust gate and exact user-run recovery when absent.
- **Files expected to change:** `scripts/doctor_pc.sh`, command-builder tests, `docs/TROUBLESHOOTING.md`.
- **Validation:** Alias-missing, untrusted/mismatched host key, connection-failed, PowerShell, cmd.exe, and direct-WSL cases; sanitizer tests.
- **Acceptance:** One route selected from evidence; connection bounded; no host-key bypass; tracked/logged output is sanitized.
- **Planned commit:** `feat(ssh): discover remote execution environment`.
- **Actual findings:** Local `ssh -G` indicates `robot-gpu` is not configured.
- **Remaining blockers:** User must add a valid alias; values cannot be guessed.
- **Completion status:** Blocked.
