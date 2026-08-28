# Mac/PC execution logistics

Codex remains in the Mac project workspace. PC-side work runs remotely over the `robot-gpu` SSH alias inside WSL2, so you normally do not need to move between two Codex sessions. WSL always runs a secret-scanned exact Git commit from this project, never an ad-hoc copy or a pristine upstream clone.

## Machine schedule

| Gate | Phases | Mac | RTX PC | What you do |
| --- | --- | --- | --- | --- |
| M0 — Mac only | 00–01 + 02 local staging | On; development, pure tests, and remote-test candidate commit happen here | May remain off | Repair GitHub login when convenient; otherwise no PC action |
| P1 — Power/SSH trust | 02 after candidate SHA is ready | On | Turn on, sign in to Windows, connect to the same LAN, keep awake; complete one-time SSH trust if requested | Reply `PC ready`, then follow only the fingerprint-verification prompt if needed |
| P2 — Remote setup/conversion | 02 remote validation–03 | Codex runs bounded diagnostics, the partial-BF16 recovery experiment, and SSH commands here | On; WSL/OpenPI/converter/server run the exact candidate SHA; no PC-side CI | Stay at Mac unless Codex reports a Windows-admin blocker |
| B1 — Both machines | 04–05 | MuJoCo, client, tunnel, video, local telemetry | Policy server and GPU telemetry | Keep PC on/awake; no routine console work |
| P3 — Final validation | 06 | Tests, docs, security scan, PR work | On only for final GPU/inference checks | Turn on if it was powered down; reply `PC ready` |
| OFF — Shutdown | After final validation or a durable blocker | Stop tunnel/runtime | Stop server/GPU sampler, then PC may power off | Wait for `PC SAFE TO POWER OFF` |

## Notifications Codex will send

### `PC ACTION REQUIRED — POWER ON`

Sent only after phase 02 local code/tests are captured in a secret-scanned remote-test candidate SHA, or when final hardware validation must resume. Before replying `PC ready`:

1. Turn on the RTX PC and sign in to Windows.
2. Ensure it is on the same local network as the Mac and will not sleep during the session.
3. Do not expose or forward port 8000.
4. Have the real Windows SSH connection details available privately; do not add them to tracked files.

After `PC ready`, Codex first checks the alias/trust gate below. It then runs bounded batch SSH and the read-only portion of `make doctor-pc` to discover the remote shell, WSL distro, CUDA visibility, RTX 3090, disk, port, and OpenPI state before changing anything.

### `PC ACTION REQUIRED — SSH SETUP`

Sent only when `robot-gpu` or its trusted host key is absent. The user creates the alias outside the repository and independently compares the server's public host-key fingerprint at the PC with the fingerprint shown by the first Mac connection. Make one interactive connection only after they match. Never print config values or key material, and never disable host-key checking. If multiple WSL distros exist, the user selects one; Codex does not guess.

### `PC CONSOLE ACTION REQUIRED`

Sent only if read-only diagnostics prove SSH, WSL, NVIDIA driver access, or Windows→WSL localhost forwarding needs a local/admin action. Codex will provide the exact failing check and smallest recovery step. It will not change drivers, firewall, WSL networking, or the SSH service without explicit approval.

### `PC REMOTE WORK STARTED`

Sent only after host trust, bounded batch SSH, shell routing, WSL, and basic GPU diagnostics pass. It confirms that Codex is installing/testing OpenPI inside WSL from the Mac. Keep the PC on; no physical switch is needed.

### `PC SAFE TO POWER OFF`

Sent after final validation or a durable external blocker, but only when `make stop` attempts cleanup for every owned component and verifies the Mac tunnel plus WSL policy/GPU-metrics processes are stopped. Any live, mismatched, or unverifiable process makes `make stop` nonzero and suppresses this notification. Powering off earlier preserves data already written but interrupts the active phase; the next session must restart server and tunnel.

## One-time handoff checklist

At gate P1, Codex validates steps 1–8 for Phase 02 and stops at the first real blocker. Those steps are complete; step 9 resumes in Phase 03:

1. `robot-gpu` alias exists without printing its private values; if first-use trust is absent, complete the fingerprint gate above.
2. Bounded batch SSH connects with host-key checking intact.
3. Remote shell is detected as Windows PowerShell, Windows `cmd.exe`, or direct WSL Bash.
4. The WSL distro is detected, never guessed.
5. `nvidia-smi` sees the RTX 3090 inside WSL.
6. An absolute WSL POSIX project path is resolved inside WSL. A fixed `~/` default may be expanded against WSL `$HOME`; all other relative paths are rejected, with no `eval`.
7. The project branch is fetched from public `origin`, or from a secret-scanned Git bundle if GitHub remains blocked; its exact SHA equals the Mac remote-test candidate and contains the audited upstream pin.
8. A WSL-local request proves the selected policy binds loopback, passes `/healthz`, and performs GPU inference.
9. In Phase 03, the Mac reaches it only through the local SSH tunnel.

## Source and process ownership contract

- Stop verified owned processes before changing the WSL checkout SHA.
- Store each server, tunnel, and GPU-sampler runtime record atomically in ignored state. It includes PID, OS start identity, exact command signature, run/profile, port, and source SHA; launch wrappers use `exec` so the PID is the owned process.
- A stop command never signals a PID whose identity differs. `make stop` is best-effort across all components, reports every failure, verifies processes/listeners afterward, and returns nonzero on uncertainty.
- Raw server/sampler logs stay ignored on their originating machine. Copy them to the ignored run directory when required; only an allowlisted sanitized summary may be published.

Machine-specific values live only in local SSH configuration, ignored `.env`, or ignored raw evidence; plans, commits, PRs, published summaries, and example telemetry use placeholders.
