# Phase 03 — Secure connectivity

- **Objective:** Reach the loopback-only WSL policy server from the Mac exclusively through an authenticated SSH local forward.
- **Scope:** SSH alias/trust validation, remote shell routing, Windows↔WSL localhost validation, tunnel PID lifecycle, bounded WebSocket client I/O/close, independent policy smoke test.
- **Non-goals:** Public/LAN port exposure, disabling host-key checking, firewall/portproxy/WSL networking changes without approval, embedding credentials.
- **Dependencies:** Phase 02 local interfaces for unblocked implementation. Hardware acceptance additionally requires phase 02 WSL-local GPU inference, a configured/fingerprint-verified `robot-gpu`, and a server ready on remote loopback.
- **Planned files:** `tools/remote_aloha/{remote,connection_check,policy_smoke}.py`, minimal client timeout/close and server Origin/frame/error patches, focused SSH/PowerShell/cmd/WSL/tunnel/WebSocket tests, `Makefile`, docs.
- **Planned commits:** `feat(ssh): add validated policy tunnel management`; `fix(client): bound WebSocket waits and shutdown`; `test(ssh): validate remote policy contract`.
- **Branch:** `codex/03-secure-connectivity`.
- **PR base:** `codex/02-remote-gpu-server`.
- **PR title:** `feat(ssh): add secure policy tunnel`.
- **Acceptance criteria:** Real shell route detected; Windows host can reach WSL loopback or exact blocker recorded; validated loopback hosts/ports drive a local forward that starts once, is health-checked, persists, and stops safely; client connect/metadata/inference waits are finite and `close()` unblocks shutdown; policy smoke verifies identity and finite `(50,14)` for both profiles; port is not publicly exposed.
- **Test commands:** `make tunnel`; `make smoke-policy` per profile; `make stop`; `lsof -nP -iTCP:8000 -sTCP:LISTEN`; remote listener/routing checks; unit tests.
- **Risks:** Wrong shell quoting, stale PID, local port collision, Windows-to-WSL localhost forwarding differences, mirrored networking exposure, leaking identifiers in logs.
- **Rollback:** Stop only validated tunnel PID; revert scripts; do not remove SSH keys/config or alter firewall/networking.
- **Current status:** Locally implemented and validated; exact-candidate two-profile idle-survival/tunnel acceptance remains.
- **Actual results:** The first real Phase 03 run proved Windows-loopback routing, an exact Mac IPv4-loopback listener, and a four-call tunneled π₀ smoke. It also found that this Windows host stops the WSL VM and its background policy process after the final Windows-side `wsl.exe` exits. A bounded diagnostic proved that one synchronous Windows WSL client prevents teardown. The project now makes the existing SSH ControlMaster run that fixed holder command, tied to the original verified server record and a random run ID. Local tests cover holder quoting/identity, record removal/replacement, launch cleanup, rollback order, client deadlines, Origin/frame/error handling, and safe repeated stop. Final π₀ and π₀.₅ hardware acceptance remains.
- **Deviations:** A single Python ControlMaster manager replaces separate start/stop shell scripts. The ControlMaster is both the exact loopback tunnel and the WSL lifetime holder, avoiding detached Windows processes or system configuration. Shutdown remains authenticated through its private control socket; WSL server signaling remains PIDfd-gated.
- **PR:** Draft [PR 4](https://github.com/therealjaysun/pi-robotics/pull/4).
- **Final commit SHA:** Pending.

## Machine handoff

After the exact Phase 03 candidate is pushed and secret-scanned, both machines stay on while Codex runs the route, tunnel, and profile smokes remotely from the Mac. No Windows networking or firewall mutation is authorized.
