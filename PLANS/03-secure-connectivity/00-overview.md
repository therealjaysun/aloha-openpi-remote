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
- **Current status:** Locally implemented and validated; exact-candidate Windows-loopback, Mac tunnel, and two-profile hardware acceptance remain.
- **Actual results:** The private alias, strict key/host trust, bounded SSH, remote cmd route, explicit WSL distro, and WSL loopback policy health are proven in Phase 02. Phase 03 adds quiet alias validation, fixed Windows-loopback health/non-wildcard checks, an exact Mac IPv4-loopback OpenSSH ControlMaster forward, private atomic ownership, bounded WebSocket stages, explicit close, browser-Origin rejection, bounded default frames, generic remote errors, and sanitized Mac smoke evidence. Local tests pass; no Phase 03 tunnel has run yet.
- **Deviations:** A single Python ControlMaster manager replaces separate start/stop shell scripts. This uses OpenSSH's authenticated control socket for shutdown instead of a macOS verify-then-signal PID race; the existing WSL smoke shell remains an internal Phase 02 diagnostic.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Machine handoff

After the exact Phase 03 candidate is pushed and secret-scanned, both machines stay on while Codex runs the route, tunnel, and profile smokes remotely from the Mac. No Windows networking or firewall mutation is authorized.
