# Phase 03 — Secure connectivity

- **Objective:** Reach the loopback-only WSL policy server from the Mac exclusively through an authenticated SSH local forward.
- **Scope:** SSH alias/trust validation, remote shell routing, Windows↔WSL localhost validation, tunnel PID lifecycle, bounded WebSocket client I/O/close, independent policy smoke test.
- **Non-goals:** Public/LAN port exposure, disabling host-key checking, firewall/portproxy/WSL networking changes without approval, embedding credentials.
- **Dependencies:** Phase 02 local interfaces for unblocked implementation. Hardware acceptance additionally requires phase 02 WSL-local GPU inference, a configured/fingerprint-verified `robot-gpu`, and a server ready on remote loopback.
- **Planned files:** `scripts/{start_tunnel,stop_tunnel,smoke_policy}.sh`, `tools/remote_aloha/connection_check.py`, minimal `packages/openpi-client/.../websocket_client_policy.py` timeout/close patch and focused tests, tests for SSH/PowerShell/cmd/WSL command construction and port/PID checks, `Makefile`, docs.
- **Planned commits:** `feat(ssh): add validated policy tunnel management`; `fix(client): bound WebSocket waits and shutdown`; `test(ssh): validate remote policy contract`.
- **Branch:** `codex/03-secure-connectivity`.
- **PR base:** `codex/02-remote-gpu-server`.
- **PR title:** `feat(ssh): add secure policy tunnel`.
- **Acceptance criteria:** Real shell route detected; Windows host can reach WSL loopback or exact blocker recorded; validated loopback hosts/ports drive a local forward that starts once, is health-checked, persists, and stops safely; client connect/metadata/inference waits are finite and `close()` unblocks shutdown; policy smoke verifies identity and finite `(50,14)` for both profiles; port is not publicly exposed.
- **Test commands:** `make tunnel`; `make smoke-policy` per profile; `make stop`; `lsof -nP -iTCP:8000 -sTCP:LISTEN`; remote listener/routing checks; unit tests.
- **Risks:** Wrong shell quoting, stale PID, local port collision, Windows-to-WSL localhost forwarding differences, mirrored networking exposure, leaking identifiers in logs.
- **Rollback:** Stop only validated tunnel PID; revert scripts; do not remove SSH keys/config or alter firewall/networking.
- **Current status:** Plan complete; implementation not started. Pure client/routing/lifecycle work is unblocked; hardware acceptance is blocked on SSH/server.
- **Actual results:** No alias, tunnel, route, port, or smoke test exists.
- **Deviations:** None.
- **PR:** Pending.
- **Final commit SHA:** Pending.

## Machine handoff

Both machines stay on, but Codex continues from the Mac. The user only visits the PC if Windows→WSL routing diagnostics produce `PC CONSOLE ACTION REQUIRED` with an explicit recovery step.
