# 03.03 — Tunnel management

- **Objective:** Start, validate, and stop a single Mac-local SSH forward safely.
- **Inputs/prerequisites:** Reachable alias; verified remote route; free local configured port.
- **Implementation tasks:** Validate configured local/remote ports in `1..65535` and require both hosts to be literal IPv4 loopback; preflight the local port; reject SSH-config forwards; launch one explicit `ssh -T -N -f -L` with strict trust, no agent/X11/local-command features, keepalive, and `ExitOnForwardFailure`; manage it through a private ControlMaster socket; atomically record PID/start identity/actual command hash/ports/SHA; validate control PID, process identity, exact `lsof` listener, and `/healthz`; reject duplicate, forged, or unknown partial state; remove validated stale state without signaling; stop only through the authenticated control socket.
- **Files expected to change:** `tools/remote_aloha/connection_check.py`, lifecycle tests, `Makefile`; `.runtime` was already ignored.
- **Validation:** Start twice, local collision, stale PID, unrelated PID, remote close, keepalive, stop twice; inspect listener bound only to `127.0.0.1`.
- **Acceptance:** Tunnel persists after command returns, is uniquely identified, local health passes, and stop never kills an unrelated process.
- **Planned commit:** `feat(ssh): add validated policy tunnel management`.
- **Actual findings:** The ControlMaster lifecycle and fail-closed state validation pass local focused tests. A separate manager is necessary because the Linux `/proc`/pidfd helper cannot safely signal macOS tunnel processes.
- **Remaining blockers:** Real Windows-loopback acceptance, then start/duplicate/health/listener/stop checks on the Mac.
- **Completion status:** Locally implemented; hardware/network acceptance pending.
