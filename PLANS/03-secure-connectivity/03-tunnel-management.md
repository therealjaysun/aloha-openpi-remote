# 03.03 — Tunnel management

- **Objective:** Start, validate, and stop a single Mac-local SSH forward safely.
- **Inputs/prerequisites:** Reachable alias; verified remote route; free local configured port.
- **Implementation tasks:** Validate configured local/remote ports in `1..65535` and require both normal policy hosts to be loopback (reject `0.0.0.0` and LAN names); preflight local port; interpolate those values into `ssh -N -L`; use `ExitOnForwardFailure`, server-alive settings, batch mode, and bounded connect timeout; launch with `exec` and store the full ownership record atomically in ignored runtime state; validate PID/start identity/command/ports/SHA and `/healthz`; reject duplicate/stale/reused/unrelated PID; trap startup failure; safe stop.
- **Files expected to change:** `scripts/start_tunnel.sh`, `scripts/stop_tunnel.sh`, PID/port helper tests, `.gitignore`, `Makefile`.
- **Validation:** Start twice, local collision, stale PID, unrelated PID, remote close, keepalive, stop twice; inspect listener bound only to `127.0.0.1`.
- **Acceptance:** Tunnel persists after command returns, is uniquely identified, local health passes, and stop never kills an unrelated process.
- **Planned commit:** `feat(ssh): add validated policy tunnel management`.
- **Actual findings:** No tunnel started.
- **Remaining blockers:** Phases 03.01–02.
- **Completion status:** Planned.
