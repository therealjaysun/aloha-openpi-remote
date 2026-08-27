# 03.04 — Policy smoke test

- **Objective:** Validate each server profile from the Mac independently of MuJoCo.
- **Inputs/prerequisites:** Tunnel health; Mac sim/client venv; selected profile server.
- **Implementation tasks:** Expose optional total connect deadline/retry interval, WebSocket `open_timeout`/`close_timeout`, metadata/inference `recv(timeout=...)`, and public idempotent `close()` using monotonic deadlines and WebSockets 14.1 while preserving upstream-compatible optional defaults; project commands always pass finite positive values. Reuse OpenPI client/image tools; construct deterministic ALOHA observation (`state (14,)`, `cam_high uint8 (3,224,224)`, prompt); connect only to configured local loopback; time request 1 as cold, run two warmups, then warmed measured calls; validate response keys/dtype/finite exact `(50,14)` plus server profile/config/checkpoint/SHA/horizon metadata; write sanitized JSON summary; repeat π₀ and π₀.₅ profiles.
- **Files expected to change:** `packages/openpi-client/src/openpi_client/websocket_client_policy.py`, focused client tests, `tools/remote_aloha/connection_check.py`, `scripts/smoke_policy.sh`, contract/connection tests, `Makefile`.
- **Validation:** Valid calls for both profiles; wrong shape/NaN/text-error/open/metadata/inference timeout/closed-server/close-idempotence tests; GPU evidence correlation from phase 02.
- **Acceptance:** Both profiles return valid chunks through the tunnel; first/warm latency recorded; profile/config/checkpoint identity recorded; no MuJoCo dependency in test logic.
- **Planned commits:** `fix(client): bound WebSocket waits and shutdown`; `test(ssh): validate remote OpenPI policy contract`.
- **Actual findings:** Upstream simple client already supplies correct ALOHA random observations and timing, but adds unnecessary Polars/Rich dependencies and lacks strict shape/finite assertions. Plan reuses its contract, not its reporting stack.
- **Remaining blockers:** π₀ currently OOMs before returning an action and π₀.₅ is untested; tunnel implementation and hardware acceptance follow E-PC-CONVERT. WebSockets 14.1 already exposes the required timeout/close API.
- **Completion status:** Planned.
