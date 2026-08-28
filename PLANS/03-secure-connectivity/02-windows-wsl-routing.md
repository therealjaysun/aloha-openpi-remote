# 03.02 — Windows-to-WSL routing

- **Objective:** Verify the SSH server's network namespace can reach the WSL loopback policy port.
- **Inputs/prerequisites:** Detected shell route; policy `/healthz` ready on WSL `127.0.0.1:8000`.
- **Implementation tasks:** If SSH lands in WSL, curl loopback directly. If it lands in PowerShell or cmd.exe, invoke the detected distro through a safely quoted fixed wrapper and separately test Windows `127.0.0.1:<validated-port>`; inspect, but do not alter, WSL networking mode/portproxy/firewall exposure; prefer direct WSL SSH or built-in Windows→WSL localhost forwarding. If neither reaches a WSL-loopback server, stop with evidence and request approval before any relay, interface bind, portproxy, firewall, or networking change; a WSL IP alone is not a route to a loopback-bound server. Recheck LAN exposure after any approved remedy.
- **Files expected to change:** Remote command helpers/tests, `scripts/doctor_pc.sh`, `docs/TROUBLESHOOTING.md`.
- **Validation:** `/healthz` from WSL and from the SSH host namespace; listener inspection; negative closed-port case.
- **Acceptance:** Remote side of SSH can reach policy server without LAN/public binding; route and security boundary documented.
- **Planned commit:** `feat(ssh): validate Windows to WSL policy routing`.
- **Actual findings:** The real Windows cmd→explicit Ubuntu-24.04 WSL command route is proven. A Phase 3 candidate passed both WSL-native and Windows-native loopback health, and Windows exposed no non-loopback listener. The implementation revalidates the owned WSL listener and rejects every visible Windows listener address except `127.0.0.1` and `::1` without changing networking.
- **Remaining blockers:** None.
- **Completion status:** Complete on final implementation candidate `0c64187` for both profiles; see E-PC-TUNNEL.

No `netsh portproxy`, firewall, mirrored-network, or SSH-service change is authorized by this plan.
