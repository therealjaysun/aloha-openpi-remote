# 03.02 — Windows-to-WSL routing

- **Objective:** Verify the SSH server's network namespace can reach the WSL loopback policy port.
- **Inputs/prerequisites:** Detected shell route; policy `/healthz` ready on WSL `127.0.0.1:8000`.
- **Implementation tasks:** If SSH lands in WSL, curl loopback directly. If it lands in PowerShell or cmd.exe, invoke the detected distro through a safely quoted fixed wrapper and separately test Windows `127.0.0.1:<validated-port>`; inspect, but do not alter, WSL networking mode/portproxy/firewall exposure; prefer direct WSL SSH or built-in Windows→WSL localhost forwarding. If neither reaches a WSL-loopback server, stop with evidence and request approval before any relay, interface bind, portproxy, firewall, or networking change; a WSL IP alone is not a route to a loopback-bound server. Recheck LAN exposure after any approved remedy.
- **Files expected to change:** Remote command helpers/tests, `scripts/doctor_pc.sh`, `docs/TROUBLESHOOTING.md`.
- **Validation:** `/healthz` from WSL and from the SSH host namespace; listener inspection; negative closed-port case.
- **Acceptance:** Remote side of SSH can reach policy server without LAN/public binding; route and security boundary documented.
- **Planned commit:** `feat(ssh): validate Windows to WSL policy routing`.
- **Actual findings:** The real Windows cmd→explicit Ubuntu-24.04 WSL command route is proven, and the policy server passed health checks on WSL loopback. Windows-host loopback and Mac local-forward tests remain Phase 03 work.
- **Remaining blockers:** A policy must return valid actions before tunneled acceptance can run; see E-PC-JAX.
- **Completion status:** Route discovery complete; tunnel path planned.

No `netsh portproxy`, firewall, mirrored-network, or SSH-service change is authorized by this plan.
