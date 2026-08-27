# 02.04 — GPU validation

- **Objective:** Prove each selected policy performs real RTX 3090 inference rather than merely starting.
- **Inputs/prerequisites:** Ready server; WSL-local OpenPI client; verified available outer process-deadline tool (normally GNU `timeout`); GPU metrics access.
- **Implementation tasks:** Capture sanitized `nvidia-smi` baseline; run every pre-phase-03 WSL-local smoke invocation under a validated finite outer process deadline with TERM then bounded KILL fallback and record timeout exits, so the stock client cannot hang phase 02; time request 1 as cold, then issue two warmups and measured warmed inferences per profile; validate handshake identity, JAX device, response dtype, and finite `(50,14)` actions; capture process GPU memory/utilization before/during; record cold/warm latency and checkpoint/profile/SHA; detect CPU fallback and fail. Phase 03 replaces this coarse guard with per-stage client deadlines/close.
- **Files expected to change:** `scripts/doctor_pc.sh`, `scripts/start_policy_server.sh`, `scripts/collect_gpu_metrics.sh`, evidence references in plans/docs (raw output ignored).
- **Validation:** Profile-specific smoke inference; absent/hung-server outer-timeout exit with no surviving client; `nvidia-smi`/JAX device evidence; server timing response.
- **Acceptance:** Both profiles allocate RTX 3090 memory and return valid actions; π₀.₅ task success is not inferred from shape validity; evidence is sanitized.
- **Planned commit:** `test(remote): verify RTX policy inference`.
- **Actual findings:** None; hardware unavailable.
- **Remaining blockers:** Phases 02.01–03 and real PC access. Mac-through-tunnel validation belongs to phase 03 and is not a phase 02 dependency.
- **Completion status:** Blocked pending hardware.
