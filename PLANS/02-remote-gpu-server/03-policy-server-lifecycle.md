# 02.03 — Policy server lifecycle

- **Objective:** Start exactly one selected policy server on loopback, wait for readiness, preserve logs, and stop only that process.
- **Inputs/prerequisites:** Installed OpenPI; free remote port; valid `OPENPI_POLICY_PROFILE`.
- **Implementation tasks:** Add server `--host` with upstream-compatible default, pass `127.0.0.1`, and replace hostname/IP discovery logging with the chosen bind host/port; map the two fixed profile names via a shell `case` (never `eval` arbitrary config); augment handshake metadata with allowlisted `policy_profile`, `config_name`, public checkpoint label, action horizon/dimension, and source SHA; fail closed to the validated JAX GPU platform and record server-process device evidence; launch with `exec` and an atomic ownership record containing PID/start identity/command/profile/port/SHA; reject duplicate/stale/mismatched records; wait with bounded retries on `/healthz`; show checkpoint progress and ignored raw log location; stop only after identity validation; trap partial startup; never automatically delete a partial cache.
- **Files expected to change:** `scripts/serve_policy.py`, its focused test, `scripts/start_policy_server.sh`, `scripts/stop_policy_server.sh`, `.env.example`, `.gitignore`, `Makefile`.
- **Validation:** Start/health/duplicate/stop/stale-or-reused-PID tests for both profiles; verify listener is `127.0.0.1:8000`; process survives setup SSH exit; metadata/logs identify profile/config/checkpoint/SHA and GPU without private paths or host/IP discovery.
- **Acceptance:** Safe idempotent lifecycle; bounded startup; loopback bind; profile reported; no CPU inference; stop cannot kill an unrelated PID.
- **Planned commits:** `fix(server): allow loopback policy binding`; `feat(remote): manage selectable policy profiles`.
- **Actual findings:** Current server hard-codes `0.0.0.0` and has `/healthz`. `--env=ALOHA_SIM` selects task-specific π₀; `--env=ALOHA` selects `pi05_aloha` with `pi05_base`, which provides the native π₀.₅ option but is not sim-task-fine-tuned.
- **Remaining blockers:** Remote runtime and exact tyro help must be exercised after install.
- **Completion status:** Planned.

Representative commands after verification:

```bash
uv run scripts/serve_policy.py --host=127.0.0.1 --env=ALOHA_SIM
uv run scripts/serve_policy.py --host=127.0.0.1 --env=ALOHA --default-prompt="Transfer cube"
```
