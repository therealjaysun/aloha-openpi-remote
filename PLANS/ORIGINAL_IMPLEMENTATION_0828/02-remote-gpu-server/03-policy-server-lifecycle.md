# 02.03 — Policy server lifecycle

- **Objective:** Start exactly one selected policy server on loopback, wait for readiness, preserve logs, and stop only that process.
- **Inputs/prerequisites:** Installed OpenPI; free remote port; valid `OPENPI_POLICY_PROFILE`.
- **Implementation tasks:** Add server `--host` with upstream-compatible default, pass `127.0.0.1`, and replace hostname/IP discovery logging with the chosen bind host/port; map the two fixed profile names via a shell `case` (never `eval` arbitrary config); augment handshake metadata with allowlisted `policy_profile`, `config_name`, public checkpoint label, action horizon/dimension, and source SHA; fail closed to the validated JAX GPU platform and record server-process device evidence; launch with `exec` and an atomic ownership record containing PID/start identity/command/profile/port/SHA; reject duplicate/stale/mismatched records; wait with bounded retries on `/healthz`; show checkpoint progress and ignored raw log location; stop only after identity validation; trap partial startup; never automatically delete a partial cache.
- **Files expected to change:** `scripts/serve_policy.py`, its focused test, `scripts/start_policy_server.sh`, `scripts/stop_policy_server.sh`, `.env.example`, `.gitignore`, `Makefile`.
- **Validation:** Start/health/duplicate/stop/stale-or-reused-PID tests for both profiles; verify listener is `127.0.0.1:8000`; process survives setup SSH exit; metadata/logs identify profile/config/checkpoint/SHA and prove either JAX process-level GPU attribution or the PyTorch model's actual CUDA device without private paths or host/IP discovery.
- **Acceptance:** Safe idempotent lifecycle; bounded startup; loopback bind; profile reported; no CPU inference; stop cannot kill an unrelated PID.
- **Planned commits:** `fix(server): allow loopback policy binding`; `feat(remote): manage selectable policy profiles`.
- **Actual findings:** The upstream-compatible default remains `0.0.0.0`, while the wrapper requires literal `127.0.0.1`. `--env=ALOHA_SIM` selects task-specific π₀; `--env=ALOHA` selects `pi05_aloha` with `pi05_base`, which provides the native π₀.₅ option but is not sim-task-fine-tuned. Start/check/stop share a user-state lifecycle lock; the background child closes the lock descriptor; an atomic mode-600 record binds PID start identity, command hash, profile, port, source SHA, and log; Linux PIDfd signaling closes the PID-reuse race. A second SSH session revalidated record, listener owner, loopback bind, and health while WSL remained active. Phase 03 later found that this host tears down WSL after the final Windows-side WSL client exits, so idle-host persistence belongs to the Phase 03 tunnel holder rather than the detached Linux server alone. JAX inference OOM persisted at 75%, 90%, and 95% preallocation and with the documented on-demand and minimum-footprint allocators. The one-camera smoke input was still encoding two generated black views whose masks were false; the wrapper now removes only fully masked views before inference and records that mode in handshake metadata, preserving valid-image inputs and mask positions while reducing the compiled prefix.
- **Remaining blockers:** None. The bounded PyTorch path passed lifecycle and finite-action inference for both profiles; the π₀.₅ profile remains an infrastructure/demo option rather than a simulator-task-quality claim.
- **Completion status:** Complete for both profiles, including post-inference second-session survival and safe cleanup; see E-PC-BF16.

Representative commands after verification:

```bash
uv run scripts/serve_policy.py --host=127.0.0.1 --env=ALOHA_SIM
uv run scripts/serve_policy.py --host=127.0.0.1 --env=ALOHA --default-prompt="Transfer cube"
```
