# Troubleshooting

Start with the exact recovery printed by the failed command. Raw Mac, SSH, Windows, WSL, server, sampler, and model output belongs only in ignored evidence; do not paste hostnames, addresses, usernames, absolute home paths, keys, or raw logs into tracked files or public issues.

## Mac setup or rendering fails

Confirm a native Apple Silicon shell, `uv` on `PATH`, and no Linux rendering override:

```bash
uname -m
command -v uv
unset MUJOCO_GL
make setup-mac
make doctor-mac
```

The expected architecture is `arm64`. Setup owns only `examples/aloha_sim/.venv` and applies the pinned MuJoCo OpenGL-framework path or Mac-only FFmpeg override when the platform needs it. If an existing environment is not native Python 3.10, move that ignored directory aside rather than altering the system Python. Run rendering from the logged-in desktop session; SSH/headless macOS sessions may not have a usable CGL context.

Do not move simulation to the PC as a silent fallback. If `make doctor-mac` still fails, preserve its sanitized package/version/error evidence and follow the native-engine escalation in the phase plan.

## SSH alias or public-key authentication fails

The alias must be exactly one safe host name such as `robot-gpu`; connection details stay in `~/.ssh/config`, not `.env`. Check without expanding or printing the private configuration:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes robot-gpu exit
```

If the address is in doubt, run `ipconfig` locally on the PC to find the active adapter's IPv4 address. On the Mac, the following prints the address currently configured for the alias:

```bash
ssh -G robot-gpu | awk '$1 == "hostname" { print $2; exit }'
```

Treat both outputs as private and do not paste them into tracked files or public issues.

If the host key is new or changed, stop. Read the current host-key fingerprint locally at the PC, compare it independently, and update trust only when the change is expected. Never disable strict host-key checking.

On Windows, confirm OpenSSH Server is running, `PubkeyAuthentication yes` is effective, and the key is in the account's configured `AuthorizedKeysFile`. Administrative accounts commonly use `%ProgramData%\ssh\administrators_authorized_keys`, whose Windows ACL must remain acceptable to `sshd`. A Windows account name containing spaces belongs quoted in the private SSH config/command; it does not belong in the repository.

## WSL distro selection is ambiguous

The project never guesses when multiple Ubuntu distros exist. List them on Windows, choose the intended Ubuntu instance, and set only its display name in ignored `.env` or the command:

```powershell
wsl.exe --list --verbose
```

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 make doctor-pc
```

The validated project setup uses `Ubuntu-24.04` after the complete doctor and locked setup pass. Do not delete an accidental distro through this project; manage it separately after confirming it has no needed data.

## WSL does not see the configured PC memory

On the 48 GB demo PC, merge this into `%UserProfile%\.wslconfig` on Windows; do not commit that global host file:

```ini
[wsl2]
memory=32GB
```

Stop owned project services and any other WSL/Docker work first, then run `wsl.exe --shutdown`, relaunch `Ubuntu-24.04`, and verify with `free -h` plus `make doctor-pc`. The setting is a ceiling shared by all WSL2 distros, not a 32 GiB reservation; actual `MemAvailable` is lower. Preserve unrelated `.wslconfig` entries. More host RAM does not increase RTX VRAM, and existing converted artifacts do not need reconversion.

## RTX 3090 or CUDA is not visible in WSL

Run the bounded doctor from the Mac:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 make doctor-pc
```

If it cannot see an RTX 3090, stop before setup or inference. At the PC, verify `nvidia-smi` in the selected WSL distro and update the NVIDIA **Windows** driver if required. Do not install a Linux NVIDIA display driver in WSL and do not accept CPU fallback. Driver, WSL, or Windows-admin changes require explicit approval.

## Candidate is dirty, unpushed, on the wrong branch, or unscanned

Remote orchestration is deliberately stricter than local simulation. Inspect, preserve, and commit only intended source changes; never hide weights/logs/secrets merely to satisfy the gate:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse "origin/$(git branch --show-current)"
make ci
make secret-scan
```

The working tree must then be clean, local HEAD must equal the active pushed phase branch, and `.runtime/secret-scan.sha` must name that exact HEAD. Generated evidence belongs under ignored `outputs/`; a custom in-repository `RUN_OUTPUT_DIR` must itself be ignored.

## PC setup reports disk, port, or tool failure

Do not bypass the doctor. Free space only from a precisely identified safe target, or choose a different approved WSL path. The project requires at least the configured `OPENPI_MIN_FREE_GIB` on both checkout and checkpoint-cache filesystems. Install the specifically named missing WSL tool, then rerun:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 make doctor-pc
OPENPI_WSL_DISTRO=Ubuntu-24.04 make setup-pc
```

If port 8000 is occupied, stop its legitimate owner or select a matching unused loopback port in ignored `.env`. The project never kills an unrecorded listener.

## Conversion is OOM-killed or the source checkpoint is absent

Use the measured bounded path on a low-RAM PC:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> \
  OPENPI_CONVERSION_RESTORE_MODE=partial-bfloat16 make convert-pc
```

The default `auto` already selects this below 16 GiB Linux `MemAvailable`; do not force full FP32 there. A failed conversion must not publish a partial artifact. If the source JAX checkpoint is absent, allow the selected, pinned profile to populate its cache through the normal bounded server setup, stop it, and retry conversion. Do not delete or redownload a partial/corrupt cache until its exact private path and error have been recorded and targeted deletion is explicitly approved.

The original converter is estimated to need a host with roughly 32 GiB available RAM in this project, but that is an evidence-based practical fallback, not an upstream minimum.

## JAX loads but inference OOMs

This is the measured result on the prepared RTX 3090, not a reason to add undocumented allocator flags. If the matching converted artifact already exists, start PyTorch directly. Run conversion only when that artifact is absent or the source/runtime pin changed:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> make convert-pc
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> \
  OPENPI_POLICY_BACKEND=pytorch make server
```

JAX remains an explicit diagnostic option; the Phase 4+ demo runner rejects it on this validated 24 GiB setup. There is no silent backend fallback.

## WSL server exits when a Windows-side WSL command ends

Always use `make server`. It starts the server and makes the same owned SSH ControlMaster/tunnel hold one synchronous `wsl.exe` command open for exactly the server lifetime. Do not create a detached WSL process, service, scheduled task, `.wslconfig` lifetime workaround, mirrored-network change, or public relay.

If startup fails, run `make stop`. When ownership is uncertain, cleanup intentionally retains the holder and exits nonzero rather than tearing down WSL around an unverified process.

## Windows cannot reach WSL loopback

Keep the WSL server on `127.0.0.1`. Do not add `netsh portproxy`, a firewall exception, wildcard bind, scheduled relay, or WSL networking change. Preserve the bounded route failure and request approval before the smallest Windows networking mutation. A passing route must show only Windows loopback listeners.

## Mac tunnel port is occupied

Stop the legitimate owner of `LOCAL_POLICY_PORT` or choose another unused local port in ignored `.env`; keep both local and remote service hosts at literal `127.0.0.1`. The tunnel refuses an occupied port and never signals an unrecorded process.

## Tunnel or process ownership state is stale

Run the one coordinated stop:

```bash
make stop
```

A valid stale record is removed only after the process/listener is proved absent. A changed PID start identity, command signature, profile, SHA, malformed record, unknown socket, or non-socket control path fails closed. Do not use `pkill`, `killall`, a PID copied from old output, `git reset --hard`, or manual deletion of ownership files. Inspect first and preserve recoverable state.

Stop order matters. During `make run`, its `finally` path stops and verifies the run-owned GPU sampler and copies its evidence before returning. After the run has completed or failed, `make stop` stops the verified remote server first and releases the WSL holder/tunnel second. If either cleanup path is nonzero, the PC is not yet declared safe to power off.

## Server starts slowly or a policy connection times out

Check the owned components in order:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> \
  OPENPI_POLICY_BACKEND=pytorch make tunnel
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=<profile> \
  OPENPI_POLICY_BACKEND=pytorch make smoke-policy
```

These verify the WSL server, Windows loopback route, Mac listener, exact profile/backend/checkpoint/SHA, and finite inference. Increase startup/connect/metadata/inference/close timeouts only from measured evidence; never set an infinite wait.

Before simulator reset, connection, timeout, EOF, and WebSocket-close errors during initial client construction and metadata validation receive bounded retries. The default is two attempts after the original connection with a fixed two-second backoff; each creates and identity-checks a fresh client. Once inference may have been sent, a timeout or disconnect aborts the episode, closes the transport, discards buffered state, and preserves partial evidence rather than replaying an uncertain request. A schema, NaN, shape, model, or application error is never retried.

## Buffer underruns or control rate below 50 Hz

This does not imply that simulation or policy inference failed. Inspect the per-seed buffer waits, warmed request p95, active step rate, and `uninterrupted_50hz_claimed` field. The validated Phase 5 runs averaged 46.91 Hz for π₀ and 48.13 Hz for π₀.₅; neither passed the complete cadence/underrun gate, so they make no uninterrupted claim.

Tune only from Mac-through-tunnel measurements and preserve:

```text
1 <= ALOHA_PREFETCH_STEPS < ALOHA_ACTION_HORIZON <= 50
warmed p95 + explicit margin < prefetch steps × 20 ms
```

If the gate still fails at a justified larger threshold/horizon, report the waits and measured rate. Do not repeat the last action, append a late chunk, lower the margin to manufacture a pass, or call a WSL-local timing the end-to-end p95.

## GPU telemetry is missing, duplicated, or stops early

The runner owns one sampler for the complete run; users should not start `scripts/collect_gpu_metrics.sh` manually. It refuses a duplicate run/profile lock and stops if the server PID/start identity/profile/SHA changes or a bounded `nvidia-smi` query fails. The terminal sampler event records pass, interruption, or failure.

After the run, use the read-only aggregator:

```bash
OPENPI_POLICY_PROFILE=<profile> make metrics
```

This command contacts neither the PC nor the sampler. It validates the selected profile's latest local Phase 5 evidence and atomically rebuilds only the derived summaries.

If sampling failed, retain the private Phase 5 JSONL and server tail, run `make stop`, fix the exact server/driver/identity issue, and repeat the run. Do not SSH or call `nvidia-smi` once per control step. A sampler failure must remain distinct from task success.

## Telemetry or a summary is partial

Per-seed JSONL is line-buffered. The parser accepts every complete valid line and ignores only an incomplete final fragment; malformed complete/interior lines are errors. Terminal and summary files are written in `finally` and published atomically, so a failed or interrupted run should remain explicitly partial rather than disappear. If the telemetry writer cannot close cleanly, the manifest records `writer_closed=false` and no derived per-seed telemetry summary is published; treat the raw terminal line as unverified and use the manifest as authoritative.

Check the ignored run without copying it into the repository:

```bash
OPENPI_POLICY_PROFILE=<profile> make metrics
git status --short
```

NumPy scalars are normalized, but arrays, unsupported deep structures, NaN, and infinity are rejected. Fix the producer instead of weakening JSON. Raw evidence may include private fields; publish only the separately generated allowlisted summary, never a hand-edited raw log.

## Joint trajectory is missing or invalid

Each seed with at least one valid step row should have an ignored `joint-trajectory.png`. Its manifest must report 14 joints, sample count equal to successfully applied steps, exact 1.0 coverage, and `plot_status=passed`. Interrupted runs still plot valid partial rows; a zero-step run honestly reports no samples.

If plotting fails, retain the JSONL and manifest, then run the focused trajectory tests and fix the exact malformed row or local Matplotlib error. Do not normalize from observed extrema, hand-edit telemetry, publish raw paths, or rerun GPU inference merely to redraw a plot; plotting is local and post-episode.

## Video encoding or validation fails

The manifest records video failure separately and still preserves the episode result. A passing artifact must reopen with the exact applied-step frame count, 50 fps metadata, and 224×224 RGB frames. Rerun `make doctor-mac` to verify FFmpeg and native rendering. Do not mark infrastructure passed from file existence alone.

## GitHub authentication, CI, or stacked PR review fails

Reauthenticate without printing tokens:

```bash
gh auth login -h github.com
gh auth status
gh repo view therealjaysun/pi-robotics
```

Run `make ci`, `make secret-scan`, `make public-audit`, and `make pr-status` locally. The publication audit checks both current files and project-added history; if it reports a real secret, stop publication and rotate it before rewriting unpublished history. Do not silence a finding or edit generated evidence into a pass.

Review the seven PRs in numerical order from [PR stack](../PLANS/PR_STACK.md), using [Review and merge](../PLANS/REVIEW_AND_MERGE.md). A child diff that contains earlier phases must be retargeted/repaired before review; do not merge out of order or enable auto-merge.
