# pi-robotics

`pi-robotics` is a personal, non-commercial robotics demo that runs an ALOHA Transfer Cube simulation on an Apple Silicon Mac and OpenPI inference on an RTX 3090 PC through WSL2. It is an independent integration project, not an official Physical Intelligence project and not endorsed by Physical Intelligence.

Public repository: [therealjaysun/pi-robotics](https://github.com/therealjaysun/pi-robotics)

```text
Apple Silicon Mac                         RTX 3090 PC
┌──────────────────────────┐             ┌────────────────────────────────┐
│ MuJoCo / gym-aloha       │             │ Windows OpenSSH                │
│ observation + 50 Hz loop │──SSH -L────▶│   └─ Ubuntu WSL2               │
│ action buffer + video    │  loopback   │      └─ OpenPI + CUDA :8000    │
│ JSONL + summaries/plots  │◀────────────│         + GPU sampler          │
└──────────────────────────┘             └────────────────────────────────┘
```

The policy port is loopback-only at every boundary. Model weights, CUDA, and the server stay on the PC; the Mac holds only simulator and lightweight client dependencies. See [Architecture](docs/ARCHITECTURE.md) for the contracts and lifecycle.

## Validated results

The complete workflow was validated on the exact Phase 5 candidate `2065dd9` with upstream pinned at `215abfb`. Both runs used the converted PyTorch backend and seeds 0–2:

| Profile | Intended use | Infrastructure | Task result | Total steps | Active rate (mean / p95) | Warm request p95 | Peak GPU memory |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `pi0_aloha_sim` (default) | π₀ fine-tuned for this simulator | 3/3 | 2/3 success | 761 | 46.91 / 47.02 Hz | 359.21 ms | 15,401 MiB |
| `pi05_aloha_base` | Experimental π₀.₅ base model with ALOHA transforms | 3/3 | 0/3 success | 900 | 48.13 / 48.38 Hz | 502.35 ms | 15,857 MiB |

GPU coverage passed for both runs (31 and 24 samples respectively), with no request retries or failures. The 1,661 successful simulator steps produced exactly 1,661 trajectory samples and six plots; each plot contains all 14 actual joints plus thinner dashed commands, normalized against pinned simulator limits. The π₀.₅ base checkpoint is not simulator-fine-tuned, so its zero task successes do not negate the separately reported infrastructure result. Neither profile met the complete cadence/underrun gate, so this project does **not** claim uninterrupted 50 Hz. The current execution cursor and complete evidence ledger are in [Project status](PLANS/STATUS.md).

The pre-trajectory hardening candidate `90b0fed` also passed exact-SHA WSL setup and a fresh four-call tunneled smoke for each profile, then verified cleanup and a free policy port. Phase 5 candidate `2065dd9` remains the newer hardware proof and the source of the full-episode figures above.

Both profiles return the same finite floating `(50, 14)` wire chunk. The default runner executes at most 30 actions from a chunk and starts one background prefetch with 25 actions remaining. It never clips the absolute joint commands, appends a late chunk, or repeats the last action on underrun.

## Prerequisites

Mac:

- Apple Silicon macOS in a native `arm64` shell.
- Git, GitHub CLI, [Gitleaks](https://github.com/gitleaks/gitleaks), and [`uv`](https://docs.astral.sh/uv/) on `PATH`.
- A logged-in desktop session for native MuJoCo rendering and enough free space for the repository, ignored videos, and evidence.
- `MUJOCO_GL` unset; Linux `egl` mode is rejected on macOS.

PC:

- Windows with WSL2, an x86-64 Ubuntu distro, and Windows OpenSSH Server using public-key authentication.
- RTX 3090 with 24 GiB VRAM and a current NVIDIA Windows driver that supports CUDA in WSL. Do not install a Linux NVIDIA display driver inside WSL.
- At least 40 GiB free on the WSL checkout and checkpoint-cache filesystems. The validated setup uses an explicitly selected `Ubuntu-24.04` distro; do not rely on the default when multiple WSL distros exist.
- The PC and Mac on the same trusted network, with the PC signed in, awake, and not exposing port 8000.

Inside the selected Ubuntu WSL Bash shell, install the doctor prerequisites and `uv` once. The project prepends `~/.local/bin` itself, so no manual PATH edit or PowerShell `source` command is needed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git iproute2 linux-libc-dev time util-linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### One-time SSH trust

Keep all real connection values outside the repository. Create a dedicated Mac key under `~/.ssh/`, install only its public half for the intended Windows account, and configure the private alias in `~/.ssh/config`:

In an Administrator PowerShell on the PC, confirm that OpenSSH Server is installed and make `sshd` persistent:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
# If its State is NotPresent:
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

```sshconfig
Host robot-gpu
    HostName <private-pc-address>
    User <windows-account-name>
    IdentityFile ~/.ssh/<dedicated-private-key>
    IdentitiesOnly yes
```

On Windows, enable OpenSSH Server and place the public key in the account's configured `AuthorizedKeysFile`; administrative accounts commonly use `%ProgramData%\ssh\administrators_authorized_keys`. Preserve the Windows ACLs required by `sshd`. Before accepting the Mac's first connection, compare its displayed host-key fingerprint with the fingerprint read locally from the PC. Never use `StrictHostKeyChecking=no`, commit the address/user/key, or paste private SSH output into an issue.

Confirm only the alias and batch authentication, without printing its expansion:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes robot-gpu exit
```

## First setup

Start on the Mac. During stacked review, use the current published phase branch named in [Project status](PLANS/STATUS.md); hardware commands deliberately reject another branch, a dirty checkout, an unpushed SHA, or a candidate without a fresh secret-scan receipt.

```bash
git clone https://github.com/therealjaysun/pi-robotics.git
cd pi-robotics
git remote add upstream https://github.com/Physical-Intelligence/openpi.git
git remote set-url --push upstream DISABLED
git fetch --no-tags upstream main:refs/remotes/upstream/main
# While the stack is open, use the exact active branch from PLANS/STATUS.md.
git switch --track origin/codex/06-hardening-docs
cp .env.example .env
# Edit only .env: select Ubuntu-24.04 and any intentional non-secret overrides.

make setup-mac
make doctor
make ci
make secret-scan
```

`make setup-mac` creates an ignored native Python 3.10 environment and applies the narrowly scoped macOS OpenGL/FFmpeg compatibility fixes only when needed. `make doctor` validates the Mac plus repository/remotes/GitHub prerequisites without contacting the PC.

Then switch on the PC, sign in, start the selected WSL distro, and confirm Windows OpenSSH is running. Back on the Mac, verify the private alias and run the PC gates:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes robot-gpu exit
OPENPI_WSL_DISTRO=Ubuntu-24.04 make doctor-pc
OPENPI_WSL_DISTRO=Ubuntu-24.04 make setup-pc
```

`make doctor-pc` discovers WSL, architecture, disk, RAM, NVIDIA/CUDA visibility, ports, and tools before setup changes the WSL project environment. It must identify an RTX 3090; CPU fallback is rejected. The PC must remain awake only for conversion, policy smoke tests, and full runs; after `make stop` succeeds, it is safe to switch off.

Converted weights are PC-local and ignored. Convert each desired profile once, or again after its artifact is removed or the source/runtime pin changes:

```bash
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=pi0_aloha_sim make convert-pc
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE=pi05_aloha_base make convert-pc
```

Conversion mode `auto` uses the one-layer-at-a-time partial-BF16 restore when Linux `MemAvailable` is below 16 GiB; otherwise it preserves the full-FP32 restore. The bounded path passed for both profiles on the prepared 16 GB PC. The original JAX server remains an explicit diagnostic option, but its first inference repeatedly exhausted GPU memory on this machine. The demo therefore defaults to the matching converted PyTorch checkpoint and never falls back silently.

## Run modes

Simulation only—no PC required:

```bash
make smoke-sim
```

Policy only—starts the owned WSL server and Mac tunnel, verifies one bounded client workload, then stops server-first:

```bash
profile=pi0_aloha_sim
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make server
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make tunnel
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make smoke-policy
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make stop
```

Complete system—keep both machines on and run the same profile in every command:

```bash
profile=pi0_aloha_sim
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make server
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make smoke-policy
ALOHA_SEED=0 ALOHA_EPISODES=3 OPENPI_WSL_DISTRO=Ubuntu-24.04 \
  OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make run
OPENPI_WSL_DISTRO=Ubuntu-24.04 OPENPI_POLICY_PROFILE="$profile" OPENPI_POLICY_BACKEND=pytorch make stop
```

Repeat after `make stop` with `profile=pi05_aloha_base`. `make server` owns the server, synchronous WSL lifetime holder, and loopback SSH tunnel together; `make tunnel` only revalidates them. Do not close the tunnel before the server. `make stop` is idempotent and refuses to signal an unverified PID.

One automatic, low-frequency RTX sampler covers the whole `make run`. It validates the owned server identity before every bounded `nvidia-smi` query and stops in `finally`; there is no SSH or GPU query per simulation step. Start/end clock probes record the Mac↔WSL UTC offset with a round-trip uncertainty bound. `make metrics` only validates the latest run for the selected `OPENPI_POLICY_PROFILE` and atomically rebuilds derived summaries—it does not contact the PC, change raw evidence, or start a sampler.

Initial client setup retries are also bounded: the default is two retries with a fixed two-second backoff, only for connection/timeout/EOF/WebSocket-close failures while connecting and validating metadata before the simulator resets. Once any inference may have been sent, a timeout or disconnect aborts the episode, closes the transport, discards the buffer, and preserves partial evidence; it is never replayed automatically. Invalid schemas, non-finite actions, and application errors also fail immediately. No stale action is applied.

Connect, metadata receive, inference receive, and close have explicit deadlines. The pinned synchronous WebSocket library has no separate deadline for a backpressured `send`; observations are bounded below 1 MiB, and interruption/close is the recovery boundary. Do not claim every client I/O operation is independently time-bounded.

## Outputs and checks

`RUN_OUTPUT_DIR` defaults to ignored `outputs/`; an in-repository override must also be Git-ignored, while an external override must be absolute. Raw evidence can contain machine details and must not be committed.

- `outputs/phase01/<UTC>/`: simulator smoke manifest and the recorded seed's video/reset frame.
- `outputs/phase02/` and `outputs/phase03/`: ignored setup, conversion, server, route, and policy-smoke evidence.
- `outputs/phase04/<UTC>/<profile>/`: validated Phase 4 summary, per-seed manifest, and video.
- `outputs/phase05/<UTC>/<profile>/`: `summary.json`, `performance-summary.json`, `performance-summary.md`, `gpu-metrics.jsonl`, `clock-correlation.json`, and capped `server-tail.log`; each `seed-N/` contains `manifest.json`, `telemetry.jsonl`, `telemetry-summary.json`, `telemetry-summary.md`, `joint-trajectory.png`, and `episode.mp4`.
- `.runtime/`: private ownership records, locks, control socket, scan receipt, and remote sampler/server state.

JSONL is line-buffered without per-step `fsync`; every successful step records its exact step number, monotonic elapsed time, and finite 14-value actual/commanded vectors. A damaged final fragment can be discarded while earlier complete events remain readable. After each complete or partial episode, the runner atomically plots all 14 joints against fixed gym-aloha 0.1.1 ranges. Raw rows, plot paths, and images stay private; publishable summaries expose only counts, coverage, status, and safe local IDs.

```bash
make test
make lint
make secret-scan
make public-audit
make pr-status
# Select the profile whose latest run should be checked:
OPENPI_POLICY_PROFILE=pi0_aloha_sim make metrics
git status --short
```

See [Troubleshooting](docs/TROUBLESHOOTING.md) for fail-closed recovery. Do not delete checkpoint caches, change Windows networking/firewall/driver state, or kill broad process names without first proving the exact target and obtaining approval for destructive recovery.

## Review, source, and license

Development is a seven-PR stack. Review and merge it in numerical order, using merge commits while descendants remain stacked; after each merge retarget and verify the next incremental diff. Do not enable auto-merge or automatic branch deletion. See [PR stack](PLANS/PR_STACK.md) and [Review and merge](PLANS/REVIEW_AND_MERGE.md).

This project is derived from [Physical Intelligence's OpenPI repository](https://github.com/Physical-Intelligence/openpi) at pinned commit [`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479). Read the [original OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) for model documentation, upstream setup, research context, and limitations.

The original Git history, [`LICENSE`](LICENSE), and other upstream attribution are retained. Project additions include Mac/WSL setup, strict profile/config and data contracts, memory-bounded BF16 conversion, loopback server metadata and health checks, finite receive/close deadlines, SSH/WSL process ownership, buffered seeded control, atomic evidence/video/trajectory validation, bounded retries, and local/GPU telemetry. The project does not train or redistribute weights and does not imply upstream endorsement.

Security summary: secrets and machine identities stay in the Mac SSH config, ignored `.env`, or ignored raw evidence; weights/caches/videos/logs/telemetry are ignored; `origin` is the user repository and official `upstream` has push disabled; CI dependencies are pinned and permissions restricted; secret scanning fails closed before a remote candidate is accepted.
