# pi-robotics

`pi-robotics` is a personal robotics demo designed to pair an ALOHA simulation on an Apple Silicon Mac with remote OpenPI policy inference on an RTX 3090 PC through WSL2.

It is an independent integration project, not an official Physical Intelligence project or a commercial product.

## What it demonstrates

- Native `gym-aloha` simulation, offscreen rendering, and episode video on macOS.
- Reproducible OpenPI installation and CUDA validation in Ubuntu WSL2.
- Selectable π₀ and π₀.₅ server profiles for RTX inference.
- Private SSH orchestration with a loopback-only policy server.
- A latency-aware end-to-end path from simulated observations to remote policy actions.

```text
Mac: ALOHA simulation and video
              |
              | private SSH tunnel
              v
PC: Windows OpenSSH -> Ubuntu WSL2 -> RTX 3090 -> OpenPI
```

## Policy options

| Profile | Purpose | Status |
| --- | --- | --- |
| `pi0_aloha_sim` | π₀ checkpoint fine-tuned for the ALOHA Transfer Cube simulator | Converted BF16 PyTorch path returns finite RTX actions; JAX path OOMs |
| `pi05_aloha_base` | π₀.₅ base checkpoint using ALOHA transforms | Converted BF16 PyTorch path returns finite RTX actions; not simulator-fine-tuned |

Both profiles use the same `(50, 14)` ALOHA action-chunk contract. Results are recorded separately because the checkpoints are not equivalent.

## Current status

- Phase 0: repository, plans, security gates, and CI complete.
- Phase 1: native Mac simulation and video validated.
- Phase 2: WSL, CUDA, locked setup, bounded checkpoint conversion, both RTX inference profiles, loopback lifecycle, and clean shutdown validated.
- Phase 3: secure tunnel, WSL lifetime ownership, bounded client, and real two-profile tunneled inference validated.
- Phase 4: end-to-end buffered control and native Mac simulation validated; two-profile remote-policy episode acceptance is in progress.
- Phases 5–6: observability, reliability, and final hardening planned.

See [`PLANS/STATUS.md`](PLANS/STATUS.md) for the live execution cursor and [`PLANS/README.md`](PLANS/README.md) for the AI-readable implementation plans.

## Mac simulation

```bash
make setup-mac
make doctor-mac
make smoke-sim
```

The Mac installs only the simulator and lightweight client dependencies. Runtime model weights, JAX, and CUDA stay off the Mac; converted runtime weights normally reside on the PC.

## RTX 3090 inference

Create an ignored `.env` from [`.env.example`](.env.example), configure the private `robot-gpu` SSH alias, and select the WSL distro. Then run:

```bash
make doctor-pc
make setup-pc

# Auto-selects partial BF16 below 16 GiB available RAM.
OPENPI_POLICY_PROFILE=pi0_aloha_sim make convert-pc

# Starts the WSL server and its Mac loopback tunnel together.
OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make server
# Optional recheck of the already-running route and tunnel.
OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make tunnel
OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make smoke-policy
ALOHA_SEED=0 ALOHA_EPISODES=3 OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make run
make stop
```

`make run` requires an already-running, exact-candidate server/tunnel; it does not start or replace shared remote state. It runs one fresh seeded simulator and one fresh client per episode, records post-step video, and separates infrastructure success from the environment's `is_success`. Repeat the server, smoke, run, and stop sequence with `pi05_aloha_base`. Ubuntu 22.04 is the upstream-supported target; this project also permits an explicitly selected Ubuntu 24.04 environment only after it passes the same locked dependency and GPU checks.

### Memory-bounded recovery

The prepared PC has an RTX 3090 with 24 GB VRAM and 16 GB system RAM. The π₀ JAX server loads but every measured first request on this pinned WSL setup failed with CUDA OOM. The stock JAX→PyTorch converter also exceeds available WSL RAM. `make convert-pc` now selects the bounded partial-BF16 restore automatically when Linux `MemAvailable` is below 16 GiB; at or above the threshold it preserves the full-FP32 restore. Set `OPENPI_CONVERSION_RESTORE_MODE=full-float32` or `partial-bfloat16` only for an intentional override. The bounded path restores one stored leaf at a time, copies mapped tensors layer-by-layer into a GPU-resident model, and writes standard 1 GB SafeTensors shards. It passed on this PC for both profiles, including fresh loads and finite-action inference. `OPENPI_POLICY_BACKEND` therefore defaults to `pytorch` and selects only the matching converted local checkpoint without silent fallback. JAX remains an explicit diagnostic option outside the Phase 4 runner. The server disables optional PyTorch compile autotuning to avoid a first-call memory spike on this demo hardware.

On this PC, WSL stops background Linux processes after the final Windows-side WSL client exits. The project therefore keeps one synchronous WSL client inside the same owned SSH ControlMaster that provides the tunnel. No Windows service, scheduled task, global WSL setting, or public listener is added.

Never expose policy port 8000 publicly or install a Linux NVIDIA display driver inside WSL. WSL uses the Windows NVIDIA driver.

## Development checks

```bash
make test
make lint
make secret-scan
```

Runtime evidence, videos, machine identifiers, private addresses, credentials, model weights, and `.env` remain untracked.
See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for fail-closed recovery guidance.

## OpenPI source and license

This project is derived from [Physical Intelligence's OpenPI repository](https://github.com/Physical-Intelligence/openpi) at pinned commit [`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479).

Read the [original OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) for model documentation, upstream installation guidance, research context, and limitations.

The original Git history and [`LICENSE`](LICENSE) are retained. OpenPI is licensed under Apache License 2.0; no endorsement by Physical Intelligence is implied.
