# pi-robotics

`pi-robotics` is a personal robotics demo designed to pair an ALOHA simulation on an Apple Silicon Mac with remote OpenPI policy inference on an RTX 3090 PC through WSL2.

It is an independent integration project, not an official Physical Intelligence project or a commercial product.

## What it demonstrates

- Native `gym-aloha` simulation, offscreen rendering, and episode video on macOS.
- Reproducible OpenPI installation and CUDA validation in Ubuntu WSL2.
- Selectable π₀ and π₀.₅ server profiles for RTX inference.
- Private SSH orchestration with a loopback-only policy server.
- A planned end-to-end path from simulated observations to remote policy actions.

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
| `pi0_aloha_sim` | π₀ checkpoint fine-tuned for the ALOHA Transfer Cube simulator | Loads on RTX 3090; first JAX inference currently OOMs |
| `pi05_aloha_base` | π₀.₅ base checkpoint using ALOHA transforms | Selectable but not yet hardware-tested; not simulator-fine-tuned |

Both profiles use the same `(50, 14)` ALOHA action-chunk contract. Results are recorded separately because the checkpoints are not equivalent.

## Current status

- Phase 0: repository, plans, security gates, and CI complete.
- Phase 1: native Mac simulation and video validated.
- Phase 2: WSL, CUDA, locked OpenPI setup, loopback lifecycle, and clean shutdown validated; inference blocked by memory.
- Phases 3–6: secure connectivity, end-to-end control, reliability, and final hardening planned.

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

# Opt-in bounded conversion experiment; the stock converter remains the default.
OPENPI_POLICY_PROFILE=pi0_aloha_sim make convert-pc

OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make server
OPENPI_POLICY_PROFILE=pi0_aloha_sim OPENPI_POLICY_BACKEND=pytorch make smoke-policy
make stop
```

Only after π₀ returns finite RTX actions, repeat the conversion and PyTorch smoke with profile `pi05_aloha_base`. Ubuntu 22.04 is the upstream-supported target; this project also permits an explicitly selected Ubuntu 24.04 environment only after it passes the same locked dependency and GPU checks.

### Current hardware limit

The prepared PC has an RTX 3090 with 24 GB VRAM and 16 GB system RAM. The π₀ JAX server loads and becomes healthy, but every measured first request on this exact pinned WSL setup failed with CUDA OOM, including JAX's documented minimum-footprint allocator and the project-local masked-camera optimization. OpenPI's documented PyTorch path requires a one-time checkpoint conversion. This project adds an opt-in, bounded partial-BF16 conversion experiment that restores one stored checkpoint leaf at a time and writes standard sharded SafeTensors; it is not the default and has not yet passed hardware validation. `OPENPI_POLICY_BACKEND` defaults to `jax`; `pytorch` explicitly selects only the matching converted local checkpoint and must never fall back silently. If conversion fails, at least 32 GiB available RAM is an evidence-based practical target for the stock converter, not a guarantee; 64 GB total host RAM is preferred for Windows/WSL headroom. Conversion or checkpoint loading alone does not prove inference.

Never expose policy port 8000 publicly or install a Linux NVIDIA display driver inside WSL. WSL uses the Windows NVIDIA driver.

## Development checks

```bash
make test
make lint
make secret-scan
```

Runtime evidence, videos, machine identifiers, private addresses, credentials, model weights, and `.env` remain untracked.

## OpenPI source and license

This project is derived from [Physical Intelligence's OpenPI repository](https://github.com/Physical-Intelligence/openpi) at pinned commit [`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479).

Read the [original OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) for model documentation, upstream installation guidance, research context, and limitations.

The original Git history and [`LICENSE`](LICENSE) are retained. OpenPI is licensed under Apache License 2.0; no endorsement by Physical Intelligence is implied.
