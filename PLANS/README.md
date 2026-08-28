# Implementation plans

This directory is the execution contract for the Mac ALOHA simulator and remote RTX 3090 OpenPI inference project. Plans are intentionally short and numbered so an AI agent can load `STATUS.md`, one phase, and only the referenced requirement rows without consuming the whole repository context.

## Reading order

1. Read [`STATUS.md`](STATUS.md) for live state and blockers.
2. Read [`EXECUTION_LOGISTICS.md`](EXECUTION_LOGISTICS.md) for Mac/PC power and handoff gates.
3. Read [`PR_STACK.md`](PR_STACK.md) for branch ancestry.
4. Read the active phase's `00-overview.md` and subphase file.
5. Load only that phase's rows from [`REQUIREMENTS.md`](REQUIREMENTS.md); phase 06 loads the whole matrix.
6. Update `Actual findings`, `Remaining blockers`, status, evidence, commit SHA, and PR URL immediately after work.

## Global execution rules

- Audit pin: OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; re-audit before intentionally upgrading.
- Keep model inference in WSL2 on the RTX 3090. The Mac installs only simulation and `openpi-client` dependencies.
- Never commit credentials, private addresses, OS/SSH usernames or hostnames, absolute home paths, weights, videos, raw logs/telemetry, or `.env`. A verified public GitHub owner/repository URL is allowed.
- Use `robot-gpu` and other placeholders in tracked examples. Discover real values at runtime.
- Bind the server to loopback and reach it through `127.0.0.1:8000` over SSH forwarding.
- Prefer existing OpenPI ALOHA environment, transforms, WebSocket client/server, runtime interfaces, and video saver. Add code only for requirements they do not meet.
- Hardware claims require captured command evidence. `code complete`, `locally tested`, `remotely tested`, and `hardware validated` are separate states.
- A child branch may begin when its parent is locally code-complete/tested **or** externally blocked with evidence and an exact recovery command. Keep hardware-blocked PRs draft; never wait for a parent PR to merge.
- Before any PC test, create a secret-scanned remote-test candidate commit and make that exact SHA available to WSL. Verify Mac SHA = WSL SHA; never test uncommitted or mismatched source.
- Keep the PC off through phases 00–01 and phase 02's local staging. At the documented P1 gate, notify the user before PC-dependent validation, then operate WSL remotely from the Mac whenever possible.
- Do not merge PRs or enable auto-merge.

## Phases

| Phase | Plan | Outcome |
| --- | --- | --- |
| 00 | [`00-bootstrap/00-overview.md`](00-bootstrap/00-overview.md) | Audited baseline, public fork, governance, CI |
| 01 | [`01-mac-simulation/00-overview.md`](01-mac-simulation/00-overview.md) | Native Mac ALOHA simulation and video |
| 02 | [`02-remote-gpu-server/00-overview.md`](02-remote-gpu-server/00-overview.md) | WSL OpenPI server on RTX 3090 |
| 03 | [`03-secure-connectivity/00-overview.md`](03-secure-connectivity/00-overview.md) | Validated loopback-only SSH tunnel |
| 04 | [`04-end-to-end-control/00-overview.md`](04-end-to-end-control/00-overview.md) | Policy-controlled episodes with measured cadence |
| 05 | [`05-observability-and-reliability/00-overview.md`](05-observability-and-reliability/00-overview.md) | JSONL telemetry, GPU metrics, recovery |
| 06 | [`06-hardening-and-documentation/00-overview.md`](06-hardening-and-documentation/00-overview.md) | Tests, security audit, docs, release evidence |

## Deferred extensions

| Suite | Plan | Status | Relationship to completed stack |
| --- | --- | --- | --- |
| S0827 Push-π | [`SCENARIOS_0827/00-overview.md`](SCENARIOS_0827/00-overview.md) | Reviewed; ready after branch gate; not implemented | Custom PushT-inspired ALOHA experiment. Prefer one new PR from post-merge `main`; it is not an eighth member of the completed seven-PR phase stack. |

Deferred suites own their pending requirements in `REQUIREMENTS.md`. They must preserve the stable interfaces above, pass the same privacy/security gates, and update `STATUS.md` when scheduled; their pending rows do not weaken or reopen completed phase evidence.

## Stable implementation interface

Do not rename these targets without updating every plan and user document:

```text
make doctor doctor-mac doctor-pc setup-mac setup-pc
make convert-pc server tunnel smoke-sim smoke-policy run metrics stop
make test lint secret-scan public-audit pr-status
```

Configuration stays in environment variables loaded from an ignored `.env`; the complete keys, defaults, and validation rules are in the `Configuration contract` table in [`REQUIREMENTS.md`](REQUIREMENTS.md). Reject non-loopback policy hosts and use a fixed profile-to-config/checkpoint `case` statement rather than executing configuration strings.

## Verified primary references

- [OpenPI at the audited commit](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)
- [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/remote_inference.md)
- [OpenPI ALOHA Sim example](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim)
- [gym-aloha](https://github.com/huggingface/gym-aloha)
- [dm_control rendering and macOS notes](https://github.com/google-deepmind/dm_control#rendering)
- [WSL networking](https://learn.microsoft.com/windows/wsl/networking)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create)
- [uv environments](https://docs.astral.sh/uv/pip/environments/)
