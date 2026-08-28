# Security policy

## System and scope

This repository is an independent experimental integration derived from OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`: an ALOHA simulator and control loop run on a Mac while OpenPI inference runs inside WSL on an operator-owned GPU PC. The policy service, SSH forwarding, process ownership, action validation, generated evidence, and publication tooling are in scope.

## Threat model and trust boundaries

The operator and their two machines are trusted. Network peers, remote responses, repository contributions, paths, environment values, and copied evidence are untrusted until validated. The only network entry point is Windows OpenSSH on a trusted private network. The policy port must remain loopback-only inside WSL and on the Mac; SSH local forwarding is its security boundary. Host-key verification and a dedicated key authenticate that boundary.

Private SSH keys, account names, hostnames, private addresses, absolute user paths, model weights, checkpoints, videos, raw logs, and raw telemetry are sensitive assets even when they are not conventional credentials.

## Security invariants

- Never bind or forward the policy port beyond loopback, disable SSH host-key checking, weaken firewall or SSH ACLs, or signal an unverified process.
- Validate server/source/profile identity and observation/action schemas before applying actions. Timeouts, malformed data, non-finite actions, and identity changes fail closed.
- Keep secrets and machine-specific values in the operator's SSH configuration, ignored `.env`, or ignored private runtime/output directories.
- Never commit or attach weights, checkpoints, videos, raw logs, raw telemetry, runtime ownership records, or connection transcripts. Publish only newly constructed, allowlisted summaries.
- Preserve the upstream Git history, license files, submodule pins and URLs, derivative notice, and disabled upstream push URL.
- Run `make secret-scan` and `make public-audit` before publication. Scanner absence or an unverifiable repository boundary is a failure.

## Reporting vulnerabilities

Use GitHub private vulnerability reporting when available. Otherwise contact the repository owner through a verified private channel instead of opening a public issue. Identify the affected revision, realistic impact, reproduction conditions, and a minimally scoped remediation. Do not include live credentials, private connection values, raw evidence, or unnecessary exploit detail.

## Scope and limitations

Unchanged OpenPI and submodule code remains attributable to its upstream project, but vulnerabilities made reachable or worsened by this integration are reportable here. This demo assumes an operator-controlled trusted network and is not a hardened multi-user or internet-facing service. Do not test systems or hardware you do not own or have permission to use.
