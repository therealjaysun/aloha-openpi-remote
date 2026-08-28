# Contributing to pi-robotics

`pi-robotics` is an independent Mac/WSL ALOHA demo derived from [OpenPI](https://github.com/Physical-Intelligence/openpi). It is not an official Physical Intelligence project. Submit integration-specific issues and pull requests here; report a reproducible problem in unchanged OpenPI to the upstream project.

## Issues

Search existing issues first, then include:

- the affected project revision and policy profile;
- operating-system and Python versions;
- minimal reproduction commands and the expected/actual result;
- sanitized error text or an allowlisted summary.

Do not paste SSH output, private addresses, account or host names, absolute user paths, credentials, videos, weights, checkpoints, raw logs, or raw telemetry. Report security-sensitive details through [SECURITY.md](SECURITY.md).

## Pull requests

Keep changes focused on this integration and preserve its fail-closed boundaries: loopback-only policy service, strict SSH host-key checking, exact process ownership, finite timeouts, validated action schemas, ignored private evidence, and disabled upstream pushes.

Before requesting review, run:

```bash
make test
make lint
make secret-scan
make public-audit
```

External contributors should run tests, lint, and the secret scan. The canonical `make public-audit` also verifies this repository's exact remotes/history, so maintainers run it before accepting a change; forks may fail that identity gate by design. Run `make pr-status` only when changing the seven-PR development stack. The publication audit checks project-added history as well as current files. If it finds a real secret, rotate it and repair unpublished history; deleting it only from the latest revision is insufficient.

Describe the scope, tests, security impact, hardware evidence if relevant, limitations, and rollback. Never commit generated outputs, runtime records, machine-specific configuration, or model artifacts. Changes to upstream-derived files must retain the original license and attribution.
