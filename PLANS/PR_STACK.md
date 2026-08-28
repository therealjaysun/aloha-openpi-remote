# Pull request stack

Create each branch from the locally code-complete branch directly above it. An external blocker does not prevent unblocked child work: record the blocker, keep affected PRs draft, and continue without waiting for merge or publication.

```text
main
└── codex/00-bootstrap
    └── codex/01-mac-simulation
        └── codex/02-remote-gpu-server
            └── codex/03-secure-connectivity
                └── codex/04-end-to-end-control
                    └── codex/05-observability
                        └── codex/06-hardening-docs
```

| PR | Head | Base | Title |
| --- | --- | --- | --- |
| 1/7 | `codex/00-bootstrap` | `main` | `chore: establish remote ALOHA project baseline` |
| 2/7 | `codex/01-mac-simulation` | `codex/00-bootstrap` | `feat(sim): run native ALOHA simulation on macOS` |
| 3/7 | `codex/02-remote-gpu-server` | `codex/01-mac-simulation` | `feat(remote): run OpenPI policy server in WSL` |
| 4/7 | `codex/03-secure-connectivity` | `codex/02-remote-gpu-server` | `feat(ssh): add secure policy tunnel` |
| 5/7 | `codex/04-end-to-end-control` | `codex/03-secure-connectivity` | `feat(runtime): connect ALOHA simulation to OpenPI` |
| 6/7 | `codex/05-observability` | `codex/04-end-to-end-control` | `feat(telemetry): record control and GPU metrics` |
| 7/7 | `codex/06-hardening-docs` | `codex/05-observability` | `docs: harden and document the complete workflow` |

All seven PRs are open, ready, green, and correctly stacked for human review. [PR 7](https://github.com/therealjaysun/pi-robotics/pull/7) contains final implementation candidate `a8a3ca1`; none is configured for auto-merge.

## Per-phase procedure

1. Confirm the parent branch is clean and locally code-complete/tested, or has an external blocker with evidence and an exact recovery command. A push is not required while GitHub is blocked.
2. `git switch -c <head> <base>`.
3. Implement only the active phase and its tests.
4. Run feasible phase checks, `git diff --check`, staged-file inspection, and the fail-closed secret scan over staged files, the project commit range, and non-ignored candidates.
5. Commit coherent concepts; never commit outputs, weights, secrets, or machine identifiers.
6. For PC-dependent phases, commit a coherent locally tested remote-test candidate, push it, check out that exact SHA in a clean WSL worktree, and verify Mac SHA = WSL SHA. Add hardware evidence/fixes in follow-up commits. If GitHub is still blocked, transfer a secret-scanned `git bundle` over SSH and fetch the exact commit; never copy an uncommitted tree.
7. `git push -u origin <head>` when origin is available; otherwise record the local branch/SHA as ready to push.
8. Create a draft PR with explicit `--base` and `--head`; record URL and SHA in plans.
9. Keep it draft until every phase acceptance criterion passes. A hardware-blocked PR may be marked ready only when the user explicitly accepts review with that named blocker.

## Required PR body

Every PR includes: Summary; Phase plan; Stack position; Base PR dependency; Scope; Non-goals; Implementation details; Files changed; Tests executed; Test results; Evidence; Known limitations; Security considerations; Review guide; Rollback procedure; Checklist.

For PRs 1–7 only, include exactly: `This is PR X of 7 in a stacked series. Review and merge the stack in numerical order.`

## Post-stack experiment

S0827 Push-PI is not PR 8. The explicitly scheduled fallback uses one standalone `codex/push-pi-scenarios` PR based on `codex/06-hardening-docs`; its body states that dependency and omits the `PR X of 7` sentence. Keep it draft through exact hardware validation. After PR 7 merges, retarget to `main` and verify the incremental diff before review.
