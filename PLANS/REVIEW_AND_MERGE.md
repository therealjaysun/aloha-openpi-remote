# Review and merge

Begin merging only after the final holistic stack review. Review the seven PRs in numerical order. While the stack is active, use merge commits so child branches retain their parent ancestry. Do not enable auto-merge or automatic branch deletion.

## Normal sequence

1. Review PR 1 and its phase plan/evidence.
2. Merge PR 1 into `main` with a merge commit.
3. Retarget PR 2 to `main`.
4. Verify PR 2 now contains only phase 01 changes: `gh pr diff <PR>` and compare `main...codex/01-mac-simulation`.
5. Review and merge PR 2.
6. Repeat retarget, diff verification, review, and merge through PR 7.

For every PR, distinguish code completeness from Mac, remote, and full hardware validation. A missing hardware check is a blocker/evidence gap, not a reason to misstate test results.

## If a parent was squash-merged, rebased, or changed after children exist

Repair **every descendant**, oldest to newest. For each branch, record its old parent tip, rebase onto the repaired parent, push with lease, retarget if needed, then use its new tip as the next branch's parent. Replace placeholders only after discovering actual branch names and SHAs:

```bash
git fetch origin
git switch <head>
git rebase --onto <new-parent-tip> <old-parent-tip> <head>
git push --force-with-lease origin <head>
gh pr edit <pr> --base <new-base-branch>
git diff --stat <new-parent-tip>...<head>
gh pr diff <pr>
```

For the immediate child of the merged parent, `<new-parent-tip>`/`<new-base-branch>` are normally `origin/main`/`main`. For each deeper descendant, `<new-parent-tip>` is the newly repaired tip of its immediate parent and its PR base remains that parent branch. Repeat for all descendants. Record every old parent tip, new head SHA, and force-push in `STATUS.md`; verify ancestry and every incremental PR diff. Stop if any diff includes earlier phases or unrelated files.

## Review checklist

- Base/head match `PR_STACK.md`; incremental diff contains one phase.
- Phase acceptance commands and actual outputs are recorded.
- No weights, outputs, `.env`, keys, addresses, usernames, or private paths are tracked.
- Server remains loopback-only and the client uses the SSH tunnel.
- Hardware claims have command evidence; success rate is reported separately from infrastructure health.
- License/attribution files remain intact and the derivative does not imply endorsement.
- Rollback is feasible by reverting the phase merge commit.
