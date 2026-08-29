#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 0 ]] || { echo 'pr_status.sh accepts no arguments.' >&2; exit 2; }

repo=therealjaysun/pi-robotics
bases=(
    main
    codex/00-bootstrap
    codex/01-mac-simulation
    codex/02-remote-gpu-server
    codex/03-secure-connectivity
    codex/04-end-to-end-control
    codex/05-observability
)
heads=(
    codex/00-bootstrap
    codex/01-mac-simulation
    codex/02-remote-gpu-server
    codex/03-secure-connectivity
    codex/04-end-to-end-control
    codex/05-observability
    codex/06-hardening-docs
)

fail() {
    echo "PR status failed: $1" >&2
    echo "Recovery: $2" >&2
    exit 1
}

command -v gh >/dev/null || fail 'GitHub CLI is required.' 'Install gh from https://cli.github.com/, then rerun: make pr-status'
gh auth status --hostname github.com >/dev/null 2>&1 ||
    fail 'GitHub CLI is not authenticated for github.com.' 'Run: gh auth login --hostname github.com'

for index in "${!heads[@]}"; do
    number=$((index + 1))
    if ! row="$(gh pr view "$number" --repo "$repo" \
        --json number,state,isDraft,headRefName,baseRefName,autoMergeRequest,statusCheckRollup \
        --jq '[.number, .state, .isDraft, .headRefName, .baseRefName, (.autoMergeRequest != null), ([.statusCheckRollup[] | if .__typename == "CheckRun" then (.conclusion // .status // "") else (.state // "") end] | join(","))] | map(tostring) | join("|")' 2>/dev/null)"; then
        fail "PR $number is unavailable." "Run: gh pr view $number --repo $repo"
    fi
    IFS='|' read -r actual state draft head base auto_merge checks <<<"$row"
    [[ "$actual" == "$number" ]] || fail "requested PR $number but received '$actual'." "Run: gh pr view $number --repo $repo"
    [[ "$state" == OPEN ]] || fail "PR $number is $state, not OPEN." "Run: gh pr reopen $number --repo $repo"
    [[ "$draft" == false ]] || fail "PR $number is a draft." "Run: gh pr ready $number --repo $repo"
    [[ "$head" == "${heads[$index]}" ]] ||
        fail "PR $number head is '$head', expected '${heads[$index]}'." "Recreate PR $number with: gh pr create --repo $repo --head ${heads[$index]} --base ${bases[$index]}"
    [[ "$base" == "${bases[$index]}" ]] ||
        fail "PR $number base is '$base', expected '${bases[$index]}'." "Run: gh pr edit $number --repo $repo --base ${bases[$index]}"
    [[ "$auto_merge" == false ]] ||
        fail "PR $number has auto-merge enabled." "Run: gh pr merge $number --repo $repo --disable-auto"
    [[ -n "$checks" ]] || fail "PR $number has no reported checks." "Run: gh pr checks $number --repo $repo --watch"
    IFS=',' read -r -a conclusions <<<"$checks"
    for conclusion in "${conclusions[@]}"; do
        case "$conclusion" in
            SUCCESS | NEUTRAL | SKIPPED) ;;
            *) fail "PR $number has a non-green check state: '$conclusion'." "Run: gh pr checks $number --repo $repo --watch" ;;
        esac
    done
    echo "PR $number passed: ${heads[$index]} -> ${bases[$index]}"
done

echo 'PR status passed: all seven pull requests are open, ready, correctly stacked, green, and manual-merge only.'
