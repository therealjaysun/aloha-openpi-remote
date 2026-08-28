#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 0 ]] || { echo 'doctor_repo.sh accepts no arguments.' >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
expected_repo=therealjaysun/pi-robotics
expected_origin=https://github.com/therealjaysun/pi-robotics.git
expected_upstream=https://github.com/Physical-Intelligence/openpi.git

fail() {
    echo "Repository doctor failed: $1" >&2
    echo "Recovery: $2" >&2
    exit 1
}

command -v git >/dev/null || fail 'git is required.' 'Install Git, then rerun: make doctor'
command -v gh >/dev/null || fail 'GitHub CLI is required.' 'Install gh from https://cli.github.com/, then rerun: make doctor'

cd "$repo_root"
[[ "$(git rev-parse --show-toplevel 2>/dev/null)" == "$repo_root" ]] ||
    fail 'this checkout is not the canonical repository root.' "Run: cd '$repo_root'"
if ! worktree_status="$(git status --porcelain=v1 --untracked-files=all 2>/dev/null)"; then
    fail 'git could not inspect the worktree.' 'Run: git status --short'
fi
[[ -z "$worktree_status" ]] ||
    fail 'the worktree is not clean.' 'Run: git status --short'
[[ "$(git remote get-url origin 2>/dev/null)" == "$expected_origin" ]] ||
    fail 'origin fetch URL is not the independent project.' "Run: git remote set-url origin $expected_origin"
[[ "$(git remote get-url --push origin 2>/dev/null)" == "$expected_origin" ]] ||
    fail 'origin push URL is not the independent project.' "Run: git remote set-url --push origin $expected_origin"
[[ "$(git remote get-url upstream 2>/dev/null)" == "$expected_upstream" ]] ||
    fail 'upstream fetch URL is not official OpenPI.' "Run: git remote set-url upstream $expected_upstream"
[[ "$(git remote get-url --push upstream 2>/dev/null)" == DISABLED ]] ||
    fail 'upstream push is not disabled.' 'Run: git remote set-url --push upstream DISABLED'

gh auth status --hostname github.com >/dev/null 2>&1 ||
    fail 'GitHub CLI is not authenticated for github.com.' 'Run: gh auth login --hostname github.com'
if ! repo_status="$(gh repo view "$expected_repo" --json nameWithOwner,visibility --jq '[.nameWithOwner, .visibility] | join("|")' 2>/dev/null)"; then
    fail 'the public project is unavailable to GitHub CLI.' "Run: gh repo view $expected_repo"
fi
[[ "$repo_status" == "$expected_repo|PUBLIC" ]] ||
    fail "expected $expected_repo to be public; received '$repo_status'." "Run: gh repo view $expected_repo --json nameWithOwner,visibility"

echo "Repository doctor passed: clean $expected_repo checkout with protected upstream remote."
