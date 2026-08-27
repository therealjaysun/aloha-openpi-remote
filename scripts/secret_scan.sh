#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_base="${SECRET_SCAN_BASE_REF:-215abfb217dbac7d5f1273282331b9b1866c0479}"
scan_tmp="$(mktemp -d "${TMPDIR:-/tmp}/aloha-secret-scan.XXXXXX")"

cleanup() {
    if [[ -n "$scan_tmp" && -d "$scan_tmp" ]]; then
        rm -rf -- "$scan_tmp"
    fi
}
trap cleanup EXIT

cd "$repo_root"
command -v gitleaks >/dev/null || {
    echo 'gitleaks is required; install it with: brew install gitleaks' >&2
    exit 1
}
git rev-parse --verify "$upstream_base^{commit}" >/dev/null || {
    echo "Secret-scan base is unavailable: $upstream_base" >&2
    exit 1
}

gitleaks git --no-banner --redact --timeout 180 --log-opts="$upstream_base..HEAD" .
gitleaks git --no-banner --redact --timeout 180 --staged .

candidate_bundle="$scan_tmp/publishable-candidates"
: >"$candidate_bundle"
while IFS= read -r -d '' path; do
    printf '\nFILE %s\n' "$path" >>"$candidate_bundle"
    if [[ -L "$path" ]]; then
        readlink "$path" >>"$candidate_bundle"
    elif [[ -f "$path" ]]; then
        cat -- "$path" >>"$candidate_bundle"
    fi
done < <(git ls-files --cached --others --exclude-standard -z)
gitleaks dir --no-banner --redact --timeout 180 "$candidate_bundle"

echo 'Secret scan passed: commit range, staged changes, and non-ignored candidates.'
