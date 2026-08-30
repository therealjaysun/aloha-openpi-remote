#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1-}"
port="${2-}"
expected_sha="${3-}"
run_id="${4-}"

[[ "$(uname -s)" == Linux && "$(uname -r)" == *[Mm]icrosoft* ]] || exit 1
[[ "$profile" == pi0_aloha_sim || "$profile" == pi05_aloha_base || "$profile" == pi05_trossen_block_transfer ]] || exit 2
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || exit 2
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$run_id" =~ ^[0-9a-f]{32}$ && "${OPENPI_HOLDER_RUN_ID-}" == "$run_id" ]] || exit 2
[[ -x "$repo_root/.venv/bin/python" ]] || exit 1

cd "$repo_root"
exec .venv/bin/python -m tools.remote_aloha.process_record hold \
    .runtime/server.json "$profile" "$port" "$expected_sha"
