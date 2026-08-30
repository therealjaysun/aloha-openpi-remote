#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1-}"
host="${2-}"
port="${3-}"
expected_sha="${4-}"
record="$repo_root/.runtime/server.json"
python="$repo_root/.venv/bin/python"

[[ "$profile" == pi0_aloha_sim || "$profile" == pi05_aloha_base || "$profile" == pi05_trossen_block_transfer || "$profile" == pi05_libero ]] || { echo 'Invalid profile.' >&2; exit 1; }
[[ "$host" == 127.0.0.1 ]] || { echo 'Policy listener must use loopback.' >&2; exit 1; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo 'Invalid policy port.' >&2; exit 1; }
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid expected SHA.' >&2; exit 1; }
[[ -x "$python" ]] || { echo 'Missing WSL project environment.' >&2; exit 1; }

pid="$($python -m tools.remote_aloha.process_record verify "$record")"
[[ "$($python -m tools.remote_aloha.process_record field "$record" profile)" == "$profile" ]] || {
    echo 'Recorded policy profile differs.' >&2
    exit 1
}
[[ "$($python -m tools.remote_aloha.process_record field "$record" port)" == "$port" ]] || {
    echo 'Recorded policy port differs.' >&2
    exit 1
}
[[ "$($python -m tools.remote_aloha.process_record field "$record" source_sha)" == "$expected_sha" ]] || {
    echo 'Recorded source SHA differs.' >&2
    exit 1
}
listener="$(ss -H -ltnp "sport = :$port")"
grep -Fq "$host:$port" <<<"$listener" || { echo 'Policy listener is not on IPv4 loopback.' >&2; exit 1; }
grep -Fq "pid=$pid," <<<"$listener" || { echo 'Policy listener is not owned by the recorded process.' >&2; exit 1; }
curl --fail --silent --max-time 5 "http://$host:$port/healthz" >/dev/null
printf '__ALOHA_SERVER__=ready\n'
