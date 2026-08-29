#!/usr/bin/env bash
set -euo pipefail
umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
expected_port="${1-}"
state_dir="$repo_root/.runtime"
record="$state_dir/server.json"
lifecycle_state="$HOME/.local/state/aloha-openpi-remote"
python="$repo_root/.venv/bin/python"

[[ "$expected_port" =~ ^[0-9]+$ ]] && (( expected_port >= 1 && expected_port <= 65535 )) || {
    echo 'Invalid policy port.' >&2
    exit 1
}
mkdir -p "$state_dir"
chmod 700 "$state_dir"
mkdir -p "$lifecycle_state"
chmod 700 "$lifecycle_state"
exec 9>"$lifecycle_state/lifecycle.lock"
flock -n 9 || { echo 'Another policy lifecycle operation is active.' >&2; exit 1; }

if [[ ! -e "$record" && ! -L "$record" ]]; then
    if ss -H -ltn "sport = :$expected_port" | grep -q .; then
        echo 'No ownership record exists, but the policy port is occupied; refusing to signal.' >&2
        exit 1
    fi
    echo 'No owned policy server is running.'
    exit 0
fi
[[ -x "$python" ]] || { echo 'Cannot validate the process record without the WSL project environment.' >&2; exit 1; }
recorded_port="$($python -m tools.remote_aloha.process_record field "$record" port)"
[[ "$recorded_port" == "$expected_port" ]] || { echo 'Recorded port differs; refusing to signal.' >&2; exit 1; }

set +e
pid="$($python -m tools.remote_aloha.process_record verify "$record" 2>/dev/null)"
verify_status=$?
set -e
if (( verify_status == 3 )); then
    ss -H -ltn "sport = :$expected_port" | grep -q . && {
        echo 'Recorded process is gone, but the port is occupied; refusing cleanup.' >&2
        exit 1
    }
    rm -- "$record"
    echo 'Cleared a stale ownership record; no process was signaled.'
    exit 0
elif (( verify_status != 0 )); then
    echo 'Process identity is corrupt or mismatched; refusing to signal.' >&2
    exit 1
fi

"$python" -m tools.remote_aloha.process_record signal "$record" TERM >/dev/null
deadline=$((SECONDS + 30))
while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
    current_pid="$($python -m tools.remote_aloha.process_record verify "$record")" || {
        echo 'PID identity changed before KILL; refusing to signal.' >&2
        exit 1
    }
    [[ "$current_pid" == "$pid" ]] || { echo 'PID identity changed before KILL; refusing to signal.' >&2; exit 1; }
    "$python" -m tools.remote_aloha.process_record signal "$record" KILL >/dev/null
    for _ in {1..10}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
fi
kill -0 "$pid" 2>/dev/null && { echo 'Owned policy process did not stop.' >&2; exit 1; }
ss -H -ltn "sport = :$expected_port" | grep -q . && { echo 'Policy port remains occupied.' >&2; exit 1; }
rm -- "$record"
echo 'Owned policy server stopped; log preserved.'
