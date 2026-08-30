#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

[[ $# -eq 7 ]] || {
    echo 'Expected output path, run ID, profile, server PID, source SHA, interval seconds, and policy port.' >&2
    exit 2
}
output="$1"
run_id="$2"
profile="$3"
server_pid="$4"
source_sha="$5"
interval_seconds="$6"
policy_port="$7"

[[ "$output" == /* && "$output" == *.jsonl && "$output" != *$'\n'* && "$output" != *$'\r'* ]] || {
    echo 'Output must be an absolute JSONL path without control characters.' >&2
    exit 2
}
[[ "$run_id" =~ ^[0-9a-f]{32}$ ]] || { echo 'Run ID must be a 32-character lowercase hex token.' >&2; exit 2; }
[[ "$profile" == pi0_aloha_sim || "$profile" == pi05_aloha_base || "$profile" == pi05_trossen_block_transfer || "$profile" == pi05_libero ]] || { echo 'Invalid profile.' >&2; exit 2; }
[[ "$server_pid" =~ ^[0-9]+$ ]] && (( 10#$server_pid > 1 && 10#$server_pid <= 4194304 )) || {
    echo 'Invalid server PID.' >&2
    exit 2
}
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid source SHA.' >&2; exit 2; }
[[ "$policy_port" =~ ^[0-9]+$ ]] && (( 10#$policy_port >= 1 && 10#$policy_port <= 65535 )) || {
    echo 'Invalid policy port.' >&2
    exit 2
}
[[ "$interval_seconds" =~ ^[0-9]+([.][0-9]{1,3})?$ ]] &&
    awk -v value="$interval_seconds" 'BEGIN { exit !(value >= 0.1 && value <= 60) }' || {
    echo 'Interval must be between 0.1 and 60 seconds with at most three decimal places.' >&2
    exit 2
}
interval_ms="$(awk -v value="$interval_seconds" 'BEGIN { printf "%.0f", value * 1000 }')"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
runtime="$repo_root/.runtime"
record="$runtime/server.json"
python="$repo_root/.venv/bin/python"
[[ -d "$runtime" && ! -L "$runtime" ]] || { echo 'The private runtime directory is missing or unsafe.' >&2; exit 1; }
runtime="$(realpath -e -- "$runtime")"
[[ "$(stat -c '%u:%a' -- "$runtime")" == "$(id -u):700" ]] || {
    echo 'The runtime directory must be owned by the current user with mode 0700.' >&2
    exit 1
}
[[ -x "$python" ]] || { echo 'The locked project Python environment is missing.' >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required.' >&2; exit 1; }
command -v timeout >/dev/null || { echo 'GNU timeout is required.' >&2; exit 1; }
smi="$(command -v nvidia-smi || true)"
[[ -n "$smi" ]] || [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]] || smi=/usr/lib/wsl/lib/nvidia-smi
[[ -n "$smi" ]] || { echo 'nvidia-smi is unavailable in WSL.' >&2; exit 1; }

output_parent="$(dirname -- "$output")"
[[ -d "$output_parent" && ! -L "$output_parent" ]] || { echo 'Output parent must be an existing directory.' >&2; exit 2; }
output_parent="$(realpath -e -- "$output_parent")"
case "$output_parent/" in "$runtime/"*) ;; *) echo 'Output must remain inside the private runtime directory.' >&2; exit 2 ;; esac
[[ "$(stat -c '%u:%a' -- "$output_parent")" == "$(id -u):700" ]] || {
    echo 'Output parent must be owned by the current user with mode 0700.' >&2
    exit 1
}
output="$output_parent/$(basename -- "$output")"
[[ ! -e "$output" && ! -L "$output" ]] || { echo 'Refusing to overwrite an existing metrics file.' >&2; exit 1; }

lock="$runtime/gpu-sampler.lock"
[[ ! -L "$lock" ]] || { echo 'Sampler lock must not be a symbolic link.' >&2; exit 1; }
exec 9>>"$lock"
chmod 600 "$lock"
flock -n 9 || { echo 'A GPU sampler is already active.' >&2; exit 1; }
sampler_record="$runtime/gpu-sampler.json"
if [[ -e "$sampler_record" || -L "$sampler_record" ]]; then
    set +e
    "$python" -m tools.remote_aloha.process_record verify "$sampler_record" >/dev/null 2>&1
    record_status=$?
    set -e
    if (( record_status == 3 )); then
        rm -- "$sampler_record"
    elif (( record_status == 0 )); then
        echo 'A verified GPU sampler is already running.' >&2
        exit 1
    else
        echo 'The GPU sampler ownership record is unsafe or mismatched.' >&2
        exit 1
    fi
fi

open_output() {
    set -o noclobber
    exec 8>"$output"
}
open_output 2>/dev/null || { echo 'Could not exclusively create the metrics file.' >&2; exit 1; }
set +o noclobber
chmod 600 "$output"

identity_snapshot() {
    "$python" -c '
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from tools.remote_aloha.process_record import verify_record

try:
    record = verify_record(Path(sys.argv[1]))
    status = (Path("/proc") / str(record.pid) / "status").read_text(encoding="utf-8")
    rss_values = []
    for line in status.splitlines():
        fields = line.split()
        if fields[:1] == ["VmRSS:"]:
            if len(fields) != 3 or not fields[1].isdigit() or fields[2] != "kB":
                raise ValueError
            rss_values.append(int(fields[1]))
    if len(rss_values) != 1 or verify_record(Path(sys.argv[1])) != record:
        raise ValueError
except Exception:
    raise SystemExit(1) from None
utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
print(record.pid, record.profile, record.source_sha, time.monotonic_ns(), utc, rss_values[0], sep="\t")
' "$record"
}

clock_snapshot() {
    timeout --signal=TERM --kill-after=1s 2s "$python" -c '
from datetime import datetime, timezone
import time

utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
print(time.monotonic_ns(), utc, sep="\t")
' 2>/dev/null
}

read_identity() {
    local snapshot
    snapshot="$(identity_snapshot)" || return 1
    IFS=$'\t' read -r current_pid current_profile current_sha current_monotonic_ns current_utc current_server_rss_kib \
        <<<"$snapshot"
    [[ "$current_pid" == "$server_pid" && "$current_profile" == "$profile" && "$current_sha" == "$source_sha" ]] || return 1
    [[ "$current_monotonic_ns" =~ ^[0-9]+$ && "$current_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T &&
        "$current_server_rss_kib" =~ ^[0-9]+$ ]] || return 1
}

read_identity || { echo 'Policy-server ownership does not match the requested sampler.' >&2; exit 1; }
start_monotonic_ns="$current_monotonic_ns"
last_monotonic_ns="$current_monotonic_ns"
last_utc="$current_utc"
sleeper_pid=
record_owned=no

cleanup() {
    local status=$?
    local terminal_status=failed
    set +e
    trap - EXIT HUP INT TERM
    if [[ -n "$sleeper_pid" ]]; then
        kill -TERM "$sleeper_pid" 2>/dev/null || true
        wait "$sleeper_pid" 2>/dev/null || true
    fi
    if [[ "$record_owned" == yes ]]; then
        local recorded_pid
        recorded_pid="$($python -m tools.remote_aloha.process_record verify "$sampler_record" 2>/dev/null)"
        if [[ "$recorded_pid" == "$$" ]]; then
            rm -- "$sampler_record"
        fi
    fi
    local terminal_clock
    local terminal_monotonic_ns
    local terminal_utc
    if terminal_clock="$(clock_snapshot)"; then
        IFS=$'\t' read -r terminal_monotonic_ns terminal_utc <<<"$terminal_clock"
        if [[ "$terminal_monotonic_ns" =~ ^[0-9]+$ && "$terminal_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]; then
            last_monotonic_ns="$terminal_monotonic_ns"
            last_utc="$terminal_utc"
        fi
    fi
    [[ $status -eq 0 ]] && terminal_status=passed
    [[ $status -eq 129 || $status -eq 130 || $status -eq 143 ]] && terminal_status=interrupted
    printf '{"schema":1,"event":"sampler_stopped","utc":"%s","monotonic_ns":%s,"run_id":"%s","profile":"%s","server_pid":%s,"source_sha":"%s","interval_ms":%s,"status":"%s","exit_status":%s}\n' \
        "$last_utc" "$last_monotonic_ns" "$run_id" "$profile" "$server_pid" "$source_sha" "$interval_ms" \
        "$terminal_status" "$status" >&8
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

metrics_relative=".runtime/$(basename -- "$output")"
"$python" -m tools.remote_aloha.process_record create "$sampler_record" "$$" "$profile" "$policy_port" \
    "$source_sha" "$metrics_relative"
record_owned=yes

printf '{"schema":1,"event":"sampler_started","utc":"%s","monotonic_ns":%s,"run_id":"%s","profile":"%s","server_pid":%s,"source_sha":"%s","interval_ms":%s}\n' \
    "$current_utc" "$current_monotonic_ns" "$run_id" "$profile" "$server_pid" "$source_sha" "$interval_ms" >&8

sample_index=0
while true; do
    read_identity || { echo 'Policy-server identity changed; stopping GPU sampling.' >&2; exit 1; }
    last_monotonic_ns="$current_monotonic_ns"
    last_utc="$current_utc"
    raw="$({ timeout --signal=TERM --kill-after=2s 5s "$smi" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; } 2>/dev/null)" || {
        echo 'Bounded nvidia-smi sampling failed.' >&2
        exit 1
    }
    read_identity || { echo 'Policy-server identity changed during GPU sampling.' >&2; exit 1; }
    last_monotonic_ns="$current_monotonic_ns"
    last_utc="$current_utc"
    [[ "$raw" != *$'\n'* ]] || { echo 'Expected exactly one GPU sample.' >&2; exit 1; }
    IFS=, read -r memory_used_mib utilization_percent extra <<<"$raw"
    memory_used_mib="${memory_used_mib//[[:space:]]/}"
    utilization_percent="${utilization_percent//[[:space:]]/}"
    [[ -z "${extra-}" && "$memory_used_mib" =~ ^[0-9]+$ && "$utilization_percent" =~ ^[0-9]+$ ]] || {
        echo 'nvidia-smi returned an invalid sample.' >&2
        exit 1
    }
    (( 10#$utilization_percent <= 100 )) || { echo 'nvidia-smi returned an invalid utilization.' >&2; exit 1; }
    elapsed_ms=$(( (10#$current_monotonic_ns - 10#$start_monotonic_ns) / 1000000 ))
    (( elapsed_ms >= 0 )) || { echo 'The monotonic clock moved backwards.' >&2; exit 1; }
    printf '{"schema":1,"event":"gpu_sample","utc":"%s","monotonic_ns":%s,"elapsed_ms":%s,"run_id":"%s","profile":"%s","server_pid":%s,"source_sha":"%s","interval_ms":%s,"memory_used_mib":%s,"utilization_percent":%s,"server_rss_kib":%s,"sample_index":%s}\n' \
        "$current_utc" "$current_monotonic_ns" "$elapsed_ms" "$run_id" "$profile" "$server_pid" "$source_sha" \
        "$interval_ms" "$memory_used_mib" "$utilization_percent" "$current_server_rss_kib" "$sample_index" >&8
    sample_index=$((sample_index + 1))
    sleep "$interval_seconds" &
    sleeper_pid=$!
    wait "$sleeper_pid"
    sleeper_pid=
done
