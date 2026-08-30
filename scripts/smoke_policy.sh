#!/usr/bin/env bash
set -euo pipefail
umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1-}"
backend="${2-}"
host="${3-}"
port="${4-}"
inference_timeout="${5-}"
expected_sha="${6-}"

[[ "$profile" == pi0_aloha_sim || "$profile" == pi05_aloha_base || "$profile" == pi05_trossen_block_transfer || "$profile" == pi05_libero ]] || { echo 'Invalid profile.' >&2; exit 1; }
[[ "$backend" == jax || "$backend" == pytorch ]] || { echo 'Invalid backend.' >&2; exit 1; }
[[ "$host" == 127.0.0.1 ]] || { echo 'Policy smoke requires loopback.' >&2; exit 1; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo 'Invalid port.' >&2; exit 1; }
[[ "$inference_timeout" =~ ^[0-9]+$ ]] && (( inference_timeout >= 1 && inference_timeout <= 1800 )) || {
    echo 'Invalid inference timeout.' >&2
    exit 1
}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid expected SHA.' >&2; exit 1; }
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_sha" ]] || { echo 'WSL and Mac candidate SHAs differ.' >&2; exit 1; }
command -v timeout >/dev/null || { echo 'GNU timeout is required for the Phase 2 smoke test.' >&2; exit 1; }
smi="$(command -v nvidia-smi || true)"
[[ -n "$smi" ]] || [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]] || smi=/usr/lib/wsl/lib/nvidia-smi
[[ -n "$smi" ]] || { echo 'nvidia-smi is unavailable in WSL.' >&2; exit 1; }
record="$repo_root/.runtime/server.json"
"$repo_root/scripts/check_policy_server.sh" "$profile" "$host" "$port" "$expected_sha" >/dev/null
server_pid="$($repo_root/.venv/bin/python -m tools.remote_aloha.process_record verify "$record")"
cd "$repo_root"
metrics=".runtime/gpu-smoke-${expected_sha:0:12}-$profile.csv"
host_metrics=".runtime/host-smoke-${expected_sha:0:12}-$profile.csv"
timeout --signal=TERM --kill-after=5s "$((inference_timeout + 15))s" \
    "$smi" --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=500 \
    >"$metrics" 2>&1 &
sampler_pid=$!
(
    while [[ -r "/proc/$server_pid/status" ]]; do
        awk '/VmRSS:/ {rss=$2} /VmSwap:/ {swap=$2} END {print rss + 0 "," swap + 0}' "/proc/$server_pid/status"
        sleep 0.5
    done
) >"$host_metrics" &
host_sampler_pid=$!
cleanup() {
    kill -TERM "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    kill -TERM "$host_sampler_pid" 2>/dev/null || true
    wait "$host_sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
timeout --signal=TERM --kill-after=10s "${inference_timeout}s" \
    .venv/bin/python -m tools.remote_aloha.policy_smoke \
    --profile "$profile" --backend "$backend" --host "$host" --port "$port" --source-sha "$expected_sha" \
    --connect-timeout 60 --metadata-timeout 30 --inference-timeout "$inference_timeout" --close-timeout 10
cleanup
trap - EXIT INT TERM
host_peak_rss="$(awk -F, '$1+0>m {m=$1+0} END {print m+0}' "$host_metrics")"
[[ "$host_peak_rss" =~ ^[0-9]+$ ]] && (( host_peak_rss > 0 )) || {
    echo 'Host-memory sampling evidence is invalid.' >&2
    exit 1
}
"$repo_root/.venv/bin/python" -m tools.remote_aloha.process_record verify "$record" >/dev/null || {
    echo "Policy server exited during inference; peak RSS was ${host_peak_rss} KiB." >&2
    exit 1
}
grep -Fq '3090' "$metrics" || { echo 'GPU sampler did not observe the RTX 3090.' >&2; exit 1; }
if [[ "$backend" == jax ]]; then
    compute_apps="$($smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)" || {
        echo 'nvidia-smi cannot provide required per-process GPU-memory evidence.' >&2
        exit 1
    }
    awk -F, -v pid="$server_pid" '$1 + 0 == pid && $2 + 0 > 0 {found=1} END {exit !found}' <<<"$compute_apps" || {
        echo 'nvidia-smi did not attribute GPU memory to the owned policy server.' >&2
        exit 1
    }
fi
echo "GPU and host evidence captured: $metrics $host_metrics peak_rss_kib=$host_peak_rss"
