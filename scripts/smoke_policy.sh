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

[[ "$profile" == pi0_aloha_sim || "$profile" == pi05_aloha_base ]] || { echo 'Invalid profile.' >&2; exit 1; }
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
timeout --signal=TERM --kill-after=5s "$((inference_timeout + 15))s" \
    "$smi" --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=500 \
    >"$metrics" 2>&1 &
sampler_pid=$!
cleanup() {
    kill -TERM "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
timeout --signal=TERM --kill-after=10s "${inference_timeout}s" \
    .venv/bin/python -m tools.remote_aloha.policy_smoke \
    --profile "$profile" --backend "$backend" --host "$host" --port "$port" --source-sha "$expected_sha"
cleanup
trap - EXIT INT TERM
grep -Fq '3090' "$metrics" || { echo 'GPU sampler did not observe the RTX 3090.' >&2; exit 1; }
compute_apps="$($smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)" || {
    echo 'nvidia-smi cannot provide required per-process GPU-memory evidence.' >&2
    exit 1
}
awk -F, -v pid="$server_pid" '$1 + 0 == pid && $2 + 0 > 0 {found=1} END {exit !found}' <<<"$compute_apps" || {
    echo 'nvidia-smi did not attribute GPU memory to the owned policy server.' >&2
    exit 1
}
echo "GPU evidence captured: $metrics"
