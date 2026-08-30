#!/usr/bin/env bash
set -euo pipefail
umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1-}"
backend="${2-}"
host="${3-}"
port="${4-}"
startup_timeout="${5-}"
data_home="${6-}"
jax_mem_fraction="${7-}"
expected_sha="${8-}"
state_dir="$repo_root/.runtime"
record="$state_dir/server.json"
lifecycle_state="$HOME/.local/state/aloha-openpi-remote"

[[ "$(uname -s)" == Linux && "$(uname -r)" == *[Mm]icrosoft* ]] || {
    echo 'start_policy_server.sh must run inside WSL.' >&2
    exit 1
}
[[ "$host" == 127.0.0.1 ]] || { echo 'Policy server must bind literal loopback 127.0.0.1.' >&2; exit 1; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo 'Invalid policy port.' >&2; exit 1; }
[[ "$startup_timeout" =~ ^[0-9]+$ ]] && (( startup_timeout >= 1 && startup_timeout <= 7200 )) || {
    echo 'Invalid startup timeout.' >&2
    exit 1
}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid expected project SHA.' >&2; exit 1; }
[[ "$backend" == jax || "$backend" == pytorch ]] || { echo 'Invalid policy backend.' >&2; exit 1; }
[[ -z "$data_home" || "$data_home" == /* ]] || { echo 'OPENPI_DATA_HOME must be empty or absolute.' >&2; exit 1; }
case "$jax_mem_fraction" in
    0.75|0.80|0.85|0.90|0.95) ;;
    *) echo 'OPENPI_JAX_MEM_FRACTION must be one of: 0.75, 0.80, 0.85, 0.90, 0.95.' >&2; exit 1 ;;
esac

case "$profile" in
    pi0_aloha_sim)
        environment=ALOHA_SIM
        prompt=()
        ;;
    pi05_aloha_base)
        environment=ALOHA
        prompt=(--default-prompt="Transfer cube")
        ;;
    pi05_trossen_block_transfer)
        environment=ALOHA
        prompt=(--default-prompt="grab and handover the red cube")
        ;;
    pi05_libero)
        environment=LIBERO
        prompt=()
        ;;
    *)
        echo 'Invalid OPENPI_POLICY_PROFILE.' >&2
        exit 1
        ;;
esac

cd "$repo_root"
[[ -x .venv/bin/python ]] || { echo 'Missing WSL project environment; run make setup-pc.' >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$expected_sha" ]] || { echo 'WSL and Mac candidate SHAs differ.' >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo 'WSL checkout is dirty.' >&2; exit 1; }
mkdir -p "$state_dir"
chmod 700 "$state_dir"
mkdir -p "$lifecycle_state"
chmod 700 "$lifecycle_state"
exec 9>"$lifecycle_state/lifecycle.lock"
flock -n 9 || { echo 'Another policy lifecycle operation is active.' >&2; exit 1; }

if [[ -e "$record" || -L "$record" ]]; then
    if .venv/bin/python -m tools.remote_aloha.process_record verify "$record" >/dev/null 2>&1; then
        echo 'An owned policy server is already running; run make stop first.' >&2
    else
        echo 'A stale, corrupt, or mismatched process record exists; run make stop for safe recovery.' >&2
    fi
    exit 1
fi
if ss -H -ltn "sport = :$port" | grep -q .; then
    echo 'The remote policy port is already in use.' >&2
    exit 1
fi

data_home="${data_home:-$HOME/.cache/openpi}"
mkdir -p "$data_home"
backend_args=("--policy-backend=$backend")
compiler_env=()
if [[ "$backend" == pytorch ]]; then
    checkpoint_label="$profile"
    [[ "$profile" != pi05_aloha_base ]] || checkpoint_label=pi05_base
    checkpoint="$data_home/openpi-assets/checkpoints/${checkpoint_label}_pytorch"
    [[ ( -f "$checkpoint/model.safetensors" || -f "$checkpoint/model.safetensors.index.json" ) &&
        -f "$checkpoint/config.json" && -d "$checkpoint/assets" ]] || {
        echo 'Selected PyTorch checkpoint is incomplete; run make convert-pc first.' >&2
        exit 1
    }
    jax_platform=cpu
    backend_args+=("--pytorch-checkpoint-dir=$checkpoint" --require-torch-device=3090)
    if [[ "$profile" == pi05_* ]]; then
        inductor_cache="$(mktemp -d "$state_dir/torchinductor-${expected_sha:0:12}.XXXXXX")"
        chmod 700 "$inductor_cache"
        compiler_env+=("TORCHINDUCTOR_CACHE_DIR=$inductor_cache" TORCH_LOGS=dynamo,recompiles,graph_breaks)
    fi
else
    jax_platform=cuda
    backend_args+=(--require-jax-platform=gpu --require-jax-device=3090)
fi
log_relative=".runtime/server-${expected_sha:0:12}-$profile.log"
command=(
    env
    "OPENPI_DATA_HOME=$data_home"
    "XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_mem_fraction"
    "JAX_PLATFORMS=$jax_platform"
    CUDA_VISIBLE_DEVICES=0
    "${compiler_env[@]}"
    "$repo_root/.venv/bin/python"
    "$repo_root/scripts/serve_policy.py"
    "--host=$host"
    "--port=$port"
    "--env=$environment"
    "--policy-profile=$profile"
    --compact-masked-images
    "${backend_args[@]}"
    "${prompt[@]}"
)
pid="$(.venv/bin/python -m tools.remote_aloha.process_record launch \
    "$record" "$profile" "$port" "$expected_sha" "$log_relative" \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/serve_policy.py" \
    -- "${command[@]}")"
echo "[server] loading profile=$profile backend=$backend; a temporary RAM increase is expected"

cleanup_failed_start() {
    local status=$?
    trap - EXIT
    if (( status != 0 )) && [[ -e "$record" ]]; then
        flock -u 9 || true
        "$repo_root/scripts/stop_policy_server.sh" "$port" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_failed_start EXIT

deadline=$((SECONDS + startup_timeout))
next_progress=$SECONDS
while ! curl --fail --silent --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; do
    if ! .venv/bin/python -m tools.remote_aloha.process_record verify "$record" >/dev/null 2>&1; then
        echo 'Policy server exited before readiness; inspect the ignored log.' >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo 'Policy server startup timed out; partial cache and log were preserved.' >&2
        flock -u 9
        "$repo_root/scripts/stop_policy_server.sh" "$port" || true
        exit 1
    fi
    if (( SECONDS >= next_progress )); then
        echo "[server] still loading; elapsed=${SECONDS}s"
        next_progress=$((SECONDS + 10))
    fi
    sleep 2
done

"$repo_root/scripts/check_policy_server.sh" "$profile" "$host" "$port" "$expected_sha" >/dev/null
trap - EXIT
echo "Policy server ready: profile=$profile backend=$backend source_sha=$expected_sha log=$log_relative"
