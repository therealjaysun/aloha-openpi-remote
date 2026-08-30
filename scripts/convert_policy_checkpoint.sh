#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# -eq 5 ]] || { echo 'Expected profile, data home, source SHA, policy port, and restore mode.' >&2; exit 2; }
profile="$1"
data_input="$2"
expected_sha="$3"
policy_port="$4"
requested_restore_mode="$5"
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid source SHA.' >&2; exit 2; }
[[ "$policy_port" =~ ^[0-9]+$ ]] && (( policy_port >= 1 && policy_port <= 65535 )) || {
    echo 'Invalid policy port.' >&2
    exit 2
}
case "$requested_restore_mode" in auto|full-float32|partial-bfloat16) ;; *)
    echo 'Invalid conversion restore mode.' >&2
    exit 2
    ;;
esac

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$expected_sha" ]] || { echo 'Remote source SHA mismatch.' >&2; exit 1; }
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || { echo 'Remote checkout is dirty.' >&2; exit 1; }
[[ -x "$repo/.venv/bin/python" ]] || { echo 'The locked OpenPI environment is missing.' >&2; exit 1; }
[[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU time' || {
    echo 'GNU /usr/bin/time is required.' >&2
    exit 1
}
command -v timeout >/dev/null || { echo 'timeout is required.' >&2; exit 1; }
command -v realpath >/dev/null || { echo 'realpath is required.' >&2; exit 1; }

available_ram_kib="$(awk '$1 == "MemAvailable:" {print $2; exit}' /proc/meminfo)"
[[ "$available_ram_kib" =~ ^[0-9]+$ ]] && (( 10#$available_ram_kib > 0 )) || {
    echo 'Could not determine available system RAM from /proc/meminfo.' >&2
    exit 1
}
restore_mode="$requested_restore_mode"
if [[ "$restore_mode" == auto ]]; then
    if (( 10#$available_ram_kib < 16 * 1024 * 1024 )); then
        restore_mode=partial-bfloat16
    else
        restore_mode=full-float32
    fi
fi

if [[ -z "$data_input" ]]; then data_home="$HOME/.cache/openpi"; else data_home="$data_input"; fi
[[ "$data_home" == /* && "$data_home" != / && -d "$data_home" && ! -L "$data_home" ]] || {
    echo 'Data home must be an absolute, non-symlink directory.' >&2
    exit 2
}
data_home="$(realpath -e -- "$data_home")"

case "$profile" in
    pi0_aloha_sim)
        checkpoint_label=pi0_aloha_sim
        config_name=pi0_aloha_sim
        ;;
    pi05_aloha_base)
        checkpoint_label=pi05_base
        config_name=pi05_aloha
        ;;
    pi05_trossen_block_transfer)
        checkpoint_label=pi05_trossen_block_transfer
        config_name=pi05_trossen_transfer_block
        hf_repo=TrossenRoboticsCommunity/pi05-block-transfer-trossen-ai-openpi
        hf_revision=40aee785d8907e868976454a3ca51c76175f6d4c
        ;;
    *) echo 'Unsupported conversion profile.' >&2; exit 2 ;;
esac

parent_input="$data_home/openpi-assets/checkpoints"
[[ -d "$parent_input" && ! -L "$parent_input" ]] || { echo 'Checkpoint directory is invalid.' >&2; exit 1; }
parent="$(realpath -e -- "$parent_input")"
case "$parent/" in "$data_home"/*) ;; *) echo 'Checkpoint directory escapes data home.' >&2; exit 1 ;; esac
available_kib="$(df -Pk "$parent" | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= 60 * 1024 * 1024 )) || {
    echo 'At least 60 GiB of free checkpoint-disk space is required.' >&2
    exit 1
}
state_dir="$HOME/.local/state/aloha-openpi-remote"
mkdir -p "$state_dir" "$repo/.runtime/conversion"
chmod 700 "$state_dir" "$repo/.runtime" "$repo/.runtime/conversion"
exec 9>"$state_dir/conversion.lock"
flock -n 9 || { echo 'Another checkpoint conversion is active.' >&2; exit 1; }
source_input="$parent/$checkpoint_label"
if [[ -n "${hf_repo-}" && ! -e "$source_input" && ! -L "$source_input" ]]; then
    download_root="$parent/.${checkpoint_label}.download.$hf_revision"
    [[ ! -L "$download_root" ]] || { echo 'Checkpoint download staging path is unsafe.' >&2; exit 1; }
    mkdir -p "$download_root"
    chmod 700 "$download_root"
    "$repo/.venv/bin/python" - "$hf_repo" "$hf_revision" "$download_root" <<'PY'
import pathlib
import sys

from huggingface_hub import snapshot_download

repo_id, revision, destination = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    allow_patterns=["params/**", "assets/**", "_CHECKPOINT_METADATA"],
    local_dir=destination,
)
root = pathlib.Path(destination)
if not (root / "params" / "_METADATA").is_file() or not (
    root / "assets" / "trossen" / "norm_stats.json"
).is_file():
    raise SystemExit("Downloaded checkpoint is incomplete")
PY
    find "$download_root" -type l -print -quit | grep -q . && {
        echo 'Downloaded checkpoint must not contain symbolic links.' >&2
        exit 1
    }
    mv -- "$download_root" "$source_input"
fi
[[ -d "$source_input" && ! -L "$source_input" ]] || { echo 'Source checkpoint is invalid.' >&2; exit 1; }
source_checkpoint="$(realpath -e -- "$source_input")"
[[ "$source_checkpoint" == "$source_input" ]] || { echo 'Source checkpoint path is not canonical.' >&2; exit 1; }
final_checkpoint="$parent/${checkpoint_label}_pytorch"
[[ -d "$source_checkpoint/params" && ! -L "$source_checkpoint/params" &&
    -d "$source_checkpoint/assets" && ! -L "$source_checkpoint/assets" ]] || {
    echo 'The complete source checkpoint and assets are required.' >&2
    exit 1
}
[[ ! -e "$final_checkpoint" && ! -L "$final_checkpoint" ]] || {
    echo 'Refusing to overwrite an existing converted checkpoint.' >&2
    exit 1
}
[[ ! -e "$repo/.runtime/server.json" && ! -L "$repo/.runtime/server.json" ]] || {
    echo 'Stop the policy server before conversion.' >&2
    exit 1
}
ss -H -ltn "sport = :$policy_port" | grep -q . && { echo 'The policy port is occupied.' >&2; exit 1; }
run_prefix=".${checkpoint_label}_pytorch.${expected_sha:0:12}.run."
run_root="$(mktemp -d "$parent/${run_prefix}XXXXXX")"
run_root="$(realpath -e -- "$run_root")"
[[ "$(dirname -- "$run_root")" == "$parent" && "$(basename -- "$run_root")" == "$run_prefix"* ]] || {
    echo 'Temporary conversion path failed containment checks.' >&2
    exit 1
}
chmod 700 "$run_root"
printf '%s\n%s\n' "$expected_sha" "$profile" >"$run_root/owner"
chmod 600 "$run_root/owner"
temporary_checkpoint="$run_root/checkpoint.partial"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$repo/.runtime/conversion/$run_id"
mkdir -m 700 "$evidence"
printf '__ALOHA_REMOTE_EVIDENCE__=.runtime/conversion/%s\n' "$run_id"
sampler_pid=
published=no

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    if [[ "$published" != yes && -d "$run_root" && ! -L "$run_root" ]] &&
        [[ "$(dirname -- "$run_root")" == "$parent" ]] &&
        [[ "$(basename -- "$run_root")" == "$run_prefix"* ]] &&
        [[ "$(cat "$run_root/owner" 2>/dev/null || true)" == "$expected_sha"$'\n'"$profile" ]]; then
        rm -rf -- "$run_root"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

run_measured() {
    name="$1"
    deadline="$2"
    shift 2
    set +e
    /usr/bin/time -v -o "$evidence/$name.time" \
        timeout --signal=TERM --kill-after=30s "${deadline}s" "$@" >"$evidence/$name.log" 2>&1
    status=$?
    set -e
    if (( status != 0 )); then
        tail -40 "$evidence/$name.log" >&2
        return "$status"
    fi
}

probe_rss=0
if [[ "$restore_mode" == partial-bfloat16 ]]; then
    run_measured probe 600 env JAX_PLATFORMS=cpu "$repo/.venv/bin/python" \
        "$repo/examples/convert_jax_model_to_pytorch.py" \
        --checkpoint-dir "$source_checkpoint" \
        --config-name "$config_name" \
        --precision bfloat16 \
        --restore-mode partial-bfloat16 \
        --partial-probe-only
    probe_rss="$(awk -F: '/Maximum resident set size/ {gsub(/ /, "", $2); print $2}' "$evidence/probe.time")"
fi

smi="$(command -v nvidia-smi || true)"
[[ -n "$smi" ]] || [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]] || smi=/usr/lib/wsl/lib/nvidia-smi
[[ -n "$smi" ]] || { echo 'nvidia-smi is unavailable.' >&2; exit 1; }
timeout 6600s "$smi" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=500 \
    >"$evidence/gpu.csv" 2>"$evidence/gpu.err" &
sampler_pid=$!

run_measured full 6300 env JAX_PLATFORMS=cpu "$repo/.venv/bin/python" \
    "$repo/examples/convert_jax_model_to_pytorch.py" \
    --checkpoint-dir "$source_checkpoint" \
    --config-name "$config_name" \
    --output-path "$temporary_checkpoint" \
    --precision bfloat16 \
    --restore-mode "$restore_mode"

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
[[ -s "$temporary_checkpoint/config.json" && -d "$temporary_checkpoint/assets" ]] || {
    echo 'Converted config or assets are missing.' >&2
    exit 1
}
if [[ "$restore_mode" == partial-bfloat16 ]]; then
    [[ -s "$temporary_checkpoint/model.safetensors.index.json" ]] || {
        echo 'Sharded index is missing.' >&2
        exit 1
    }
    find "$temporary_checkpoint" -maxdepth 1 -type f -name 'model-*.safetensors' -print -quit | grep -q . || {
        echo 'Converted weight shards are missing.' >&2
        exit 1
    }
else
    [[ -s "$temporary_checkpoint/model.safetensors" ]] || {
        echo 'Converted monolithic weights are missing.' >&2
        exit 1
    }
fi
find "$temporary_checkpoint" -type l -print -quit | grep -q . && {
    echo 'Converted artifact must not contain symlinks.' >&2
    exit 1
}
model_hash="$(cd "$temporary_checkpoint" && find . -type f -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
full_rss="$(awk -F: '/Maximum resident set size/ {gsub(/ /, "", $2); print $2}' "$evidence/full.time")"
gpu_peak="$(awk -F, 'BEGIN {m=0} {gsub(/ /, "", $1); if ($1+0>m) m=$1+0} END {print m}' "$evidence/gpu.csv")"
gpu_samples="$(wc -l <"$evidence/gpu.csv" | tr -d ' ')"
for metric in "$full_rss" "$gpu_peak" "$gpu_samples"; do
    [[ "$metric" =~ ^[0-9]+$ ]] && (( 10#$metric > 0 )) || {
        echo 'Conversion resource evidence is missing or invalid.' >&2
        exit 1
    }
done
[[ "$probe_rss" =~ ^[0-9]+$ ]] && {
    [[ "$restore_mode" == full-float32 && "$probe_rss" == 0 ]] || (( 10#$probe_rss > 0 ))
} || { echo 'Conversion probe evidence is invalid.' >&2; exit 1; }
awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/) exit 1}' \
    "$evidence/gpu.csv" || { echo 'GPU sampling evidence is invalid.' >&2; exit 1; }
[[ ! -s "$evidence/gpu.err" ]] || { echo 'GPU sampling reported an error.' >&2; exit 1; }

mv -- "$temporary_checkpoint" "$final_checkpoint"
published=yes
rm -f -- "$run_root/owner"
rmdir -- "$run_root" || true
printf '__ALOHA_CONVERSION__=passed\n'
printf '__ALOHA_PROFILE__=%s\n' "$profile"
printf '__ALOHA_PROJECT_SHA__=%s\n' "$expected_sha"
printf '__ALOHA_CONVERSION_RESTORE_MODE__=%s\n' "$restore_mode"
printf '__ALOHA_AVAILABLE_RAM_KIB__=%s\n' "$available_ram_kib"
printf '__ALOHA_MODEL_HASH__=%s\n' "$model_hash"
printf '__ALOHA_PROBE_MAX_RSS_KIB__=%s\n' "$probe_rss"
printf '__ALOHA_FULL_MAX_RSS_KIB__=%s\n' "$full_rss"
printf '__ALOHA_GPU_PEAK_MIB__=%s\n' "$gpu_peak"
printf '__ALOHA_GPU_SAMPLES__=%s\n' "$gpu_samples"
printf '__ALOHA_CONVERSION_PARTIAL__=absent\n'
