#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_root/examples/aloha_sim/.venv/bin/python"

[[ -x "$python" ]] || { echo 'Missing project test environment; run: make setup-mac' >&2; exit 1; }
cd "$repo_root"
exec "$python" -m tools.remote_aloha.remote doctor
