#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_root/examples/aloha_sim/.venv/bin/python"

cd "$repo_root"
[[ "${MUJOCO_GL-}" != "egl" ]] || { echo 'MUJOCO_GL=egl is unsupported on macOS; unset it.' >&2; exit 1; }
[[ -x "$python" ]] || { echo 'Missing simulator environment; run: make setup-mac' >&2; exit 1; }
exec env -u MUJOCO_GL "$python" -m tools.remote_aloha.run "$@"
