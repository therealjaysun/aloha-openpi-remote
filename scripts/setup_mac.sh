#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$repo_root/examples/aloha_sim/.venv"
python="$venv/bin/python"
export UV_CACHE_DIR="$repo_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$repo_root/.cache/uv/python"

cd "$repo_root"
command -v uv >/dev/null || { echo 'uv is required: https://docs.astral.sh/uv/' >&2; exit 1; }
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if [[ -x "$python" ]] && ! "$python" -c 'import platform, sys; raise SystemExit(not (sys.version_info[:2] == (3, 10) and platform.machine() == "arm64"))'; then
    echo "Existing $venv is not native arm64 Python 3.10; move it aside and rerun make setup-mac." >&2
    exit 1
fi

if [[ ! -x "$python" ]]; then
    uv venv --no-project --python 3.10 "$venv"
fi
uv pip sync --python "$python" examples/aloha_sim/requirements.txt
uv pip install --python "$python" --require-hashes -r requirements/project-test.txt
uv pip install --python "$python" --no-deps -e packages/openpi-client

mujoco_cgl="$venv/lib/python3.10/site-packages/mujoco/cgl/cgl.py"
old_opengl_path='/System/Library/OpenGL.framework/OpenGL'
new_opengl_path='/System/Library/Frameworks/OpenGL.framework/OpenGL'
if [[ ! -e "$old_opengl_path" && -d /System/Library/Frameworks/OpenGL.framework ]] \
    && grep -Fq "$old_opengl_path" "$mujoco_cgl"; then
    echo 'Pinned MuJoCo 2.3.7 uses the pre-macOS-26 OpenGL path; applying the native framework-path compatibility patch.'
    sed -i '' "s#$old_opengl_path#$new_opengl_path#" "$mujoco_cgl"
fi

if ! "$python" -c 'import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()' >/dev/null 2>&1; then
    echo 'Pinned imageio-ffmpeg 0.5.1 has no usable arm64 binary; applying the planned 0.6.0 Mac-only override.'
    uv pip install --python "$python" 'imageio-ffmpeg==0.6.0'
fi

uv pip check --python "$python"
"$python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 10); assert platform.machine() == "arm64"'
echo "Mac simulator environment ready: $python"
