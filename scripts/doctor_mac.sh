#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_root/examples/aloha_sim/.venv/bin/python"

cd "$repo_root"
[[ "$(uname -s)" == "Darwin" ]] || { echo 'doctor-mac requires macOS.' >&2; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || { echo 'doctor-mac requires a native arm64 shell.' >&2; exit 1; }
[[ "${MUJOCO_GL-}" != "egl" ]] || { echo 'MUJOCO_GL=egl is unsupported on macOS; unset it.' >&2; exit 1; }
[[ -x "$python" ]] || { echo 'Missing simulator environment; run: make setup-mac' >&2; exit 1; }

echo "macOS $(sw_vers -productVersion), $(uname -m)"
memory="$(sysctl -n hw.memsize 2>/dev/null || true)"
if [[ -z "$memory" ]]; then
    memory="$(system_profiler SPHardwareDataType | awk -F': ' '/Memory:/ {print $2; exit}')"
fi
echo "Memory: $memory"
echo "Workspace disk: $(df -h . | awk 'NR == 2 {print $4 " free of " $2}')"
echo "Git commit: $(git rev-parse HEAD)"
uv --version
file -L "$python"

env -u MUJOCO_GL "$python" -c '
import importlib.metadata
import platform
import sys

import gym_aloha
import imageio_ffmpeg
import mujoco
import openpi_client

assert sys.version_info[:2] == (3, 10)
assert platform.machine() == "arm64"
assert importlib.metadata.version("gym-aloha") == "0.1.1"
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 64, 64)
try:
    renderer.update_scene(data)
    assert renderer.render().shape == (64, 64, 3)
finally:
    renderer._gl_context.free()
print("Python", platform.python_version())
print("MuJoCo", importlib.metadata.version("mujoco"))
print("gym-aloha", importlib.metadata.version("gym-aloha"))
print("FFmpeg", imageio_ffmpeg.get_ffmpeg_version())
'

echo 'Mac doctor passed.'
