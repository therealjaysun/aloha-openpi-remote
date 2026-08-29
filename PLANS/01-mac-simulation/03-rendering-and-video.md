# 01.03 — Rendering and video

- **Objective:** Capture offscreen frames and a valid episode video natively on macOS.
- **Inputs/prerequisites:** Passing smoke environment; writable ignored output directory.
- **Implementation tasks:** Reuse stock image resize/conversion and `VideoSaver`; test the official lock and default backend first with no `MUJOCO_GL`; never use EGL on Mac; if the pinned `imageio-ffmpeg==0.5.1` cannot install/run natively, apply the narrow Mac-only `imageio-ffmpeg==0.6.0` override (verify its arm64 wheel at implementation) before considering Homebrew or Rosetta; save a PNG smoke frame and MP4; verify dimensions/frame count/fps; add optional display only if stable and requested later; bound retained artifacts.
- **Files expected to change:** `tools/remote_aloha/sim_smoke_test.py`, `scripts/smoke_sim.sh`, `.gitignore`; minimal patch to `VideoSaver` only if a proven bug blocks output.
- **Validation:** imageio reads first/last frame; expected fps 50; `ffprobe` when installed; generated files stay untracked.
- **Acceptance:** Nonempty image/video, readable frames, correct uint8 RGB, no EGL setting, clean renderer shutdown.
- **Planned commit:** `feat(render): record offscreen ALOHA episodes`.
- **Actual findings:** Stock gym render is RGB-array only; stock saver records the policy-ready 224×224 `cam_high` frames at 50 fps. Interactive human mode is not implemented by gym-aloha. The pinned imageio-ffmpeg 0.5.1 lacks a macOS arm64 wheel; 0.6.0 is the first planned narrow fallback, subject to implementation-time index verification.
- **Remaining blockers:** FFmpeg/ffprobe are absent locally and the actual native writer/render path has not run.
- **Completion status:** Planned.
