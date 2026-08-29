# 01.03 — Rendering and video

- **Objective:** Capture offscreen frames and a valid episode video natively on macOS.
- **Inputs/prerequisites:** Passing smoke environment; writable ignored output directory.
- **Implementation tasks:** Reuse stock image resize/conversion and `VideoSaver`; test the official lock and default backend first with no `MUJOCO_GL`; never use EGL on Mac; if the pinned `imageio-ffmpeg==0.5.1` cannot install/run natively, apply the narrow Mac-only `imageio-ffmpeg==0.6.0` override (verify its arm64 wheel at implementation) before considering Homebrew or Rosetta; save a PNG smoke frame and MP4; verify dimensions/frame count/fps; add optional display only if stable and requested later; bound retained artifacts.
- **Files expected to change:** `tools/remote_aloha/sim_smoke_test.py`, `scripts/smoke_sim.sh`, `.gitignore`; minimal patch to `VideoSaver` only if a proven bug blocks output.
- **Validation:** imageio reads first/last frame; expected fps 50; `ffprobe` when installed; generated files stay untracked.
- **Acceptance:** Nonempty image/video, readable frames, correct uint8 RGB, no EGL setting, clean renderer shutdown.
- **Planned commit:** `feat(render): record offscreen ALOHA episodes`.
- **Actual findings:** Stock `VideoSaver` encoded one bounded episode per run. ImageIO decoded first/last frames and counted 300 frames at 224×224 and 50 fps; the saved reset PNG is RGB 640×480. The latest ignored MP4 SHA-256 is `4c3d283559aab448c87c72223c51e36572900cdf8c0db1d5cd8017e4c47925f2`. Bundled FFmpeg 7.1 from `imageio-ffmpeg==0.6.0` was used; system `ffprobe` remains optional and absent.
- **Remaining blockers:** None; external ffprobe was not needed because the actual decoder validated frame count, dimensions, and fps.
- **Completion status:** Complete; evidence `E-MAC01`.
