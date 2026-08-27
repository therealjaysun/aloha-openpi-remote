from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import load_mac_sim_config

EXPECTED_RAW_IMAGE_SHAPE = (480, 640, 3)
EXPECTED_POLICY_IMAGE_SHAPE = (3, 224, 224)
EXPECTED_STATE_SHAPE = (14,)
EXPECTED_FPS = 50.0
EPISODE_STEPS = 300
P95_BUDGET_MS = 20.0


def validate_observation(observation: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observation, dict):
        raise ValueError("observation must be a dictionary")
    try:
        image = np.asarray(observation["pixels"]["top"])
        state = np.asarray(observation["agent_pos"])
    except (KeyError, TypeError) as error:
        raise ValueError("observation must contain pixels.top and agent_pos") from error
    if image.shape != EXPECTED_RAW_IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"pixels.top must be uint8 {EXPECTED_RAW_IMAGE_SHAPE}, got {image.dtype} {image.shape}")
    if state.shape != EXPECTED_STATE_SHAPE or not np.isfinite(state).all():
        raise ValueError(f"agent_pos must be finite with shape {EXPECTED_STATE_SHAPE}")
    return image, state


def validate_action(action: np.ndarray) -> None:
    if action.shape != EXPECTED_STATE_SHAPE or not np.isfinite(action).all():
        raise ValueError(f"action must be finite with shape {EXPECTED_STATE_SHAPE}")


def percentile_ms(samples_ms: list[float], percentile: float) -> float:
    if not samples_ms:
        raise ValueError("latency samples cannot be empty")
    return float(np.percentile(np.asarray(samples_ms), percentile))


def _policy_image(image: np.ndarray) -> np.ndarray:
    from openpi_client import image_tools

    resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    converted = np.transpose(resized, (2, 0, 1))
    if converted.shape != EXPECTED_POLICY_IMAGE_SHAPE or converted.dtype != np.uint8:
        raise ValueError("policy image conversion produced an invalid result")
    return converted


def _package_versions() -> dict[str, str]:
    names = ("dm-control", "gym-aloha", "gymnasium", "imageio", "imageio-ffmpeg", "mujoco", "numpy")
    return {name: importlib.metadata.version(name) for name in names}


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def _verify_video(path: Path, expected_frames: int) -> dict[str, Any]:
    import imageio.v2 as imageio

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"video was not created: {path}")
    reader = imageio.get_reader(path)
    try:
        frame_count = reader.count_frames()
        metadata = reader.get_meta_data()
        first = np.asarray(reader.get_data(0))
        last = np.asarray(reader.get_data(frame_count - 1))
    finally:
        reader.close()
    if frame_count != expected_frames:
        raise ValueError(f"video has {frame_count} frames; expected {expected_frames}")
    if first.shape != (224, 224, 3) or last.shape != first.shape:
        raise ValueError(f"video frames have invalid dimensions: {first.shape}, {last.shape}")
    fps = float(metadata.get("fps", 0.0))
    if abs(fps - EXPECTED_FPS) > 0.1:
        raise ValueError(f"video reports {fps} fps; expected {EXPECTED_FPS}")
    return {"bytes": path.stat().st_size, "fps": fps, "frames": frame_count, "shape": list(first.shape)}


def _run_episode(config: MacSimConfig, seed: int, run_dir: Path, *, record: bool) -> tuple[dict[str, Any], list[float]]:
    import gym_aloha  # noqa: F401
    import gymnasium
    import imageio.v2 as imageio

    env = gymnasium.make(config.task, obs_type="pixels_agent_pos", render_mode="rgb_array")
    saver = None
    video_dir = run_dir / f"seed-{seed}"
    latencies_ms: list[float] = []
    result: dict[str, Any] = {"seed": seed, "steps": 0, "max_reward": 0.0}
    try:
        if (
            env.metadata.get("render_fps") != EXPECTED_FPS
            or env.spec is None
            or env.spec.max_episode_steps != EPISODE_STEPS
        ):
            raise ValueError("pinned simulator must expose a 50 fps, 300-step contract")
        observation, info = env.reset(seed=seed)
        image, state = validate_observation(observation)
        if tuple(env.action_space.shape) != EXPECTED_STATE_SHAPE:
            raise ValueError(f"action space must have shape {EXPECTED_STATE_SHAPE}")
        if record:
            from examples.aloha_sim.saver import VideoSaver

            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(video_dir / "reset.png", image)
            saver = VideoSaver(video_dir)
            saver.on_episode_start()

        source_success = "is_success" in info
        success_seen = bool(info.get("is_success", False)) if source_success else None
        terminated = truncated = False
        for step in range(1, EPISODE_STEPS + 1):
            action = np.asarray(state, dtype=np.float64).copy()
            validate_action(action)
            started = time.perf_counter_ns()
            observation, reward, terminated, truncated, info = env.step(action)
            image, state = validate_observation(observation)
            policy_image = _policy_image(image)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            if saver is not None:
                saver.on_step({"images": {"cam_high": policy_image}}, {"actions": action})
            result["steps"] = step
            result["max_reward"] = max(float(result["max_reward"]), float(reward))
            if "is_success" in info:
                source_success = True
                success_seen = bool(success_seen) or bool(info["is_success"])
            if terminated or truncated:
                break

        if saver is not None:
            saver.on_episode_end()
            video_path = video_dir / "out_0.mp4"
            result["artifacts"] = {
                "reset_frame": str((video_dir / "reset.png").relative_to(run_dir)),
                "video": str(video_path.relative_to(run_dir)),
                "video_validation": _verify_video(video_path, int(result["steps"])),
            }
        result.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success_metric_available": source_success,
                "success_seen": success_seen,
                "latency_ms": {
                    "p50": percentile_ms(latencies_ms, 50),
                    "p95": percentile_ms(latencies_ms, 95),
                },
            }
        )
        return result, latencies_ms
    finally:
        env.close()


def run(config: MacSimConfig, *, enforce_budget: bool = True) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    run_dir = config.output_dir / "phase01" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "task": config.task,
        "seeds": list(range(config.seed, config.seed + config.episodes)),
        "project_sha": _git_sha(),
        "machine": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine()},
        "python": platform.python_version(),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        "package_versions": _package_versions(),
        "episodes": [],
    }
    all_latencies: list[float] = []
    try:
        for index, seed in enumerate(manifest["seeds"]):
            episode, latencies = _run_episode(config, seed, run_dir, record=index == 0)
            manifest["episodes"].append(episode)
            all_latencies.extend(latencies)
        aggregate = {"p50": percentile_ms(all_latencies, 50), "p95": percentile_ms(all_latencies, 95)}
        manifest["step_render_convert_latency_ms"] = aggregate
        manifest["p95_budget_ms"] = P95_BUDGET_MS
        manifest["full_300_step_episode"] = any(episode["steps"] == EPISODE_STEPS for episode in manifest["episodes"])
        if not manifest["full_300_step_episode"]:
            raise RuntimeError("no full 300-step episode completed")
        if enforce_budget and aggregate["p95"] > P95_BUDGET_MS:
            raise RuntimeError(f"p95 latency {aggregate['p95']:.3f} ms exceeds {P95_BUDGET_MS:.1f} ms")
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 01 manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native Mac ALOHA simulation smoke test.")
    parser.add_argument("--no-enforce-budget", action="store_true", help="Record but do not fail the 20 ms p95 gate.")
    args = parser.parse_args()
    manifest = run(load_mac_sim_config(), enforce_budget=not args.no_enforce_budget)
    print(f"Phase 01 simulation passed: {manifest}")


if __name__ == "__main__":
    main()
