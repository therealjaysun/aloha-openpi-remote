from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.observation_contract import POLICY_CAMERA_VIEWS
from tools.remote_aloha.observation_contract import convert_gym_observation
from tools.remote_aloha.scenarios import CALIBRATION_MAX_HEIGHT_ERROR_METERS
from tools.remote_aloha.scenarios import CALIBRATION_MIN_PUSH_METERS
from tools.remote_aloha.scenarios import CALIBRATION_SEGMENT_STEPS
from tools.remote_aloha.scenarios import CANONICAL_LAYOUTS
from tools.remote_aloha.scenarios import COLOR_MASK_RULES
from tools.remote_aloha.scenarios import DISPLAY_EVERY_STEPS
from tools.remote_aloha.scenarios import LEFT_FINGER_GEOMS
from tools.remote_aloha.scenarios import LEFT_PUSH_WAYPOINTS
from tools.remote_aloha.scenarios import MIN_VISIBLE_PIXELS
from tools.remote_aloha.scenarios import PARKED_JOINT_TOLERANCE
from tools.remote_aloha.scenarios import PUSHER_POSITION
from tools.remote_aloha.scenarios import RIGHT_FINGER_GEOMS
from tools.remote_aloha.scenarios import RIGHT_PUSH_WAYPOINTS
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import body_descriptors
from tools.remote_aloha.scenarios import descriptor_payload
from tools.remote_aloha.scenarios import descriptor_sha256
from tools.remote_aloha.scenarios import project_action

EXPECTED_RAW_IMAGE_SHAPE = (480, 640, 3)
EXPECTED_POLICY_IMAGE_SHAPE = (3, 224, 224)
EXPECTED_STATE_SHAPE = (14,)
EXPECTED_FPS = 50.0
EPISODE_STEPS = 300
P95_BUDGET_MS = 20.0


def _atomic_manifest(path: Path, payload: object) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def calibration_commands(home: np.ndarray, side: str) -> list[np.ndarray]:
    validate_action(home)
    if side == "left":
        arm = slice(0, 6)
        waypoints = LEFT_PUSH_WAYPOINTS
    elif side == "right":
        arm = slice(7, 13)
        waypoints = RIGHT_PUSH_WAYPOINTS
    else:
        raise ValueError("calibration side must be left or right")
    start = home[arm].copy()
    commands = []
    for target_values, steps in zip(waypoints, CALIBRATION_SEGMENT_STEPS, strict=True):
        target = np.asarray(target_values, dtype=np.float64)
        for step in range(1, steps + 1):
            command = home.copy()
            command[arm] = start + (target - start) * (step / steps)
            command[6] = command[13] = PUSHER_POSITION
            commands.append(command)
        start = target
    return commands


def _policy_segmentation(segmentation: np.ndarray) -> np.ndarray:
    from PIL import Image

    if segmentation.shape != (480, 640, 2):
        raise ValueError("segmentation render has an invalid shape")
    geom_ids = segmentation[..., 0]
    resized = Image.fromarray(geom_ids.astype(np.int32), mode="I").resize((224, 168), Image.Resampling.NEAREST)
    result = np.full((224, 224), -1, dtype=np.int32)
    result[28:196] = np.asarray(resized, dtype=np.int32)
    return result


def _color_mask(frame: np.ndarray, body_name: str) -> np.ndarray:
    try:
        rules = COLOR_MASK_RULES[body_name]
    except KeyError as error:
        raise ValueError("unknown calibration body") from error
    channels = {name: frame[..., index].astype(np.int16) for index, name in enumerate("rgb")}
    return np.logical_and.reduce([channels[first] - channels[second] >= minimum for first, second, minimum in rules])


def validate_visibility(environment: object, image: np.ndarray, object_kind: str) -> dict[str, int]:
    policy_frame = np.transpose(_policy_image(image), (1, 2, 0))
    physics = environment.unwrapped._env.physics  # noqa: SLF001
    segmentation = _policy_segmentation(physics.render(height=480, width=640, camera_id="top", segmentation=True))
    counts = {}
    for body in body_descriptors(object_kind):
        mask = _color_mask(policy_frame, body.name)
        for role, prefix in (("movable", f"push_pi/{body.name}_"), ("target", f"push_pi/target_{body.name}_")):
            geom_ids = [
                index
                for index in range(physics.model.ngeom)
                if (physics.model.id2name(index, "geom") or "").startswith(prefix)
            ]
            count = int(np.count_nonzero(mask & np.isin(segmentation, geom_ids)))
            counts[f"{body.name}_{role}_pixels"] = count
            if count < MIN_VISIBLE_PIXELS:
                raise ValueError(f"{body.name} {role} has only {count} visible policy pixels")
    return counts


def calibration_contact_seen(contacts: list[tuple[str, str]], body_name: str, side: str) -> bool:
    movable = {f"push_pi/{body_name}_{index}" for index, _ in enumerate(body_descriptors("pi")[0].parts)}
    if body_name in {"P", "I"}:
        descriptor = next(body for body in body_descriptors("letters") if body.name == body_name)
        movable = {f"push_pi/{body_name}_{index}" for index, _ in enumerate(descriptor.parts)}
    fingers = LEFT_FINGER_GEOMS if side == "left" else RIGHT_FINGER_GEOMS if side == "right" else frozenset()
    return any(
        (first in movable and second in fingers) or (second in movable and first in fingers)
        for first, second in contacts
    )


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


def package_versions() -> dict[str, str]:
    names = (
        "dm-control",
        "gym-aloha",
        "gymnasium",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "mujoco",
        "numpy",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _git_sha(*, require_clean: bool = False) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10)
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.stdout.strip():
            raise RuntimeError("Push-PI release calibration requires a clean exact-candidate checkout")
    return result.stdout.strip()


def verify_video(
    path: Path,
    expected_frames: int,
    expected_shape: tuple[int, int, int] = (224, 224, 3),
) -> dict[str, Any]:
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
    if first.shape != expected_shape or last.shape != first.shape:
        raise ValueError(f"video frames have invalid dimensions: {first.shape}, {last.shape}")
    fps = float(metadata.get("fps", 0.0))
    if abs(fps - EXPECTED_FPS) > 0.1:
        raise ValueError(f"video reports {fps} fps; expected {EXPECTED_FPS}")
    return {"bytes": path.stat().st_size, "fps": fps, "frames": frame_count, "shape": list(first.shape)}


def _atomic_png(path: Path, frame: np.ndarray) -> None:
    import imageio.v2 as imageio

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        imageio.imwrite(temporary, frame)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_calibration(output_dir: Path) -> Path:
    import gymnasium

    import examples.aloha_sim.push_pi_env  # noqa: F401

    project_sha = _git_sha(require_clean=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    run_dir = validate_output_root(output_dir) / "scenarios_0827" / "calibration" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "project_sha": project_sha,
        "descriptor_sha256": descriptor_sha256(),
        "descriptor": descriptor_payload(),
        "random_visibility": [],
        "canonical_pushes": [],
    }
    _atomic_manifest(manifest_path, manifest)
    cases = (
        ("pi_left", "push_pi_dual", "pi", "pi", "left"),
        ("pi_right", "push_pi_dual", "pi", "pi", "right"),
        ("P_left", "push_letters_dual", "letters", "P", "left"),
        ("I_right", "push_letters_dual", "letters", "I", "right"),
    )
    try:
        for object_kind, scenario in (("pi", "push_pi_single"), ("letters", "push_letters_single")):
            for seed in range(3):
                environment = gymnasium.make(
                    SCENARIOS[scenario].gym_id,
                    obs_type="pixels_agent_pos",
                    render_mode="rgb_array",
                )
                try:
                    observation, _ = environment.reset(seed=seed)
                    image, _ = validate_observation(observation)
                    counts = validate_visibility(environment, image, object_kind)
                    manifest["random_visibility"].append({"scenario": scenario, "seed": seed, "counts": counts})
                finally:
                    environment.close()
                _atomic_manifest(manifest_path, manifest)

        for name, scenario, object_kind, pushed_body, side in cases:
            environment = gymnasium.make(
                SCENARIOS[scenario].gym_id,
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
            )
            try:
                observation, reset_info = environment.reset(seed=0, options={"layout": CANONICAL_LAYOUTS[name]})
                image, home = validate_observation(observation)
                visibility = validate_visibility(environment, image, object_kind)
                reset_id = f"{name}-reset.png"
                _atomic_png(run_dir / reset_id, np.transpose(_policy_image(image), (1, 2, 0)))
                task = environment.unwrapped._push_task  # noqa: SLF001
                physics = environment.unwrapped._env.physics  # noqa: SLF001
                start_y = task._body_states(physics)[pushed_body].y  # noqa: SLF001
                body_index = [body.name for body in task.bodies].index(pushed_body)
                parked = slice(7, 13) if side == "left" else slice(0, 6)
                parked_error = 0.0
                max_height_error = 0.0
                minimum_xy_error = math.inf
                minimum_yaw_error = math.inf
                time_to_success_step = None
                named_contact = False
                final_info = reset_info
                commands = calibration_commands(home, side)
                for step, command in enumerate(commands, start=1):
                    observation, _, terminated, truncated, final_info = environment.unwrapped.step(command)
                    image, state = validate_observation(observation)
                    named_contact = named_contact or calibration_contact_seen(
                        task._contacts(physics),  # noqa: SLF001
                        pushed_body,
                        side,
                    )
                    parked_error = max(parked_error, float(np.max(np.abs(state[parked] - home[parked]))))
                    max_height_error = max(max_height_error, float(final_info[f"body_{body_index}_height_error"]))
                    minimum_xy_error = min(minimum_xy_error, float(final_info[f"body_{body_index}_xy_error"]))
                    minimum_yaw_error = min(minimum_yaw_error, float(final_info[f"body_{body_index}_yaw_error"]))
                    if time_to_success_step is None and final_info["is_success"] is True:
                        time_to_success_step = step
                    if terminated:
                        raise ValueError(f"{name} calibration terminated before completing its waypoints")
                    if bool(truncated) != (step == EPISODE_STEPS):
                        raise ValueError(f"{name} calibration violated the exact 300-step time limit")
                final_y = task._body_states(physics)[pushed_body].y  # noqa: SLF001
                push_distance = float(final_y - start_y)
                contact_key = f"{side}_contact_ever"
                if (
                    not named_contact
                    or final_info[contact_key] is not True
                    or push_distance < CALIBRATION_MIN_PUSH_METERS
                    or max_height_error > CALIBRATION_MAX_HEIGHT_ERROR_METERS
                    or final_info["lifted_ever"] is True
                    or final_info["fallen"] is True
                    or final_info["off_table"] is True
                    or parked_error > PARKED_JOINT_TOLERANCE
                    or final_info["terminal_reason"] != "time_limit"
                ):
                    raise ValueError(f"{name} canonical push missed its calibration bounds")
                final_id = f"{name}-final.png"
                _atomic_png(run_dir / final_id, np.transpose(_policy_image(image), (1, 2, 0)))
                manifest["canonical_pushes"].append(
                    {
                        "case": name,
                        "scenario": scenario,
                        "body": pushed_body,
                        "side": side,
                        "steps": len(commands),
                        "push_distance_meters": push_distance,
                        "max_height_error_meters": max_height_error,
                        "minimum_xy_error_meters": minimum_xy_error,
                        "minimum_yaw_error_radians": minimum_yaw_error,
                        "final_xy_error_meters": final_info[f"body_{body_index}_xy_error"],
                        "final_yaw_error_radians": final_info[f"body_{body_index}_yaw_error"],
                        "time_to_success_step": time_to_success_step,
                        "parked_joint_error_radians": parked_error,
                        "named_contact": named_contact,
                        "visibility": visibility,
                        "reset_frame_id": reset_id,
                        "final_frame_id": final_id,
                    }
                )
            finally:
                environment.close()
            _atomic_manifest(manifest_path, manifest)
        if {item["case"] for item in manifest["canonical_pushes"]} != {case[0] for case in cases}:
            raise RuntimeError("canonical calibration coverage is incomplete")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        raise
    finally:
        _atomic_manifest(manifest_path, manifest)
        print(f"Push-PI calibration manifest: {manifest_path}")
    return manifest_path


def _run_episode(config: MacSimConfig, seed: int, run_dir: Path, *, record: bool) -> tuple[dict[str, Any], list[float]]:
    import gym_aloha  # noqa: F401
    import gymnasium
    import imageio.v2 as imageio

    if config.scenario.is_custom:
        import examples.aloha_sim.push_pi_env  # noqa: F401

    env = gymnasium.make(config.task, obs_type="pixels_agent_pos", render_mode="rgb_array")
    saver = None
    from examples.aloha_sim.saver import LiveDisplay

    display = LiveDisplay(
        enabled=config.display,
        every_steps=DISPLAY_EVERY_STEPS,
        camera_views=POLICY_CAMERA_VIEWS,
    )
    video_dir = run_dir / f"seed-{seed}"
    latencies_ms: list[float] = []
    result: dict[str, Any] = {
        "seed": seed,
        "steps": 0,
        "max_reward": 0.0,
        "left_peak_joint_error": 0.0,
        "right_peak_joint_error": 0.0,
    }
    try:
        if (
            env.metadata.get("render_fps") != EXPECTED_FPS
            or env.spec is None
            or env.spec.max_episode_steps != EPISODE_STEPS
        ):
            raise ValueError("pinned simulator must expose a 50 fps, 300-step contract")
        observation, info = env.reset(seed=seed)
        image, state = validate_observation(observation)
        home = state.copy()
        if config.scenario.is_custom:
            result["reset_info"] = info
            result["reset_visibility"] = validate_visibility(env, image, str(config.scenario.object_kind))
        display.on_episode_start()
        if tuple(env.action_space.shape) != EXPECTED_STATE_SHAPE:
            raise ValueError(f"action space must have shape {EXPECTED_STATE_SHAPE}")
        if record:
            from examples.aloha_sim.saver import VideoSaver

            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(video_dir / "reset.png", image)
            saver = VideoSaver(video_dir, camera_views=POLICY_CAMERA_VIEWS)
            saver.on_episode_start()

        source_success = "is_success" in info
        success_seen = bool(info.get("is_success", False)) if source_success else None
        terminated = truncated = False
        for step in range(1, EPISODE_STEPS + 1):
            action = project_action(home if config.scenario.is_custom else state, config.scenario, home)
            validate_action(action)
            started = time.perf_counter_ns()
            observation, reward, terminated, truncated, info = env.step(action)
            image, state = validate_observation(observation)
            policy_image = _policy_image(image)
            result["left_peak_joint_error"] = max(
                float(result["left_peak_joint_error"]), float(np.max(np.abs(state[:6] - home[:6])))
            )
            result["right_peak_joint_error"] = max(
                float(result["right_peak_joint_error"]), float(np.max(np.abs(state[7:13] - home[7:13])))
            )
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            policy_images = {"cam_high": policy_image}
            if saver is not None or config.display:
                physics = env.unwrapped._env.physics  # noqa: SLF001 - pinned gym-aloha has no camera API
                composite_observation = {
                    **observation,
                    "pixels": {
                        "top": image,
                        "left_wrist": physics.render(height=480, width=640, camera_id="left_wrist"),
                        "right_wrist": physics.render(height=480, width=640, camera_id="right_wrist"),
                    },
                }
                policy_images = convert_gym_observation(composite_observation)["images"]
            if saver is not None:
                saver.on_step({"images": policy_images}, {"actions": action})
            display.on_step({"images": policy_images}, {"actions": action})
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
                "video_validation": verify_video(video_path, int(result["steps"]), (224, 672, 3)),
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
        if config.scenario.is_custom:
            result["final_info"] = info
        return result, latencies_ms
    finally:
        display.on_episode_end()
        env.close()


def run(config: MacSimConfig, *, enforce_budget: bool = True) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    output = validate_output_root(config.output_dir)
    run_dir = (
        output / "scenarios_0827" / "smoke" / timestamp / config.scenario.key
        if config.scenario.is_custom
        else output / "phase01" / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "task": config.task,
        "scenario": config.scenario.key,
        "seeds": list(range(config.seed, config.seed + config.episodes)),
        "project_sha": _git_sha(),
        "machine": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine()},
        "python": platform.python_version(),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        "package_versions": package_versions(),
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
        if config.scenario.is_custom:
            manifest["full_300_step_episode"] = all(
                episode["steps"] == EPISODE_STEPS
                and episode["terminated"] is False
                and episode["truncated"] is True
                and episode["final_info"]["terminal_reason"] == "time_limit"
                for episode in manifest["episodes"]
            )
            if not manifest["full_300_step_episode"]:
                raise RuntimeError("every custom hold episode must truncate cleanly at exactly 300 steps")
            if config.scenario.arm_mode == "left" and any(
                episode["right_peak_joint_error"] > PARKED_JOINT_TOLERANCE for episode in manifest["episodes"]
            ):
                raise RuntimeError("inactive arm exceeded the frozen parked-joint tolerance")
        else:
            manifest["full_300_step_episode"] = any(
                episode["steps"] == EPISODE_STEPS for episode in manifest["episodes"]
            )
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
        _atomic_manifest(manifest_path, manifest)
        print(f"Phase 01 manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native Mac ALOHA simulation smoke test.")
    parser.add_argument("--no-enforce-budget", action="store_true", help="Record but do not fail the 20 ms p95 gate.")
    parser.add_argument("--calibrate", action="store_true", help="Run the fixed Push-PI Mac calibration gate.")
    args = parser.parse_args()
    config = load_mac_sim_config()
    manifest = (
        run_calibration(config.output_dir) if args.calibrate else run(config, enforce_budget=not args.no_enforce_budget)
    )
    print(f"Phase 01 simulation passed: {manifest}")


if __name__ == "__main__":
    main()
