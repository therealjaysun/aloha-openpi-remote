from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from datetime import timezone
import importlib.metadata
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy

from examples.aloha_sim.saver import VideoSaver
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.connection_check import verify_ready_tunnel
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_policy_timing
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing
from tools.remote_aloha.policy_contract import validate_timing_reconciliation
from tools.remote_aloha.remote import UPSTREAM_SHA
from tools.remote_aloha.remote import start_gpu_sampler
from tools.remote_aloha.run import _atomic_json
from tools.remote_aloha.run import _gpu_coverage
from tools.remote_aloha.run import _gpu_events
from tools.remote_aloha.sim_smoke_test import verify_video
from tools.remote_aloha.telemetry import JsonlWriter
from tools.remote_aloha.telemetry import aggregate_events
from tools.remote_aloha.telemetry import aggregate_jsonl
from tools.remote_aloha.telemetry import json_safe
from tools.remote_aloha.telemetry import read_jsonl
from tools.remote_aloha.telemetry import write_markdown_summary
from tools.remote_aloha.telemetry import write_summary
from tools.remote_aloha.trajectory import summarize_trajectory
from tools.remote_aloha.trajectory import validate_joint_vector
from tools.remote_aloha.trajectory import write_trajectory_plot

CAMERA_VIEWS = ("agentview", "eye_in_hand")
PANDA_JOINT_LIMITS = (
    ("joint1", -2.8973, 2.8973),
    ("joint2", -1.7628, 1.7628),
    ("joint3", -2.8973, 2.8973),
    ("joint4", -3.0718, -0.0698),
    ("joint5", -2.8973, 2.8973),
    ("joint6", -0.0175, 3.7525),
    ("joint7", -2.8973, 2.8973),
)
_PACKAGE_NAMES = ("imageio", "libero", "matplotlib", "mujoco", "numpy", "robosuite")


def policy_step_limit(duration_seconds: int, *, smoke: bool, control_hz: int) -> tuple[int, int]:
    seconds = 6 if smoke else duration_seconds
    if not 1 <= seconds <= 300:
        raise ValueError("duration_seconds must be between 1 and 300")
    return seconds, seconds * control_hz


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGE_NAMES:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def _atomic_observation(path: Path, observation: Mapping[str, object]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(msgpack_numpy.Packer().pack(dict(observation)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _video_observation(element: Mapping[str, object]) -> dict[str, object]:
    images = {}
    for source, target in (
        ("observation/image", CAMERA_VIEWS[0]),
        ("observation/wrist_image", CAMERA_VIEWS[1]),
    ):
        image = element[source]
        if not isinstance(image, np.ndarray) or image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"{source} must be uint8 HWC with shape (224, 224, 3)")
        images[target] = np.transpose(image, (2, 0, 1))
    return {"images": images}


def _timing_metrics(
    request_index: int,
    latency_ms: float,
    response: Mapping[str, object],
) -> dict[str, float]:
    server = validate_server_timing(response)
    policy = validate_policy_timing(response)
    validate_timing_reconciliation(policy, server)
    metrics = {
        "cold_inference_ms" if request_index == 0 else "warm_inference_ms": latency_ms,
        "server_infer_ms": float(server["infer_ms"]),
    }
    metrics.update({f"policy_{name}": float(value) for name, value in policy.items()})
    return metrics


def run_scenario(
    *,
    scenario: str,
    duration_seconds: int,
    smoke: bool,
    seed: int,
    settle_steps: int,
    resize_size: int,
    replan_steps: int,
    output_dir: str,
    host: str,
    port: int,
    control_hz: int,
    dummy_action: list[float],
    create_env: Callable[..., tuple[object, str]],
    policy_element: Callable[[Mapping[str, object], str, int], tuple[dict[str, object], np.ndarray]],
    scene_hash: str,
) -> dict[str, object]:
    seconds, step_limit = policy_step_limit(duration_seconds, smoke=smoke, control_hz=control_hz)
    remote = replace(
        load_remote_config(),
        local_policy_host="127.0.0.1" if host == "0.0.0.0" else host,
        local_policy_port=port,
    )
    if remote.policy_backend != "pytorch" or remote.policy_profile.name != "pi05_libero":
        raise ValueError("LIBERO telemetry runs require OPENPI_POLICY_PROFILE=pi05_libero and PyTorch")
    _, source_sha = verify_ready_tunnel(remote)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    run_id = uuid.uuid4().hex
    root = validate_output_root(Path(output_dir)) / "libero_0829" / timestamp / remote.policy_profile.name / scenario
    episode_dir = root / f"seed-{seed}"
    episode_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest_path = episode_dir / "manifest.json"
    telemetry_path = episode_dir / "telemetry.jsonl"
    video = VideoSaver(
        episode_dir,
        filename="episode.mp4",
        camera_views=CAMERA_VIEWS,
        fps=control_hz,
        streaming=True,
    )
    writer = JsonlWriter(telemetry_path)
    telemetry_overheads_ms: list[float] = []
    client = environment = sampler = None
    server_metadata = None
    gpu_path = None
    gpu_events: list[dict[str, object]] = []
    gpu_result: dict[str, object] = {}
    errors: list[dict[str, str]] = []
    action_plan: deque[np.ndarray] = deque()
    request_latencies_ms: list[float] = []
    rewards: list[float] = []
    best_coverage = 0.0
    initial_coverage = 0.0
    best_coverage_step = None
    final_coverage: dict[str, float] = {}
    sticky_success = False
    applied_steps = 0
    started = time.monotonic()
    status = "running"
    primary: BaseException | None = None
    cleanup_pending = False
    video_validation = None
    trajectory: dict[str, object] = {
        "sample_count": 0,
        "joint_count": len(PANDA_JOINT_LIMITS),
        "step_coverage": 1.0,
        "plot_status": "no_samples",
        "plot_id": None,
        "actual_series_count": 0,
        "commanded_series_count": 0,
    }

    def emit(event: str, **fields: object) -> None:
        telemetry_overheads_ms.append(writer.write(event, **fields))

    initial_manifest = {
        "schema": 1,
        "status": status,
        "cleanup_pending": True,
        "profile": remote.policy_profile.name,
        "source_sha": source_sha,
        "seed": seed,
        "scenario": scenario,
        "errors": errors,
    }
    _atomic_json(manifest_path, initial_manifest)
    _atomic_json(
        root / "summary.json",
        {
            "status": status,
            "run_id": run_id,
            "profile": remote.policy_profile.name,
            "scenario": scenario,
            "source_sha": source_sha,
            "gpu_metrics_interval_seconds": remote.gpu_metrics_interval_seconds,
            "episodes": [],
            "error": None,
        },
    )

    versions = _package_versions()
    emit(
        "metadata",
        run_id=run_id,
        profile=remote.policy_profile.name,
        checkpoint_label=remote.policy_profile.checkpoint_label,
        source_sha=source_sha,
        upstream_sha=UPSTREAM_SHA,
        seeds=[seed],
        task=f"libero/{scenario}",
        scenario=scenario,
        scene_hash=scene_hash,
        target_area_coverage_method="exact-planar-union-v1",
        camera_views=list(CAMERA_VIEWS),
        action_horizon=replan_steps,
        model_action_horizon=remote.policy_profile.action_horizon,
        prefetch_steps=0,
        chunk_crossfade_steps=0,
        package_versions=versions,
    )
    try:
        sampler = start_gpu_sampler(remote, run_id, source_sha, root)
        environment, prompt = create_env(
            scenario,
            resolution=256,
            seed=seed,
            horizon=step_limit + settle_steps + 1,
        )
        client = websocket_client_policy.WebsocketClientPolicy(
            host=remote.local_policy_host,
            port=remote.local_policy_port,
            connect_timeout=remote.policy_connect_timeout_seconds,
            metadata_timeout=remote.policy_metadata_timeout_seconds,
            inference_timeout=remote.policy_inference_timeout_seconds,
            close_timeout=remote.policy_close_timeout_seconds,
            retry_interval=1,
        )
        server_metadata = dict(client.get_server_metadata())
        validate_server_metadata(server_metadata, remote.policy_profile, source_sha, remote.policy_backend)
        observation = environment.reset()
        for _ in range(settle_steps):
            observation, _, _, _ = environment.step(dummy_action)
        initial_coverage = float(environment.env.coverage()["overall"])
        best_coverage = initial_coverage
        first_element, _ = policy_element(observation, prompt, resize_size)
        _atomic_observation(episode_dir / "policy-observation.msgpack", first_element)
        video.on_episode_start()
        for step in range(step_limit):
            element, _ = policy_element(observation, prompt, resize_size)
            if not action_plan:
                request_index = len(request_latencies_ms)
                emit("policy_request", request_index=request_index, step=step)
                request_started = time.perf_counter_ns()
                response = client.infer(element)
                latency_ms = (time.perf_counter_ns() - request_started) / 1_000_000
                request_latencies_ms.append(latency_ms)
                actions = validate_policy_response(response, remote.policy_profile)
                if len(actions) < replan_steps:
                    raise ValueError(f"policy must return at least {replan_steps} actions")
                action_plan.extend(actions[:replan_steps])
                emit(
                    "policy_result",
                    request_index=request_index,
                    step=step,
                    usable_fresh_actions=replan_steps,
                    metrics=_timing_metrics(request_index, latency_ms, response),
                )
            action = np.asarray(action_plan.popleft())
            step_started = time.perf_counter_ns()
            observation, reward, done, _ = environment.step(action.tolist())
            applied_steps = step + 1
            post_element, _ = policy_element(observation, prompt, resize_size)
            video.on_step(_video_observation(post_element), {"actions": action})
            coverage = environment.env.coverage()
            final_coverage = {name: float(value) for name, value in coverage.items()}
            overall = final_coverage["overall"]
            if best_coverage_step is None or overall > best_coverage:
                best_coverage = overall
                best_coverage_step = applied_steps
            sticky_success = sticky_success or bool(done)
            joints = validate_joint_vector(
                np.asarray(observation["robot0_joint_pos"]).tolist(),
                "actual_joint_positions",
                PANDA_JOINT_LIMITS,
            )
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("LIBERO reward must be finite")
            rewards.append(reward_value)
            emit(
                "step",
                seed=seed,
                step=step,
                applied_step=applied_steps,
                elapsed_seconds=applied_steps / control_hz,
                actual_joint_positions=joints,
                osc_action=action.tolist(),
                scenario_info={"target_area_coverage": overall, **final_coverage},
                metrics={
                    "sim_step_ms": (time.perf_counter_ns() - step_started) / 1_000_000,
                    "reward": reward_value,
                },
            )
            if sampler is not None and applied_steps % (control_hz * 30) == 0:
                sampler.check()
        status = "complete"
    except BaseException as error:
        primary = error
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        errors.append({"stage": "control", "type": type(error).__name__, "message": str(error)[:500]})
        with suppress(BaseException):
            emit("error", stage="control", error_type=type(error).__name__)
    finally:
        for stage, resource in (("policy_close", client), ("environment_close", environment)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                cleanup_pending = True
                errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)[:500]})
                if primary is None:
                    primary = error
                    status = "failed"

        video_status = "no_frames"
        try:
            video.on_episode_end()
            if video.frame_count:
                video_validation = verify_video(
                    video.output_path,
                    video.frame_count,
                    (224, 448, 3),
                    expected_fps=control_hz,
                )
                video_status = "complete" if status == "complete" else "partial"
        except BaseException as error:
            if video.frame_count or str(error) != "cannot save an episode video without frames":
                video_status = "encode_failed"
                errors.append({"stage": "video", "type": type(error).__name__, "message": str(error)[:500]})
                if primary is None:
                    primary = error
                    status = "failed"

        trajectory_path = episode_dir / "joint-trajectory.png"
        trajectory_id = f"{run_id}-seed-{seed}-joint-trajectory"
        try:
            trajectory_events = read_jsonl(telemetry_path).events
            trajectory = summarize_trajectory(
                trajectory_events,
                applied_steps,
                joint_limits=PANDA_JOINT_LIMITS,
            )
            trajectory = write_trajectory_plot(
                trajectory_events,
                applied_steps,
                trajectory_path,
                trajectory_id,
                joint_limits=PANDA_JOINT_LIMITS,
                title="LIBERO Panda joint trajectory",
                footnote="Actual Panda joint positions normalized to pinned robosuite 1.4.1 limits; OSC actions remain in telemetry and are not joint commands.",
            )
        except BaseException as error:
            errors.append({"stage": "trajectory", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "failed"

        if sampler is not None:
            try:
                gpu_path = sampler.stop(root)
                gpu_events, gpu_result = _gpu_events(
                    gpu_path,
                    run_id,
                    remote.policy_profile.name,
                    source_sha,
                    remote.gpu_metrics_interval_seconds,
                )
                gpu_result.update(
                    _gpu_coverage(
                        gpu_events,
                        [list(read_jsonl(telemetry_path).events)],
                        root / "clock-correlation.json",
                        remote.gpu_metrics_interval_seconds,
                    )
                )
                if not gpu_result.get("gpu_coverage_pass"):
                    raise ValueError("GPU telemetry does not cover the LIBERO episode")
            except BaseException as error:
                cleanup_pending = cleanup_pending or gpu_path is None
                errors.append({"stage": "gpu_sampler", "type": type(error).__name__, "message": str(error)[:500]})
                if primary is None:
                    primary = error
                    status = "failed"

        infrastructure_pass = bool(
            status == "complete"
            and not errors
            and gpu_result.get("gpu_coverage_pass") is True
            and video_validation is not None
            and video.frame_count == applied_steps
            and trajectory.get("sample_count") == applied_steps
            and trajectory.get("plot_status") == "passed"
        )
        wall_seconds = time.monotonic() - started
        telemetry_p95 = float(np.percentile(telemetry_overheads_ms, 95)) if telemetry_overheads_ms else None
        try:
            emit(
                "terminal",
                status=status,
                episodes=1,
                infrastructure_pass=infrastructure_pass,
                steps_applied=applied_steps,
                request_count=len(request_latencies_ms),
                retries=0,
                failures=int(status != "complete"),
                task_success=sticky_success,
                reward_sum=float(sum(rewards)),
                reward_max=max(rewards, default=None),
                reward_final=rewards[-1] if rewards else None,
                trajectory_sample_count=trajectory["sample_count"],
                trajectory_joint_count=trajectory["joint_count"],
                trajectory_step_coverage=trajectory["step_coverage"],
                trajectory_plot_status=trajectory["plot_status"],
                trajectory_plot_id=trajectory["plot_id"],
                video_ids=[f"{run_id}-seed-{seed}"],
                push_success=int(sticky_success),
                videos_passed=int(video_status == "complete"),
                coverage_sample_count=applied_steps,
                initial_target_area_coverage_percent=initial_coverage * 100,
                final_target_area_coverage_percent=(final_coverage.get("overall", 0.0) * 100),
                best_target_area_coverage_percent=(best_coverage * 100 if best_coverage_step is not None else None),
                best_target_area_coverage_step=best_coverage_step,
                time_to_best_target_area_coverage_seconds=(
                    best_coverage_step / control_hz if best_coverage_step is not None else None
                ),
                episode_elapsed_seconds=applied_steps / control_hz,
                **gpu_result,
                metrics={
                    "telemetry_write_ms": telemetry_p95 or 0.0,
                    "wall_episode_hz": applied_steps / wall_seconds if wall_seconds else 0.0,
                    "final_target_area_coverage_percent": final_coverage.get("overall", 0.0) * 100,
                    "best_target_area_coverage_percent": best_coverage * 100,
                    "episode_elapsed_seconds": applied_steps / control_hz,
                },
            )
        except BaseException as error:
            errors.append({"stage": "telemetry", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "failed"
        writer_closed = False
        try:
            writer.close()
            writer_closed = True
        except BaseException as error:
            cleanup_pending = True
            errors.append({"stage": "telemetry_close", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "failed"

        telemetry_summary = None
        try:
            telemetry_summary = aggregate_jsonl(telemetry_path)
            write_summary(episode_dir / "telemetry-summary.json", telemetry_summary, publishable=True)
            write_markdown_summary(episode_dir / "telemetry-summary.md", telemetry_summary)
            telemetry_events = list(read_jsonl(telemetry_path).events)
            performance = aggregate_events([*telemetry_events[:-1], *gpu_events, telemetry_events[-1]])
            write_summary(root / "performance-summary.json", performance, publishable=True)
            write_markdown_summary(root / "performance-summary.md", performance)
        except BaseException as error:
            errors.append({"stage": "telemetry_summary", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "failed"

        infrastructure_pass = infrastructure_pass and primary is None and not errors

        episode = {
            "seed": seed,
            "steps_applied": applied_steps,
            "step_limit": step_limit,
            "step_limit_reached": applied_steps == step_limit,
            "policy_seconds": applied_steps / control_hz,
            "control_hz": control_hz,
            "settle_steps": settle_steps,
            "task_success": sticky_success,
            "best_coverage": best_coverage,
            "initial_coverage": initial_coverage,
            "final_coverage": final_coverage,
            "request_count": len(request_latencies_ms),
            "reward_sum": float(sum(rewards)),
            "reward_max": max(rewards, default=None),
            "reward_final": rewards[-1] if rewards else None,
            "wall_seconds": wall_seconds,
        }
        manifest = {
            "schema": 1,
            "status": status,
            "cleanup_pending": cleanup_pending,
            "infrastructure_pass": infrastructure_pass,
            "profile": remote.policy_profile.name,
            "experimental_profile": remote.policy_profile.experimental,
            "policy_backend": remote.policy_backend,
            "checkpoint_label": remote.policy_profile.checkpoint_label,
            "wire_action_horizon": remote.policy_profile.action_horizon,
            "wire_action_dimension": remote.policy_profile.action_dimension,
            "server_metadata": json_safe(server_metadata) if server_metadata is not None else None,
            "package_versions": versions,
            "source_sha": source_sha,
            "upstream_sha": UPSTREAM_SHA,
            "task": f"libero/{scenario}",
            "scenario": scenario,
            "scene_hash": scene_hash,
            "seed": seed,
            "action_horizon": replan_steps,
            "episode": episode,
            "telemetry": {
                "path": str(telemetry_path),
                "summary": telemetry_summary,
                "writer_closed": writer_closed,
                "write_p95_ms": telemetry_p95,
            },
            "trajectory": {
                **trajectory,
                "path": str(trajectory_path) if trajectory_path.is_file() else None,
            },
            "video": {
                "id": f"{run_id}-seed-{seed}",
                "camera_views": list(CAMERA_VIEWS),
                "layout": "horizontal",
                "status": video_status,
                "path": str(video.output_path) if video.output_path is not None else None,
                "frames": video.frame_count,
                "validation": video_validation,
            },
            "gpu": {**gpu_result, "path": str(gpu_path) if gpu_path is not None else None},
            "errors": errors,
        }
        _atomic_json(manifest_path, manifest)
        summary = {
            "status": "passed" if infrastructure_pass else status,
            "run_id": run_id,
            "profile": remote.policy_profile.name,
            "scenario": scenario,
            "source_sha": source_sha,
            "gpu_metrics_interval_seconds": remote.gpu_metrics_interval_seconds,
            "episodes": [{**manifest, "manifest": str(manifest_path)}],
            "error": errors[0] if errors else None,
            "summary": str(root / "summary.json"),
        }
        _atomic_json(root / "summary.json", summary)

    if primary is not None:
        raise primary
    return {
        "status": summary["status"],
        "scenario": scenario,
        "steps_applied": applied_steps,
        "policy_seconds": applied_steps / control_hz,
        "task_success": sticky_success,
        "best_coverage": best_coverage,
        "request_count": len(request_latencies_ms),
        "output": str(root),
    }
