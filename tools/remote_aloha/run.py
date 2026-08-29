from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from datetime import timezone
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from openpi_client import websocket_client_policy

from examples.aloha_sim.saver import VideoSaver
from tools.remote_aloha.buffered_policy import BufferedPolicy
from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import PolicyProfile
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.connection_check import verify_ready_tunnel
from tools.remote_aloha.observation_contract import convert_gym_observation
from tools.remote_aloha.policy_contract import validate_policy_action
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.remote import UPSTREAM_SHA
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.sim_smoke_test import package_versions
from tools.remote_aloha.sim_smoke_test import verify_video

_MAX_EPISODE_STEPS = 300
_STEP_SECONDS = 0.02
_INFERENCE_MARGIN_MS = 100.0


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_safe(value: object, depth: int = 0) -> object:
    if depth > 3:
        raise ValueError("environment info is nested too deeply")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("environment info contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > 32 or not all(isinstance(key, str) for key in value):
            raise ValueError("environment info mapping is not bounded string-keyed data")
        return {key: _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple) and len(value) <= 100:
        return [_json_safe(item, depth + 1) for item in value]
    raise ValueError("environment info contains unsupported data")


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _validated_output_root(path: Path) -> Path:
    repository = Path.cwd().resolve()
    resolved = path.resolve()
    if resolved == repository or repository in resolved.parents:
        relative = resolved.relative_to(repository)
        ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(relative)], timeout=10, check=False)
        if ignored.returncode:
            raise ValueError("RUN_OUTPUT_DIR inside the repository must be ignored by Git")
    elif not path.is_absolute():
        raise ValueError("RUN_OUTPUT_DIR outside the repository must be absolute")
    return path


def control_episode(
    environment,
    policy: BufferedPolicy,
    video: VideoSaver,
    *,
    seed: int,
    prompt: str | None,
    profile: PolicyProfile,
    max_steps: int = _MAX_EPISODE_STEPS,
    monotonic=time.monotonic,
    sleep=time.sleep,
    progress: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_observation, reset_info = environment.reset(seed=seed)
    observation = convert_gym_observation(raw_observation, prompt)
    video.on_episode_start()
    started = monotonic()
    first_step_started = None
    last_step_started = None
    step_start_intervals_ms: list[float] = []
    rate_limit_sleep_ms = 0.0
    rewards: list[float] = []
    terminated = truncated = False
    last_info: object = reset_info
    safe_reset_info = _json_safe(reset_info)
    reset_success = safe_reset_info.get("is_success") if isinstance(safe_reset_info, dict) else None
    if reset_success is not None and not isinstance(reset_success, bool):
        raise ValueError("environment reset info.is_success must be boolean")
    success_metric_available = reset_success is not None
    success_seen = bool(reset_success) if success_metric_available else None
    result = progress if progress is not None else {}
    result.update(
        {
            "seed": seed,
            "steps_applied": 0,
            "step_limit": max_steps,
            "step_limit_reached": False,
            "terminated": False,
            "truncated": False,
            "reset_info": safe_reset_info,
            "final_info": safe_reset_info,
            "success_metric_available": success_metric_available,
            "task_success": success_seen,
            "reward_sum": 0.0,
            "reward_max": None,
            "reward_final": None,
            "wall_seconds": 0.0,
            "wall_step_hz": 0.0,
            "active_step_hz": 0.0,
            "active_rate_definition": "step-start rate after the first applied action; includes underrun waits",
            "step_start_interval_p95_ms": None,
            "step_start_interval_max_ms": None,
            "step_start_interval_min_ms": None,
            "faster_than_20ms_count": 0,
            "rate_limit_sleep_ms": 0.0,
        }
    )

    for step in range(max_steps):
        action = validate_policy_action(policy.infer(observation, step), profile)
        if last_step_started is not None:
            delay = last_step_started + _STEP_SECONDS - monotonic()
            if delay > 0:
                sleep(delay)
                rate_limit_sleep_ms += delay * 1000
        step_started = monotonic()
        if last_step_started is not None:
            step_start_intervals_ms.append((step_started - last_step_started) * 1000)
        first_step_started = step_started if first_step_started is None else first_step_started
        last_step_started = step_started
        raw_observation, reward, terminated, truncated, last_info = environment.step(action)
        applied = monotonic()
        applied_steps = step + 1
        result.update(
            {
                "steps_applied": applied_steps,
                "step_limit_reached": applied_steps == max_steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "wall_seconds": applied - started,
                "wall_step_hz": applied_steps / (applied - started) if applied > started else 0.0,
                "active_step_hz": applied_steps
                / max(_STEP_SECONDS, last_step_started - first_step_started + _STEP_SECONDS),
                "step_start_interval_p95_ms": _percentile(step_start_intervals_ms, 95),
                "step_start_interval_max_ms": max(step_start_intervals_ms, default=None),
                "step_start_interval_min_ms": min(step_start_intervals_ms, default=None),
                "faster_than_20ms_count": sum(value < 19.9 for value in step_start_intervals_ms),
                "rate_limit_sleep_ms": rate_limit_sleep_ms,
            }
        )
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("environment reward must be finite")
        rewards.append(reward)
        info = _json_safe(last_info)
        current_success = info.get("is_success") if isinstance(info, dict) else None
        if current_success is not None and not isinstance(current_success, bool):
            raise ValueError("environment info.is_success must be boolean")
        if current_success is not None:
            success_metric_available = True
            success_seen = bool(success_seen) or current_success
        result.update(
            {
                "final_info": info,
                "success_metric_available": success_metric_available,
                "task_success": success_seen if success_metric_available else None,
                "reward_sum": float(sum(rewards)),
                "reward_max": max(rewards),
                "reward_final": rewards[-1],
            }
        )
        observation = convert_gym_observation(raw_observation, prompt)
        video.on_step(observation, {"actions": action})
        if terminated or truncated:
            break

    finished = monotonic()
    steps = int(result["steps_applied"])
    active_seconds = (
        0.0
        if first_step_started is None
        else max(_STEP_SECONDS, last_step_started - first_step_started + _STEP_SECONDS)
    )
    result.update(
        {
            "wall_seconds": finished - started,
            "wall_step_hz": steps / (finished - started) if finished > started else 0.0,
            "active_step_hz": steps / active_seconds if active_seconds else 0.0,
        }
    )
    return result


def _connect(config: RemoteConfig, source_sha: str):
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=config.local_policy_host,
        port=config.local_policy_port,
        connect_timeout=config.policy_connect_timeout_seconds,
        metadata_timeout=config.policy_metadata_timeout_seconds,
        inference_timeout=config.policy_inference_timeout_seconds,
        close_timeout=config.policy_close_timeout_seconds,
        retry_interval=1,
    )
    try:
        metadata = policy.get_server_metadata()
        validate_server_metadata(metadata, config.policy_profile, source_sha, config.policy_backend)
        return policy, _json_safe(dict(metadata))
    except BaseException:
        policy.close()
        raise


def _make_environment(task: str):
    import gym_aloha  # noqa: F401
    import gymnasium

    return gymnasium.make(task, obs_type="pixels_agent_pos")


def _run_seed(
    sim_config: MacSimConfig,
    remote_config: RemoteConfig,
    source_sha: str,
    upstream_sha: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    video = VideoSaver(output_dir, filename="episode.mp4")
    environment = None
    policy = None
    transport = None
    server_metadata = None
    result: dict[str, object] = {"steps_applied": 0, "task_success": None}
    errors: list[dict[str, str]] = []
    primary: BaseException | None = None
    status = "running"
    _atomic_json(
        manifest_path,
        {
            "schema": 1,
            "status": status,
            "cleanup_pending": True,
            "profile": remote_config.policy_profile.name,
            "source_sha": source_sha,
            "seed": seed,
            "episode": result,
            "errors": errors,
        },
    )
    try:
        transport, server_metadata = _connect(remote_config, source_sha)
        policy = BufferedPolicy(
            transport,
            remote_config.policy_profile,
            sim_config.action_horizon,
            sim_config.prefetch_steps,
            remote_config.policy_close_timeout_seconds,
        )
        environment = _make_environment(sim_config.task)
        if (
            environment.spec is None
            or environment.spec.max_episode_steps != _MAX_EPISODE_STEPS
            or environment.metadata.get("render_fps") != 50
            or tuple(environment.action_space.shape) != (remote_config.policy_profile.action_dimension,)
        ):
            raise ValueError("pinned ALOHA environment must expose the 300-step, 50 fps, 14-action contract")
        result = control_episode(
            environment,
            policy,
            video,
            seed=seed,
            prompt=remote_config.policy_profile.default_prompt,
            profile=remote_config.policy_profile,
            progress=result,
        )
        status = "complete"
    except BaseException as error:
        primary = error
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        errors.append({"stage": "control", "type": type(error).__name__, "message": str(error)[:500]})
        _atomic_json(
            manifest_path,
            {
                "schema": 1,
                "status": status,
                "cleanup_pending": True,
                "profile": remote_config.policy_profile.name,
                "source_sha": source_sha,
                "seed": seed,
                "episode": result,
                "errors": errors,
            },
        )
    finally:
        resources = (
            ("policy_close", policy if policy is not None else transport),
            ("environment_close", environment),
        )
        for stage, resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)[:500]})
                if primary is None:
                    primary = error
                    status = "failed"
        video_error = None
        video_validation = None
        try:
            video.on_episode_end()
            if video.output_path is not None:
                video_validation = verify_video(video.output_path, video.frame_count)
        except BaseException as error:
            video_error = error
            errors.append({"stage": "video", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"

        stats = policy.stats if policy is not None else {}
        latencies = stats.get("request_latencies_ms", [])
        warmed_latencies = latencies[1:] if isinstance(latencies, list) else []
        warmed_p95_ms = _percentile(warmed_latencies, 95)
        budget_ms = sim_config.prefetch_steps * _STEP_SECONDS * 1000
        prefetch_qualified = bool(
            warmed_p95_ms is not None
            and warmed_p95_ms + _INFERENCE_MARGIN_MS < budget_ms
            and stats.get("underrun_count") == 0
        )
        active_hz = result.get("active_step_hz")
        uninterrupted_50hz = bool(
            status == "complete"
            and prefetch_qualified
            and isinstance(active_hz, float | int)
            and active_hz >= 49.0
            and result.get("faster_than_20ms_count") == 0
        )
        manifest = {
            "schema": 1,
            "status": status,
            "cleanup_pending": False,
            "infrastructure_pass": status == "complete" and video_validation is not None and not errors,
            "profile": remote_config.policy_profile.name,
            "experimental_profile": remote_config.policy_profile.experimental,
            "policy_backend": remote_config.policy_backend,
            "checkpoint_label": remote_config.policy_profile.checkpoint_label,
            "wire_action_horizon": remote_config.policy_profile.action_horizon,
            "wire_action_dimension": remote_config.policy_profile.action_dimension,
            "server_metadata": server_metadata,
            "package_versions": package_versions(),
            "source_sha": source_sha,
            "upstream_sha": upstream_sha,
            "task": sim_config.task,
            "seed": seed,
            "action_horizon": sim_config.action_horizon,
            "prefetch_steps": sim_config.prefetch_steps,
            "prefetch_budget_ms": budget_ms,
            "inference_margin_ms": _INFERENCE_MARGIN_MS,
            "warmed_request_p95_ms": warmed_p95_ms,
            "prefetch_budget_qualified": prefetch_qualified,
            "uninterrupted_50hz_claimed": uninterrupted_50hz,
            "buffer": stats,
            "episode": result,
            "video": {
                "status": "passed" if video_validation is not None else "failed",
                "path": str(video.output_path) if video.output_path is not None else None,
                "frames": video.frame_count,
                "validation": video_validation,
                "error": str(video_error)[:500] if video_error is not None else None,
            },
            "errors": errors,
        }
        _atomic_json(manifest_path, manifest)
    if primary is not None:
        raise primary
    return {**manifest, "manifest": str(manifest_path)}


def run() -> dict[str, object]:
    sim_config = load_mac_sim_config()
    remote_config = load_remote_config()
    if remote_config.policy_backend != "pytorch":
        raise RemoteError("Phase 4 requires OPENPI_POLICY_BACKEND=pytorch on the validated 24 GiB PC")
    _, source_sha = verify_ready_tunnel(remote_config)
    upstream_sha = UPSTREAM_SHA
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    root = _validated_output_root(sim_config.output_dir) / "phase04" / timestamp / remote_config.policy_profile.name
    summary = {
        "status": "running",
        "profile": remote_config.policy_profile.name,
        "source_sha": source_sha,
        "episodes": [],
        "error": None,
    }
    summary_path = root / "summary.json"
    _atomic_json(summary_path, summary)
    try:
        for seed in range(sim_config.seed, sim_config.seed + sim_config.episodes):
            summary["episodes"].append(
                _run_seed(sim_config, remote_config, source_sha, upstream_sha, seed, root / f"seed-{seed}")
            )
            _atomic_json(summary_path, summary)
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        _atomic_json(summary_path, summary)
        raise
    summary["status"] = "passed" if all(item["infrastructure_pass"] for item in summary["episodes"]) else "failed"
    _atomic_json(summary_path, summary)
    return {**summary, "summary": str(summary_path)}


def main() -> None:
    try:
        print(json.dumps(run(), allow_nan=False, sort_keys=True))
    except (RemoteError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
