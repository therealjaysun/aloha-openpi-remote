from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime
from datetime import timezone
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from openpi_client import websocket_client_policy
from websockets.exceptions import ConnectionClosed

from examples.aloha_sim.saver import LiveDisplay
from examples.aloha_sim.saver import VideoSaver
from tools.remote_aloha.buffered_policy import BufferedPolicy
from tools.remote_aloha.config import FIXED_PROMPT_SCHEDULE
from tools.remote_aloha.config import STAGED_PROMPT_BOUNDARIES
from tools.remote_aloha.config import STAGED_PROMPT_SCHEDULE
from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import PolicyProfile
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.connection_check import verify_ready_tunnel
from tools.remote_aloha.observation_contract import POLICY_CAMERA_VIEWS
from tools.remote_aloha.observation_contract import convert_gym_artifact_observation
from tools.remote_aloha.observation_contract import convert_gym_observation
from tools.remote_aloha.policy_contract import validate_policy_action
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.remote import UPSTREAM_SHA
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.remote import start_gpu_sampler
from tools.remote_aloha.scenarios import DISPLAY_EVERY_STEPS
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import TARGET_AREA_COVERAGE_METHOD
from tools.remote_aloha.scenarios import ScenarioSpec
from tools.remote_aloha.scenarios import project_action
from tools.remote_aloha.sim_smoke_test import package_versions
from tools.remote_aloha.sim_smoke_test import verify_video
from tools.remote_aloha.telemetry import JsonlWriter
from tools.remote_aloha.telemetry import aggregate_events
from tools.remote_aloha.telemetry import aggregate_jsonl
from tools.remote_aloha.telemetry import read_jsonl
from tools.remote_aloha.telemetry import write_markdown_summary
from tools.remote_aloha.telemetry import write_summary
from tools.remote_aloha.trajectory import summarize_trajectory
from tools.remote_aloha.trajectory import validate_joint_vector
from tools.remote_aloha.trajectory import write_trajectory_plot

_DEFAULT_EPISODE_STEPS = 300
_STEP_SECONDS = 0.02
_INFERENCE_MARGIN_MS = 100.0
_STAGED_PROMPTS = (
    (
        "orient",
        "Look at the pi-shaped block and matching target. Using only the left arm, tilt the wrist so the gripper "
        "points down. Do not touch the block yet.",
    ),
    (
        "approach",
        "Using only the left arm, keep the wrist down and lower the gripper just above the table on the block's "
        "side away from the target. Stop before contact.",
    ),
    (
        "push",
        "Using only the left arm, keep the gripper low. Make short pushes toward the matching target, rechecking "
        "after each push; do not lift the pi-shaped block.",
    ),
)


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


def _scenario_info_fields(scenario: ScenarioSpec) -> set[str]:
    body_count = 1 if scenario.object_kind == "pi" else 2
    return {
        "is_success",
        "scenario",
        "scene_hash",
        "layout_hash",
        "body_count",
        "held_steps",
        "lifted_ever",
        "off_table",
        "fallen",
        "terminal_reason",
        "left_contact_ever",
        "right_contact_ever",
        "both_arms_participated",
        "interference_ever",
        "left_joint_travel",
        "right_joint_travel",
        "target_area_coverage",
        *{
            f"body_{index}_{suffix}"
            for index in range(body_count)
            for suffix in ("xy_error", "yaw_error", "roll", "pitch", "height_error", "target_area_coverage")
        },
    }


def _scenario_step_info(value: object, scenario: ScenarioSpec) -> object:
    safe = _json_safe(value)
    if not scenario.is_custom:
        return safe
    if not isinstance(safe, dict):
        raise ValueError("custom scenario info must be a mapping")
    body_count = 1 if scenario.object_kind == "pi" else 2
    expected = _scenario_info_fields(scenario)
    if set(safe) != expected:
        raise ValueError("custom scenario info fields are invalid")
    if (
        safe["scenario"] != scenario.key
        or isinstance(safe["body_count"], bool)
        or not isinstance(safe["body_count"], int)
        or safe["body_count"] != body_count
    ):
        raise ValueError("custom scenario info identity is invalid")
    for key in ("scene_hash", "layout_hash"):
        field = safe[key]
        if (
            not isinstance(field, str)
            or len(field) != 64
            or any(character not in "0123456789abcdef" for character in field)
        ):
            raise ValueError("custom scenario info hash is invalid")
    boolean_fields = (
        "is_success",
        "lifted_ever",
        "off_table",
        "fallen",
        "left_contact_ever",
        "right_contact_ever",
        "both_arms_participated",
        "interference_ever",
    )
    if any(not isinstance(safe[key], bool) for key in boolean_fields):
        raise ValueError("custom scenario info flags are invalid")
    if safe["both_arms_participated"] != (safe["left_contact_ever"] and safe["right_contact_ever"]):
        raise ValueError("custom scenario participation is inconsistent")
    if isinstance(safe["held_steps"], bool) or not isinstance(safe["held_steps"], int) or safe["held_steps"] < 0:
        raise ValueError("custom scenario info held-step count is invalid")
    if safe["terminal_reason"] not in {"running", "success", "off_table", "fallen", "time_limit"}:
        raise ValueError("custom scenario terminal reason is invalid")
    if (
        (safe["terminal_reason"] == "success") != safe["is_success"]
        or (safe["terminal_reason"] == "off_table" and not safe["off_table"])
        or (safe["terminal_reason"] == "fallen" and not safe["fallen"])
    ):
        raise ValueError("custom scenario terminal state is inconsistent")
    numeric = (
        expected
        - set(boolean_fields)
        - {
            "scenario",
            "scene_hash",
            "layout_hash",
            "body_count",
            "held_steps",
            "terminal_reason",
        }
    )
    if any(
        isinstance(safe[key], bool)
        or not isinstance(safe[key], int | float)
        or not math.isfinite(safe[key])
        or safe[key] < 0
        for key in numeric
    ):
        raise ValueError("custom scenario info metrics are invalid")
    coverage_fields = {"target_area_coverage"} | {f"body_{index}_target_area_coverage" for index in range(body_count)}
    if any(safe[key] > 1 for key in coverage_fields):
        raise ValueError("custom scenario area coverage is invalid")
    return safe


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _convert_environment_observation(environment, raw_observation: object, prompt: str | None) -> dict:
    pixels = raw_observation.get("pixels") if isinstance(raw_observation, Mapping) else None
    if isinstance(pixels, Mapping) and set(pixels) == {"top"}:
        physics = environment.unwrapped._env.physics  # noqa: SLF001 - pinned gym-aloha exposes no camera API
        # ponytail: two extra renders are simplest; replace the task's unused angle/vis renders only if measured costly.
        raw_observation = {
            **raw_observation,
            "pixels": {
                **pixels,
                "left_wrist": physics.render(height=480, width=640, camera_id="left_wrist"),
                "right_wrist": physics.render(height=480, width=640, camera_id="right_wrist"),
            },
        }
    return convert_gym_observation(raw_observation, prompt)


def _prompt_stage(
    prompt_schedule: str, default_prompt: str | None, step: int
) -> tuple[int | None, str | None, str | None]:
    if prompt_schedule == FIXED_PROMPT_SCHEDULE:
        return None, None, default_prompt
    if prompt_schedule != STAGED_PROMPT_SCHEDULE or not 0 <= step < STAGED_PROMPT_BOUNDARIES[-1]:
        raise ValueError("prompt schedule or step is invalid")
    for index, ((stage_id, prompt), start, end) in enumerate(
        zip(_STAGED_PROMPTS, STAGED_PROMPT_BOUNDARIES[:-1], STAGED_PROMPT_BOUNDARIES[1:], strict=True)
    ):
        if start <= step < end:
            return index, stage_id, prompt
    raise AssertionError("staged prompt boundaries must cover the full diagnostic")


def control_episode(
    environment,
    policy: BufferedPolicy,
    video: VideoSaver,
    *,
    seed: int,
    prompt: str | None,
    profile: PolicyProfile,
    prompt_schedule: str = FIXED_PROMPT_SCHEDULE,
    scenario: ScenarioSpec = SCENARIOS["transfer_cube"],
    expected_scene_hash: str | None = None,
    display: LiveDisplay | None = None,
    max_steps: int = _DEFAULT_EPISODE_STEPS,
    monotonic=time.monotonic,
    sleep=time.sleep,
    progress: dict[str, object] | None = None,
    emit: Callable[..., None] | None = None,
) -> dict[str, object]:
    raw_observation, reset_info = environment.reset(seed=seed)
    home_joint_positions = validate_joint_vector(
        np.asarray(raw_observation["agent_pos"]).tolist(), "home_joint_positions"
    )
    _, _, initial_prompt = _prompt_stage(prompt_schedule, prompt, 0)
    observation = _convert_environment_observation(environment, raw_observation, initial_prompt)
    video.on_episode_start()
    display_error = None
    if display is not None:
        try:
            display.on_episode_start()
        except Exception as error:
            display_error = {"type": type(error).__name__, "message": str(error)[:500]}
            display = None
    started = monotonic()
    first_step_started = None
    last_step_started = None
    step_start_intervals_ms: list[float] = []
    rate_limit_sleep_ms = 0.0
    rewards: list[float] = []
    terminated = truncated = False
    last_info: object = reset_info
    safe_reset_info = _json_safe(reset_info)
    reset_scenario_identity = None
    reset_step_info = None
    if scenario.is_custom:
        if not isinstance(safe_reset_info, dict) or not _scenario_info_fields(scenario) <= set(safe_reset_info):
            raise ValueError("custom scenario reset info fields are invalid")
        reset_step_info = _scenario_step_info(
            {key: safe_reset_info[key] for key in _scenario_info_fields(scenario)}, scenario
        )
        reset_scenario_identity = (reset_step_info["scene_hash"], reset_step_info["layout_hash"])
        if expected_scene_hash is not None and reset_step_info["scene_hash"] != expected_scene_hash:
            raise ValueError("custom scenario reset scene hash does not match the environment")
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
            "display_error": display_error,
        }
    )
    if reset_step_info is not None:
        result.update(
            {
                "coverage_method": TARGET_AREA_COVERAGE_METHOD,
                "coverage_sample_count": 0,
                "initial_target_area_coverage_percent": reset_step_info["target_area_coverage"] * 100,
                "final_target_area_coverage_percent": None,
                "best_target_area_coverage_percent": None,
                "best_target_area_coverage_step": None,
                "time_to_best_target_area_coverage_seconds": None,
            }
        )
    if emit is not None:
        emit("episode", seed=seed, status="started")

    active_prompt_stage = None
    for step in range(max_steps):
        prompt_stage_index, prompt_stage_id, step_prompt = _prompt_stage(prompt_schedule, prompt, step)
        if step_prompt is None:
            observation.pop("prompt", None)
        else:
            observation["prompt"] = step_prompt
        if prompt_stage_id is not None and prompt_stage_id != active_prompt_stage:
            transition = policy.transition_prompt_stage(observation, step, prompt_stage_id)
            if emit is not None:
                emit(
                    "prompt_stage",
                    prompt_stage_id=prompt_stage_id,
                    prompt_stage_index=prompt_stage_index,
                    start_step=step,
                    first_applied_step=step + 1,
                    discarded_action_count=transition["discarded_action_count"],
                    metrics={"prompt_transition_wait_ms": transition["transition_wait_ms"]},
                )
            active_prompt_stage = prompt_stage_id
        raw_action = validate_policy_action(policy.infer(observation, step), profile)
        action = (
            validate_policy_action(project_action(raw_action, scenario, home_joint_positions), profile)
            if scenario.is_custom
            else raw_action
        )
        if last_step_started is not None:
            delay = last_step_started + _STEP_SECONDS - monotonic()
            if delay > 0:
                sleep(delay)
                rate_limit_sleep_ms += delay * 1000
        step_started = monotonic()
        interval_ms = None
        if last_step_started is not None:
            interval_ms = (step_started - last_step_started) * 1000
            step_start_intervals_ms.append(interval_ms)
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
        camera_error = None
        try:
            next_step = min(step + 1, max_steps - 1)
            _, _, next_prompt = _prompt_stage(prompt_schedule, prompt, next_step)
            next_observation = _convert_environment_observation(environment, raw_observation, next_prompt)
        except BaseException as error:
            camera_error = error
            next_observation = convert_gym_artifact_observation(raw_observation)
        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            reward_value = None
        step_info_error = None
        try:
            step_info = _scenario_step_info(last_info, scenario)
            if scenario.is_custom and (step_info["scene_hash"], step_info["layout_hash"]) != reset_scenario_identity:
                raise ValueError("custom scenario step identity changed during the episode")
        except BaseException as error:
            step_info = None
            step_info_error = error
        if scenario.is_custom and step_info is not None:
            coverage_percent = step_info["target_area_coverage"] * 100
            result["coverage_sample_count"] = applied_steps
            result["final_target_area_coverage_percent"] = coverage_percent
            best_coverage = result["best_target_area_coverage_percent"]
            if best_coverage is None or coverage_percent > best_coverage:
                result["best_target_area_coverage_percent"] = coverage_percent
                result["best_target_area_coverage_step"] = applied_steps
                result["time_to_best_target_area_coverage_seconds"] = applied - started
        video_step_error = None
        try:
            video.on_step(next_observation, {"actions": action})
        except BaseException as error:
            video_step_error = error
        if emit is not None:
            metrics = {"sim_step_ms": (applied - step_started) * 1000}
            if reward_value is not None and math.isfinite(reward_value):
                metrics["reward"] = reward_value
            if interval_ms is not None:
                metrics["active_step_interval_ms"] = interval_ms
            fields = {
                "seed": seed,
                "step": step,
                "applied_step": applied_steps,
                "elapsed_seconds": applied - started,
                "actual_joint_positions": validate_joint_vector(
                    next_observation["state"].tolist(), "actual_joint_positions"
                ),
                "commanded_joint_positions": validate_joint_vector(action.tolist(), "commanded_joint_positions"),
                "metrics": metrics,
            }
            if scenario.is_custom and step_info is not None:
                fields["scenario_info"] = step_info
            if prompt_stage_id is not None:
                fields["prompt_stage_id"] = prompt_stage_id
                fields["prompt_stage_index"] = prompt_stage_index
            emit(
                "step",
                **fields,
            )
        if display is not None:
            try:
                display.on_step(next_observation, {"actions": action})
            except Exception as error:
                result["display_error"] = {"type": type(error).__name__, "message": str(error)[:500]}
                with suppress(Exception):
                    display.on_episode_end()
                display = None
        if camera_error is not None:
            raise camera_error
        if video_step_error is not None:
            raise video_step_error
        if step_info_error is not None:
            raise step_info_error
        observation = next_observation
        if reward_value is None or not math.isfinite(reward_value):
            raise ValueError("environment reward must be finite")
        reward = reward_value
        rewards.append(reward)
        info = step_info
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


def _connect_with_retry(
    config: RemoteConfig,
    source_sha: str,
    progress: dict[str, int],
    *,
    emit: Callable[..., None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
):
    for attempt in range(config.policy_retry_count + 1):
        try:
            return _connect(config, source_sha)
        except (ConnectionError, TimeoutError, EOFError, ConnectionClosed) as error:
            progress["failures"] += 1
            if attempt == config.policy_retry_count:
                raise
            progress["retries"] += 1
            if emit is not None:
                emit("retry", attempt=attempt + 1, error_type=type(error).__name__)
            sleep(config.policy_retry_backoff_seconds)
    raise AssertionError("bounded connection retry loop exhausted without returning or raising")


def _make_environment(config: MacSimConfig):
    import gym_aloha  # noqa: F401
    import gymnasium

    if config.scenario.is_custom:
        import examples.aloha_sim.push_pi_env  # noqa: F401

    kwargs: dict[str, object] = {
        "obs_type": "pixels_agent_pos",
        "max_episode_steps": config.episode_steps,
    }
    if config.scenario.is_custom:
        kwargs["episode_steps"] = config.episode_steps
    return gymnasium.make(config.task, **kwargs)


def _run_seed(
    sim_config: MacSimConfig,
    remote_config: RemoteConfig,
    source_sha: str,
    upstream_sha: str,
    seed: int,
    output_dir: Path,
    run_id: str,
) -> dict[str, object]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    video = VideoSaver(output_dir, filename="episode.mp4", camera_views=POLICY_CAMERA_VIEWS)
    display = LiveDisplay(
        enabled=sim_config.display,
        every_steps=DISPLAY_EVERY_STEPS,
        camera_views=POLICY_CAMERA_VIEWS,
    )
    environment = None
    policy = None
    transport = None
    server_metadata = None
    result: dict[str, object] = {"steps_applied": 0, "task_success": None}
    errors: list[dict[str, str]] = []
    connection = {"failures": 0, "retries": 0}
    telemetry_overheads_ms: list[float] = []
    telemetry_writer: JsonlWriter | None = None
    trajectory: dict[str, object] = {
        "sample_count": 0,
        "joint_count": 14,
        "step_coverage": 1.0,
        "plot_status": "no_samples",
        "plot_id": None,
        "actual_series_count": 0,
        "commanded_series_count": 0,
    }
    versions: dict[str, str] = {}
    scene_id: str | None = None
    control_finished = False
    display_cleanup_error = None
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
    telemetry_path = output_dir / "telemetry.jsonl"

    def emit(event: str, **fields: object) -> None:
        if telemetry_writer is None:
            raise RuntimeError("telemetry writer is unavailable")
        telemetry_overheads_ms.append(telemetry_writer.write(event, **fields))

    try:
        if (
            sim_config.prompt_schedule == STAGED_PROMPT_SCHEDULE
            and remote_config.policy_profile.name != "pi05_aloha_base"
        ):
            raise ValueError("the staged prompt schedule is limited to the pi05_aloha_base diagnostic")
        telemetry_writer = JsonlWriter(telemetry_path)
        versions = package_versions()
        environment = _make_environment(sim_config)
        scene_id = getattr(environment.unwrapped, "scene_hash", None) if sim_config.scenario.is_custom else None
        if sim_config.scenario.is_custom and (
            not isinstance(scene_id, str)
            or len(scene_id) != 64
            or any(character not in "0123456789abcdef" for character in scene_id)
        ):
            raise ValueError("custom environment scene hash is invalid")
        emit(
            "metadata",
            run_id=run_id,
            profile=remote_config.policy_profile.name,
            checkpoint_label=remote_config.policy_profile.checkpoint_label,
            source_sha=source_sha,
            upstream_sha=upstream_sha,
            seeds=[seed],
            task=sim_config.task,
            scenario=sim_config.scenario.key,
            scene_hash=scene_id,
            target_area_coverage_method=(TARGET_AREA_COVERAGE_METHOD if sim_config.scenario.is_custom else None),
            camera_views=list(POLICY_CAMERA_VIEWS),
            prompt_schedule=sim_config.prompt_schedule,
            prompt_stage_count=(len(_STAGED_PROMPTS) if sim_config.prompt_schedule == STAGED_PROMPT_SCHEDULE else 1),
            prompt_stage_boundaries=(
                list(STAGED_PROMPT_BOUNDARIES) if sim_config.prompt_schedule == STAGED_PROMPT_SCHEDULE else None
            ),
            action_horizon=sim_config.action_horizon,
            model_action_horizon=remote_config.policy_profile.action_horizon,
            prefetch_steps=sim_config.prefetch_steps,
            package_versions=versions,
        )
        transport, server_metadata = _connect_with_retry(remote_config, source_sha, connection, emit=emit)
        policy = BufferedPolicy(
            transport,
            remote_config.policy_profile,
            sim_config.action_horizon,
            sim_config.prefetch_steps,
            remote_config.policy_close_timeout_seconds,
            emit,
        )
        if (
            environment.spec is None
            or environment.spec.max_episode_steps != sim_config.episode_steps
            or environment.metadata.get("render_fps") != 50
            or tuple(environment.action_space.shape) != (remote_config.policy_profile.action_dimension,)
        ):
            raise ValueError("pinned ALOHA environment must expose the configured step limit, 50 fps, and 14 actions")
        result = control_episode(
            environment,
            policy,
            video,
            seed=seed,
            prompt=sim_config.scenario.prompt or remote_config.policy_profile.default_prompt,
            profile=remote_config.policy_profile,
            prompt_schedule=sim_config.prompt_schedule,
            scenario=sim_config.scenario,
            expected_scene_hash=scene_id,
            display=display,
            max_steps=sim_config.episode_steps,
            progress=result,
            emit=emit,
        )
        control_finished = True
        status = "complete"
    except BaseException as error:
        primary = error
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        errors.append({"stage": "control", "type": type(error).__name__, "message": str(error)[:500]})
        try:
            emit("error", stage="control", error_type=type(error).__name__)
        except BaseException as telemetry_error:
            errors.append(
                {
                    "stage": "telemetry_error",
                    "type": type(telemetry_error).__name__,
                    "message": str(telemetry_error)[:500],
                }
            )
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
        cleanup_pending = False
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
                cleanup_pending = True
                errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)[:500]})
                if primary is None:
                    primary = error
                    status = "failed"
        try:
            display.on_episode_end()
        except Exception as error:
            display_cleanup_error = {"type": type(error).__name__, "message": str(error)[:500]}
        video_error = None
        video_validation = None
        video_status = "no_frames" if video.frame_count == 0 else "encode_failed"
        try:
            video.on_episode_end()
            if video.output_path is not None:
                video_validation = verify_video(video.output_path, video.frame_count, (224, 672, 3))
                video_status = "complete" if control_finished else "partial"
        except BaseException as error:
            video_error = error
            errors.append({"stage": "video", "type": type(error).__name__, "message": str(error)[:500]})
            if primary is None:
                primary = error
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if video.frame_count != int(result.get("steps_applied", 0)):
            error = ValueError("video frames must match successfully applied steps")
            errors.append({"stage": "video_coverage", "type": type(error).__name__, "message": str(error)})
            if primary is None:
                primary = error
                status = "failed"

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
        infrastructure_pass = status == "complete" and video_validation is not None and not errors
        trajectory_path = output_dir / "joint-trajectory.png"
        trajectory_plot_id = f"{run_id}-seed-{seed}-joint-trajectory"
        if telemetry_writer is not None:
            try:
                trajectory_events = read_jsonl(telemetry_path).events
                trajectory = summarize_trajectory(
                    trajectory_events,
                    int(result.get("steps_applied", 0)),
                )
                trajectory = write_trajectory_plot(
                    trajectory_events,
                    int(result.get("steps_applied", 0)),
                    trajectory_path,
                    trajectory_plot_id,
                )
            except BaseException as error:
                trajectory = {**trajectory, "plot_status": "failed", "plot_id": None}
                errors.append({"stage": "trajectory", "type": type(error).__name__, "message": str(error)[:500]})
                infrastructure_pass = False
                if isinstance(error, KeyboardInterrupt):
                    status = "interrupted"
                elif status == "complete":
                    status = "failed"
                if primary is None:
                    primary = error
        if status == "complete" and (
            trajectory.get("plot_status") != "passed"
            or trajectory.get("sample_count") != result.get("steps_applied")
            or trajectory.get("step_coverage") != 1.0
            or (sim_config.scenario.is_custom and result.get("coverage_sample_count") != result.get("steps_applied"))
        ):
            error = ValueError("complete episode trajectory coverage or plot status is invalid")
            errors.append({"stage": "trajectory", "type": type(error).__name__, "message": str(error)})
            infrastructure_pass = False
            status = "failed"
            if primary is None:
                primary = error
        telemetry_summary = None
        telemetry_writer_closed = False
        try:
            wait_metrics = {
                "dropped_leading_actions": int(stats.get("dropped_leading_actions", 0)),
            }
            emit("wait", metrics=wait_metrics)
            terminal_metrics = {}
            telemetry_p95_ms = _percentile(telemetry_overheads_ms, 95)
            if telemetry_p95_ms is not None:
                terminal_metrics["telemetry_write_ms"] = telemetry_p95_ms
            if isinstance(result.get("active_step_hz"), int | float):
                terminal_metrics["active_step_hz"] = result["active_step_hz"]
            if isinstance(result.get("wall_step_hz"), int | float):
                terminal_metrics["wall_episode_hz"] = result["wall_step_hz"]
            scenario_result: dict[str, object] = {}
            final_info = result.get("final_info")
            if sim_config.scenario.is_custom and isinstance(final_info, Mapping):
                scenario_result = {
                    "push_success": int(final_info.get("is_success") is True),
                    "lifted_count": int(final_info.get("lifted_ever") is True),
                    "off_table_count": int(final_info.get("off_table") is True),
                    "fallen_count": int(final_info.get("fallen") is True),
                    "left_contact_count": int(final_info.get("left_contact_ever") is True),
                    "right_contact_count": int(final_info.get("right_contact_ever") is True),
                    "both_arms_count": int(final_info.get("both_arms_participated") is True),
                    "interference_count": int(final_info.get("interference_ever") is True),
                    "time_limit_count": int(final_info.get("terminal_reason") == "time_limit"),
                    "videos_passed": int(video_validation is not None and video_status == "complete"),
                    "coverage_sample_count": int(result.get("coverage_sample_count", 0)),
                    "initial_target_area_coverage_percent": result.get("initial_target_area_coverage_percent"),
                    "final_target_area_coverage_percent": result.get("final_target_area_coverage_percent"),
                    "best_target_area_coverage_percent": result.get("best_target_area_coverage_percent"),
                    "best_target_area_coverage_step": result.get("best_target_area_coverage_step"),
                    "time_to_best_target_area_coverage_seconds": result.get(
                        "time_to_best_target_area_coverage_seconds"
                    ),
                    "episode_elapsed_seconds": result.get("wall_seconds"),
                }
                for key in (
                    "initial_target_area_coverage_percent",
                    "final_target_area_coverage_percent",
                    "best_target_area_coverage_percent",
                    "time_to_best_target_area_coverage_seconds",
                    "episode_elapsed_seconds",
                ):
                    value = scenario_result[key]
                    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
                        terminal_metrics[key] = value
                for body_index in range(2):
                    for suffix in ("xy_error", "yaw_error", "roll", "pitch", "height_error"):
                        key = f"body_{body_index}_{suffix}"
                        value = final_info.get(key)
                        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
                            terminal_metrics[key] = value
            emit(
                "terminal",
                status=status,
                episodes=1,
                infrastructure_pass=infrastructure_pass,
                steps_applied=int(result.get("steps_applied", 0)),
                request_count=int(stats.get("request_count", 0)),
                retries=connection["retries"],
                failures=connection["failures"],
                task_success=result.get("task_success"),
                reward_sum=result.get("reward_sum", 0.0),
                reward_max=result.get("reward_max"),
                reward_final=result.get("reward_final"),
                trajectory_sample_count=trajectory["sample_count"],
                trajectory_joint_count=trajectory["joint_count"],
                trajectory_step_coverage=trajectory["step_coverage"],
                trajectory_plot_status=trajectory["plot_status"],
                trajectory_plot_id=trajectory["plot_id"],
                video_ids=[f"{run_id}-seed-{seed}"],
                metrics=terminal_metrics,
                **scenario_result,
            )
        except BaseException as error:
            errors.append({"stage": "telemetry", "type": type(error).__name__, "message": str(error)[:500]})
            infrastructure_pass = False
            status = "failed" if status == "complete" else status
            if primary is None:
                primary = error
        finally:
            if telemetry_writer is not None:
                try:
                    telemetry_writer.close()
                    telemetry_writer_closed = True
                except BaseException as error:
                    errors.append(
                        {"stage": "telemetry_close", "type": type(error).__name__, "message": str(error)[:500]}
                    )
                    infrastructure_pass = False
                    status = "failed" if status == "complete" else status
                    if primary is None:
                        primary = error
        if telemetry_writer_closed:
            try:
                telemetry_summary = aggregate_jsonl(telemetry_path)
                write_summary(output_dir / "telemetry-summary.json", telemetry_summary, publishable=True)
                write_markdown_summary(output_dir / "telemetry-summary.md", telemetry_summary)
            except BaseException as error:
                errors.append({"stage": "telemetry_summary", "type": type(error).__name__, "message": str(error)[:500]})
                infrastructure_pass = False
                status = "failed" if status == "complete" else status
                if primary is None:
                    primary = error

        manifest = {
            "schema": 1,
            "status": status,
            "cleanup_pending": cleanup_pending,
            "infrastructure_pass": infrastructure_pass,
            "profile": remote_config.policy_profile.name,
            "experimental_profile": remote_config.policy_profile.experimental,
            "policy_backend": remote_config.policy_backend,
            "checkpoint_label": remote_config.policy_profile.checkpoint_label,
            "wire_action_horizon": remote_config.policy_profile.action_horizon,
            "wire_action_dimension": remote_config.policy_profile.action_dimension,
            "server_metadata": server_metadata,
            "package_versions": versions,
            "source_sha": source_sha,
            "upstream_sha": upstream_sha,
            "task": sim_config.task,
            "scenario": sim_config.scenario.key,
            "scene_hash": scene_id,
            "seed": seed,
            "action_horizon": sim_config.action_horizon,
            "prefetch_steps": sim_config.prefetch_steps,
            "prompt_schedule": sim_config.prompt_schedule,
            "prefetch_budget_ms": budget_ms,
            "inference_margin_ms": _INFERENCE_MARGIN_MS,
            "warmed_request_p95_ms": warmed_p95_ms,
            "prefetch_budget_qualified": prefetch_qualified,
            "uninterrupted_50hz_claimed": uninterrupted_50hz,
            "buffer": stats,
            "connection": connection,
            "episode": result,
            "telemetry": {
                "path": str(telemetry_path),
                "summary": telemetry_summary,
                "writer_closed": telemetry_writer_closed,
                "write_p95_ms": _percentile(telemetry_overheads_ms, 95),
            },
            "trajectory": {
                **trajectory,
                "path": str(trajectory_path) if trajectory_path.is_file() else None,
            },
            "video": {
                "id": f"{run_id}-seed-{seed}",
                "camera_views": list(POLICY_CAMERA_VIEWS),
                "layout": "horizontal",
                "status": video_status,
                "path": str(video.output_path) if video.output_path is not None else None,
                "frames": video.frame_count,
                "validation": video_validation,
                "error": str(video_error)[:500] if video_error is not None else None,
            },
            "display": {
                "enabled": sim_config.display,
                "status": "failed"
                if result.get("display_error") is not None or display_cleanup_error is not None
                else "passed"
                if sim_config.display
                else "disabled",
                "error": result.get("display_error") or display_cleanup_error,
            },
            "errors": errors,
        }
        _atomic_json(manifest_path, manifest)
    if primary is not None:
        raise primary
    return {**manifest, "manifest": str(manifest_path)}


def _gpu_events(
    path: Path,
    run_id: str,
    profile: str,
    source_sha: str,
    expected_interval_seconds: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not path.exists():
        return [], {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (
        not rows
        or not isinstance(rows[0], dict)
        or not isinstance(rows[-1], dict)
        or rows[0].get("event") != "sampler_started"
        or rows[-1].get("event") != "sampler_stopped"
        or rows[-1].get("status") not in {"interrupted", "passed"}
    ):
        raise ValueError("GPU telemetry must contain start and terminal events")
    terminal_status = rows[-1].get("status")
    exit_status = rows[-1].get("exit_status")
    if (
        isinstance(exit_status, bool)
        or not isinstance(exit_status, int)
        or (terminal_status == "passed" and exit_status != 0)
        or (terminal_status == "interrupted" and exit_status not in {129, 130, 143})
    ):
        raise ValueError("GPU telemetry terminal status is inconsistent")
    if (
        sum(isinstance(row, dict) and row.get("event") == "sampler_started" for row in rows) != 1
        or sum(isinstance(row, dict) and row.get("event") == "sampler_stopped" for row in rows) != 1
    ):
        raise ValueError("GPU telemetry start and terminal events must be unique")
    server_pid = rows[0].get("server_pid")
    interval_ms = rows[0].get("interval_ms")
    if (
        isinstance(server_pid, bool)
        or not isinstance(server_pid, int)
        or server_pid <= 1
        or isinstance(interval_ms, bool)
        or not isinstance(interval_ms, int)
        or not 100 <= interval_ms <= 60_000
        or (expected_interval_seconds is not None and interval_ms != round(expected_interval_seconds * 1000))
    ):
        raise ValueError("GPU telemetry server or interval identity is invalid")
    result = []
    elapsed_values = []
    sample_indices = []
    monotonic_values = []
    all_monotonic_values = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema") != 1
            or row.get("run_id") != run_id
            or row.get("profile") != profile
            or row.get("source_sha") != source_sha
            or row.get("server_pid") != server_pid
            or row.get("interval_ms") != interval_ms
        ):
            raise ValueError("GPU telemetry identity is invalid")
        if row.get("event") not in {"sampler_started", "gpu_sample", "sampler_stopped"}:
            raise ValueError("GPU telemetry event is invalid")
        timestamp = row.get("utc")
        monotonic_ns = row.get("monotonic_ns")
        try:
            timestamp_is_utc = (
                isinstance(timestamp, str)
                and timestamp.endswith("Z")
                and datetime.fromisoformat(timestamp[:-1] + "+00:00").tzinfo == timezone.utc  # noqa: UP017 (Python 3.10)
            )
        except ValueError:
            timestamp_is_utc = False
        if (
            not timestamp_is_utc
            or isinstance(monotonic_ns, bool)
            or not isinstance(monotonic_ns, int)
            or monotonic_ns < 0
        ):
            raise ValueError("GPU telemetry clock fields are invalid")
        all_monotonic_values.append(monotonic_ns)
        if row.get("event") != "gpu_sample":
            continue
        metrics = {
            "gpu_memory_mib": row.get("memory_used_mib"),
            "gpu_utilization_percent": row.get("utilization_percent"),
            "server_rss_kib": row.get("server_rss_kib"),
        }
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0
                for value in metrics.values()
            )
            or metrics["gpu_utilization_percent"] > 100
        ):
            raise ValueError("GPU telemetry sample is invalid")
        elapsed_ms = row.get("elapsed_ms")
        sample_index = row.get("sample_index")
        if (
            isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, int)
            or elapsed_ms < 0
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
        ):
            raise ValueError("GPU telemetry sequence is invalid")
        elapsed_values.append(elapsed_ms)
        sample_indices.append(sample_index)
        monotonic_values.append(monotonic_ns)
        result.append(
            {
                "schema": 1,
                "event": "gpu",
                "timestamp_utc": row.get("utc"),
                "monotonic_ns": row.get("monotonic_ns"),
                "metrics": metrics,
            }
        )
    if len(result) < 2:
        raise ValueError("GPU telemetry must contain at least two samples")
    if (
        sample_indices != list(range(len(result)))
        or any(later <= earlier for earlier, later in pairwise(elapsed_values))
        or any(later <= earlier for earlier, later in pairwise(monotonic_values))
        or any(later <= earlier for earlier, later in pairwise(all_monotonic_values))
    ):
        raise ValueError("GPU telemetry samples are out of order")
    gaps = [later - earlier for earlier, later in pairwise(elapsed_values)]
    max_gap_ms = max(gaps, default=0)
    if max_gap_ms > max(interval_ms * 2.5, interval_ms + 500):
        raise ValueError("GPU telemetry cadence gap exceeds the bounded interval")
    return result, {
        "gpu_sample_count": len(result),
        "gpu_span_ms": elapsed_values[-1] - elapsed_values[0],
        "gpu_max_gap_ms": max_gap_ms,
    }


def _clock_evidence(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("clock correlation evidence is invalid")
    uncertainties = []
    offsets = []
    for key in ("start", "end"):
        sample = payload.get(key)
        if not isinstance(sample, dict):
            raise ValueError("clock correlation evidence is invalid")
        uncertainty = sample.get("round_trip_uncertainty_ms")
        offset = sample.get("wsl_minus_mac_midpoint_ms")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or (name == "uncertainty" and value < 0)
            for name, value in (("uncertainty", uncertainty), ("offset", offset))
        ):
            raise ValueError("clock correlation measurements are invalid")
        uncertainties.append(float(uncertainty))
        offsets.append(float(offset))
    return {
        "clock_max_uncertainty_ms": max(uncertainties),
        "clock_offset_change_ms": offsets[-1] - offsets[0],
    }


def _utc_milliseconds(value: object) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("telemetry UTC timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("telemetry UTC timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:  # noqa: UP017 (Python 3.10)
        raise ValueError("telemetry UTC timestamp is invalid")
    return parsed.timestamp() * 1000


def _gpu_coverage(
    gpu_events: list[dict[str, object]],
    mac_event_groups: list[list[dict[str, object]]],
    clock_path: Path,
    interval_seconds: float,
) -> dict[str, object]:
    if not gpu_events or not mac_event_groups or any(not group for group in mac_event_groups):
        raise ValueError("GPU coverage requires GPU and Mac telemetry")
    evidence = _clock_evidence(clock_path)
    clock = json.loads(clock_path.read_text(encoding="utf-8"))
    start = clock["start"]
    end = clock["end"]
    start_offset = float(start["wsl_minus_mac_midpoint_ms"])
    end_offset = float(end["wsl_minus_mac_midpoint_ms"])
    gpu_start_ms = _utc_milliseconds(gpu_events[0]["timestamp_utc"]) - start_offset
    gpu_end_ms = _utc_milliseconds(gpu_events[-1]["timestamp_utc"]) - end_offset
    mac_start_ms = min(_utc_milliseconds(group[0]["timestamp_utc"]) for group in mac_event_groups)
    mac_end_ms = max(_utc_milliseconds(group[-1]["timestamp_utc"]) for group in mac_event_groups)
    tolerance_ms = interval_seconds * 1000 + float(evidence["clock_max_uncertainty_ms"])
    evidence["gpu_coverage_pass"] = (
        gpu_start_ms <= mac_start_ms + tolerance_ms and gpu_end_ms >= mac_end_ms - tolerance_ms
    )
    return evidence


def _write_performance_summary(root: Path, summary: Mapping[str, object], gpu_path: Path | None = None) -> None:
    episodes = summary.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("a performance summary requires at least one episode manifest")
    manifests = [episode for episode in episodes if isinstance(episode, Mapping)]
    if len(manifests) != len(episodes):
        raise ValueError("performance summary episodes must be manifests")
    event_groups = [read_jsonl(Path(str(episode["telemetry"]["path"]))).events for episode in manifests]
    first = dict(event_groups[0][0])
    first["seeds"] = [episode["seed"] for episode in episodes]
    events = [first]
    for group in event_groups:
        for event in group[1:]:
            if event.get("event") == "episode":
                continue
            combined_event = event
            if event.get("event") == "terminal":
                combined_event = {**event, "event": "episode"}
                combined_event.pop("status", None)
            events.append(combined_event)
    if gpu_path is not None:
        gpu_events, gpu_result = _gpu_events(
            gpu_path,
            str(first["run_id"]),
            str(first["profile"]),
            str(first["source_sha"]),
            float(summary["gpu_metrics_interval_seconds"]),
        )
        events.extend(gpu_events)
        gpu_result.update(
            _gpu_coverage(
                gpu_events,
                event_groups,
                gpu_path.parent / "clock-correlation.json",
                float(summary["gpu_metrics_interval_seconds"]),
            )
        )
        if not gpu_result["gpu_coverage_pass"]:
            raise ValueError("GPU telemetry does not span and bracket the Mac run within clock uncertainty")
    else:
        gpu_result = {}
    final_infos = [episode["episode"].get("final_info", {}) for episode in manifests]
    scenario_result = {}
    scenario_key = first.get("scenario")
    if isinstance(scenario_key, str) and scenario_key in SCENARIOS and SCENARIOS[scenario_key].is_custom:
        scenario_result = {
            "push_success": sum(isinstance(info, Mapping) and info.get("is_success") is True for info in final_infos),
            "lifted_count": sum(isinstance(info, Mapping) and info.get("lifted_ever") is True for info in final_infos),
            "off_table_count": sum(isinstance(info, Mapping) and info.get("off_table") is True for info in final_infos),
            "fallen_count": sum(isinstance(info, Mapping) and info.get("fallen") is True for info in final_infos),
            "left_contact_count": sum(
                isinstance(info, Mapping) and info.get("left_contact_ever") is True for info in final_infos
            ),
            "right_contact_count": sum(
                isinstance(info, Mapping) and info.get("right_contact_ever") is True for info in final_infos
            ),
            "both_arms_count": sum(
                isinstance(info, Mapping) and info.get("both_arms_participated") is True for info in final_infos
            ),
            "interference_count": sum(
                isinstance(info, Mapping) and info.get("interference_ever") is True for info in final_infos
            ),
            "time_limit_count": sum(
                isinstance(info, Mapping) and info.get("terminal_reason") == "time_limit" for info in final_infos
            ),
            "videos_passed": sum(
                episode.get("video", {}).get("status") == "complete"
                and episode.get("video", {}).get("frames") == episode.get("episode", {}).get("steps_applied")
                for episode in manifests
            ),
            "coverage_sample_count": sum(
                int(episode["episode"].get("coverage_sample_count", 0)) for episode in manifests
            ),
        }
    rewards = [episode["episode"].get("reward_max") for episode in manifests]
    reward_max = max((value for value in rewards if isinstance(value, int | float)), default=None)
    trajectory_sample_count = sum(int(episode.get("trajectory", {}).get("sample_count", 0)) for episode in manifests)
    trajectory_steps = sum(int(episode["episode"].get("steps_applied", 0)) for episode in manifests)
    trajectory_plot_ids = [
        str(episode["trajectory"]["plot_id"])
        for episode in manifests
        if isinstance(episode.get("trajectory"), Mapping) and episode["trajectory"].get("plot_status") == "passed"
    ]
    trajectory_statuses = {
        episode["trajectory"].get("plot_status")
        for episode in manifests
        if isinstance(episode.get("trajectory"), Mapping)
    }
    if len(trajectory_plot_ids) == len(manifests):
        trajectory_plot_status = "passed"
    elif "failed" in trajectory_statuses:
        trajectory_plot_status = "failed"
    elif not trajectory_plot_ids:
        trajectory_plot_status = "no_samples"
    else:
        trajectory_plot_status = "partial"
    events.append(
        {
            "schema": 1,
            "event": "terminal",
            "timestamp_utc": datetime.now(timezone.utc)  # noqa: UP017 (Python 3.10)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns(),
            "status": summary.get("status", "failed"),
            "episodes": len(manifests),
            "infrastructure_pass": all(bool(episode.get("infrastructure_pass")) for episode in manifests),
            "request_count": sum(int(episode["buffer"].get("request_count", 0)) for episode in manifests),
            "retries": sum(int(episode["connection"].get("retries", 0)) for episode in manifests),
            "failures": sum(int(episode["connection"].get("failures", 0)) for episode in manifests),
            "steps_applied": sum(int(episode["episode"].get("steps_applied", 0)) for episode in manifests),
            "task_success": sum(episode["episode"].get("task_success") is True for episode in manifests),
            "reward_sum": sum(float(episode["episode"].get("reward_sum", 0)) for episode in manifests),
            "reward_max": reward_max,
            "trajectory_sample_count": trajectory_sample_count,
            "trajectory_joint_count": 14,
            "trajectory_step_coverage": (1.0 if trajectory_steps == 0 else trajectory_sample_count / trajectory_steps),
            "trajectory_plots_passed": len(trajectory_plot_ids),
            "trajectory_plot_status": trajectory_plot_status,
            "trajectory_plot_ids": trajectory_plot_ids,
            "video_ids": [f"{first['run_id']}-seed-{episode['seed']}" for episode in manifests],
            **scenario_result,
            **gpu_result,
        }
    )
    performance = aggregate_events(events)
    write_summary(root / "performance-summary.json", performance, publishable=True)
    write_markdown_summary(root / "performance-summary.md", performance)


def run() -> dict[str, object]:
    sim_config = load_mac_sim_config()
    remote_config = load_remote_config()
    if sim_config.prompt_schedule == STAGED_PROMPT_SCHEDULE and remote_config.policy_profile.name != "pi05_aloha_base":
        raise ValueError("the staged prompt schedule is limited to the pi05_aloha_base diagnostic")
    if remote_config.policy_backend != "pytorch":
        raise RemoteError("Phase 5 requires OPENPI_POLICY_BACKEND=pytorch on the validated 24 GiB PC")
    _, source_sha = verify_ready_tunnel(remote_config)
    upstream_sha = UPSTREAM_SHA
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    run_id = uuid.uuid4().hex
    output = validate_output_root(sim_config.output_dir)
    root = (
        output / "scenarios_0827" / timestamp / remote_config.policy_profile.name / sim_config.scenario.key
        if sim_config.scenario.is_custom
        else output / "phase05" / timestamp / remote_config.policy_profile.name
    )
    summary = {
        "status": "running",
        "run_id": run_id,
        "profile": remote_config.policy_profile.name,
        "scenario": sim_config.scenario.key,
        "source_sha": source_sha,
        "gpu_metrics_interval_seconds": remote_config.gpu_metrics_interval_seconds,
        "episodes": [],
        "error": None,
    }
    summary_path = root / "summary.json"
    _atomic_json(summary_path, summary)
    sampler = None
    gpu_path = None
    primary: BaseException | None = None
    try:
        sampler = start_gpu_sampler(remote_config, run_id, source_sha, root)
        for seed in range(sim_config.seed, sim_config.seed + sim_config.episodes):
            sampler.check()
            seed_dir = root / f"seed-{seed}"
            try:
                episode = _run_seed(
                    sim_config,
                    remote_config,
                    source_sha,
                    upstream_sha,
                    seed,
                    seed_dir,
                    run_id,
                )
            except BaseException:
                manifest_path = seed_dir / "manifest.json"
                if manifest_path.exists():
                    failed_episode = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(failed_episode, dict):
                        summary["episodes"].append({**failed_episode, "manifest": str(manifest_path)})
                raise
            summary["episodes"].append(episode)
            sampler.check()
            _atomic_json(summary_path, summary)
    except BaseException as error:
        primary = error
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        if sampler is not None:
            try:
                gpu_path = sampler.stop(root)
            except BaseException as error:
                summary["status"] = "failed"
                summary["sampler_cleanup_error"] = {"type": type(error).__name__, "message": str(error)[:500]}
                if primary is None:
                    primary = error
        _atomic_json(summary_path, summary)
    if primary is None:
        summary["status"] = "passed" if all(item["infrastructure_pass"] for item in summary["episodes"]) else "failed"
    _atomic_json(summary_path, summary)
    try:
        _write_performance_summary(root, summary, gpu_path)
    except BaseException as error:
        summary["status"] = "failed"
        summary["performance_summary_error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        _atomic_json(summary_path, summary)
        if primary is None:
            primary = error
    if primary is not None:
        raise primary
    return {**summary, "summary": str(summary_path)}


def main() -> None:
    try:
        print(json.dumps(run(), allow_nan=False, sort_keys=True))
    except (RemoteError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
