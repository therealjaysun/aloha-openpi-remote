from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from datetime import timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.connection_check import verify_ready_tunnel
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.remote import UPSTREAM_SHA
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.remote import _candidate_sha
from tools.remote_aloha.remote import start_gpu_sampler
from tools.remote_aloha.run import _atomic_json
from tools.remote_aloha.run import _gpu_coverage
from tools.remote_aloha.run import _gpu_events
from tools.remote_aloha.run import _run_seed
from tools.remote_aloha.run import _scenario_info_fields
from tools.remote_aloha.run import _scenario_step_info
from tools.remote_aloha.run import _status
from tools.remote_aloha.run import _write_performance_summary
from tools.remote_aloha.scenarios import CUSTOM_SCENARIOS
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import TARGET_AREA_COVERAGE_METHOD
from tools.remote_aloha.scenarios import body_descriptors
from tools.remote_aloha.scenarios import descriptor_sha256
from tools.remote_aloha.scenarios import effective_layout_seed
from tools.remote_aloha.scenarios import sample_layout
from tools.remote_aloha.sim_smoke_test import verify_video
from tools.remote_aloha.telemetry import read_jsonl
from tools.remote_aloha.trajectory import summarize_trajectory
from tools.remote_aloha.trajectory import validate_joint_vector

_SEEDS = (0, 1, 2)
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _pose_rows(value: object, names: tuple[str, ...], label: str) -> list[list[object]]:
    if not isinstance(value, list) or len(value) != len(names):
        raise ValueError(f"{label} must contain one named pose per body")
    rows = []
    for expected_name, row in zip(names, value, strict=True):
        if not isinstance(row, list) or len(row) != 8 or row[0] != expected_name:
            raise ValueError(f"{label} pose identity is invalid")
        numbers = row[1:]
        if any(
            isinstance(number, bool) or not isinstance(number, int | float) or not math.isfinite(number)
            for number in numbers
        ):
            raise ValueError(f"{label} pose values are invalid")
        if (
            numbers[4] != 0
            or numbers[5] != 0
            or not math.isclose(sum(float(number) ** 2 for number in numbers[3:]), 1.0, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(f"{label} pose quaternion is invalid")
        rows.append(row)
    return rows


def _pose_rows_hash(rows: list[list[object]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _episode_result(
    manifest: Mapping[str, object], scenario_key: str, seed: int, run_id: str, batch_root: Path | None
) -> dict[str, object]:
    scenario = SCENARIOS[scenario_key]
    profile_name = manifest.get("profile")
    source_sha = manifest.get("source_sha")
    if (
        manifest.get("status") != "complete"
        or manifest.get("infrastructure_pass") is not True
        or manifest.get("cleanup_pending") is not False
        or manifest.get("errors") != []
        or manifest.get("policy_backend") != "pytorch"
        or manifest.get("scenario") != scenario_key
        or manifest.get("task") != scenario.gym_id
        or manifest.get("seed") != seed
        or profile_name not in POLICY_PROFILES
        or not isinstance(source_sha, str)
    ):
        raise ValueError("matrix episode identity or infrastructure status is invalid")
    try:
        validate_server_metadata(manifest.get("server_metadata"), POLICY_PROFILES[profile_name], source_sha, "pytorch")
    except ValueError as error:
        raise ValueError("matrix server metadata is invalid") from error
    scene_hash = manifest.get("scene_hash")
    if not isinstance(scene_hash, str) or not _HASH.fullmatch(scene_hash):
        raise ValueError("matrix episode scene hash is invalid")

    episode = _mapping(manifest.get("episode"), "matrix episode result")
    steps = episode.get("steps_applied")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 300:
        raise ValueError("matrix episode step count is invalid")
    reset = _mapping(episode.get("reset_info"), "matrix reset info")
    final = _mapping(episode.get("final_info"), "matrix final info")
    fields = _scenario_info_fields(scenario)
    if not fields <= set(reset):
        raise ValueError("matrix reset scenario fields are incomplete")
    reset_step = _scenario_step_info({key: reset[key] for key in fields}, scenario)
    final_step = _scenario_step_info(final, scenario)
    names = tuple(body.name for body in body_descriptors(str(scenario.object_kind)))
    sampled_poses = _pose_rows(reset.get("sampled_poses"), names, "matrix sampled poses")
    settled_poses = _pose_rows(reset.get("settled_poses"), names, "matrix settled poses")
    provenance = _mapping(reset.get("layout_provenance"), "matrix layout provenance")
    attempt = provenance.get("attempt")
    if (
        set(provenance) != {"requested_seed", "effective_seed", "attempt"}
        or provenance.get("requested_seed") != seed
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or provenance.get("effective_seed") != effective_layout_seed(seed, attempt)
        or sampled_poses
        != [
            [pose.name, *pose.vector()]
            for pose in sample_layout(str(scenario.object_kind), provenance["effective_seed"])
        ]
    ):
        raise ValueError("matrix layout provenance is invalid")
    if (
        reset_step["scene_hash"] != scene_hash
        or final_step["scene_hash"] != scene_hash
        or reset_step["layout_hash"] != final_step["layout_hash"]
        or reset.get("descriptor_sha256") != descriptor_sha256()
        or _pose_rows_hash(settled_poses) != reset_step["layout_hash"]
    ):
        raise ValueError("matrix episode scenario identity changed")
    reason = final_step["terminal_reason"]
    if reason not in {"success", "off_table", "fallen", "time_limit"} or final_step["is_success"] is not (
        reason == "success"
    ):
        raise ValueError("matrix final terminal reason is invalid")
    if reason == "time_limit":
        if steps != 300 or episode.get("truncated") is not True or episode.get("terminated") is not False:
            raise ValueError("matrix time-limit episode is inconsistent")
    elif episode.get("terminated") is not True or episode.get("truncated") is not False:
        raise ValueError("matrix task-terminal episode must be terminated")

    validate_joint_vector(reset.get("home_joint_positions"), "home_joint_positions")
    telemetry = _mapping(manifest.get("telemetry"), "matrix telemetry")
    telemetry_path = Path(str(telemetry.get("path")))
    read_result = read_jsonl(telemetry_path)
    events = read_result.events
    if (
        telemetry.get("writer_closed") is not True
        or read_result.partial_final_line_ignored
        or not events
        or events[0].get("event") != "metadata"
        or events[-1].get("event") != "terminal"
    ):
        raise ValueError("matrix telemetry lifecycle is invalid")
    metadata = events[0]
    if (
        metadata.get("scenario") != scenario_key
        or metadata.get("task") != scenario.gym_id
        or metadata.get("scene_hash") != scene_hash
        or metadata.get("seeds") != [seed]
        or metadata.get("profile") != manifest.get("profile")
        or metadata.get("checkpoint_label") != manifest.get("checkpoint_label")
        or metadata.get("source_sha") != manifest.get("source_sha")
        or metadata.get("upstream_sha") != manifest.get("upstream_sha")
        or metadata.get("run_id") != run_id
        or metadata.get("target_area_coverage_method") != TARGET_AREA_COVERAGE_METHOD
    ):
        raise ValueError("matrix telemetry metadata is invalid")
    step_events = [event for event in events if event.get("event") == "step"]
    if len(step_events) != steps:
        raise ValueError("matrix telemetry step coverage is invalid")
    minimum_errors = {name: {"xy": math.inf, "yaw": math.inf} for name in names}
    maximum_coverage = {name: 0.0 for name in names}
    best_coverage = -1.0
    best_coverage_step = None
    best_coverage_elapsed = None
    previous_elapsed = -1.0
    time_to_success_step = None
    last_step_info = None
    for index, event in enumerate(step_events):
        if event.get("step") != index or event.get("applied_step") != index + 1:
            raise ValueError("matrix telemetry step sequence is invalid")
        elapsed = event.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < previous_elapsed
        ):
            raise ValueError("matrix telemetry elapsed time is invalid")
        previous_elapsed = float(elapsed)
        validate_joint_vector(event.get("commanded_joint_positions"), "commanded_joint_positions")
        validate_joint_vector(event.get("actual_joint_positions"), "actual_joint_positions")
        info = _scenario_step_info(event.get("scenario_info"), scenario)
        if info["scene_hash"] != scene_hash or info["layout_hash"] != reset_step["layout_hash"]:
            raise ValueError("matrix telemetry scenario identity changed")
        last_step_info = info
        for body_index, name in enumerate(names):
            minimum_errors[name]["xy"] = min(minimum_errors[name]["xy"], info[f"body_{body_index}_xy_error"])
            minimum_errors[name]["yaw"] = min(minimum_errors[name]["yaw"], info[f"body_{body_index}_yaw_error"])
            maximum_coverage[name] = max(maximum_coverage[name], info[f"body_{body_index}_target_area_coverage"])
        if info["target_area_coverage"] > best_coverage:
            best_coverage = info["target_area_coverage"]
            best_coverage_step = index + 1
            best_coverage_elapsed = float(elapsed)
        if time_to_success_step is None and info["is_success"] is True:
            time_to_success_step = index + 1
    if last_step_info != final_step or episode.get("task_success") is not final_step["is_success"]:
        raise ValueError("matrix final telemetry does not match the episode result")
    if (
        episode.get("coverage_method") != TARGET_AREA_COVERAGE_METHOD
        or episode.get("coverage_sample_count") != steps
        or episode.get("initial_target_area_coverage_percent") != reset_step["target_area_coverage"] * 100
        or episode.get("final_target_area_coverage_percent") != final_step["target_area_coverage"] * 100
        or episode.get("best_target_area_coverage_percent") != best_coverage * 100
        or episode.get("best_target_area_coverage_step") != best_coverage_step
        or episode.get("time_to_best_target_area_coverage_seconds") != best_coverage_elapsed
    ):
        raise ValueError("matrix area coverage summary is invalid")

    trajectory = _mapping(manifest.get("trajectory"), "matrix trajectory")
    calculated_trajectory = summarize_trajectory(events, steps)
    if (
        trajectory.get("sample_count") != steps
        or trajectory.get("joint_count") != 14
        or trajectory.get("step_coverage") != 1.0
        or trajectory.get("plot_status") != "passed"
        or trajectory.get("actual_series_count") != 14
        or trajectory.get("commanded_series_count") != 14
        or calculated_trajectory["sample_count"] != steps
    ):
        raise ValueError("matrix trajectory coverage is invalid")
    plot_id = trajectory.get("plot_id")
    plot_path = trajectory.get("path")
    plot_file = Path(str(plot_path))
    if (
        not isinstance(plot_id, str)
        or not _SAFE_ID.fullmatch(plot_id)
        or not isinstance(plot_path, str)
        or not plot_file.is_file()
        or plot_file.stat().st_size <= 0
    ):
        raise ValueError("matrix trajectory artifact is invalid")

    video = _mapping(manifest.get("video"), "matrix video")
    validation = _mapping(video.get("validation"), "matrix video validation")
    video_id = video.get("id")
    video_path = video.get("path")
    video_file = Path(str(video_path))
    if (
        video.get("status") != "complete"
        or video.get("frames") != steps
        or video.get("camera_views") != ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        or video.get("layout") != "horizontal"
        or validation.get("frames") != steps
        or validation.get("fps") != 50.0
        or validation.get("shape") != [224, 672, 3]
        or not isinstance(video_id, str)
        or not _SAFE_ID.fullmatch(video_id)
        or not isinstance(video_path, str)
        or not video_file.is_file()
        or video_file.stat().st_size <= 0
    ):
        raise ValueError("matrix video coverage is invalid")
    resolved = (telemetry_path.resolve(), plot_file.resolve(), video_file.resolve())
    if (
        tuple(path.name for path in resolved) != ("telemetry.jsonl", "joint-trajectory.png", "episode.mp4")
        or len(set(resolved)) != 3
        or len({path.parent for path in resolved}) != 1
        or (batch_root is not None and resolved[0].parent != (batch_root / scenario_key / f"seed-{seed}").resolve())
    ):
        raise ValueError("matrix artifact paths are not bound to the episode")
    if batch_root is not None:
        import imageio.v2 as imageio

        decoded_video = verify_video(video_file, steps, (224, 672, 3))
        decoded_plot = imageio.imread(plot_file)
        if decoded_video != validation or decoded_plot.ndim not in {2, 3} or decoded_plot.size == 0:
            raise ValueError("matrix artifacts do not decode to their recorded validation")
    overhead = telemetry.get("write_p95_ms")
    if (
        isinstance(overhead, bool)
        or not isinstance(overhead, int | float)
        or not math.isfinite(overhead)
        or not 0 <= overhead < 1.0
    ):
        raise ValueError("matrix telemetry overhead exceeds the one-millisecond budget")
    terminal = events[-1]
    expected_counts = {
        "push_success": int(final_step["is_success"] is True),
        "lifted_count": int(final_step["lifted_ever"] is True),
        "off_table_count": int(final_step["off_table"] is True),
        "fallen_count": int(final_step["fallen"] is True),
        "left_contact_count": int(final_step["left_contact_ever"] is True),
        "right_contact_count": int(final_step["right_contact_ever"] is True),
        "both_arms_count": int(final_step["both_arms_participated"] is True),
        "interference_count": int(final_step["interference_ever"] is True),
        "time_limit_count": int(reason == "time_limit"),
        "videos_passed": 1,
        "coverage_sample_count": steps,
    }
    expected_coverage = {
        "initial_target_area_coverage_percent": episode["initial_target_area_coverage_percent"],
        "final_target_area_coverage_percent": episode["final_target_area_coverage_percent"],
        "best_target_area_coverage_percent": episode["best_target_area_coverage_percent"],
        "best_target_area_coverage_step": episode["best_target_area_coverage_step"],
        "time_to_best_target_area_coverage_seconds": episode["time_to_best_target_area_coverage_seconds"],
        "episode_elapsed_seconds": episode["wall_seconds"],
    }
    if (
        terminal.get("status") != "complete"
        or terminal.get("episodes") != 1
        or terminal.get("infrastructure_pass") is not True
        or terminal.get("steps_applied") != steps
        or terminal.get("trajectory_sample_count") != steps
        or terminal.get("trajectory_joint_count") != 14
        or terminal.get("trajectory_step_coverage") != 1.0
        or terminal.get("trajectory_plot_status") != "passed"
        or terminal.get("trajectory_plot_id") != plot_id
        or terminal.get("video_ids") != [video_id]
        or any(terminal.get(key) != value for key, value in expected_counts.items())
        or any(terminal.get(key) != value for key, value in expected_coverage.items())
    ):
        raise ValueError("matrix terminal telemetry does not match the episode manifest")
    return {
        "manifest": manifest,
        "steps": steps,
        "scene_hash": scene_hash,
        "layout_hash": reset_step["layout_hash"],
        "sampled_poses": sampled_poses,
        "settled_poses": settled_poses,
        "layout_provenance": provenance,
        "final": final_step,
        "minimum_errors": minimum_errors,
        "maximum_coverage": maximum_coverage,
        "best_coverage": best_coverage,
        "best_coverage_step": best_coverage_step,
        "best_coverage_elapsed": best_coverage_elapsed,
        "time_to_success_step": time_to_success_step,
        "trajectory_sample_count": steps,
        "telemetry_write_p95_ms": float(overhead),
    }


def validate_matrix(raw: object, *, require_gpu: bool = True, batch_root: Path | None = None) -> dict[str, object]:
    matrix = _mapping(raw, "matrix")
    batch_id = matrix.get("batch_id")
    profile = matrix.get("profile")
    checkpoint = matrix.get("checkpoint_label")
    source_sha = matrix.get("source_sha")
    upstream_sha = matrix.get("upstream_sha")
    run_id = matrix.get("run_id")
    if (
        matrix.get("schema") != 1
        or matrix.get("status") != "passed"
        or matrix.get("error") is not None
        or matrix.get("cleanup_error") is not None
        or not isinstance(batch_id, str)
        or not _SAFE_ID.fullmatch(batch_id)
        or profile not in {"pi0_aloha_sim", "pi05_aloha_base"}
        or checkpoint != ("pi0_aloha_sim" if profile == "pi0_aloha_sim" else "pi05_base")
        or not isinstance(source_sha, str)
        or not _SHA.fullmatch(source_sha)
        or not isinstance(upstream_sha, str)
        or not _SHA.fullmatch(upstream_sha)
        or upstream_sha != UPSTREAM_SHA
        or matrix.get("descriptor_sha256") != descriptor_sha256()
        or not isinstance(run_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", run_id)
        or matrix.get("seeds") != list(_SEEDS)
        or matrix.get("scenarios") != list(CUSTOM_SCENARIOS)
        or isinstance(matrix.get("gpu_metrics_interval_seconds"), bool)
        or not isinstance(matrix.get("gpu_metrics_interval_seconds"), int | float)
        or not math.isfinite(matrix["gpu_metrics_interval_seconds"])
        or matrix["gpu_metrics_interval_seconds"] <= 0
        or (require_gpu and matrix.get("gpu_coverage_pass") is not True)
    ):
        raise ValueError("matrix identity is invalid")
    runs = _mapping(matrix.get("scenario_runs"), "matrix scenario runs")
    if set(runs) != set(CUSTOM_SCENARIOS):
        raise ValueError("matrix scenario set is invalid")

    checked: dict[tuple[str, int], dict[str, object]] = {}
    results = []
    for scenario_key in CUSTOM_SCENARIOS:
        scenario = SCENARIOS[scenario_key]
        run = _mapping(runs[scenario_key], "matrix scenario run")
        episodes = run.get("episodes")
        if run.get("status") != "passed" or not isinstance(episodes, list) or len(episodes) != len(_SEEDS):
            raise ValueError("matrix scenario run is incomplete")
        if require_gpu and run.get("gpu_coverage_pass") is not True:
            raise ValueError("matrix scenario GPU coverage is invalid")
        for seed, manifest in zip(_SEEDS, episodes, strict=True):
            item = _mapping(manifest, "matrix episode manifest")
            if item.get("profile") != profile or item.get("checkpoint_label") != checkpoint:
                raise ValueError("matrix episode profile is invalid")
            if item.get("source_sha") != source_sha or item.get("upstream_sha") != upstream_sha:
                raise ValueError("matrix episode source identity is invalid")
            checked[(scenario_key, seed)] = _episode_result(item, scenario_key, seed, run_id, batch_root)
        scene_hashes = {checked[(scenario_key, seed)]["scene_hash"] for seed in _SEEDS}
        if len(scene_hashes) != 1:
            raise ValueError("matrix scenario scene hash changed across seeds")
        if len({checked[(scenario_key, seed)]["layout_provenance"]["effective_seed"] for seed in _SEEDS}) != len(
            _SEEDS
        ):
            raise ValueError("matrix scenario reused an effective layout seed")
        final_infos = [checked[(scenario_key, seed)]["final"] for seed in _SEEDS]
        body_results = []
        for body_index, body in enumerate(body_descriptors(str(scenario.object_kind))):
            body_results.append(
                {
                    "body": body.name,
                    "minimum_xy_error_meters": min(
                        checked[(scenario_key, seed)]["minimum_errors"][body.name]["xy"] for seed in _SEEDS
                    ),
                    "minimum_yaw_error_radians": min(
                        checked[(scenario_key, seed)]["minimum_errors"][body.name]["yaw"] for seed in _SEEDS
                    ),
                    "final_xy_error_meters_mean": sum(info[f"body_{body_index}_xy_error"] for info in final_infos)
                    / len(final_infos),
                    "final_yaw_error_radians_mean": sum(info[f"body_{body_index}_yaw_error"] for info in final_infos)
                    / len(final_infos),
                    "maximum_target_area_coverage_percent": max(
                        checked[(scenario_key, seed)]["maximum_coverage"][body.name] * 100 for seed in _SEEDS
                    ),
                    "final_target_area_coverage_percent_mean": sum(
                        info[f"body_{body_index}_target_area_coverage"] * 100 for info in final_infos
                    )
                    / len(final_infos),
                }
            )
        success_steps = [
            checked[(scenario_key, seed)]["time_to_success_step"]
            for seed in _SEEDS
            if checked[(scenario_key, seed)]["time_to_success_step"] is not None
        ]
        results.append(
            {
                "scenario": scenario_key,
                "task": scenario.gym_id,
                "scene_hash": checked[(scenario_key, 0)]["scene_hash"],
                "run_id": run_id,
                "episodes": 3,
                "steps_applied": sum(int(checked[(scenario_key, seed)]["steps"]) for seed in _SEEDS),
                "push_success": sum(info["is_success"] is True for info in final_infos),
                "lifted_count": sum(info["lifted_ever"] is True for info in final_infos),
                "fallen_count": sum(info["fallen"] is True for info in final_infos),
                "off_table_count": sum(info["off_table"] is True for info in final_infos),
                "time_limit_count": sum(info["terminal_reason"] == "time_limit" for info in final_infos),
                "left_contact_count": sum(info["left_contact_ever"] is True for info in final_infos),
                "right_contact_count": sum(info["right_contact_ever"] is True for info in final_infos),
                "both_arms_count": sum(info["both_arms_participated"] is True for info in final_infos),
                "interference_count": sum(info["interference_ever"] is True for info in final_infos),
                "trajectory_sample_count": sum(
                    int(checked[(scenario_key, seed)]["trajectory_sample_count"]) for seed in _SEEDS
                ),
                "trajectory_plots_passed": 3,
                "videos_passed": 3,
                "telemetry_write_p95_ms_max": max(
                    float(checked[(scenario_key, seed)]["telemetry_write_p95_ms"]) for seed in _SEEDS
                ),
                "gpu_coverage_pass": run.get("gpu_coverage_pass") is True,
                "body_errors": body_results,
                "time_to_success_step_min": min(success_steps, default=None),
            }
        )

    for left, right in (("push_pi_single", "push_pi_dual"), ("push_letters_single", "push_letters_dual")):
        for seed in _SEEDS:
            first, second = checked[(left, seed)], checked[(right, seed)]
            if any(
                first[key] != second[key] for key in ("scene_hash", "layout_hash", "sampled_poses", "settled_poses")
            ):
                raise ValueError("matrix single/dual reset pairing is invalid")

    return {
        "schema": 1,
        "status": "passed",
        "batch_id": batch_id,
        "profile": profile,
        "checkpoint_label": checkpoint,
        "source_sha": source_sha,
        "upstream_sha": upstream_sha,
        "descriptor_sha256": descriptor_sha256(),
        "seeds": list(_SEEDS),
        "episode_count": len(CUSTOM_SCENARIOS) * len(_SEEDS),
        "infrastructure_pass": True,
        "pairing_pass": True,
        "scenarios": list(CUSTOM_SCENARIOS),
        "results": results,
    }


def _validate_batch_gpu(root: Path, matrix: Mapping[str, object]) -> None:
    interval = float(matrix["gpu_metrics_interval_seconds"])
    events, gpu = _gpu_events(
        root / "gpu-metrics.jsonl",
        str(matrix["run_id"]),
        str(matrix["profile"]),
        str(matrix["source_sha"]),
        interval,
    )
    if not events:
        raise ValueError("matrix GPU evidence is empty")
    event_groups = []
    runs = _mapping(matrix["scenario_runs"], "matrix scenario runs")
    for scenario_key in CUSTOM_SCENARIOS:
        run = _mapping(runs[scenario_key], "matrix scenario run")
        for manifest in run["episodes"]:
            item = _mapping(manifest, "matrix episode")
            rows = read_jsonl(Path(str(_mapping(item["telemetry"], "matrix telemetry")["path"]))).events
            event_groups.append(rows)
    gpu.update(_gpu_coverage(events, event_groups, root / "clock-correlation.json", interval))
    if gpu["gpu_coverage_pass"] is not True:
        raise ValueError("matrix GPU telemetry does not bracket the full batch")


def run_matrix() -> Path:
    base = load_mac_sim_config()
    remote = load_remote_config()
    if base.episode_steps != 300:
        raise RemoteError("the Push-PI acceptance matrix requires the fixed 300-step limit")
    if remote.policy_backend != "pytorch":
        raise RemoteError("the Push-PI matrix requires the validated PyTorch backend")
    _, source_sha = verify_ready_tunnel(remote)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017
    run_id = uuid.uuid4().hex
    root = validate_output_root(base.output_dir) / "scenarios_0827" / batch_id / remote.policy_profile.name
    raw_path = root / "matrix.json"
    raw: dict[str, object] = {
        "schema": 1,
        "status": "running",
        "batch_id": batch_id,
        "run_id": run_id,
        "profile": remote.policy_profile.name,
        "checkpoint_label": remote.policy_profile.checkpoint_label,
        "source_sha": source_sha,
        "upstream_sha": UPSTREAM_SHA,
        "descriptor_sha256": descriptor_sha256(),
        "gpu_metrics_interval_seconds": remote.gpu_metrics_interval_seconds,
        "gpu_coverage_pass": False,
        "seeds": list(_SEEDS),
        "scenarios": list(CUSTOM_SCENARIOS),
        "scenario_runs": {},
        "error": None,
        "cleanup_error": None,
    }
    _atomic_json(raw_path, raw)
    _status(
        f"matrix start profile={remote.policy_profile.name} scenarios={len(CUSTOM_SCENARIOS)} "
        f"seeds={len(_SEEDS)} episodes={len(CUSTOM_SCENARIOS) * len(_SEEDS)}"
    )
    sampler = None
    primary: BaseException | None = None
    try:
        sampler = start_gpu_sampler(remote, run_id, source_sha, root)
        runs = raw["scenario_runs"]
        assert isinstance(runs, dict)
        for scenario_key in CUSTOM_SCENARIOS:
            scenario = SCENARIOS[scenario_key]
            scenario_root = root / scenario_key
            scenario_run: dict[str, object] = {
                "status": "running",
                "profile": remote.policy_profile.name,
                "scenario": scenario_key,
                "source_sha": source_sha,
                "gpu_metrics_interval_seconds": remote.gpu_metrics_interval_seconds,
                "episodes": [],
            }
            runs[scenario_key] = scenario_run
            config = replace(base, task=scenario.gym_id, scenario=scenario, display=False, seed=0, episodes=3)
            for seed in _SEEDS:
                sampler.check()
                seed_root = scenario_root / f"seed-{seed}"
                try:
                    episode = _run_seed(config, remote, source_sha, UPSTREAM_SHA, seed, seed_root, run_id)
                except BaseException:
                    manifest_path = seed_root / "manifest.json"
                    if manifest_path.exists():
                        scenario_run["episodes"].append(json.loads(manifest_path.read_text(encoding="utf-8")))
                    scenario_run["status"] = "failed"
                    raise
                scenario_run["episodes"].append(episode)
                sampler.check()
                _atomic_json(scenario_root / "summary.json", scenario_run)
                _atomic_json(raw_path, raw)
            scenario_run["status"] = "passed"
            _atomic_json(scenario_root / "summary.json", scenario_run)
    except BaseException as error:
        primary = error
        raw["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raw["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        if sampler is not None:
            try:
                sampler.stop(root)
            except BaseException as error:
                if primary is None:
                    primary = error
                    raw["status"] = "failed"
                    raw["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
                else:
                    raw["cleanup_error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        _atomic_json(raw_path, raw)
    if primary is not None:
        episodes = sum(len(run.get("episodes", [])) for run in raw["scenario_runs"].values() if isinstance(run, dict))
        _status(f"matrix end status={raw['status']} episodes={episodes}/{len(CUSTOM_SCENARIOS) * len(_SEEDS)}")
        raise primary

    _status("matrix validating evidence")
    try:
        runs = _mapping(raw["scenario_runs"], "matrix scenario runs")
        _validate_batch_gpu(root, raw)
        raw["gpu_coverage_pass"] = True
        for scenario_key in CUSTOM_SCENARIOS:
            scenario_run = runs[scenario_key]
            assert isinstance(scenario_run, dict)
            scenario_run["gpu_coverage_pass"] = True
        raw["status"] = "passed"
        public = validate_matrix(raw, batch_root=root)
        for scenario_key in CUSTOM_SCENARIOS:
            scenario_root = root / scenario_key
            scenario_run = runs[scenario_key]
            assert isinstance(scenario_run, dict)
            _write_performance_summary(scenario_root, scenario_run)
            _atomic_json(scenario_root / "summary.json", scenario_run)
        _atomic_json(root / "matrix-summary.json", public)
        _atomic_json(raw_path, raw)
    except BaseException as error:
        raw["status"] = "failed"
        raw["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        _atomic_json(raw_path, raw)
        _status(
            f"matrix end status=failed episodes={len(CUSTOM_SCENARIOS) * len(_SEEDS)}/{len(CUSTOM_SCENARIOS) * len(_SEEDS)}"
        )
        raise
    _status(
        f"matrix end status=passed episodes={len(CUSTOM_SCENARIOS) * len(_SEEDS)}/"
        f"{len(CUSTOM_SCENARIOS) * len(_SEEDS)}"
    )
    return root / "matrix-summary.json"


def summarize_latest() -> Path:
    base = load_mac_sim_config()
    remote = load_remote_config()
    source_sha = _candidate_sha()
    output = validate_output_root(base.output_dir)
    candidates = sorted((output / "scenarios_0827").glob(f"*/{remote.policy_profile.name}/matrix.json"))
    if not candidates:
        raise RemoteError(f"no Push-PI matrix exists for {remote.policy_profile.name}; run make scenario-matrix first")
    raw_path = candidates[-1]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("source_sha") != source_sha:
        raise RemoteError("the latest Push-PI matrix is not from the exact current candidate")
    _validate_batch_gpu(raw_path.parent, raw)
    raw["gpu_coverage_pass"] = True
    runs = _mapping(raw["scenario_runs"], "matrix scenario runs")
    for scenario_key in CUSTOM_SCENARIOS:
        scenario_run = runs[scenario_key]
        assert isinstance(scenario_run, dict)
        scenario_run["gpu_coverage_pass"] = True
    public = validate_matrix(raw, batch_root=raw_path.parent)
    for scenario_key in CUSTOM_SCENARIOS:
        scenario_run = runs[scenario_key]
        assert isinstance(scenario_run, dict)
        _write_performance_summary(raw_path.parent / scenario_key, scenario_run)
    output_path = raw_path.parent / "matrix-summary.json"
    _atomic_json(output_path, public)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or revalidate the exact Push-PI scenario matrix.")
    parser.add_argument("command", choices=("run", "metrics"))
    command = parser.parse_args().command
    try:
        print(run_matrix() if command == "run" else summarize_latest())
    except (OSError, ValueError, json.JSONDecodeError, RemoteError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
