from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import TextIO

import numpy as np

from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import TARGET_AREA_COVERAGE_METHOD
from tools.remote_aloha.scenarios import TASK_TO_SCENARIO

_EVENT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_VERSION = re.compile(r"[0-9][A-Za-z0-9.+_-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_TERMINAL_STATUSES = {"complete", "failed", "interrupted", "passed"}
_PACKAGE_NAMES = {
    "dm-control",
    "gym-aloha",
    "gymnasium",
    "imageio",
    "imageio-ffmpeg",
    "matplotlib",
    "mujoco",
    "numpy",
}

_PUBLISHABLE_METADATA = {
    "action_horizon",
    "camera_views",
    "checkpoint_label",
    "model_action_horizon",
    "package_versions",
    "prefetch_steps",
    "profile",
    "run_id",
    "scenario",
    "scene_hash",
    "seeds",
    "source_sha",
    "task",
    "target_area_coverage_method",
    "upstream_sha",
}
_PUBLISHABLE_RESULTS = {
    "episodes",
    "failures",
    "gpu_coverage_pass",
    "gpu_max_gap_ms",
    "gpu_sample_count",
    "gpu_span_ms",
    "infrastructure_pass",
    "request_count",
    "retries",
    "reward_final",
    "reward_max",
    "reward_sum",
    "steps_applied",
    "task_success",
    "trajectory_joint_count",
    "trajectory_plot_id",
    "trajectory_plot_ids",
    "trajectory_plot_status",
    "trajectory_plots_passed",
    "trajectory_sample_count",
    "trajectory_step_coverage",
    "video_ids",
    "clock_max_uncertainty_ms",
    "clock_offset_change_ms",
    "both_arms_count",
    "fallen_count",
    "interference_count",
    "left_contact_count",
    "lifted_count",
    "off_table_count",
    "push_success",
    "right_contact_count",
    "time_limit_count",
    "videos_passed",
    "coverage_sample_count",
    "initial_target_area_coverage_percent",
    "final_target_area_coverage_percent",
    "best_target_area_coverage_percent",
    "best_target_area_coverage_step",
    "time_to_best_target_area_coverage_seconds",
    "episode_elapsed_seconds",
}
_PUBLISHABLE_EVENTS = {
    "episode",
    "error",
    "gpu",
    "metadata",
    "policy_request",
    "policy_result",
    "retry",
    "step",
    "terminal",
    "wait",
}
_PUBLISHABLE_METRICS = {
    "active_step_hz",
    "active_step_interval_ms",
    "buffer_wait_ms",
    "cold_inference_ms",
    "dropped_leading_actions",
    "gpu_memory_mib",
    "gpu_utilization_percent",
    "reward",
    "server_infer_ms",
    "server_rss_kib",
    "server_total_ms",
    "sim_step_ms",
    "telemetry_write_ms",
    "wall_episode_hz",
    "warm_inference_ms",
    "body_0_height_error",
    "body_0_pitch",
    "body_0_roll",
    "body_0_xy_error",
    "body_0_yaw_error",
    "body_1_height_error",
    "body_1_pitch",
    "body_1_roll",
    "body_1_xy_error",
    "body_1_yaw_error",
    "initial_target_area_coverage_percent",
    "final_target_area_coverage_percent",
    "best_target_area_coverage_percent",
    "time_to_best_target_area_coverage_seconds",
    "episode_elapsed_seconds",
}


def json_safe(value: object, depth: int = 0) -> object:
    """Return bounded JSON data, normalizing NumPy scalars and rejecting arrays/non-finite values."""
    if depth > 3:
        raise ValueError("telemetry data is nested too deeply")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("telemetry numbers must be finite")
        return value
    if isinstance(value, np.ndarray):
        raise ValueError("NumPy arrays are not valid telemetry fields")
    if isinstance(value, Mapping):
        if len(value) > 40 or not all(isinstance(key, str) for key in value):
            raise ValueError("telemetry mappings must have at most 40 string keys")
        return {key: json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple) and len(value) <= 100:
        return [json_safe(item, depth + 1) for item in value]
    raise ValueError(f"unsupported telemetry value: {type(value).__name__}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")  # noqa: UP017


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").tzinfo == timezone.utc  # noqa: UP017 (Python 3.10)
    except ValueError:
        return False


class JsonlWriter:
    """A line-buffered, local-only telemetry writer with no per-event fsync."""

    def __init__(
        self,
        path: str | Path,
        *,
        utc_now=_utc_now,
        monotonic_ns=time.monotonic_ns,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        self._stream: TextIO = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)
        self._utc_now = utc_now
        self._monotonic_ns = monotonic_ns

    def write(self, event: str, **fields: object) -> float:
        started = time.perf_counter_ns()
        if self._stream.closed:
            raise RuntimeError("telemetry writer is closed")
        if not _EVENT_NAME.fullmatch(event):
            raise ValueError("telemetry event name is invalid")
        reserved = {"schema", "event", "timestamp_utc", "monotonic_ns"} & fields.keys()
        if reserved:
            raise ValueError(f"telemetry fields use reserved keys: {', '.join(sorted(reserved))}")
        payload = json_safe(
            {
                "schema": 1,
                "event": event,
                "timestamp_utc": self._utc_now(),
                "monotonic_ns": self._monotonic_ns(),
                **fields,
            }
        )
        line = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        self._stream.write(line)
        return (time.perf_counter_ns() - started) / 1_000_000

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class JsonlReadResult:
    events: tuple[dict[str, object], ...]
    partial_final_line_ignored: bool = False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _validate_event(value: object, line_number: int) -> dict[str, object]:
    schema = value.get("schema") if isinstance(value, dict) else None
    if not isinstance(value, dict) or isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError(f"telemetry line {line_number} must be a schema-1 object")
    event = value.get("event")
    timestamp = value.get("timestamp_utc")
    monotonic_ns = value.get("monotonic_ns")
    if not isinstance(event, str) or not _EVENT_NAME.fullmatch(event):
        raise ValueError(f"telemetry line {line_number} has an invalid event name")
    if not _is_utc_timestamp(timestamp):
        raise ValueError(f"telemetry line {line_number} has an invalid UTC timestamp")
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
        raise ValueError(f"telemetry line {line_number} has an invalid monotonic timestamp")
    return json_safe(value)  # type: ignore[return-value]


def read_jsonl(path: str | Path) -> JsonlReadResult:
    raw_lines = Path(path).read_bytes().splitlines(keepends=True)
    events: list[dict[str, object]] = []
    partial_ignored = False
    for index, raw_line in enumerate(raw_lines, start=1):
        complete = raw_line.endswith(b"\n")
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            if index == len(raw_lines) and not complete:
                partial_ignored = True
                break
            raise ValueError(f"invalid telemetry JSON on line {index}") from None
        events.append(_validate_event(value, index))
    return JsonlReadResult(tuple(events), partial_ignored)


def summarize_values(values: Iterable[object]) -> dict[str, int | float | None]:
    samples: list[float] = []
    for value in values:
        sample = value.item() if isinstance(value, np.generic) else value
        if isinstance(sample, bool) or not isinstance(sample, int | float) or not math.isfinite(sample):
            raise ValueError("metric samples must be finite real numbers")
        samples.append(float(sample))
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    samples.sort()

    def percentile(percent: float) -> float:
        rank = (len(samples) - 1) * percent / 100
        lower = math.floor(rank)
        upper = math.ceil(rank)
        return samples[lower] + (samples[upper] - samples[lower]) * (rank - lower)

    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(50),
        "p95": percentile(95),
        "max": samples[-1],
    }


def aggregate_events(
    events: Iterable[Mapping[str, object]],
    *,
    partial_final_line_ignored: bool = False,
) -> dict[str, object]:
    items = [dict(event) for event in events]
    if items and items[0].get("event") != "metadata":
        raise ValueError("the first telemetry event must be metadata")
    terminal_indices = [index for index, event in enumerate(items) if event.get("event") == "terminal"]
    if terminal_indices and terminal_indices != [len(items) - 1]:
        raise ValueError("the terminal telemetry event must be last and unique")

    event_counts: dict[str, int] = {}
    metric_samples: dict[str, list[object]] = {}
    for event in items:
        validated = _validate_event(event, 0)
        name = str(validated["event"])
        event_counts[name] = event_counts.get(name, 0) + 1
        metrics = validated.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("telemetry metrics must be a mapping")
        for metric, value in metrics.items():
            if not isinstance(metric, str) or not _EVENT_NAME.fullmatch(metric):
                raise ValueError("telemetry metric name is invalid")
            metric_samples.setdefault(metric, []).append(value)

    metadata = {}
    if items:
        metadata = {
            key: value
            for key, value in items[0].items()
            if key not in {"schema", "event", "timestamp_utc", "monotonic_ns", "metrics"}
        }
    terminal = items[-1] if terminal_indices else {}
    result = {
        key: value
        for key, value in terminal.items()
        if key not in {"schema", "event", "timestamp_utc", "monotonic_ns", "metrics", "status"}
    }
    status = terminal.get("status", "partial")
    if not isinstance(status, str) or (terminal and status not in _TERMINAL_STATUSES):
        raise ValueError("terminal telemetry status is invalid")

    steps_applied = result.get("steps_applied")
    step_events = event_counts.get("step", 0)
    if isinstance(steps_applied, bool) or not isinstance(steps_applied, int) or steps_applied < 0:
        step_coverage = None
    else:
        step_coverage = 1.0 if steps_applied == 0 else step_events / steps_applied
    return {
        "schema": 1,
        "status": status,
        "metadata": json_safe(metadata),
        "result": json_safe(result),
        "event_count": len(items),
        "event_counts": event_counts,
        "metrics": {name: summarize_values(values) for name, values in sorted(metric_samples.items())},
        "telemetry": {
            "valid_lines": len(items),
            "partial_final_line_ignored": partial_final_line_ignored,
            "terminal_event_present": bool(terminal_indices),
            "step_coverage": step_coverage,
        },
    }


def aggregate_jsonl(path: str | Path) -> dict[str, object]:
    result = read_jsonl(path)
    return aggregate_events(result.events, partial_final_line_ignored=result.partial_final_line_ignored)


def _valid_publishable_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    result = {key: metadata[key] for key in _PUBLISHABLE_METADATA if key in metadata}
    profile = result.get("profile")
    if profile not in {"pi0_aloha_sim", "pi05_aloha_base"}:
        raise ValueError("publishable telemetry must name a safe model profile")
    for key in ("source_sha", "upstream_sha"):
        if key in result and (not isinstance(result[key], str) or not _SHA.fullmatch(result[key])):
            raise ValueError(f"publishable telemetry {key} is invalid")
    if "run_id" in result and (
        not isinstance(result["run_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", result["run_id"])
    ):
        raise ValueError("publishable telemetry run_id is invalid")
    expected_checkpoint = "pi0_aloha_sim" if profile == "pi0_aloha_sim" else "pi05_base"
    if "checkpoint_label" in result and result["checkpoint_label"] != expected_checkpoint:
        raise ValueError("publishable telemetry checkpoint label is invalid")
    task = result.get("task")
    scenario = result.get("scenario")
    if (task is None) != (scenario is None):
        raise ValueError("publishable telemetry task and scenario must be provided together")
    if task is not None and task not in TASK_TO_SCENARIO:
        raise ValueError("publishable telemetry task is invalid")
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError("publishable telemetry scenario is invalid")
    if task is not None and scenario is not None and TASK_TO_SCENARIO[task] != scenario:
        raise ValueError("publishable telemetry scenario/task pair is invalid")
    scene_id = result.get("scene_hash")
    custom = scenario is not None and SCENARIOS[str(scenario)].is_custom
    if custom and (not isinstance(scene_id, str) or not re.fullmatch(r"[0-9a-f]{64}", scene_id)):
        raise ValueError("publishable telemetry custom scenario hash is invalid")
    if not custom and scene_id is not None:
        raise ValueError("publishable telemetry stock scenario must not have a scene hash")
    coverage_method = result.get("target_area_coverage_method")
    if custom and coverage_method != TARGET_AREA_COVERAGE_METHOD:
        raise ValueError("publishable telemetry area coverage method is invalid")
    if not custom and coverage_method is not None:
        raise ValueError("publishable stock telemetry must not define area coverage")
    for key in ("action_horizon", "model_action_horizon", "prefetch_steps"):
        value = result.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50):
            raise ValueError(f"publishable telemetry {key} is invalid")
    if "camera_views" in result and result["camera_views"] != ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        raise ValueError("publishable telemetry camera views are invalid")
    seeds = result.get("seeds")
    if seeds is not None and (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**32 - 1 for value in seeds)
    ):
        raise ValueError("publishable telemetry seeds are invalid")
    versions = result.get("package_versions")
    if versions is not None and (
        not isinstance(versions, Mapping)
        or not versions
        or any(
            name not in _PACKAGE_NAMES or not isinstance(version, str) or not _SAFE_VERSION.fullmatch(version)
            for name, version in versions.items()
        )
    ):
        raise ValueError("publishable package versions are invalid")
    return result


def _valid_publishable_result(result: Mapping[str, object]) -> dict[str, object]:
    safe = {key: result[key] for key in _PUBLISHABLE_RESULTS if key in result}
    for key in (
        "episodes",
        "failures",
        "gpu_sample_count",
        "request_count",
        "retries",
        "steps_applied",
        "trajectory_joint_count",
        "trajectory_plots_passed",
        "trajectory_sample_count",
        "both_arms_count",
        "fallen_count",
        "interference_count",
        "left_contact_count",
        "lifted_count",
        "off_table_count",
        "push_success",
        "right_contact_count",
        "time_limit_count",
        "videos_passed",
        "coverage_sample_count",
        "best_target_area_coverage_step",
    ):
        value = safe.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"publishable result {key} is invalid")
    if safe.get("trajectory_joint_count") not in {None, 14}:
        raise ValueError("publishable trajectory joint count must be 14")
    sample_count = safe.get("trajectory_sample_count")
    steps_applied = safe.get("steps_applied")
    if sample_count is not None and steps_applied is not None and sample_count != steps_applied:
        raise ValueError("publishable trajectory samples must match applied steps")
    coverage_samples = safe.get("coverage_sample_count")
    if coverage_samples is not None and steps_applied is not None and coverage_samples != steps_applied:
        raise ValueError("publishable area coverage samples must match applied steps")
    plots_passed = safe.get("trajectory_plots_passed")
    episodes = safe.get("episodes")
    count_limit = episodes if episodes is not None else 1
    if plots_passed is not None and episodes is not None and plots_passed > episodes:
        raise ValueError("publishable trajectory plot count exceeds episodes")
    for key in (
        "both_arms_count",
        "fallen_count",
        "interference_count",
        "left_contact_count",
        "lifted_count",
        "off_table_count",
        "push_success",
        "right_contact_count",
        "time_limit_count",
        "videos_passed",
    ):
        if safe.get(key, 0) > count_limit:
            raise ValueError(f"publishable result {key} exceeds episode count")
    if "infrastructure_pass" in safe and not isinstance(safe["infrastructure_pass"], bool):
        raise ValueError("publishable infrastructure result is invalid")
    if "gpu_coverage_pass" in safe and not isinstance(safe["gpu_coverage_pass"], bool):
        raise ValueError("publishable GPU coverage result is invalid")
    task_success = safe.get("task_success")
    if task_success is not None and (
        not isinstance(task_success, bool) and (not isinstance(task_success, int) or task_success < 0)
    ):
        raise ValueError("publishable task result is invalid")
    for key in (
        "clock_max_uncertainty_ms",
        "clock_offset_change_ms",
        "gpu_max_gap_ms",
        "gpu_span_ms",
        "reward_final",
        "reward_max",
        "reward_sum",
        "initial_target_area_coverage_percent",
        "final_target_area_coverage_percent",
        "best_target_area_coverage_percent",
        "time_to_best_target_area_coverage_seconds",
        "episode_elapsed_seconds",
    ):
        value = safe.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
        ):
            raise ValueError(f"publishable result {key} is invalid")
    for key in (
        "initial_target_area_coverage_percent",
        "final_target_area_coverage_percent",
        "best_target_area_coverage_percent",
    ):
        value = safe.get(key)
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"publishable result {key} is invalid")
    for key in ("time_to_best_target_area_coverage_seconds", "episode_elapsed_seconds"):
        value = safe.get(key)
        if value is not None and value < 0:
            raise ValueError(f"publishable result {key} is invalid")
    best_step = safe.get("best_target_area_coverage_step")
    if best_step is not None and (steps_applied is None or not 1 <= best_step <= steps_applied):
        raise ValueError("publishable area coverage best step is invalid")
    best_fields = (
        safe.get("best_target_area_coverage_percent"),
        best_step,
        safe.get("time_to_best_target_area_coverage_seconds"),
    )
    if any(value is None for value in best_fields) != all(value is None for value in best_fields):
        raise ValueError("publishable area coverage best fields are incomplete")
    if (
        safe.get("final_target_area_coverage_percent") is not None
        and safe.get("best_target_area_coverage_percent") is not None
        and safe["final_target_area_coverage_percent"] > safe["best_target_area_coverage_percent"]
    ):
        raise ValueError("publishable final area coverage exceeds its best value")
    if (
        safe.get("time_to_best_target_area_coverage_seconds") is not None
        and safe.get("episode_elapsed_seconds") is not None
        and safe["time_to_best_target_area_coverage_seconds"] > safe["episode_elapsed_seconds"]
    ):
        raise ValueError("publishable area coverage time exceeds episode time")
    video_ids = safe.get("video_ids")
    if video_ids is not None and (
        not isinstance(video_ids, list)
        or not all(isinstance(value, str) and _SAFE_ID.fullmatch(value) for value in video_ids)
    ):
        raise ValueError("publishable video IDs must be safe local identifiers")
    plot_id = safe.get("trajectory_plot_id")
    if plot_id is not None and (not isinstance(plot_id, str) or not _SAFE_ID.fullmatch(plot_id)):
        raise ValueError("publishable trajectory plot ID is invalid")
    plot_ids = safe.get("trajectory_plot_ids")
    if plot_ids is not None and (
        not isinstance(plot_ids, list)
        or not all(isinstance(value, str) and _SAFE_ID.fullmatch(value) for value in plot_ids)
    ):
        raise ValueError("publishable trajectory plot IDs are invalid")
    plot_status = safe.get("trajectory_plot_status")
    if plot_status is not None and plot_status not in {"passed", "partial", "no_samples", "failed"}:
        raise ValueError("publishable trajectory plot status is invalid")
    trajectory_coverage = safe.get("trajectory_step_coverage")
    if trajectory_coverage is not None and (
        isinstance(trajectory_coverage, bool)
        or not isinstance(trajectory_coverage, int | float)
        or not math.isfinite(trajectory_coverage)
        or not 0 <= trajectory_coverage <= 1
    ):
        raise ValueError("publishable trajectory coverage is invalid")
    if sample_count is not None and steps_applied is not None:
        expected_coverage = 1.0 if steps_applied == 0 else sample_count / steps_applied
        if trajectory_coverage is not None and trajectory_coverage != expected_coverage:
            raise ValueError("publishable trajectory coverage does not match applied steps")
    return safe


def _valid_publishable_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for name, stats in metrics.items():
        if name not in _PUBLISHABLE_METRICS:
            continue
        if not isinstance(stats, Mapping) or set(stats) != {"count", "mean", "p50", "p95", "max"}:
            raise ValueError("publishable metric statistics are invalid")
        count = stats["count"]
        values = (stats[key] for key in ("mean", "p50", "p95", "max"))
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value))
                for value in values
            )
        ):
            raise ValueError("publishable metric statistics are invalid")
        result[name] = dict(stats)
        if name.endswith("_target_area_coverage_percent") and any(
            stats[key] is not None and not 0 <= stats[key] <= 100 for key in ("mean", "p50", "p95", "max")
        ):
            raise ValueError("publishable area coverage metric is invalid")
        if name.endswith("_seconds") and any(
            stats[key] is not None and stats[key] < 0 for key in ("mean", "p50", "p95", "max")
        ):
            raise ValueError("publishable duration metric is invalid")
    return result


def publishable_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Build a new allowlisted summary; raw local evidence is never mutated."""
    safe = json_safe(summary)
    if not isinstance(safe, Mapping):
        raise ValueError("telemetry summary must be a mapping")
    metadata = safe.get("metadata", {})
    result = safe.get("result", {})
    event_counts = safe.get("event_counts", {})
    metrics = safe.get("metrics", {})
    telemetry = safe.get("telemetry", {})
    if not all(isinstance(value, Mapping) for value in (metadata, result, event_counts, metrics, telemetry)):
        raise ValueError("telemetry summary sections must be mappings")
    status = safe.get("status", "partial")
    event_count = safe.get("event_count", 0)
    if status not in _TERMINAL_STATUSES | {"partial"}:
        raise ValueError("publishable telemetry status is invalid")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ValueError("publishable telemetry event count is invalid")
    safe_event_counts = {key: value for key, value in event_counts.items() if key in _PUBLISHABLE_EVENTS}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in safe_event_counts.values()):
        raise ValueError("publishable telemetry event counts are invalid")
    safe_telemetry = {
        key: telemetry[key]
        for key in ("partial_final_line_ignored", "step_coverage", "terminal_event_present", "valid_lines")
        if key in telemetry
    }
    for key in ("partial_final_line_ignored", "terminal_event_present"):
        if key in safe_telemetry and not isinstance(safe_telemetry[key], bool):
            raise ValueError("publishable telemetry integrity fields are invalid")
    if "valid_lines" in safe_telemetry and (
        isinstance(safe_telemetry["valid_lines"], bool)
        or not isinstance(safe_telemetry["valid_lines"], int)
        or safe_telemetry["valid_lines"] < 0
    ):
        raise ValueError("publishable telemetry valid-line count is invalid")
    coverage = safe_telemetry.get("step_coverage")
    if coverage is not None and (
        isinstance(coverage, bool)
        or not isinstance(coverage, int | float)
        or not math.isfinite(coverage)
        or coverage < 0
    ):
        raise ValueError("publishable telemetry step coverage is invalid")
    return {
        "schema": 1,
        "status": status,
        "metadata": _valid_publishable_metadata(metadata),
        "result": _valid_publishable_result(result),
        "event_count": event_count,
        "event_counts": safe_event_counts,
        "metrics": _valid_publishable_metrics(metrics),
        "telemetry": safe_telemetry,
    }


def _atomic_write(path: Path, content: str) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def write_summary(path: str | Path, summary: Mapping[str, object], *, publishable: bool = False) -> Path:
    payload = publishable_summary(summary) if publishable else json_safe(summary)
    return _atomic_write(Path(path), json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


def render_markdown(summary: Mapping[str, object]) -> str:
    public = publishable_summary(summary)
    metadata = public["metadata"]
    lines = [
        "# Telemetry summary",
        "",
        f"- Profile: `{metadata['profile']}`",
        f"- Status: `{public['status']}`",
        f"- Valid events: {public['event_count']}",
        "",
        "| Metric | Count | Mean | p50 | p95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in public["metrics"].items():
        lines.append(
            f"| {name} | {stats['count']} | {stats['mean']} | {stats['p50']} | {stats['p95']} | {stats['max']} |"
        )
    result = public["result"]
    if "trajectory_sample_count" in result:
        lines.extend(
            [
                "",
                "Trajectory: "
                f"{result['trajectory_sample_count']} samples, "
                f"{result.get('trajectory_joint_count', 14)} joints, "
                f"{result.get('trajectory_step_coverage')} step coverage, "
                f"plot `{result.get('trajectory_plot_status')}`.",
            ]
        )
    plot_ids = result.get("trajectory_plot_ids") or (
        [result["trajectory_plot_id"]] if result.get("trajectory_plot_id") else []
    )
    if plot_ids:
        lines.append("Plot IDs: " + ", ".join(f"`{value}`" for value in plot_ids))
    if result.get("video_ids"):
        lines.extend(["", "Video IDs: " + ", ".join(f"`{value}`" for value in result["video_ids"])])
    return "\n".join(lines) + "\n"


def write_markdown_summary(path: str | Path, summary: Mapping[str, object]) -> Path:
    return _atomic_write(Path(path), render_markdown(summary))
