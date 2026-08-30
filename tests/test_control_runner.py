from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.aloha_sim.saver import VideoSaver
from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.run import _connect_with_retry
from tools.remote_aloha.run import _convert_environment_observation
from tools.remote_aloha.run import _gpu_events
from tools.remote_aloha.run import _json_safe
from tools.remote_aloha.run import _prefetch_evidence
from tools.remote_aloha.run import _run_seed
from tools.remote_aloha.run import _scenario_step_info
from tools.remote_aloha.run import _status
from tools.remote_aloha.run import _write_performance_summary
from tools.remote_aloha.run import control_episode
from tools.remote_aloha.run import run
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.telemetry import JsonlWriter


def _raw_observation() -> dict:
    return {
        "pixels": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "left_wrist", "right_wrist")},
        "agent_pos": np.zeros(14, dtype=np.float64),
    }


def test_environment_info_mapping_allows_full_letters_reset_but_remains_bounded() -> None:
    assert _json_safe({str(index): index for index in range(64)}) == {str(index): index for index in range(64)}
    with pytest.raises(ValueError, match="bounded string-keyed"):
        _json_safe({str(index): index for index in range(65)})


def test_environment_observation_captures_both_wrist_views() -> None:
    class Physics:
        def __init__(self) -> None:
            self.cameras = []

        def render(self, *, height: int, width: int, camera_id: str) -> np.ndarray:
            self.cameras.append((height, width, camera_id))
            value = 1 if camera_id == "left_wrist" else 2
            return np.full((height, width, 3), value, dtype=np.uint8)

    physics = Physics()
    environment = SimpleNamespace(unwrapped=SimpleNamespace(_env=SimpleNamespace(physics=physics)))
    raw = _raw_observation()
    raw["pixels"] = {"top": raw["pixels"]["top"]}
    converted = _convert_environment_observation(environment, raw, "prompt")
    assert physics.cameras == [(480, 640, "left_wrist"), (480, 640, "right_wrist")]
    assert set(converted["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert converted["images"]["cam_left_wrist"].max() == 1
    assert converted["images"]["cam_right_wrist"].max() == 2


def _custom_info(scenario: str = "push_letters_single") -> dict[str, object]:
    body_count = 1 if SCENARIOS[scenario].object_kind == "pi" else 2
    return {
        "is_success": False,
        "scenario": scenario,
        "scene_hash": "a" * 64,
        "layout_hash": "b" * 64,
        "body_count": body_count,
        "held_steps": 0,
        "lifted_ever": False,
        "off_table": False,
        "fallen": False,
        "terminal_reason": "running",
        "left_contact_ever": True,
        "right_contact_ever": False,
        "both_arms_participated": False,
        "interference_ever": False,
        "left_joint_travel": 0.1,
        "right_joint_travel": 0.0,
        "target_area_coverage": 0.1,
        **{
            f"body_{index}_{suffix}": 0.1
            for index in range(body_count)
            for suffix in ("xy_error", "yaw_error", "roll", "pitch", "height_error", "target_area_coverage")
        },
    }


def _gpu_rows() -> list[dict[str, object]]:
    identity = {
        "schema": 1,
        "run_id": "c" * 32,
        "profile": "pi0_aloha_sim",
        "server_pid": 4242,
        "source_sha": "a" * 40,
        "interval_ms": 1000,
    }
    return [
        {**identity, "event": "sampler_started", "utc": "2026-08-28T08:00:00.000Z", "monotonic_ns": 1},
        {
            **identity,
            "event": "gpu_sample",
            "utc": "2026-08-28T08:00:00.100Z",
            "monotonic_ns": 2,
            "elapsed_ms": 0,
            "sample_index": 0,
            "memory_used_mib": 1024,
            "utilization_percent": 25,
            "server_rss_kib": 2048,
        },
        {
            **identity,
            "event": "gpu_sample",
            "utc": "2026-08-28T08:00:01.100Z",
            "monotonic_ns": 3,
            "elapsed_ms": 1000,
            "sample_index": 1,
            "memory_used_mib": 1030,
            "utilization_percent": 30,
            "server_rss_kib": 2050,
        },
        {
            **identity,
            "event": "sampler_stopped",
            "utc": "2026-08-28T08:00:01.200Z",
            "monotonic_ns": 4,
            "status": "interrupted",
            "exit_status": 143,
        },
    ]


def test_gpu_events_require_exact_identity_sequence_and_cadence(tmp_path: Path) -> None:
    path = tmp_path / "gpu.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _gpu_rows()), encoding="utf-8")
    events, result = _gpu_events(path, "c" * 32, "pi0_aloha_sim", "a" * 40, 1.0)
    assert len(events) == result["gpu_sample_count"] == 2
    assert result["gpu_span_ms"] == result["gpu_max_gap_ms"] == 1000


def test_prefetch_gate_uses_only_completed_warm_request_submission_depths() -> None:
    with pytest.raises(ValueError, match="qualification evidence"):
        _prefetch_evidence(
            {
                "request_latencies_ms": [900.0, 200.0],
                "completed_request_buffer_depths": [0],
                "underrun_count": 0,
            }
        )
    for invalid in (
        {"request_latencies_ms": [1.0], "completed_request_buffer_depths": [51], "underrun_count": 0},
        {"request_latencies_ms": [1.0], "completed_request_buffer_depths": [0], "underrun_count": False},
    ):
        with pytest.raises(ValueError, match="qualification evidence"):
            _prefetch_evidence(invalid)

    pending_only = _prefetch_evidence(
        {
            "request_latencies_ms": [900.0],
            "request_buffer_depths": [0, 40],
            "completed_request_buffer_depths": [0],
            "underrun_count": 0,
        }
    )
    assert pending_only["budget_ms"] is None
    assert pending_only["qualified"] is False

    completed = _prefetch_evidence(
        {
            "request_latencies_ms": [900.0, 200.0],
            "completed_request_buffer_depths": [0, 20],
            "underrun_count": 0,
        }
    )
    assert completed["request_depth_min"] == completed["request_depth_p5"] == 20
    assert completed["budget_ms"] == 400.0
    assert completed["qualified"] is True


@pytest.mark.parametrize(
    ("row", "field", "value"),
    [
        (3, "status", "failed"),
        (3, "exit_status", 0),
        (2, "sample_index", 7),
        (2, "elapsed_ms", 5000),
        (1, "server_pid", 9999),
        (1, "utc", "Z"),
    ],
)
def test_gpu_events_reject_corrupt_or_failed_evidence(tmp_path: Path, row: int, field: str, value: object) -> None:
    rows = _gpu_rows()
    rows[row][field] = value
    path = tmp_path / "gpu.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU telemetry"):
        _gpu_events(path, "c" * 32, "pi0_aloha_sim", "a" * 40, 1.0)


def test_gpu_events_reject_a_single_readiness_sample(tmp_path: Path) -> None:
    rows = _gpu_rows()
    del rows[2]
    path = tmp_path / "gpu.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        _gpu_events(path, "c" * 32, "pi0_aloha_sim", "a" * 40, 1.0)


def test_performance_summary_rejects_gpu_samples_that_do_not_span_the_mac_run(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "seed-0.jsonl"
    telemetry_rows = [
        {
            "schema": 1,
            "event": "metadata",
            "timestamp_utc": "2026-08-28T09:00:00.000Z",
            "monotonic_ns": 1,
            "run_id": "c" * 32,
            "profile": "pi0_aloha_sim",
            "source_sha": "a" * 40,
        },
        {
            "schema": 1,
            "event": "step",
            "timestamp_utc": "2026-08-28T09:00:00.500Z",
            "monotonic_ns": 1_000_000_001,
            "step": 0,
            "applied_step": 1,
            "elapsed_seconds": 1.0,
            "actual_joint_positions": [0.0] * 14,
            "commanded_joint_positions": [0.0] * 14,
        },
        {
            "schema": 1,
            "event": "terminal",
            "timestamp_utc": "2026-08-28T09:00:01.000Z",
            "monotonic_ns": 1_000_000_001,
            "status": "complete",
            "metrics": {"telemetry_write_ms": 0.2},
        },
    ]
    telemetry_path.write_text("".join(json.dumps(row) + "\n" for row in telemetry_rows), encoding="utf-8")
    gpu_path = tmp_path / "gpu-metrics.jsonl"
    gpu_path.write_text("".join(json.dumps(row) + "\n" for row in _gpu_rows()), encoding="utf-8")
    (tmp_path / "clock-correlation.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "start": {"round_trip_uncertainty_ms": 1, "wsl_minus_mac_midpoint_ms": 0},
                "end": {"round_trip_uncertainty_ms": 1, "wsl_minus_mac_midpoint_ms": 0},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "passed",
        "gpu_metrics_interval_seconds": 1.0,
        "episodes": [
            {
                "seed": 0,
                "infrastructure_pass": True,
                "telemetry": {"path": str(telemetry_path)},
                "episode": {"steps_applied": 1, "reward_sum": 0.0, "reward_max": 0.0},
                "trajectory": {
                    "sample_count": 1,
                    "plot_status": "passed",
                    "plot_id": "run-seed-0-joint-trajectory",
                },
                "buffer": {
                    "request_count": 3,
                    "request_buffer_depths": [0, 7, 9],
                    "completed_request_buffer_depths": [0, 7, 9],
                    "replacement_count": 2,
                    "crossfade_replacement_count": 1,
                    "crossfade_action_count": 5,
                    "zero_overlap_replacements": 1,
                },
                "connection": {},
            }
        ],
    }
    local_root = tmp_path / "local-only"
    _write_performance_summary(local_root, summary)
    local_performance = json.loads((local_root / "performance-summary.json").read_text(encoding="utf-8"))
    assert local_performance["metrics"]["telemetry_write_ms"]["p95"] == 0.2
    expected_trajectory = {
        "trajectory_sample_count": 1,
        "trajectory_joint_count": 14,
        "trajectory_step_coverage": 1.0,
        "trajectory_plots_passed": 1,
        "trajectory_plot_status": "passed",
        "trajectory_plot_ids": ["run-seed-0-joint-trajectory"],
    }
    for key, value in expected_trajectory.items():
        assert local_performance["result"][key] == value
    assert local_performance["result"]["replacement_count"] == 2
    assert local_performance["result"]["crossfade_action_count"] == 5
    assert local_performance["result"]["request_buffer_depth_min"] == 7
    assert local_performance["result"]["request_buffer_depth_p5"] == pytest.approx(7.1)
    encoded = json.dumps(local_performance)
    assert str(tmp_path) not in encoded
    assert "trajectory.path" not in encoded
    with pytest.raises(ValueError, match="does not span"):
        _write_performance_summary(tmp_path, summary, gpu_path)


def test_connection_retry_is_bounded_before_episode_start(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    transport = object()

    def connect(config: RemoteConfig, source_sha: str):
        calls.append(source_sha)
        if len(calls) < 3:
            raise TimeoutError("not ready")
        return transport, {"ready": True}

    sleeps = []
    retries = []
    monkeypatch.setattr("tools.remote_aloha.run._connect", connect)
    progress = {"failures": 0, "retries": 0}
    result = _connect_with_retry(
        RemoteConfig(policy_retry_count=2, policy_retry_backoff_seconds=0.25),
        "a" * 40,
        progress,
        emit=lambda event, **fields: retries.append((event, fields)),
        sleep=sleeps.append,
    )
    assert result == (transport, {"ready": True})
    assert len(calls) == 3
    assert sleeps == [0.25, 0.25]
    assert progress == {"failures": 2, "retries": 2}
    assert [event for event, _ in retries] == ["retry", "retry"]


def test_identity_or_application_error_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def connect(*args: object):
        calls.append(True)
        raise ValueError("metadata mismatch")

    monkeypatch.setattr("tools.remote_aloha.run._connect", connect)
    with pytest.raises(ValueError, match="metadata mismatch"):
        _connect_with_retry(
            RemoteConfig(),
            "a" * 40,
            {"failures": 0, "retries": 0},
            sleep=lambda _: pytest.fail("must not retry"),
        )
    assert calls == [True]


def test_connection_retry_exhaustion_reports_exact_bounded_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    sleeps = []

    def connect(*args: object):
        calls.append(True)
        raise TimeoutError("still unavailable")

    monkeypatch.setattr("tools.remote_aloha.run._connect", connect)
    progress = {"failures": 0, "retries": 0}
    with pytest.raises(TimeoutError, match="still unavailable"):
        _connect_with_retry(
            RemoteConfig(policy_retry_count=2, policy_retry_backoff_seconds=0.25),
            "a" * 40,
            progress,
            sleep=sleeps.append,
        )
    assert len(calls) == 3
    assert sleeps == [0.25, 0.25]
    assert progress == {"failures": 3, "retries": 2}


def test_control_episode_uses_exact_seed_steps_and_non_catchup_cadence(tmp_path: Path) -> None:
    from openpi_client import msgpack_numpy

    class Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.sleeps.append(duration)
            self.now += duration

    class Environment:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock
            self.seed = None
            self.steps = 0

        def reset(self, *, seed: int):
            self.seed = seed
            return _raw_observation(), {"is_success": False}

        def step(self, action: np.ndarray):
            self.steps += 1
            self.clock.now += 0.01
            observation = _raw_observation()
            observation["agent_pos"] = np.full(14, self.steps, dtype=np.float64)
            return observation, float(self.steps), self.steps == 3, False, {"is_success": self.steps == 3}

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Video:
        def __init__(self) -> None:
            self.frames = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frames += 1

    clock = Clock()
    environment = Environment(clock)
    video = Video()
    events = []
    capture_path = tmp_path / "policy-observation.msgpack"
    result = control_episode(
        environment,
        Policy(),
        video,
        seed=2,
        prompt=None,
        profile=POLICY_PROFILES["pi05_aloha_base"],
        max_steps=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
        capture_path=capture_path,
    )
    assert environment.seed == 2
    assert environment.steps == result["steps_applied"] == video.frames == 3
    assert result["terminated"] is True
    assert result["task_success"] is True
    assert clock.sleeps == pytest.approx([0.01, 0.01])
    assert result["active_step_hz"] == pytest.approx(50.0)
    assert result["faster_than_20ms_count"] == 0
    steps = [event for event in events if event["event"] == "step"]
    assert [event["step"] for event in steps] == [0, 1, 2]
    assert [event["applied_step"] for event in steps] == [1, 2, 3]
    assert [event["elapsed_seconds"] for event in steps] == pytest.approx([0.01, 0.03, 0.05])
    assert [event["actual_joint_positions"] for event in steps] == [[float(value)] * 14 for value in (1, 2, 3)]
    assert all(event["commanded_joint_positions"] == [0.0] * 14 for event in steps)
    captured = msgpack_numpy.unpackb(capture_path.read_bytes())
    assert capture_path.stat().st_mode & 0o077 == 0
    assert captured["state"].shape == (14,)
    assert set(captured["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def test_custom_episode_passes_all_model_joints_into_sim_video_and_telemetry() -> None:
    scenario = SCENARIOS["push_letters_single"]
    home = np.arange(14, dtype=np.float64) + 100
    raw_action = np.arange(14, dtype=np.float64)

    class Environment:
        action = None

        def reset(self, *, seed: int):
            observation = _raw_observation()
            observation["agent_pos"] = home.copy()
            return observation, _custom_info()

        def step(self, action: np.ndarray):
            self.action = action.copy()
            observation = _raw_observation()
            observation["agent_pos"] = home + 1
            return observation, 0.0, True, False, _custom_info()

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            assert observation["prompt"] == scenario.prompt
            return raw_action

    class Subscriber:
        def __init__(self) -> None:
            self.actions = []

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.actions.append(action["actions"].copy())

    environment = Environment()
    video = Subscriber()
    display = Subscriber()
    events = []
    result = control_episode(
        environment,
        Policy(),
        video,
        seed=0,
        prompt=scenario.prompt,
        profile=POLICY_PROFILES["pi0_aloha_sim"],
        scenario=scenario,
        display=display,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
    )
    expected = raw_action.copy()
    step = next(event for event in events if event["event"] == "step")
    assert result["steps_applied"] == 1
    assert np.array_equal(environment.action, expected)
    assert np.array_equal(video.actions[0], expected)
    assert np.array_equal(display.actions[0], expected)
    assert step["commanded_joint_positions"] == expected.tolist()
    assert step["actual_joint_positions"] == (home + 1).tolist()
    assert step["scenario_info"] == _custom_info()
    assert result["coverage_sample_count"] == 1
    assert result["initial_target_area_coverage_percent"] == 10.0
    assert result["final_target_area_coverage_percent"] == 10.0
    assert result["best_target_area_coverage_percent"] == 10.0
    assert result["best_target_area_coverage_step"] == 1


def test_custom_episode_tracks_earliest_best_coverage_time_and_preserves_failure_progress(tmp_path: Path) -> None:
    scenario = SCENARIOS["push_pi_single"]

    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.now += duration

    class Environment:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock
            self.steps = 0

        def reset(self, *, seed: int):
            return _raw_observation(), _custom_info(scenario.key)

        def step(self, action: np.ndarray):
            self.steps += 1
            self.clock.now += 0.01
            info = _custom_info(scenario.key)
            info["target_area_coverage"] = (0.2, 0.6, 0.6)[self.steps - 1]
            info["body_0_target_area_coverage"] = info["target_area_coverage"]
            return _raw_observation(), 0.0, self.steps == 3, False, info

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Video:
        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            return None

    clock = Clock()
    result = control_episode(
        Environment(clock),
        Policy(),
        Video(),
        seed=0,
        prompt=scenario.prompt,
        profile=POLICY_PROFILES["pi0_aloha_sim"],
        scenario=scenario,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result["coverage_sample_count"] == 3
    assert result["final_target_area_coverage_percent"] == pytest.approx(60.0)
    assert result["best_target_area_coverage_percent"] == pytest.approx(60.0)
    assert result["best_target_area_coverage_step"] == 2
    assert result["time_to_best_target_area_coverage_seconds"] == pytest.approx(0.03)

    progress = {}

    class FailingVideo(Video):
        def on_step(self, observation: dict, action: dict) -> None:
            raise KeyboardInterrupt

    capture_path = tmp_path / "partial-policy-observation.msgpack"
    with pytest.raises(KeyboardInterrupt):
        control_episode(
            Environment(Clock()),
            Policy(),
            FailingVideo(),
            seed=0,
            prompt=scenario.prompt,
            profile=POLICY_PROFILES["pi05_aloha_base"],
            scenario=scenario,
            progress=progress,
            capture_path=capture_path,
        )
    assert progress["coverage_sample_count"] == progress["steps_applied"] == 1
    assert progress["best_target_area_coverage_step"] == 1
    assert capture_path.is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_hash", "bad"),
        ("body_count", 2.0),
        ("held_steps", -1),
        ("both_arms_participated", True),
        ("terminal_reason", "success"),
        ("body_0_xy_error", float("nan")),
        ("target_area_coverage", 1.01),
    ],
)
def test_custom_step_info_rejects_invalid_identity_counts_and_metrics(field: str, value: object) -> None:
    info = _custom_info()
    info[field] = value
    with pytest.raises(ValueError, match="custom scenario|non-finite"):
        _scenario_step_info(info, SCENARIOS["push_letters_single"])


def test_custom_episode_preserves_core_artifacts_before_rejecting_hash_drift() -> None:
    scenario = SCENARIOS["push_pi_dual"]

    class Environment:
        def reset(self, *, seed: int):
            return _raw_observation(), _custom_info(scenario.key)

        def step(self, action: np.ndarray):
            info = _custom_info(scenario.key)
            info["layout_hash"] = "c" * 64
            return _raw_observation(), 0.0, False, False, info

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Video:
        frames = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frames += 1

    video = Video()
    events = []
    with pytest.raises(ValueError, match="identity changed"):
        control_episode(
            Environment(),
            Policy(),
            video,
            seed=0,
            prompt=scenario.prompt,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            scenario=scenario,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
        )
    step = next(event for event in events if event["event"] == "step")
    assert video.frames == 1
    assert "scenario_info" not in step
    assert len(step["actual_joint_positions"]) == len(step["commanded_joint_positions"]) == 14


def test_post_step_telemetry_failure_still_captures_video_frame() -> None:
    class Environment:
        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            return _raw_observation(), 0.0, True, False, {}

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Video:
        frames = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frames += 1

    video = Video()

    def emit(event: str, **fields: object) -> None:
        if event == "step":
            raise OSError("telemetry disk full")

    with pytest.raises(OSError, match="disk full"):
        control_episode(
            Environment(),
            Policy(),
            video,
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            emit=emit,
        )
    assert video.frames == 1


def test_display_failure_is_reported_without_changing_episode_result() -> None:
    class Environment:
        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            return _raw_observation(), 0.0, True, False, {}

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Subscriber:
        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            return None

    class Display(Subscriber):
        def on_step(self, observation: dict, action: dict) -> None:
            raise RuntimeError("viewer closed")

        def on_episode_end(self) -> None:
            return None

    result = control_episode(
        Environment(),
        Policy(),
        Subscriber(),
        seed=0,
        prompt=None,
        profile=POLICY_PROFILES["pi0_aloha_sim"],
        display=Display(),
    )
    assert result["steps_applied"] == 1
    assert result["display_error"] == {"type": "RuntimeError", "message": "viewer closed"}


def test_video_saver_publishes_atomically_and_finalizes_once(tmp_path: Path, monkeypatch) -> None:
    writes = []

    def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
        writes.append((np.asarray(frames[0]), fps))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("examples.aloha_sim.saver.imageio.mimwrite", write_video)
    saver = VideoSaver(
        tmp_path,
        filename="episode.mp4",
        camera_views=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    )
    saver.on_episode_start()
    saver.on_step(
        {
            "images": {
                name: np.full((3, 224, 224), value, dtype=np.uint8)
                for name, value in (("cam_high", 0), ("cam_left_wrist", 1), ("cam_right_wrist", 2))
            }
        },
        {},
    )
    saver.on_episode_end()
    saver.on_episode_end()
    assert saver.output_path == tmp_path / "episode.mp4"
    assert saver.output_path.read_bytes() == b"video"
    frame, fps = writes[0]
    assert fps == 50
    assert frame.shape == (224, 672, 3)
    assert [frame[:, start : start + 224].max() for start in (0, 224, 448)] == [0, 1, 2]
    assert not list(tmp_path.glob("*.tmp"))


def test_video_saver_streams_two_camera_frames_at_configured_fps(tmp_path: Path, monkeypatch) -> None:
    frames = []
    configured = []

    class Writer:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

        def append_data(self, frame: np.ndarray) -> None:
            frames.append(frame.copy())

        def close(self) -> None:
            self.path.write_bytes(b"video")

    def get_writer(path: Path, *, fps: int) -> Writer:
        configured.append(fps)
        return Writer(path)

    monkeypatch.setattr("examples.aloha_sim.saver.imageio.get_writer", get_writer)
    saver = VideoSaver(
        tmp_path,
        filename="episode.mp4",
        camera_views=("agentview", "eye_in_hand"),
        fps=20,
        streaming=True,
    )
    saver.on_episode_start()
    saver.on_step(
        {
            "images": {
                "agentview": np.zeros((3, 224, 224), dtype=np.uint8),
                "eye_in_hand": np.ones((3, 224, 224), dtype=np.uint8),
            }
        },
        {},
    )
    saver.on_episode_end()
    saver.on_episode_end()
    assert configured == [20]
    assert saver.frame_count == 1
    assert frames[0].shape == (224, 448, 3)
    assert saver.output_path == tmp_path / "episode.mp4"
    assert saver.output_path.read_bytes() == b"video"


def test_streaming_video_cleans_up_writer_after_first_frame_failure(tmp_path: Path, monkeypatch) -> None:
    closed = []

    class Writer:
        def append_data(self, _: np.ndarray) -> None:
            raise OSError("encode failed")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("examples.aloha_sim.saver.imageio.get_writer", lambda *_args, **_kwargs: Writer())
    saver = VideoSaver(tmp_path, filename="episode.mp4", streaming=True)
    saver.on_episode_start()
    with pytest.raises(OSError, match="encode failed"):
        saver.on_step({"images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)}}, {})
    with pytest.raises(ValueError, match="without frames"):
        saver.on_episode_end()
    assert closed == [True]
    assert not list(tmp_path.iterdir())


def test_control_error_preserves_exact_partial_step_count() -> None:
    class Environment:
        def reset(self, *, seed: int):
            return _raw_observation(), {"is_success": False}

        def step(self, action: np.ndarray):
            return _raw_observation(), 0.0, False, False, {"is_success": False}

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            if step:
                raise RuntimeError("connection lost")
            return np.zeros(14, dtype=np.float64)

    class Video:
        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            return None

    progress = {}
    events = []
    with pytest.raises(RuntimeError, match="connection lost"):
        control_episode(
            Environment(),
            Policy(),
            Video(),
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            progress=progress,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
        )
    assert progress["steps_applied"] == 1
    steps = [event for event in events if event["event"] == "step"]
    assert len(steps) == 1
    assert steps[0]["applied_step"] == progress["steps_applied"]
    assert len(steps[0]["actual_joint_positions"]) == len(steps[0]["commanded_joint_positions"]) == 14


@pytest.mark.parametrize(
    ("reward", "info", "message"),
    [(float("nan"), {}, "reward"), (0.0, {"bad": object()}, "unsupported")],
)
def test_post_step_validation_failure_preserves_applied_count(reward: float, info: dict, message: str) -> None:
    class Environment:
        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            return _raw_observation(), reward, False, False, info

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(14, dtype=np.float64)

    class Video:
        frames = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frames += 1

    progress = {}
    events = []
    video = Video()
    with pytest.raises(ValueError, match=message):
        control_episode(
            Environment(),
            Policy(),
            video,
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            progress=progress,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
        )
    assert progress["steps_applied"] == 1
    assert [event["applied_step"] for event in events if event["event"] == "step"] == [1]
    assert video.frames == 1


def test_success_latches_and_finite_actions_are_not_clipped() -> None:
    class Environment:
        def __init__(self) -> None:
            self.steps = 0
            self.actions = []

        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            self.actions.append(action.copy())
            info = ({"is_success": True}, {"is_success": False}, {})[self.steps - 1]
            return _raw_observation(), 0.0, self.steps == 3, False, info

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.full(14, 1_000.0, dtype=np.float32)

    class Video:
        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            return None

    environment = Environment()
    result = control_episode(
        environment,
        Policy(),
        Video(),
        seed=0,
        prompt=None,
        profile=POLICY_PROFILES["pi0_aloha_sim"],
        sleep=lambda _: None,
    )
    assert result["task_success"] is True
    assert all(np.all(action == 1_000.0) for action in environment.actions)
    assert all(action.dtype == np.float32 for action in environment.actions)


def test_invalid_action_never_reaches_environment() -> None:
    class Environment:
        steps = 0

        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            raise AssertionError("invalid action was applied")

    class Policy:
        def infer(self, observation: dict, step: int) -> np.ndarray:
            return np.zeros(13)

    class Video:
        def on_episode_start(self) -> None:
            return None

    environment = Environment()
    with pytest.raises(ValueError, match="action must"):
        control_episode(
            environment,
            Policy(),
            Video(),
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
        )
    assert environment.steps == 0


def test_output_root_inside_repository_must_be_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.config.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(ValueError, match="must be ignored"):
        validate_output_root(Path("unignored-phase-output"))


@pytest.mark.parametrize(
    "outcome",
    [
        "complete",
        "interrupted",
        "telemetry-close-failure",
        "policy-close-failure",
        "plot-failure",
        "plot-interrupted",
        "camera-failure",
        "camera-interrupted",
    ],
)
def test_run_seed_finalizes_manifest_and_resources(tmp_path: Path, monkeypatch, outcome: str, capsys) -> None:
    class Transport:
        def __init__(self) -> None:
            self.closed = False

        def infer(self, observation: dict) -> dict:
            return {"actions": np.zeros((50, 14), dtype=np.float64)}

        def close(self) -> None:
            self.closed = True
            if outcome == "policy-close-failure":
                raise TimeoutError("policy cleanup remains pending")

    class Environment:
        def __init__(self) -> None:
            self.closed = False
            self.steps = 0
            self.spec = SimpleNamespace(max_episode_steps=300)
            self.metadata = {"render_fps": 50}
            self.action_space = SimpleNamespace(shape=(14,))
            self._env = SimpleNamespace(physics=self.Physics())

        @property
        def unwrapped(self):
            return self

        class Physics:
            def __init__(self) -> None:
                self.renders = 0

            def render(self, *, height: int, width: int, camera_id: str) -> np.ndarray:
                self.renders += 1
                if self.renders > 2 and outcome in {"camera-failure", "camera-interrupted"}:
                    if outcome == "camera-interrupted":
                        raise KeyboardInterrupt
                    raise OSError("wrist render failed")
                return np.zeros((height, width, 3), dtype=np.uint8)

        def reset(self, *, seed: int):
            observation = _raw_observation()
            if outcome in {"camera-failure", "camera-interrupted"}:
                observation["pixels"] = {"top": observation["pixels"]["top"]}
            return observation, {}

        def step(self, action: np.ndarray):
            self.steps += 1
            if outcome == "interrupted" and self.steps == 2:
                raise KeyboardInterrupt
            observation = _raw_observation()
            if outcome in {"camera-failure", "camera-interrupted"}:
                observation["pixels"] = {"top": observation["pixels"]["top"]}
            return observation, 0.0, outcome != "interrupted", False, {}

        def close(self) -> None:
            self.closed = True

    class Video:
        def __init__(self, output_dir: Path, filename: str, **kwargs: object) -> None:
            self.output_path = output_dir / filename
            self.frame_count = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frame_count += 1

        def on_episode_end(self) -> None:
            self.output_path.write_bytes(b"video")

    transport = Transport()
    environment = Environment()
    monkeypatch.setattr("tools.remote_aloha.run._connect", lambda *args: (transport, {"ready": True}))
    monkeypatch.setattr("tools.remote_aloha.run._make_environment", lambda task: environment)
    monkeypatch.setattr("tools.remote_aloha.run.VideoSaver", Video)
    monkeypatch.setattr(
        "tools.remote_aloha.run.verify_video",
        lambda path, frames, shape: {
            "bytes": path.stat().st_size,
            "fps": 50.0,
            "frames": frames,
            "shape": list(shape),
        },
    )
    monkeypatch.setattr("tools.remote_aloha.run.package_versions", lambda: {"numpy": "1.26.4"})
    if outcome in {"plot-failure", "plot-interrupted"}:
        error = OSError("plot failed") if outcome == "plot-failure" else KeyboardInterrupt()
        monkeypatch.setattr(
            "tools.remote_aloha.run.write_trajectory_plot",
            lambda *args: (_ for _ in ()).throw(error),
        )
    if outcome == "telemetry-close-failure":

        class FailingCloseWriter(JsonlWriter):
            def close(self) -> None:
                super().close()
                raise OSError("telemetry close failed")

        monkeypatch.setattr("tools.remote_aloha.run.JsonlWriter", FailingCloseWriter)
    output_dir = tmp_path / "seed-0"
    arguments = (
        MacSimConfig(episodes=1),
        RemoteConfig(),
        "a" * 40,
        "b" * 40,
        0,
        output_dir,
        "c" * 32,
    )
    if outcome in {"interrupted", "plot-interrupted", "camera-interrupted"}:
        with pytest.raises(KeyboardInterrupt):
            _run_seed(*arguments)
    elif outcome in {"telemetry-close-failure", "policy-close-failure", "plot-failure", "camera-failure"}:
        expected = {
            "telemetry-close-failure": "telemetry close failed",
            "policy-close-failure": "policy cleanup remains pending",
            "plot-failure": "plot failed",
            "camera-failure": "wrist render failed",
        }[outcome]
        with pytest.raises((OSError, TimeoutError), match=expected):
            _run_seed(*arguments)
    else:
        result = _run_seed(*arguments)
        assert result["infrastructure_pass"] is True
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cleanup_pending"] is (outcome == "policy-close-failure")
    expected_status = {
        "telemetry-close-failure": "failed",
        "policy-close-failure": "failed",
        "plot-failure": "failed",
        "plot-interrupted": "interrupted",
        "camera-failure": "failed",
        "camera-interrupted": "interrupted",
    }.get(outcome, outcome)
    assert manifest["status"] == expected_status
    assert transport.closed
    assert environment.closed
    telemetry = (output_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(telemetry[0])["event"] == "metadata"
    assert json.loads(telemetry[-1])["event"] == "terminal"
    telemetry_summary_paths = (output_dir / "telemetry-summary.json", output_dir / "telemetry-summary.md")
    if outcome == "telemetry-close-failure":
        assert not any(path.exists() for path in telemetry_summary_paths)
        assert manifest["telemetry"]["writer_closed"] is False
    else:
        assert all(path.exists() for path in telemetry_summary_paths)
        assert manifest["telemetry"]["writer_closed"] is True
    trajectory = manifest["trajectory"]
    assert trajectory["sample_count"] == manifest["episode"]["steps_applied"] == 1
    assert trajectory["step_coverage"] == 1.0
    assert trajectory["actual_series_count"] == trajectory["commanded_series_count"] == 14
    if outcome in {"plot-failure", "plot-interrupted"}:
        assert trajectory["plot_status"] == "failed"
        assert trajectory["path"] is None
    else:
        assert trajectory["plot_status"] == "passed"
        assert Path(trajectory["path"]).is_file()
    assert manifest["video"]["frames"] == manifest["episode"]["steps_applied"]
    expected_video_status = (
        "partial" if outcome in {"interrupted", "camera-failure", "camera-interrupted"} else "complete"
    )
    assert manifest["video"]["status"] == expected_video_status
    progress = capsys.readouterr()
    assert progress.out == ""
    assert "[simulation] episode seed=0 preparing" in progress.err
    assert "[simulation] seed=0 progress=1/300" in progress.err
    assert f"[simulation] episode seed=0 end status={expected_status}" in progress.err
    assert len([line for line in progress.err.splitlines() if "seed=0 progress=" in line]) <= 11
    assert str(tmp_path) not in progress.err


@pytest.mark.parametrize("stderr_mode", ["closed", "missing"])
def test_status_output_never_changes_the_run_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stderr_mode: str
) -> None:
    class ClosedStderr:
        def write(self, value: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

    monkeypatch.setattr("tools.remote_aloha.run.sys.stderr", ClosedStderr() if stderr_mode == "closed" else None)
    _status("safe bounded progress")
    assert capsys.readouterr().out == ""


def test_run_seed_finalizes_manifest_when_telemetry_cannot_start(tmp_path: Path, monkeypatch) -> None:
    def fail_writer(*args: object, **kwargs: object):
        raise OSError("telemetry create failed")

    monkeypatch.setattr("tools.remote_aloha.run.JsonlWriter", fail_writer)
    output_dir = tmp_path / "seed-0"
    with pytest.raises(OSError, match="telemetry create failed"):
        _run_seed(MacSimConfig(episodes=1), RemoteConfig(), "a" * 40, "b" * 40, 0, output_dir, "c" * 32)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["cleanup_pending"] is False
    assert manifest["telemetry"]["writer_closed"] is False


def test_run_seed_publishes_custom_scenario_identity_and_integer_counts(tmp_path: Path, monkeypatch) -> None:
    scenario = SCENARIOS["push_pi_single"]

    class Transport:
        def infer(self, observation: dict) -> dict:
            return {"actions": np.zeros((50, 14), dtype=np.float64)}

        def close(self) -> None:
            return None

    class Environment:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(max_episode_steps=300)
            self.metadata = {"render_fps": 50}
            self.action_space = SimpleNamespace(shape=(14,))
            self.scene_hash = "a" * 64

        @property
        def unwrapped(self):
            return self

        def reset(self, *, seed: int):
            return _raw_observation(), _custom_info(scenario.key)

        def step(self, action: np.ndarray):
            info = _custom_info(scenario.key)
            info.update({"is_success": True, "held_steps": 5, "terminal_reason": "success"})
            return _raw_observation(), 1.0, True, False, info

        def close(self) -> None:
            return None

    class Video:
        def __init__(self, output_dir: Path, filename: str, **kwargs: object) -> None:
            self.output_path = output_dir / filename
            self.frame_count = 0

        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            self.frame_count += 1

        def on_episode_end(self) -> None:
            self.output_path.write_bytes(b"video")

    monkeypatch.setattr("tools.remote_aloha.run._connect", lambda *args: (Transport(), {"ready": True}))
    monkeypatch.setattr("tools.remote_aloha.run._make_environment", lambda config: Environment())
    monkeypatch.setattr("tools.remote_aloha.run.VideoSaver", Video)
    monkeypatch.setattr(
        "tools.remote_aloha.run.verify_video",
        lambda path, frames, shape: {
            "bytes": path.stat().st_size,
            "fps": 50.0,
            "frames": frames,
            "shape": list(shape),
        },
    )
    monkeypatch.setattr("tools.remote_aloha.run.package_versions", lambda: {"numpy": "1.26.4"})
    output_dir = tmp_path / "seed-0"
    manifest = _run_seed(
        MacSimConfig(task=scenario.gym_id, scenario=scenario, episodes=1),
        RemoteConfig(),
        "a" * 40,
        "b" * 40,
        0,
        output_dir,
        "c" * 32,
    )
    public = json.loads((output_dir / "telemetry-summary.json").read_text(encoding="utf-8"))
    assert manifest["infrastructure_pass"] is True
    assert public["metadata"]["scenario"] == scenario.key
    assert public["metadata"]["scene_hash"] == "a" * 64
    assert public["metadata"]["target_area_coverage_method"] == "exact-planar-union-v1"
    assert public["result"]["push_success"] == 1
    assert public["result"]["coverage_sample_count"] == 1
    assert public["result"]["best_target_area_coverage_percent"] == 10.0
    assert public["result"]["best_target_area_coverage_step"] == 1
    assert isinstance(public["result"]["push_success"], int)


def test_run_writes_failed_root_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.run.load_mac_sim_config",
        lambda: MacSimConfig(episodes=1, output_dir=tmp_path),
    )
    monkeypatch.setattr("tools.remote_aloha.run.load_remote_config", RemoteConfig)
    monkeypatch.setattr("tools.remote_aloha.run.verify_ready_tunnel", lambda config: ({}, "source"))
    monkeypatch.setattr(
        "tools.remote_aloha.run.start_gpu_sampler",
        lambda *args: SimpleNamespace(check=lambda: None, stop=lambda output: None),
    )
    monkeypatch.setattr("tools.remote_aloha.run._run_seed", lambda *args: (_ for _ in ()).throw(RuntimeError("lost")))
    with pytest.raises(RuntimeError, match="lost"):
        run()
    summaries = list(tmp_path.glob("phase05/*/pi05_aloha_base/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == {"type": "RuntimeError", "message": "lost"}
    progress = capsys.readouterr()
    assert progress.out == ""
    assert "run start profile=pi05_aloha_base" in progress.err
    assert "run validating evidence" in progress.err
    assert "run end status=failed episodes=0/1" in progress.err
    assert str(tmp_path) not in progress.err
