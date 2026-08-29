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
from tools.remote_aloha.run import _gpu_events
from tools.remote_aloha.run import _run_seed
from tools.remote_aloha.run import _write_performance_summary
from tools.remote_aloha.run import control_episode
from tools.remote_aloha.run import run
from tools.remote_aloha.telemetry import JsonlWriter


def _raw_observation() -> dict:
    return {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros(14, dtype=np.float64),
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
            "timestamp_utc": "2026-08-28T08:00:00.000Z",
            "monotonic_ns": 1,
            "run_id": "c" * 32,
            "profile": "pi0_aloha_sim",
            "source_sha": "a" * 40,
        },
        {
            "schema": 1,
            "event": "step",
            "timestamp_utc": "2026-08-28T08:00:01.000Z",
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
            "timestamp_utc": "2026-08-28T08:00:10.000Z",
            "monotonic_ns": 10_000_000_001,
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
                "buffer": {},
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


def test_control_episode_uses_exact_seed_steps_and_non_catchup_cadence() -> None:
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
    result = control_episode(
        environment,
        Policy(),
        video,
        seed=2,
        prompt=None,
        profile=POLICY_PROFILES["pi0_aloha_sim"],
        max_steps=10,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
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


def test_video_saver_publishes_atomically_and_finalizes_once(tmp_path: Path, monkeypatch) -> None:
    writes = []

    def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
        writes.append((len(frames), fps))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("examples.aloha_sim.saver.imageio.mimwrite", write_video)
    saver = VideoSaver(tmp_path, filename="episode.mp4")
    saver.on_episode_start()
    saver.on_step({"images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)}}, {})
    saver.on_episode_end()
    saver.on_episode_end()
    assert saver.output_path == tmp_path / "episode.mp4"
    assert saver.output_path.read_bytes() == b"video"
    assert writes == [(1, 50)]
    assert not list(tmp_path.glob("*.tmp"))


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
        def on_episode_start(self) -> None:
            return None

        def on_step(self, observation: dict, action: dict) -> None:
            return None

    progress = {}
    events = []
    with pytest.raises(ValueError, match=message):
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
    assert [event["applied_step"] for event in events if event["event"] == "step"] == [1]


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
            return np.full(14, 1_000.0)

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
    ],
)
def test_run_seed_finalizes_manifest_and_resources(tmp_path: Path, monkeypatch, outcome: str) -> None:
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

        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            if outcome == "interrupted" and self.steps == 2:
                raise KeyboardInterrupt
            return _raw_observation(), 0.0, outcome != "interrupted", False, {}

        def close(self) -> None:
            self.closed = True

    class Video:
        def __init__(self, output_dir: Path, filename: str) -> None:
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
        lambda path, frames: {"bytes": path.stat().st_size, "fps": 50.0, "frames": frames},
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
    if outcome in {"interrupted", "plot-interrupted"}:
        with pytest.raises(KeyboardInterrupt):
            _run_seed(*arguments)
    elif outcome in {"telemetry-close-failure", "policy-close-failure", "plot-failure"}:
        expected = {
            "telemetry-close-failure": "telemetry close failed",
            "policy-close-failure": "policy cleanup remains pending",
            "plot-failure": "plot failed",
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


def test_run_writes_failed_root_summary(tmp_path: Path, monkeypatch) -> None:
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
    summaries = list(tmp_path.glob("phase05/*/pi0_aloha_sim/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == {"type": "RuntimeError", "message": "lost"}
