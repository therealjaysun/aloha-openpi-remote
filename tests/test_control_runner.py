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
from tools.remote_aloha.run import _run_seed
from tools.remote_aloha.run import _validated_output_root
from tools.remote_aloha.run import control_episode
from tools.remote_aloha.run import run


def _raw_observation() -> dict:
    return {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros(14, dtype=np.float64),
    }


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
            return _raw_observation(), float(self.steps), self.steps == 3, False, {"is_success": self.steps == 3}

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
    )
    assert environment.seed == 2
    assert environment.steps == result["steps_applied"] == video.frames == 3
    assert result["terminated"] is True
    assert result["task_success"] is True
    assert clock.sleeps == pytest.approx([0.01, 0.01])
    assert result["active_step_hz"] == pytest.approx(50.0)
    assert result["faster_than_20ms_count"] == 0


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
    with pytest.raises(RuntimeError, match="connection lost"):
        control_episode(
            Environment(),
            Policy(),
            Video(),
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            progress=progress,
        )
    assert progress["steps_applied"] == 1


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
    with pytest.raises(ValueError, match=message):
        control_episode(
            Environment(),
            Policy(),
            Video(),
            seed=0,
            prompt=None,
            profile=POLICY_PROFILES["pi0_aloha_sim"],
            progress=progress,
        )
    assert progress["steps_applied"] == 1


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
        "tools.remote_aloha.run.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(ValueError, match="must be ignored"):
        _validated_output_root(Path("unignored-phase-output"))


@pytest.mark.parametrize("outcome", ["complete", "interrupted"])
def test_run_seed_finalizes_manifest_and_resources(tmp_path: Path, monkeypatch, outcome: str) -> None:
    class Transport:
        def __init__(self) -> None:
            self.closed = False

        def infer(self, observation: dict) -> dict:
            return {"actions": np.zeros((50, 14), dtype=np.float64)}

        def close(self) -> None:
            self.closed = True

    class Environment:
        def __init__(self) -> None:
            self.closed = False
            self.spec = SimpleNamespace(max_episode_steps=300)
            self.metadata = {"render_fps": 50}
            self.action_space = SimpleNamespace(shape=(14,))

        def reset(self, *, seed: int):
            return _raw_observation(), {}

        def step(self, action: np.ndarray):
            if outcome == "interrupted":
                raise KeyboardInterrupt
            return _raw_observation(), 0.0, True, False, {}

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
    monkeypatch.setattr("tools.remote_aloha.run.package_versions", dict)
    output_dir = tmp_path / "seed-0"
    if outcome == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            _run_seed(MacSimConfig(episodes=1), RemoteConfig(), "source", "upstream", 0, output_dir)
    else:
        result = _run_seed(MacSimConfig(episodes=1), RemoteConfig(), "source", "upstream", 0, output_dir)
        assert result["infrastructure_pass"] is True
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cleanup_pending"] is False
    assert manifest["status"] == outcome
    assert transport.closed
    assert environment.closed


def test_run_writes_failed_root_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.run.load_mac_sim_config",
        lambda: MacSimConfig(episodes=1, output_dir=tmp_path),
    )
    monkeypatch.setattr("tools.remote_aloha.run.load_remote_config", RemoteConfig)
    monkeypatch.setattr("tools.remote_aloha.run.verify_ready_tunnel", lambda config: ({}, "source"))
    monkeypatch.setattr("tools.remote_aloha.run._run_seed", lambda *args: (_ for _ in ()).throw(RuntimeError("lost")))
    with pytest.raises(RuntimeError, match="lost"):
        run()
    summaries = list(tmp_path.glob("phase04/*/pi0_aloha_sim/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == {"type": "RuntimeError", "message": "lost"}
