from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.libero import run as libero_run
from tools.libero.run import _video_observation
from tools.libero.run import policy_step_limit
from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import RemoteConfig


def test_libero_duration_limits_cover_smoke_default_and_five_minutes() -> None:
    assert policy_step_limit(30, smoke=False, control_hz=20) == (30, 600)
    assert policy_step_limit(300, smoke=False, control_hz=20) == (300, 6000)
    assert policy_step_limit(300, smoke=True, control_hz=20) == (6, 120)
    for seconds in (0, 301):
        with pytest.raises(ValueError, match="between 1 and 300"):
            policy_step_limit(seconds, smoke=False, control_hz=20)


def test_libero_video_observation_preserves_both_policy_views() -> None:
    artifact = _video_observation(
        {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        }
    )
    assert set(artifact["images"]) == {"agentview", "eye_in_hand"}
    assert artifact["images"]["agentview"].shape == (3, 224, 224)
    assert artifact["images"]["eye_in_hand"].max() == 1


def test_libero_video_observation_rejects_bad_policy_frame() -> None:
    with pytest.raises(ValueError, match="uint8 HWC"):
        _video_observation(
            {
                "observation/image": np.zeros((224, 224, 3), dtype=np.float32),
                "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            }
        )


def test_libero_run_writes_established_episode_artifact_set(tmp_path: Path, monkeypatch) -> None:
    source_sha = "a" * 40
    profile = POLICY_PROFILES["pi05_libero"]

    class Sampler:
        def check(self) -> None:
            return None

        def stop(self, root: Path) -> Path:
            path = root / "gpu-metrics.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            return path

    class Client:
        def __init__(self, **_: object) -> None:
            return None

        def get_server_metadata(self) -> dict[str, object]:
            return {
                "policy_profile": profile.name,
                "config_name": profile.config_name,
                "checkpoint_label": profile.checkpoint_label,
                "checkpoint_variant": "pi05_libero_pytorch",
                "policy_backend": "pytorch",
                "action_horizon": 10,
                "action_dimension": 7,
                "source_sha": source_sha,
                "compact_masked_images": True,
                "torch_platform": "cuda",
                "torch_device": "NVIDIA GeForce RTX 3090",
                "torch_model_device": "cuda:0",
            }

        def infer(self, _: dict[str, object]) -> dict[str, object]:
            return {
                "actions": np.zeros((10, 7), dtype=np.float64),
                "server_timing": {"infer_ms": 10.0},
                "policy_timing": {
                    "infer_ms": 8.0,
                    "input_transfer_ms": 1.0,
                    "model_ms": 5.0,
                    "output_transfer_ms": 1.0,
                    "vision_ms": 1.0,
                    "language_embed_ms": 0.0,
                    "prefix_kv_ms": 1.0,
                    "denoise_ms": 2.0,
                    "model_stages_ms": 4.0,
                },
            }

        def close(self) -> None:
            return None

    class Task:
        @staticmethod
        def coverage() -> dict[str, float]:
            return {"pi": 0.0, "overall": 0.0}

        @staticmethod
        def layout() -> dict[str, dict[str, float]]:
            return {"pi_1": {"x": -0.1, "y": -0.1}, "pi_target_1": {"x": 0.1, "y": 0.1}}

    class Environment:
        env = Task()

        def __init__(self) -> None:
            self.steps = 0

        @staticmethod
        def observation() -> dict[str, np.ndarray]:
            return {"robot0_joint_pos": np.zeros(7, dtype=np.float64)}

        def reset(self) -> dict[str, np.ndarray]:
            return self.observation()

        @staticmethod
        def seed(_: int) -> None:
            return None

        def step(self, _: list[float]) -> tuple[dict[str, np.ndarray], float, bool, dict]:
            self.steps += 1
            return self.observation(), 0.0, self.steps == 2, {}

        def close(self) -> None:
            return None

    def element(_: dict[str, object], prompt: str, resize: int) -> tuple[dict[str, object], np.ndarray]:
        assert prompt == "push"
        assert resize == 224
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return {
            "observation/state": np.zeros(8, dtype=np.float64),
            "observation/image": image,
            "observation/wrist_image": image.copy(),
            "prompt": prompt,
        }, image

    monkeypatch.setattr(
        libero_run,
        "load_remote_config",
        lambda: RemoteConfig(policy_profile=profile, policy_backend="pytorch"),
    )
    monkeypatch.setattr(libero_run, "verify_ready_tunnel", lambda _: (None, source_sha))
    monkeypatch.setattr(libero_run, "start_gpu_sampler", lambda *_: Sampler())
    gpu_events = [
        {
            "schema": 1,
            "event": "gpu",
            "timestamp_utc": f"2026-08-29T12:00:0{index}.000Z",
            "monotonic_ns": index + 1,
            "metrics": {"gpu_memory_mib": 1000.0, "gpu_utilization_percent": 50.0, "server_rss_kib": 100.0},
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        libero_run,
        "_gpu_events",
        lambda *_: (gpu_events, {"gpu_sample_count": 2, "gpu_span_ms": 1000, "gpu_max_gap_ms": 1000}),
    )
    monkeypatch.setattr(libero_run, "_gpu_coverage", lambda *_: {"gpu_coverage_pass": True})
    monkeypatch.setattr(libero_run.websocket_client_policy, "WebsocketClientPolicy", Client)
    result = libero_run.run_scenario(
        scenario="push_pi",
        duration_seconds=1,
        smoke=False,
        seed=0,
        settle_steps=0,
        resize_size=224,
        replan_steps=5,
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=8000,
        control_hz=20,
        dummy_action=[0.0] * 7,
        create_env=lambda *_args, **_kwargs: (Environment(), "push"),
        policy_element=element,
        scene_hash="d" * 64,
        scene_metadata={"layout_hash": "c" * 64},
        layout_snapshot=lambda environment: environment.env.layout(),
    )
    root = Path(result["output"])
    assert root.relative_to(tmp_path).parts[0] == "libero_0829"
    episode = root / "seed-0"
    assert {
        "episode.mp4",
        "joint-trajectory.png",
        "manifest.json",
        "policy-observation.msgpack",
        "telemetry-summary.json",
        "telemetry-summary.md",
        "telemetry.jsonl",
    } <= {path.name for path in episode.iterdir()}
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["episode"]["steps_applied"] == manifest["video"]["frames"] == 20
    assert manifest["trajectory"]["joint_count"] == 7
    assert manifest["layout"]["layout_hash"] == "c" * 64
    assert manifest["layout"]["sampled"] == manifest["layout"]["settled"]
    rows = [json.loads(line) for line in (episode / "telemetry.jsonl").read_text().splitlines()]
    steps = [row for row in rows if row["event"] == "step"]
    assert len(steps) == 20
    assert len(steps[0]["actual_joint_positions"]) == len(steps[0]["osc_action"]) == 7
