import json

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_policy_timing
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing
from tools.remote_aloha.policy_contract import validate_timing_reconciliation
from tools.remote_aloha.policy_smoke import run_policy_smoke

SHA = "a" * 40


@pytest.mark.parametrize("profile_name", POLICY_PROFILES)
def test_policy_metadata_and_response_contract(profile_name: str) -> None:
    profile = POLICY_PROFILES[profile_name]
    metadata = {
        "policy_profile": profile.name,
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "checkpoint_variant": profile.checkpoint_label,
        "policy_backend": "jax",
        "action_horizon": profile.action_horizon,
        "action_dimension": profile.action_dimension,
        "source_sha": SHA,
        "compact_masked_images": True,
        "jax_platform": "gpu",
        "jax_device": "NVIDIA GeForce RTX 3090",
    }
    validate_server_metadata(metadata, profile, SHA)
    actions = validate_policy_response(
        {"actions": np.zeros((profile.action_horizon, profile.action_dimension), dtype=np.float32)}, profile
    )
    assert actions.shape == (profile.action_horizon, profile.action_dimension)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"actions": np.zeros((49, 14), dtype=np.float32)},
        {"actions": np.zeros((50, 13), dtype=np.float32)},
        {"actions": np.zeros((50, 14), dtype=np.int64)},
        {"actions": np.full((50, 14), np.nan, dtype=np.float32)},
    ],
)
def test_invalid_policy_response_is_rejected(response: object) -> None:
    with pytest.raises(ValueError, match="actions"):
        validate_policy_response(response, POLICY_PROFILES["pi0_aloha_sim"])


def test_mismatched_metadata_is_rejected() -> None:
    profile = POLICY_PROFILES["pi0_aloha_sim"]
    metadata = {
        "policy_profile": "pi05_aloha_base",
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "checkpoint_variant": profile.checkpoint_label,
        "policy_backend": "jax",
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": SHA,
        "compact_masked_images": True,
        "jax_platform": "gpu",
        "jax_device": "NVIDIA GeForce RTX 3090",
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_server_metadata(metadata, profile, SHA)


def test_pytorch_metadata_requires_converted_variant_and_cuda_3090() -> None:
    profile = POLICY_PROFILES["pi0_aloha_sim"]
    metadata = {
        "policy_profile": profile.name,
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "checkpoint_variant": "pi0_aloha_sim_pytorch",
        "policy_backend": "pytorch",
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": SHA,
        "compact_masked_images": True,
        "torch_platform": "cuda",
        "torch_device": "NVIDIA GeForce RTX 3090",
        "torch_model_device": "cuda:0",
    }
    validate_server_metadata(metadata, profile, SHA, "pytorch")


@pytest.mark.parametrize(
    "timing",
    [None, {}, {"infer_ms": True}, {"infer_ms": -1}, {"infer_ms": float("nan")}, {"infer_ms": 1, "prev_total_ms": -1}],
)
def test_invalid_server_timing_is_rejected(timing: object) -> None:
    with pytest.raises(ValueError, match="timing"):
        validate_server_timing({"server_timing": timing})


def test_server_timing_accepts_optional_previous_total() -> None:
    assert validate_server_timing({"server_timing": {"infer_ms": 1.5}}) == {"infer_ms": 1.5}
    assert validate_server_timing({"server_timing": {"infer_ms": 1.5, "prev_total_ms": 2}})["prev_total_ms"] == 2
    assert validate_server_timing({"server_timing": {"infer_ms": 1.5, "private": "discarded"}}) == {"infer_ms": 1.5}


def test_policy_timing_accepts_safe_stages_and_rejects_invalid_values() -> None:
    timing = validate_policy_timing(
        {
            "policy_timing": {
                "infer_ms": 10.0,
                "input_transfer_ms": 1,
                "vision_ms": 2.0,
                "private": "discarded",
            }
        }
    )
    assert timing == {"infer_ms": 10.0, "input_transfer_ms": 1, "vision_ms": 2.0}
    for invalid in ({}, {"infer_ms": -1}, {"infer_ms": float("nan")}, {"infer_ms": 1, "denoise_ms": True}):
        with pytest.raises(ValueError, match="policy timing"):
            validate_policy_timing({"policy_timing": invalid})


def test_policy_timing_reconciliation_rejects_inconsistent_cuda_stages() -> None:
    timing = {
        "infer_ms": 10.0,
        "input_transfer_ms": 1.0,
        "model_ms": 7.0,
        "output_transfer_ms": 1.0,
        "vision_ms": 1.0,
        "language_embed_ms": 1.0,
        "prefix_kv_ms": 2.0,
        "denoise_ms": 3.0,
        "model_stages_ms": 7.0,
    }
    validate_timing_reconciliation(timing, {"infer_ms": 11.0})
    validate_timing_reconciliation({**timing, "model_ms": 4.0}, {"infer_ms": 11.0})
    with pytest.raises(ValueError, match="reconcile"):
        validate_timing_reconciliation({**timing, "denoise_ms": 8.0}, {"infer_ms": 11.0})


def test_policy_smoke_closes_client_when_metadata_validation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from openpi_client import websocket_client_policy

    closed = []

    class InvalidMetadataPolicy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_server_metadata(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", InvalidMetadataPolicy)
    with pytest.raises(ValueError, match="metadata"):
        run_policy_smoke(
            profile_name="pi0_aloha_sim",
            backend="pytorch",
            host="127.0.0.1",
            port=8000,
            source_sha=SHA,
        )
    assert closed == [True]


def test_policy_smoke_supports_five_warmups_and_compact_measured_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openpi_client import websocket_client_policy

    class Policy:
        calls = 0
        closed = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def get_server_metadata(self) -> dict[str, object]:
            return {
                "policy_profile": "pi0_aloha_sim",
                "config_name": "pi0_aloha_sim",
                "checkpoint_label": "pi0_aloha_sim",
                "checkpoint_variant": "pi0_aloha_sim_pytorch",
                "policy_backend": "pytorch",
                "action_horizon": 50,
                "action_dimension": 14,
                "source_sha": SHA,
                "compact_masked_images": True,
                "torch_platform": "cuda",
                "torch_device": "NVIDIA GeForce RTX 3090",
                "torch_model_device": "cuda:0",
            }

        def infer(self, observation: dict) -> dict[str, object]:
            self.calls += 1
            return {
                "actions": np.zeros((50, 14), dtype=np.float64),
                "server_timing": {"infer_ms": 10.0 + self.calls},
                "policy_timing": {
                    "infer_ms": 10.0,
                    "input_transfer_ms": 1.0,
                    "model_ms": 7.0,
                    "output_transfer_ms": 1.0,
                    "vision_ms": 1.0,
                    "language_embed_ms": 1.0,
                    "prefix_kv_ms": 2.0,
                    "denoise_ms": 3.0,
                    "model_stages_ms": 7.0,
                },
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", Policy)
    result = run_policy_smoke(
        profile_name="pi0_aloha_sim",
        backend="pytorch",
        host="127.0.0.1",
        port=8000,
        source_sha=SHA,
        warmup_requests=5,
        measured_requests=3,
    )
    assert result["warmup_requests"] == 5
    assert result["measured_requests"] == 3
    assert result["server_infer_ms"]["count"] == 3
    assert result["server_infer_ms"]["p95"] == pytest.approx(17.9)
    assert result["policy_timing"]["vision_ms"]["count"] == 3


def test_policy_smoke_uses_native_libero_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from openpi_client import websocket_client_policy

    class Policy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_server_metadata(self) -> dict[str, object]:
            return {
                "policy_profile": "pi05_libero",
                "config_name": "pi05_libero",
                "checkpoint_label": "pi05_libero",
                "checkpoint_variant": "pi05_libero_pytorch",
                "policy_backend": "pytorch",
                "action_horizon": 10,
                "action_dimension": 7,
                "source_sha": SHA,
                "compact_masked_images": True,
                "torch_platform": "cuda",
                "torch_device": "NVIDIA GeForce RTX 3090",
                "torch_model_device": "cuda:0",
            }

        def infer(self, observation: dict) -> dict[str, object]:
            assert observation["observation/state"].shape == (8,)
            assert observation["observation/image"].shape == (224, 224, 3)
            assert observation["observation/wrist_image"].shape == (224, 224, 3)
            return {
                "actions": np.zeros((10, 7), dtype=np.float32),
                "server_timing": {"infer_ms": 10.0},
                "policy_timing": {
                    "infer_ms": 9.0,
                    "input_transfer_ms": 1.0,
                    "model_ms": 7.0,
                    "output_transfer_ms": 1.0,
                    "vision_ms": 1.0,
                    "language_embed_ms": 1.0,
                    "prefix_kv_ms": 2.0,
                    "denoise_ms": 3.0,
                    "model_stages_ms": 7.0,
                },
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", Policy)
    result = run_policy_smoke(
        profile_name="pi05_libero",
        backend="pytorch",
        host="127.0.0.1",
        port=8000,
        source_sha=SHA,
        warmup_requests=1,
        measured_requests=1,
    )
    assert result["action_shape"] == [10, 7]
    assert result["camera_views"] == ["observation/image", "observation/wrist_image"]


def test_policy_smoke_replays_fixed_input_and_noise_exactly(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from openpi_client import websocket_client_policy

    class Policy:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_server_metadata(self) -> dict[str, object]:
            return {
                "policy_profile": "pi05_aloha_base",
                "config_name": "pi05_aloha",
                "checkpoint_label": "pi05_base",
                "checkpoint_variant": "pi05_base_pytorch",
                "policy_backend": "pytorch",
                "action_horizon": 50,
                "action_dimension": 14,
                "source_sha": SHA,
                "compact_masked_images": True,
                "torch_platform": "cuda",
                "torch_device": "NVIDIA GeForce RTX 3090",
                "torch_model_device": "cuda:0",
            }

        def infer(self, observation: dict) -> dict[str, object]:
            self.calls += 1
            assert observation["__openpi_benchmark_noise_seed"] == 7
            return {
                "actions": np.full((50, 14), 0.25, dtype=np.float64),
                "server_timing": {"infer_ms": 10.0},
                "policy_timing": {
                    "infer_ms": 9.0,
                    "input_transfer_ms": 1.0,
                    "model_ms": 7.0,
                    "output_transfer_ms": 1.0,
                    "vision_ms": 1.0,
                    "language_embed_ms": 1.0,
                    "prefix_kv_ms": 2.0,
                    "denoise_ms": 3.0,
                    "model_stages_ms": 7.0,
                },
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", Policy)
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    episode = tmp_path / "seed-0"
    episode.mkdir()
    observation_path = episode / "policy-observation.msgpack"
    observation_path.write_bytes(
        msgpack_numpy.Packer().pack(
            {
                "state": np.zeros(14, dtype=np.float64),
                "images": {name: image for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "prompt": "Use both arms.",
            }
        )
    )
    observation_path.chmod(0o600)
    manifest_path = episode / "manifest.json"
    capture_sha = "b" * 40
    manifest = {
        "profile": "pi05_aloha_base",
        "scenario": "push_pi_dual",
        "seed": 0,
        "source_sha": capture_sha,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    result = run_policy_smoke(
        profile_name="pi05_aloha_base",
        backend="pytorch",
        host="127.0.0.1",
        port=8000,
        source_sha=SHA,
        warmup_requests=1,
        measured_requests=2,
        observation_path=observation_path,
        fixed_noise_seed=7,
    )
    assert result["action_replay_exact"] is True
    assert result["fixed_noise_seed"] == 7
    assert result["capture_source_sha"] == capture_sha
    assert len(result["observation_sha256"]) == len(result["action_sha256"]) == 64
    for invalid in ({**manifest, "scenario": "/private/machine"}, {**manifest, "seed": -1}):
        manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest"):
            run_policy_smoke(
                profile_name="pi05_aloha_base",
                backend="pytorch",
                host="127.0.0.1",
                port=8000,
                source_sha=SHA,
                observation_path=observation_path,
                fixed_noise_seed=7,
            )
    manifest_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        run_policy_smoke(
            profile_name="pi05_aloha_base",
            backend="pytorch",
            host="127.0.0.1",
            port=8000,
            source_sha=SHA,
            observation_path=observation_path,
            fixed_noise_seed=7,
        )
