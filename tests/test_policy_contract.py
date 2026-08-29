import numpy as np
import pytest

from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing
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
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": SHA,
        "compact_masked_images": True,
        "jax_platform": "gpu",
        "jax_device": "NVIDIA GeForce RTX 3090",
    }
    validate_server_metadata(metadata, profile, SHA)
    actions = validate_policy_response({"actions": np.zeros((50, 14), dtype=np.float32)}, profile)
    assert actions.shape == (50, 14)


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
