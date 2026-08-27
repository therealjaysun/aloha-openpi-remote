import numpy as np
import pytest

from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing

SHA = "a" * 40


@pytest.mark.parametrize("profile_name", POLICY_PROFILES)
def test_policy_metadata_and_response_contract(profile_name: str) -> None:
    profile = POLICY_PROFILES[profile_name]
    metadata = {
        "policy_profile": profile.name,
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": SHA,
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
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": SHA,
        "jax_platform": "gpu",
        "jax_device": "NVIDIA GeForce RTX 3090",
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_server_metadata(metadata, profile, SHA)


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
