import numpy as np
import pytest

from tools.remote_aloha.observation_contract import convert_gym_observation
from tools.remote_aloha.observation_contract import validate_policy_observation


def _gym_observation() -> dict:
    return {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros(14, dtype=np.float64),
    }


def test_stock_gym_observation_is_converted_without_changing_state_dtype() -> None:
    converted = convert_gym_observation(_gym_observation(), "Transfer cube")
    assert converted["state"].dtype == np.float64
    assert converted["images"]["cam_high"].shape == (3, 224, 224)
    assert converted["images"]["cam_high"].dtype == np.uint8
    assert converted["prompt"] == "Transfer cube"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=1),
        lambda value: value["state"].__setitem__(0, np.nan),
        lambda value: value.__setitem__("state", value["state"].astype(np.float32)),
        lambda value: value["images"].__setitem__("unknown", value["images"]["cam_high"]),
        lambda value: value["images"].__setitem__("cam_high", np.zeros((224, 224, 3), dtype=np.uint8)),
        lambda value: value.__setitem__("prompt", ""),
    ],
)
def test_invalid_policy_observation_is_rejected(mutate) -> None:
    value = {
        "state": np.zeros(14, dtype=np.float64),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
    }
    mutate(value)
    with pytest.raises(ValueError, match="observation"):
        validate_policy_observation(value)


def test_invalid_raw_layout_is_rejected() -> None:
    value = _gym_observation()
    value["pixels"]["top"] = np.zeros((3, 224, 224), dtype=np.uint8)
    with pytest.raises(ValueError, match="Gym observation.pixels.top"):
        convert_gym_observation(value)
