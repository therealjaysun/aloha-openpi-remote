import numpy as np
import pytest

from tools.remote_aloha.observation_contract import convert_gym_observation
from tools.remote_aloha.observation_contract import validate_policy_observation


def _gym_observation() -> dict:
    return {
        "pixels": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "left_wrist", "right_wrist")},
        "agent_pos": np.zeros(14, dtype=np.float64),
    }


def test_stock_gym_observation_is_converted_without_changing_state_dtype() -> None:
    raw = _gym_observation()
    raw["pixels"]["left_wrist"].fill(1)
    raw["pixels"]["right_wrist"].fill(2)
    converted = convert_gym_observation(raw, "Transfer cube")
    assert converted["state"].dtype == np.float64
    assert converted["images"]["cam_high"].shape == (3, 224, 224)
    assert converted["images"]["cam_high"].dtype == np.uint8
    assert set(converted["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert converted["images"]["cam_left_wrist"].max() == 1
    assert converted["images"]["cam_right_wrist"].max() == 2
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
        "images": {
            name: np.zeros((3, 224, 224), dtype=np.uint8) for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
    }
    mutate(value)
    with pytest.raises(ValueError, match="observation"):
        validate_policy_observation(value)


def test_invalid_raw_layout_is_rejected() -> None:
    value = _gym_observation()
    value["pixels"]["top"] = np.zeros((3, 224, 224), dtype=np.uint8)
    with pytest.raises(ValueError, match="Gym observation.pixels.top"):
        convert_gym_observation(value)


def test_missing_wrist_view_is_rejected() -> None:
    value = _gym_observation()
    del value["pixels"]["left_wrist"]
    with pytest.raises(ValueError, match="left_wrist"):
        convert_gym_observation(value)
