import numpy as np
import pytest

from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.sim_smoke_test import percentile_ms
from tools.remote_aloha.sim_smoke_test import run
from tools.remote_aloha.sim_smoke_test import validate_action
from tools.remote_aloha.sim_smoke_test import validate_observation


def test_observation_and_action_contract() -> None:
    observation = {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.linspace(0, 1, 14, dtype=np.float64),
    }
    image, state = validate_observation(observation)
    validate_action(state.copy())
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    "observation",
    [
        {},
        {"pixels": {"top": np.zeros((224, 224, 3), dtype=np.uint8)}, "agent_pos": np.zeros(14)},
        {"pixels": {"top": np.zeros((480, 640, 3), dtype=np.float32)}, "agent_pos": np.zeros(14)},
        {"pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)}, "agent_pos": np.full(14, np.nan)},
    ],
)
def test_invalid_observation_is_rejected(observation: object) -> None:
    with pytest.raises(ValueError, match="observation|pixels.top|agent_pos"):
        validate_observation(observation)


def test_invalid_action_and_empty_latency_are_rejected() -> None:
    with pytest.raises(ValueError, match="action must"):
        validate_action(np.zeros(13))
    with pytest.raises(ValueError, match="latency samples"):
        percentile_ms([], 95)


def test_percentile_is_reported_in_milliseconds() -> None:
    assert percentile_ms([1.0, 2.0, 3.0], 50) == 2.0


def test_smoke_validates_output_root_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.sim_smoke_test.validate_output_root",
        lambda path: (_ for _ in ()).throw(ValueError("unsafe output")),
    )
    with pytest.raises(ValueError, match="unsafe output"):
        run(MacSimConfig(), enforce_budget=False)
