import numpy as np
import pytest

from tools.remote_aloha.config import MacSimConfig
import tools.remote_aloha.scenarios as scenarios
from tools.remote_aloha.sim_smoke_test import _color_mask
from tools.remote_aloha.sim_smoke_test import _git_sha
from tools.remote_aloha.sim_smoke_test import calibration_commands
from tools.remote_aloha.sim_smoke_test import calibration_contact_seen
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


def test_calibration_commands_are_one_exact_episode() -> None:
    home = np.arange(14, dtype=np.float64)
    left = calibration_commands(home, "left")
    right = calibration_commands(home, "right")
    assert len(left) == len(right) == 300
    assert all(command.shape == (14,) and np.isfinite(command).all() for command in left + right)
    assert all(command[6] == command[13] == 0.5 for command in left + right)
    assert np.allclose(left[-1][:6], scenarios.LEFT_PUSH_WAYPOINTS[-1], atol=1e-15)
    assert np.array_equal(left[-1][7:13], home[7:13])
    assert np.allclose(right[-1][7:13], scenarios.RIGHT_PUSH_WAYPOINTS[-1], atol=1e-15)
    assert np.array_equal(right[-1][:6], home[:6])


def test_calibration_masks_and_named_contacts_are_exact() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (242, 140, 20)
    frame[0, 1] = (217, 31, 31)
    frame[0, 2] = (26, 89, 230)
    assert _color_mask(frame, "pi")[0, 0]
    assert _color_mask(frame, "P")[0, 1]
    assert _color_mask(frame, "I")[0, 2]
    assert calibration_contact_seen([("vx300s_left/10_left_gripper_finger", "push_pi/P_2")], "P", "left")
    assert not calibration_contact_seen([("vx300s_right/not_a_finger", "push_pi/P_2")], "P", "right")


def test_release_calibration_rejects_dirty_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    responses = iter((Result("a" * 40 + "\n"), Result(" M changed.py\n")))
    monkeypatch.setattr("tools.remote_aloha.sim_smoke_test.subprocess.run", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="clean exact-candidate"):
        _git_sha(require_clean=True)


def test_smoke_validates_output_root_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.sim_smoke_test.validate_output_root",
        lambda path: (_ for _ in ()).throw(ValueError("unsafe output")),
    )
    with pytest.raises(ValueError, match="unsafe output"):
        run(MacSimConfig(), enforce_budget=False)
