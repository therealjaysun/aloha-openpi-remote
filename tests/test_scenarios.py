import math

import numpy as np
import pytest

import tools.remote_aloha.scenarios as scenarios
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import BodyState
from tools.remote_aloha.scenarios import OutcomeState
from tools.remote_aloha.scenarios import Participation
from tools.remote_aloha.scenarios import advance_outcome
from tools.remote_aloha.scenarios import body_descriptors
from tools.remote_aloha.scenarios import layout_hash
from tools.remote_aloha.scenarios import project_action
from tools.remote_aloha.scenarios import sample_layout
from tools.remote_aloha.scenarios import scene_hash
from tools.remote_aloha.scenarios import update_participation


def _state(body, *, x=None, y=None, z=0.012, yaw=None, roll=0.0, com_z=None) -> BodyState:
    yaw = body.target_yaw if yaw is None else yaw
    cosine_roll, sine_roll = math.cos(roll / 2), math.sin(roll / 2)
    cosine_yaw, sine_yaw = math.cos(yaw / 2), math.sin(yaw / 2)
    return BodyState(
        body.name,
        body.target_x if x is None else x,
        body.target_y if y is None else y,
        z,
        cosine_roll * cosine_yaw,
        sine_roll * cosine_yaw,
        -sine_roll * sine_yaw,
        cosine_roll * sine_yaw,
        com_z,
    )


def test_fixed_scenario_contract_uses_uppercase_dotless_letters() -> None:
    assert tuple(SCENARIOS) == (
        "transfer_cube",
        "push_pi_single",
        "push_pi_dual",
        "push_letters_single",
        "push_letters_dual",
    )
    assert SCENARIOS["push_letters_single"].prompt == "Push the P and I blocks onto their matching targets."
    bodies = body_descriptors("letters")
    assert tuple(body.name for body in bodies) == ("P", "I")
    assert len(bodies[1].parts) == 3
    assert bodies[1].yaw_period == math.pi


def test_layout_is_deterministic_and_arm_mode_independent() -> None:
    for object_kind in ("pi", "letters"):
        for seed in range(3):
            first = sample_layout(object_kind, seed)
            assert first == sample_layout(object_kind, seed)
            assert layout_hash(first) == layout_hash(sample_layout(object_kind, seed))
    assert SCENARIOS["push_pi_single"].object_kind == SCENARIOS["push_pi_dual"].object_kind
    assert SCENARIOS["push_letters_single"].object_kind == SCENARIOS["push_letters_dual"].object_kind


def test_layout_rejects_bad_seed_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="seed"):
        sample_layout("pi", -1)
    monkeypatch.setattr(scenarios, "SPAWN_BOUNDS", (0.0, 0.01, 0.4, 0.41))
    with pytest.raises(ValueError, match="exhausted"):
        sample_layout("pi", 7, max_attempts=1)


def test_projection_preserves_stock_and_active_joints_exactly() -> None:
    action = np.arange(14, dtype=np.float64)
    home = np.arange(100, 114, dtype=np.float64)
    assert np.array_equal(project_action(action, SCENARIOS["transfer_cube"]), action)
    single = project_action(action, SCENARIOS["push_pi_single"], home)
    assert np.array_equal(single[:6], action[:6])
    assert np.array_equal(single[7:13], home[7:13])
    assert single[6] == single[13] == scenarios.PUSHER_POSITION
    dual = project_action(action, SCENARIOS["push_pi_dual"], home)
    assert np.array_equal(dual[:6], action[:6])
    assert np.array_equal(dual[7:13], action[7:13])
    assert dual[6] == dual[13] == scenarios.PUSHER_POSITION
    with pytest.raises(ValueError, match="14-vector"):
        project_action(np.zeros(13), SCENARIOS["push_pi_dual"], home)


def test_success_requires_five_held_steps_and_i_yaw_is_pi_symmetric() -> None:
    bodies = body_descriptors("letters")
    states = {body.name: _state(body, yaw=math.pi if body.name == "I" else 0.0) for body in bodies}
    rest = {body.name: 0.012 for body in bodies}
    outcome = OutcomeState()
    for index in range(5):
        outcome, metrics = advance_outcome(bodies, states, rest, outcome)
        assert outcome.success is (index == 4)
    assert metrics["body_1_yaw_error"] == pytest.approx(0.0)


def test_letters_reject_one_correct_swapped_lift_fall_and_off_table() -> None:
    bodies = body_descriptors("letters")
    rest = {body.name: 0.012 for body in bodies}
    one_correct = {"P": _state(bodies[0]), "I": _state(bodies[1], x=0.0, y=0.4)}
    outcome, _ = advance_outcome(bodies, one_correct, rest)
    assert not outcome.success
    assert outcome.held_steps == 0

    swapped = {
        "P": _state(bodies[0], x=bodies[1].target_x, y=bodies[1].target_y),
        "I": _state(bodies[1], x=bodies[0].target_x, y=bodies[0].target_y),
    }
    assert advance_outcome(bodies, swapped, rest)[0].held_steps == 0

    lifted = {body.name: _state(body, com_z=0.023) for body in bodies}
    lifted_outcome, _ = advance_outcome(bodies, lifted, rest)
    assert lifted_outcome.lifted_ever
    assert not lifted_outcome.success
    exact = {body.name: _state(body) for body in bodies}
    for _ in range(6):
        lifted_outcome, _ = advance_outcome(bodies, exact, rest, lifted_outcome)
    assert not lifted_outcome.success

    fallen = {"P": _state(bodies[0], roll=math.radians(31)), "I": _state(bodies[1])}
    assert advance_outcome(bodies, fallen, rest)[0].terminal_reason == "fallen"
    outside = {"P": _state(bodies[0], x=scenarios.TABLE_BOUNDS[1]), "I": _state(bodies[1])}
    assert advance_outcome(bodies, outside, rest)[0].terminal_reason == "off_table"


def test_contacts_are_order_independent_and_interference_is_same_step() -> None:
    geom_to_body = {"push_pi/P_0": "P"}
    left = "vx300s_left/10_left_gripper_finger"
    right = "vx300s_right/10_right_gripper_finger"
    state = update_participation([("push_pi/P_0", left)], geom_to_body)
    assert state == Participation(left_contact_ever=True)
    state = update_participation([(right, "push_pi/P_0"), (left, "push_pi/P_0")], geom_to_body, state)
    assert state.both_arms_participated
    assert state.interference_ever


def test_scene_hash_covers_xml_assets_and_kind() -> None:
    assets = {"a.xml": b"a", "mesh.stl": b"b"}
    baseline = scene_hash(b"<mujoco/>", assets, "pi")
    assert baseline == scene_hash(b"<mujoco/>", dict(reversed(list(assets.items()))), "pi")
    assert baseline != scene_hash(b"<mujoco/>", assets, "letters")
    assert baseline != scene_hash(b"<mujoco changed='1'/>", assets, "pi")
